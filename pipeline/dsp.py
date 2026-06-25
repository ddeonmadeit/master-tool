"""Shared DSP utilities.

Internal audio convention everywhere in the pipeline:
    float32 numpy array, shape (n_samples, n_channels), value range ~[-1, 1].
Mono is kept as (n_samples, 1). Sample rate is always passed alongside.

Only convert to pedalboard's (n_channels, n_samples) layout at the call site.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Shape / unit helpers
# --------------------------------------------------------------------------- #
def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Return audio as float32 (n_samples, n_channels)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    return x


def to_mono(x: np.ndarray) -> np.ndarray:
    """Collapse to mono (n_samples, 1)."""
    x = ensure_2d(x)
    if x.shape[1] == 1:
        return x
    return x.mean(axis=1, keepdims=True).astype(np.float32)


def to_stereo(x: np.ndarray) -> np.ndarray:
    """Ensure stereo (n_samples, 2) by duplicating mono."""
    x = ensure_2d(x)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.shape[1] > 2:
        return x[:, :2]
    return x


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(abs(x), EPS)))


def peak_db(x: np.ndarray) -> float:
    return lin_to_db(float(np.max(np.abs(x))) if x.size else EPS)


def match_length(a: np.ndarray, b: np.ndarray):
    """Zero-pad the shorter of two (n, ch) arrays so both share the same length."""
    a, b = ensure_2d(a), ensure_2d(b)
    n = max(a.shape[0], b.shape[0])
    if a.shape[0] < n:
        a = np.pad(a, ((0, n - a.shape[0]), (0, 0)))
    if b.shape[0] < n:
        b = np.pad(b, ((0, n - b.shape[0]), (0, 0)))
    return a, b


# --------------------------------------------------------------------------- #
# Mid / Side
# --------------------------------------------------------------------------- #
def lr_to_ms(x: np.ndarray):
    """Stereo (n,2) -> (mid (n,), side (n,))."""
    x = to_stereo(x)
    l, r = x[:, 0], x[:, 1]
    mid = (l + r) * 0.5
    side = (l - r) * 0.5
    return mid.astype(np.float32), side.astype(np.float32)


def ms_to_lr(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """(mid, side) -> stereo (n,2)."""
    l = mid + side
    r = mid - side
    return np.stack([l, r], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Filters / crossovers
# --------------------------------------------------------------------------- #
def _sos(btype: str, cutoff, sr: int, order: int = 2):
    wn = np.asarray(cutoff, dtype=float) / (sr * 0.5)
    wn = np.clip(wn, 1e-5, 0.999999)
    return signal.butter(order, wn, btype=btype, output="sos")


def _apply_sos(sos, x: np.ndarray) -> np.ndarray:
    """Causal SOS filter over a (n, ch) array (all channels in one C call)."""
    x = ensure_2d(x)
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def highpass(x: np.ndarray, sr: int, freq: float, order: int = 2) -> np.ndarray:
    return _apply_sos(_sos("highpass", freq, sr, order), x)


def lowpass(x: np.ndarray, sr: int, freq: float, order: int = 2) -> np.ndarray:
    return _apply_sos(_sos("lowpass", freq, sr, order), x)


def bandpass(x: np.ndarray, sr: int, low: float, high: float, order: int = 2) -> np.ndarray:
    return _apply_sos(_sos("bandpass", [low, high], sr, order), x)


def lr_crossover(x: np.ndarray, sr: int, freq: float):
    """Linkwitz-Riley 4th-order split -> (low, high). low + high reconstructs flat.

    Implemented as two cascaded 2nd-order Butterworth sections (LR4).
    """
    x = ensure_2d(x)
    lp = _sos("lowpass", freq, sr, order=2)
    hp = _sos("highpass", freq, sr, order=2)
    low = _apply_sos(lp, _apply_sos(lp, x))
    high = _apply_sos(hp, _apply_sos(hp, x))
    return low, high


def split_band(x: np.ndarray, sr: int, low_hz: float, high_hz: float):
    """Phase-coherent 3-way split returning (band, rest) where ``band`` is the
    content in [low_hz, high_hz] and band + rest reconstructs the input flat.

    Uses LR crossovers (not bandpass-and-subtract), so processing only the band
    and recombining stays phase-clean — important for the de-esser and the
    master harsh-tame bands feeding the "smooth highs" goal.
    """
    below_high, above_high = lr_crossover(x, sr, high_hz)
    below_low, band = lr_crossover(below_high, sr, low_hz)
    rest = (below_low + above_high).astype(np.float32)
    return band.astype(np.float32), rest


# --------------------------------------------------------------------------- #
# Envelopes / dynamics
# --------------------------------------------------------------------------- #
def _onepole(x: np.ndarray, sr: int, time_ms: float) -> np.ndarray:
    """Vectorized one-pole low-pass (leaky integrator) with the given time const."""
    a = float(np.exp(-1.0 / (max(time_ms, 1e-3) * 1e-3 * sr)))
    return signal.lfilter([1.0 - a], [1.0, -a], x).astype(np.float32)


def envelope_follower(x: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Fast attack / slow release peak envelope — fully vectorized.

    Approximates an asymmetric one-pole detector as ``max(fast, slow)`` of two
    one-pole smoothers: on a rising edge the fast pole tracks (quick attack); on
    a falling edge it drops below the slow pole, which holds (slow release).
    Replaces the old per-sample Python loop (orders of magnitude faster).
    """
    x = ensure_2d(x)
    rect = np.max(np.abs(x), axis=1).astype(np.float32)
    fast = _onepole(rect, sr, attack_ms)
    slow = _onepole(rect, sr, release_ms)
    return np.maximum(fast, slow).astype(np.float32)


def dynamic_band_reduction(
    x: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
    threshold_db: float = -28.0,
    ratio: float = 4.0,
    max_reduction_db: float = 3.0,
    attack_ms: float = 1.0,
    release_ms: float = 60.0,
) -> np.ndarray:
    """Compress only the energy inside [low_hz, high_hz]; leave the rest untouched.

    Used for the de-esser (6-8 kHz on the vocal) and the master harsh-tame bands.
    Returns audio of the same shape. Reduction is clamped to ``max_reduction_db``.
    """
    x = ensure_2d(x)
    band, rest = split_band(x, sr, low_hz, high_hz)

    env = envelope_follower(band, sr, attack_ms, release_ms)
    env_db = 20.0 * np.log10(np.maximum(env, EPS))
    over = np.maximum(env_db - threshold_db, 0.0)
    gain_red_db = over * (1.0 - 1.0 / max(ratio, 1.0))
    gain_red_db = np.minimum(gain_red_db, max_reduction_db)
    gain = (10.0 ** (-gain_red_db / 20.0)).astype(np.float32)[:, None]

    return (rest + band * gain).astype(np.float32)


def sidechain_band_duck(
    target: np.ndarray,
    sr: int,
    sidechain: np.ndarray,
    low_hz: float,
    high_hz: float,
    max_duck_db: float = 2.0,
    threshold_db: float = -30.0,
    attack_ms: float = 5.0,
    release_ms: float = 120.0,
) -> np.ndarray:
    """Dip the [low_hz, high_hz] band of ``target`` while ``sidechain`` is active.

    Carves pocket for the vocal: the instrumental's 1-4 kHz band ducks ~1-2 dB
    only when the vocal is present, so the vocal sits "in" the beat without
    obvious pumping.
    """
    target = ensure_2d(target)
    band, rest = split_band(target, sr, low_hz, high_hz)

    env = envelope_follower(to_mono(sidechain), sr, attack_ms, release_ms)
    env_db = 20.0 * np.log10(np.maximum(env, EPS))
    activity = np.clip((env_db - threshold_db) / 12.0, 0.0, 1.0)  # 0..1 ramp
    duck_db = activity * max_duck_db
    gain = (10.0 ** (-duck_db / 20.0)).astype(np.float32)[:, None]

    return (rest + band * gain).astype(np.float32)


# --------------------------------------------------------------------------- #
# Saturation / warmth
# --------------------------------------------------------------------------- #
def soft_clip_tanh(x: np.ndarray, drive: float, sr: int, oversample: int = 4) -> np.ndarray:
    """Oversampled tanh waveshaper for subtle analog-style warmth.

    ``drive`` ~0 is clean; ~1 is gentle warmth; >2 is obvious. Oversampling
    keeps harmonics from aliasing back into the audible band.
    """
    if drive <= 1e-4:
        return ensure_2d(x)
    x = ensure_2d(x)
    k = 1.0 + 4.0 * drive  # waveshaper hardness
    up = signal.resample_poly(x, oversample, 1, axis=0)
    shaped = np.tanh(up * k) / np.tanh(k)
    down = signal.resample_poly(shaped, 1, oversample, axis=0)
    # length can drift by a sample after up/down; clamp back
    if down.shape[0] != x.shape[0]:
        down = _fit_length(down, x.shape[0])
    # blend a touch of dry to keep it subtle
    mix = np.clip(0.5 + 0.5 * min(drive, 1.0), 0.5, 1.0)
    return (mix * down + (1.0 - mix) * x).astype(np.float32)


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[0] >= n:
        return x[:n]
    return np.pad(x, ((0, n - x.shape[0]), (0, 0)))


# --------------------------------------------------------------------------- #
# True peak
# --------------------------------------------------------------------------- #
def true_peak_db(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Estimate inter-sample (true) peak in dBTP via oversampling."""
    x = ensure_2d(x)
    up = signal.resample_poly(x, oversample, 1, axis=0)
    return lin_to_db(float(np.max(np.abs(up))) if up.size else EPS)


def lookahead_limiter(x: np.ndarray, sr: int, ceiling_db: float,
                      lookahead_ms: float = 1.2, release_ms: float = 80.0) -> np.ndarray:
    """Brick-wall limiter with anticipatory (lookahead) gain reduction.

    Computes the per-sample gain needed to keep the peak under ``ceiling_db``,
    takes a sliding minimum over the lookahead window so the gain is already
    pulled down as a transient arrives, then smooths it with a fast attack /
    slow release envelope so reduction ramps in cleanly and recovers smoothly.
    A final hard clip at the ceiling guarantees the brick wall.

    Run this in an oversampled domain to catch inter-sample peaks.
    """
    x = to_stereo(x)
    ceiling = db_to_lin(ceiling_db)

    peak = np.max(np.abs(x), axis=1).astype(np.float32)
    desired = np.ones_like(peak)
    hot = peak > ceiling
    desired[hot] = ceiling / peak[hot]

    # Lookahead: a centred sliding-min sees the upcoming transient, so the gain
    # is already reduced before the peak lands (no overshoot to clip away).
    look = max(1, int(lookahead_ms * 1e-3 * sr))
    g = minimum_filter1d(desired, size=2 * look + 1, mode="nearest")

    # Smooth the *reduction* (1 - g): fast attack (reduction rises quickly),
    # slow release (reduction falls slowly) — reuse the asymmetric follower.
    reduction = np.clip(1.0 - g, 0.0, 1.0)
    reduction = envelope_follower(reduction, sr, attack_ms=0.2, release_ms=release_ms)
    g = (1.0 - reduction).astype(np.float32)

    out = x * g[:, None]
    np.clip(out, -ceiling, ceiling, out=out)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Dither
# --------------------------------------------------------------------------- #
def tpdf_dither(x: np.ndarray, bit_depth: int = 24, seed: int | None = None) -> np.ndarray:
    """Add ~1 LSB triangular (TPDF) dither for the target bit depth."""
    x = ensure_2d(x)
    rng = np.random.default_rng(seed)
    lsb = 2.0 ** -(bit_depth - 1)
    # triangular = sum of two independent uniforms in [-0.5, 0.5] LSB
    noise = (rng.random(x.shape) - rng.random(x.shape)).astype(np.float32) * lsb
    return (x + noise).astype(np.float32)
