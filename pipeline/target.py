"""Built-in genre target curve + reference-free matching EQ.

A real "AI master" doesn't apply the same EQ to every song — it analyses the
mix's tonal balance and nudges it toward a target curve. This module measures
the mix's average spectrum and applies a smooth, bounded, linear-phase
correction toward a hip-hop / trap / R&B target. It powers Fully-automatic mode
and the out-of-box Genre mode (when no reference WAV is supplied), so neither
needs a copyrighted reference track to sound on-genre.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from . import dsp

# Relative target tilt in dB at anchor frequencies (only the *shape* matters;
# it is mean-normalised before use). Solid but controlled lows, a gentle
# low-mid scoop to clear mud, a tamed 4-8 kHz presence region for "smooth highs",
# and a touch of air up top.
HIPHOP_TARGET: list[tuple[float, float]] = [
    (30, 1.0), (50, 2.0), (80, 1.6), (120, 0.9), (200, -0.4), (300, -1.2),
    (500, -1.0), (800, -0.4), (1000, 0.0), (2000, -0.3), (3500, -1.2),
    (5000, -2.0), (7000, -2.2), (9000, -1.6), (12000, -0.8), (16000, -0.3),
    (20000, -1.2),
]

_REF_LO, _REF_HI = 300.0, 5000.0  # band both curves are aligned on (mean = 0)


def _avg_spectrum_db(x: np.ndarray, sr: int, nfft: int = 8192):
    mono = dsp.to_mono(x)[:, 0]
    nper = min(nfft, len(mono))
    if nper < 256:
        return None, None
    f, psd = signal.welch(mono, sr, nperseg=nper, noverlap=nper // 2, scaling="spectrum")
    psd = np.maximum(psd, 1e-12)
    return f, 10.0 * np.log10(psd)


def _interp_target(freqs: np.ndarray, anchors) -> np.ndarray:
    af = np.array([a[0] for a in anchors], dtype=float)
    ad = np.array([a[1] for a in anchors], dtype=float)
    lf = np.log10(np.clip(freqs, 1.0, None))
    return np.interp(lf, np.log10(af), ad, left=ad[0], right=ad[-1])


def _mean_in_band(f: np.ndarray, vals: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f <= hi)
    return float(vals[m].mean()) if m.any() else float(vals.mean())


def matching_eq(x: np.ndarray, sr: int, anchors=HIPHOP_TARGET,
                strength: float = 0.7, max_db: float = 4.0,
                numtaps: int = 4097) -> np.ndarray:
    """Nudge ``x`` toward the target tonal curve with a smooth linear-phase FIR.

    ``strength`` 0..1 scales how far toward the target we move; ``max_db`` caps
    the correction per band so it stays corrective, never destructive.
    """
    x2 = dsp.to_stereo(x)
    f, mix_db = _avg_spectrum_db(x2, sr)
    if f is None:
        return x2

    # Smooth the measured spectrum so we correct broad balance, not fine ripples.
    mix_db = uniform_filter1d(mix_db, size=9, mode="nearest")
    tgt_db = _interp_target(f, anchors)

    # Align both on the mid reference band, then derive a bounded correction.
    mix_db = mix_db - _mean_in_band(f, mix_db, _REF_LO, _REF_HI)
    tgt_db = tgt_db - _mean_in_band(f, tgt_db, _REF_LO, _REF_HI)
    corr_db = np.clip((tgt_db - mix_db) * float(strength), -max_db, max_db)
    corr_db = uniform_filter1d(corr_db, size=5, mode="nearest")

    # Design a linear-phase FIR from the correction and apply it.
    fn = np.clip(f / (sr * 0.5), 0.0, 1.0)
    fn, gains = _prep_fir_points(fn, 10.0 ** (corr_db / 20.0))
    if numtaps % 2 == 0:
        numtaps += 1  # Type-I (odd) so Nyquist gain is unconstrained
    fir = signal.firwin2(numtaps, fn, gains)

    delay = numtaps // 2
    n = x2.shape[0]
    out = np.empty_like(x2)
    for c in range(x2.shape[1]):
        y = signal.fftconvolve(x2[:, c], fir, mode="full")
        out[:, c] = y[delay:delay + n]
    return out.astype(np.float32)


def _prep_fir_points(fn: np.ndarray, gains: np.ndarray):
    """firwin2 needs strictly increasing freqs spanning exactly [0, 1]."""
    fn = fn.astype(float).copy()
    gains = gains.astype(float).copy()
    fn[0] = 0.0
    fn[-1] = 1.0
    # enforce strict monotonicity (welch bins are already sorted/unique, but be safe)
    keep = np.concatenate(([True], np.diff(fn) > 0))
    fn, gains = fn[keep], gains[keep]
    gains = np.clip(np.nan_to_num(gains, nan=1.0), 0.05, 20.0)
    return fn, gains
