"""Stage 4 — Master: three modes + the signature chain.

Modes:
  - "genre":        Matchering against a bundled-folder reference (default).
  - "my_reference": Matchering against a user-dropped reference.
  - "auto":         no reference; the signature chain *is* the master.

In reference modes Matchering gets tone/width/loudness into the ballpark, then
the signature chain runs as finishing colour. The mandatory loudness + true-peak
stage lives in loudness.py and always runs afterwards.
"""
from __future__ import annotations

import glob
import os
import tempfile

import numpy as np
import pedalboard as pb

from . import dsp, target
from .ingest import Audio, write_wav, decode
from .settings import Settings

REFERENCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "references", "hiphop")


# --------------------------------------------------------------------------- #
# Stereo width (mid/side, lows forced mono)
# --------------------------------------------------------------------------- #
def apply_width(x: np.ndarray, sr: int, s: Settings) -> np.ndarray:
    """Widen the side channel above ~300 Hz; force mono below ~120 Hz.

    Keeps correlation ≥ 0 by backing off the side gain if it goes out of phase.
    """
    x = dsp.to_stereo(x)
    mid, side = dsp.lr_to_ms(x)
    side2d = side[:, None]

    # Force mono below the crossover: remove all low energy from the side.
    side2d = dsp.highpass(side2d, sr, s.mono_below_hz, order=2)

    # Split remaining side at the width crossover; widen only the upper part.
    side_low, side_high = dsp.lr_crossover(side2d, sr, s.width_above_hz)
    width_gain = 1.0 + 0.8 * s.width            # 1.0 (subtle) .. 1.8 (wide)
    side2d = side_low + side_high * width_gain

    out = dsp.ms_to_lr(mid, side2d[:, 0])

    # Safety: never let the image go out of phase (correlation ≥ 0).
    l, r = out[:, 0], out[:, 1]
    if np.std(l) > 1e-9 and np.std(r) > 1e-9:
        corr = float(np.corrcoef(l, r)[0, 1])
        if corr < 0.0:
            side2d *= 0.6
            out = dsp.ms_to_lr(mid, side2d[:, 0])
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Signature chain (the character)
# --------------------------------------------------------------------------- #
def signature_chain(x: np.ndarray, sr: int, s: Settings,
                    do_target_match: bool = False) -> np.ndarray:
    x = dsp.to_stereo(x)

    # 1. Clarity EQ: kill rumble + a small mud dip.
    x = dsp.highpass(x, sr, 26.0, order=2)
    x = pb.Pedalboard([pb.PeakFilter(cutoff_frequency_hz=300.0, gain_db=-0.8, q=1.0)])(x, sr)

    # 1b. Corrective minimum-phase tonal match toward the genre target curve.
    #     Only when there is no reference (auto / out-of-box genre); a Matchering
    #     reference already defines tone, so we don't fight it here.
    if do_target_match:
        x = target.matching_eq(x, sr, strength=0.5, max_db=3.0)

    # 2. Multiband glue (the "finished record" stage): tighten the sub/808 band
    #    for punch, gently glue the mids and highs for cohesion and density. This
    #    replaces the old single bus compressor — banded control is what makes a
    #    master sound even and professional instead of raw.
    x = dsp.multiband_compress(x, sr)

    # 3. Warmth + weight: asymmetric analog saturation (even-harmonic warmth, not
    #    just edgy odd harmonics) and a low-shelf for a full, slightly bass-heavy
    #    bottom. A little low-mid body too, so it reads "full" rather than thin.
    x = dsp.analog_saturate(x, drive=0.5 * s.warmth, sr=sr, oversample=4, asym=0.35)
    x = pb.Pedalboard([
        pb.LowShelfFilter(cutoff_frequency_hz=90.0, gain_db=1.6 + 1.4 * s.warmth, q=0.7),
        pb.PeakFilter(cutoff_frequency_hz=180.0, gain_db=0.8, q=0.9),     # low-mid body
    ])(x, sr)

    # 4. Open the top *gently*. Tame only genuinely harsh peaks, then a small
    #    presence lift, a modest air shelf, and a light exciter — kept subtle so
    #    it stays warm and clean (no high-frequency grit / harshness).
    x = dsp.dynamic_band_reduction(x, sr, 5000.0, 9000.0,
                                   threshold_db=-16.0, ratio=2.0,
                                   max_reduction_db=2.0, attack_ms=1.0, release_ms=80.0)
    x = pb.Pedalboard([
        # Broad upper-mid presence (the ear's most sensitive region): this is what
        # makes a master "cut" and feel loud. Warm lows + present upper-mids is the
        # classic loudness curve — full AND forward, not dull.
        pb.PeakFilter(cutoff_frequency_hz=2800.0, gain_db=1.3, q=0.6),
        pb.HighShelfFilter(cutoff_frequency_hz=11000.0, gain_db=1.6, q=0.6),  # air
    ])(x, sr)
    x = dsp.hf_exciter(x, sr, freq=9500.0, amount=0.12)

    # 5. Stereo width (mid/side; lows mono so the bass stays centred and punchy).
    x = apply_width(x, sr, s)

    # 6. Bus glue: one gentle, slow full-band compressor over the whole mix so it
    #    "breathes as one" — this is the cohesion ("all one") that multiband
    #    banding alone doesn't give. ~1-2 dB, soft, musical.
    x = pb.Pedalboard([
        pb.Compressor(threshold_db=-14.0, ratio=1.8, attack_ms=30.0, release_ms=260.0),
    ])(x, sr)

    return dsp.to_stereo(x)


# --------------------------------------------------------------------------- #
# Matchering reference modes
# --------------------------------------------------------------------------- #
def _find_genre_reference() -> str | None:
    if not os.path.isdir(REFERENCES_DIR):
        return None
    refs = []
    for ext in ("*.wav", "*.aiff", "*.aif", "*.flac"):
        refs.extend(glob.glob(os.path.join(REFERENCES_DIR, ext)))
    return sorted(refs)[0] if refs else None


def _run_matchering(bus: Audio, reference_path: str) -> np.ndarray:
    """Match ``bus`` toward ``reference_path``; returns matched audio at bus.sr."""
    import matchering as mg

    with tempfile.TemporaryDirectory() as td:
        target_wav = os.path.join(td, "target.wav")
        out_wav = os.path.join(td, "matched.wav")
        write_wav(target_wav, bus.data, bus.sr, subtype="FLOAT")
        mg.process(
            target=target_wav,
            reference=reference_path,
            results=[mg.pcm24(out_wav)],
        )
        matched = decode(out_wav)
    if matched.sr != bus.sr:
        from .ingest import resample
        matched = resample(matched, bus.sr)
    return dsp.to_stereo(matched.data)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def master(bus: Audio, s: Settings, reference_path: str | None = None):
    """Returns (colored: Audio, info: dict, notices: list[str]).

    Output is pre-loudness — loudness.finalize() applies the target LUFS + true-peak.
    """
    sr = bus.sr
    notices: list[str] = []
    x = bus.data
    mode = s.mode

    matched_to_reference = False
    if mode in ("genre", "my_reference"):
        ref = reference_path if mode == "my_reference" else _find_genre_reference()
        if not ref or not os.path.isfile(ref):
            if mode == "genre":
                notices.append(
                    "No reference WAV in references/hiphop/ — using the built-in "
                    "hip-hop genre target curve. Drop a commercially-mastered "
                    "reference there to match a specific track instead."
                )
            else:
                notices.append(
                    "No reference provided — using the built-in genre target curve."
                )
            mode = "auto"
        else:
            try:
                x = _run_matchering(bus, ref)
                matched_to_reference = True
            except Exception as exc:  # matchering can reject short/odd files
                notices.append(f"Matchering failed ({exc}); used the genre target curve instead.")
                mode = "auto"

    # Signature chain runs in every mode. The corrective target-match runs only
    # when no reference set the tone (auto / out-of-box genre).
    x = signature_chain(x, sr, s, do_target_match=not matched_to_reference)

    info = {"mode_used": mode, "requested_mode": s.mode,
            "matched_to_reference": matched_to_reference}
    return Audio(dsp.to_stereo(x), sr), info, notices
