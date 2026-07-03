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

from . import dsp, sounds, target
from .ingest import Audio, write_wav, decode
from .settings import Settings

REFERENCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "references", "hiphop")


# --------------------------------------------------------------------------- #
# Stereo width (mid/side, lows forced mono)
# --------------------------------------------------------------------------- #
def apply_width(x: np.ndarray, sr: int, s: Settings,
                width_override: float | None = None) -> np.ndarray:
    """Widen the side channel above ~300 Hz; force mono below ~120 Hz.

    Keeps correlation ≥ 0 by backing off the side gain if it goes out of phase.
    ``width_override`` lets the active Sound scale the user's width setting.
    """
    width = s.width if width_override is None else width_override
    x = dsp.to_stereo(x)
    mid, side = dsp.lr_to_ms(x)
    side2d = side[:, None]

    # Force mono below the crossover: remove all low energy from the side.
    side2d = dsp.highpass(side2d, sr, s.mono_below_hz, order=2)

    # Split remaining side at the width crossover; widen only the upper part.
    side_low, side_high = dsp.lr_crossover(side2d, sr, s.width_above_hz)
    width_gain = 1.0 + 0.8 * width              # 1.0 (subtle) .. 1.8 (wide)
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
    prof = sounds.get_sound(s.genre_sound)   # voiced Sound (Trap / Boom-bap / Melodic)

    # 1. Clarity EQ: kill rumble + a small mud dip.
    x = dsp.highpass(x, sr, 26.0, order=2)
    x = pb.Pedalboard([pb.PeakFilter(cutoff_frequency_hz=300.0, gain_db=-0.8, q=1.0)])(x, sr)

    # 1b. Corrective minimum-phase tonal match toward the active Sound's target
    #     curve. Only when there is no reference (auto / out-of-box genre); a
    #     Matchering reference already defines tone, so we don't fight it here.
    if do_target_match:
        x = target.matching_eq(x, sr, anchors=prof.target, strength=0.65, max_db=3.5)

    # 2. Multiband glue: tighten the sub/808 for punch, glue mids/highs for
    #    cohesion and density.
    x = dsp.multiband_compress(x, sr)

    # 2b. Restore the attack the compression rounded off — a brief, differential
    #     lift on the first milliseconds of each hit so the kick/snare still
    #     cracks through a dense master (studio transient-shaper move).
    x = dsp.transient_enhance(x, sr, amount=0.3, max_boost_db=2.0)

    # 3. HOUSE CHARACTER (under every Sound) — Kanye essence, run tape-style:
    #    lows/mids take the full even-harmonic saturation (thick, warm), highs
    #    are driven far gentler so the top stays silky (tape self-erasure). Plus
    #    a tape-machine head bump (~60 Hz), a soulful low-mid body, and a full,
    #    weighty low-shelf. The Sound scales the drive and low weight.
    drive = 0.5 * s.warmth * prof.sat_mult
    x = dsp.tape_saturate(x, sr, drive=drive, oversample=4)
    x = pb.Pedalboard([
        pb.PeakFilter(cutoff_frequency_hz=62.0, gain_db=0.5 + 0.6 * drive, q=1.1),  # head bump
        pb.LowShelfFilter(cutoff_frequency_hz=90.0,
                          gain_db=1.2 + 1.0 * s.warmth + 0.7 * prof.low_weight_db, q=0.7),
        pb.PeakFilter(cutoff_frequency_hz=220.0, gain_db=1.0, q=0.9),   # Kanye low-mid body
    ])(x, sr)

    # 4. Open the top per Sound: tame harsh peaks, then presence ("cut") + air +
    #    a light exciter. The +0.4 dB air baseline is the untiljapan house layer
    #    (a smooth, open top under everything).
    x = dsp.dynamic_band_reduction(x, sr, 5000.0, 9000.0,
                                   threshold_db=-16.0, ratio=2.0,
                                   max_reduction_db=2.0, attack_ms=1.0, release_ms=80.0)
    x = pb.Pedalboard([
        pb.PeakFilter(cutoff_frequency_hz=2800.0, gain_db=prof.presence_db, q=0.6),
        pb.HighShelfFilter(cutoff_frequency_hz=11000.0, gain_db=prof.air_db + 0.4, q=0.6),
    ])(x, sr)
    x = dsp.hf_exciter(x, sr, freq=9500.0, amount=0.12)

    # 5. Stereo width per Sound (+0.05 untiljapan width baseline); lows stay mono.
    eff_width = float(np.clip(s.width * prof.width_mult + 0.05, 0.0, 1.0))
    x = apply_width(x, sr, s, width_override=eff_width)

    # 6. Bus glue so the master "breathes as one" — custom console-style comp
    #    with soft knee + program-dependent (auto) release, the SSL-bus trait
    #    that reads as "glued, not squashed". Gentler / more open for the more
    #    dynamic Sounds (boom-bap), denser for trap & melodic.
    glue_thresh = -14.0 + (prof.dynamic - 1.0) * 16.0
    glue_ratio = max(1.3, 1.8 - (prof.dynamic - 1.0) * 1.6)
    x = dsp.bus_glue(x, sr, threshold_db=glue_thresh, ratio=glue_ratio)

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
            "matched_to_reference": matched_to_reference,
            "sound": s.genre_sound if not matched_to_reference else None}
    return Audio(dsp.to_stereo(x), sr), info, notices
