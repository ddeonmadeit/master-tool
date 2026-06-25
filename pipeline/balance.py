"""Stage 2 — Balance the vocal against the beat (the heart of the mix).

Loudness-match the vocal to a target offset over the instrumental, then carve a
1-4 kHz pocket in the beat under the vocal so it sits "in" the track.
"""
from __future__ import annotations

import numpy as np

from . import dsp
from .ingest import Audio
from .meters import integrated_lufs
from .settings import Settings


def balance(vocal: Audio, instrumental: Audio, s: Settings):
    """Return (vocal_gained: Audio mono, instrumental_ducked: Audio stereo, info)."""
    sr = vocal.sr
    voc = dsp.to_mono(vocal.data)
    instr = dsp.to_stereo(instrumental.data)
    voc, instr = dsp.match_length(voc, instr)

    voc_lufs = integrated_lufs(voc, sr)
    instr_lufs = integrated_lufs(instr, sr)

    # Target: vocal ≈ instrumental + offset (default +1 LU), shifted by slider.
    target = instr_lufs + s.vocal_balance_offset_lu + s.vocal_level_db
    gain_db = 0.0
    if np.isfinite(voc_lufs) and np.isfinite(instr_lufs):
        gain_db = target - voc_lufs
        voc = (voc * dsp.db_to_lin(gain_db)).astype(np.float32)

    # Subtle ducking (default ON): dip the beat 1-2 dB @ 1-4 kHz under the vocal.
    if s.ducking and np.isfinite(voc_lufs):
        instr = dsp.sidechain_band_duck(
            instr, sr, sidechain=voc,
            low_hz=1000.0, high_hz=4000.0,
            max_duck_db=2.0, threshold_db=-32.0,
            attack_ms=5.0, release_ms=140.0,
        )

    info = {
        "vocal_lufs": _finite(voc_lufs),
        "instrumental_lufs": _finite(instr_lufs),
        "vocal_gain_db": round(gain_db, 2),
        "vocal_target_lufs": round(target, 2) if np.isfinite(target) else None,
    }
    return Audio(dsp.to_mono(voc), sr), Audio(dsp.to_stereo(instr), sr), info


def _finite(v: float):
    return round(float(v), 2) if np.isfinite(v) else None
