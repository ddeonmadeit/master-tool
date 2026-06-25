"""Stage 1 — Vocal conditioning (LIGHT: the vocal is already mixed).

The win here is corrective only: clear sub-mud, gently tame sibilance, hold the
level steady. We do NOT re-EQ the vocal from scratch.
"""
from __future__ import annotations

import numpy as np
import pedalboard as pb

from . import dsp
from .ingest import Audio
from .settings import Settings


def condition(vocal: Audio, s: Settings) -> Audio:
    x = dsp.to_mono(vocal.data)
    sr = vocal.sr

    # 1. High-pass ~90 Hz, 12 dB/oct — clear sub-mud that clashes with 808/kick.
    x = dsp.highpass(x, sr, s.vocal_hpf_hz, order=2)

    # 2. De-esser (default ON, gentle): dynamic reduction 6-8 kHz. Higher
    #    threshold + lower max so it only catches real sibilance, not the whole
    #    top end of a bright vocal.
    if s.deesser:
        x = dsp.dynamic_band_reduction(
            x, sr, low_hz=6000.0, high_hz=8500.0,
            threshold_db=-26.0, ratio=3.0, max_reduction_db=2.5,
            attack_ms=0.5, release_ms=50.0,
        )

    # 3. Leveling: low-ratio compressor (≤2:1, ≤~3 dB GR) for consistency.
    board = pb.Pedalboard([
        pb.Compressor(threshold_db=-20.0, ratio=2.0, attack_ms=10.0, release_ms=150.0),
    ])
    x = board(x, sr)

    # 4. Optional "glue reverb" — very short plate, low mix. OFF by default.
    if s.glue_reverb:
        verb = pb.Pedalboard([
            pb.Reverb(room_size=0.18, damping=0.6, wet_level=0.06, dry_level=0.97, width=0.5),
        ])
        x = verb(x, sr)

    return Audio(dsp.to_mono(x), sr)
