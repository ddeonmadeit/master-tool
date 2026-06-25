"""Stage 3 — Sum to the stereo bus.

Centered vocal + stereo instrumental -> 32-bit float stereo bus with ~-6 dBFS
headroom before the master chain, plus optional very light glue compression.
"""
from __future__ import annotations

import numpy as np
import pedalboard as pb

from . import dsp
from .ingest import Audio
from .settings import Settings


def sum_to_bus(vocal: Audio, instrumental: Audio, s: Settings) -> Audio:
    sr = vocal.sr
    voc = dsp.to_stereo(vocal.data)        # center mono -> dual-mono
    instr = dsp.to_stereo(instrumental.data)
    voc, instr = dsp.match_length(voc, instr)

    bus = (voc + instr).astype(np.float32)

    # Optional mix-bus glue: very light (≈1.5:1, slow attack, ≤2 dB GR).
    if s.mixbus_glue:
        glue = pb.Pedalboard([
            pb.Compressor(threshold_db=-18.0, ratio=1.5, attack_ms=30.0, release_ms=250.0),
        ])
        bus = glue(bus, sr)

    # Leave ~-6 dBFS headroom before the master chain.
    peak = float(np.max(np.abs(bus))) if bus.size else 0.0
    if peak > 0:
        target_lin = dsp.db_to_lin(s.premaster_headroom_db)
        bus = (bus * (target_lin / peak)).astype(np.float32)

    return Audio(dsp.to_stereo(bus), sr)
