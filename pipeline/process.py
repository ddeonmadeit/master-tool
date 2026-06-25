"""End-to-end orchestration for a single (vocal + instrumental) pair."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import balance, dsp, ingest, loudness, master, mixbus, report, vocal
from .ingest import Audio
from .meters import gain_to_lufs
from .settings import Settings

# Equal-loudness level for the A/B preview (quiet enough to never clip in-browser).
AB_PREVIEW_LUFS = -16.0


@dataclass
class Result:
    master: Audio            # final 24-bit-ready master (float, dithered)
    premaster: Audio         # pre-master mix bus (raw, for reference)
    ab_premaster: Audio      # loudness-matched preview of the pre-master mix
    ab_master: Audio         # loudness-matched preview of the master
    report: dict
    info: dict
    notices: list


def run(vocal_path: str, instrumental_path: str, s: Settings,
        reference_path: str | None = None,
        target_lufs: float | None = None,
        progress_cb=None) -> Result:

    def _p(stage: str, pct: int):
        if progress_cb is not None:
            progress_cb(stage, pct)

    # Stage 0 — ingest
    _p("ingest", 0)
    voc, instr, sr = ingest.ingest_pair(vocal_path, instrumental_path, s.offset_ms)

    # Stage 1 — vocal conditioning
    _p("vocal", 8)
    voc = vocal.condition(voc, s)

    # Stage 2 — balance + ducking
    _p("balance", 18)
    voc, instr, balance_info = balance.balance(voc, instr, s)

    # Stage 3 — sum to bus (this is the pre-master mix for A/B)
    _p("mixbus", 32)
    bus = mixbus.sum_to_bus(voc, instr, s)

    # Stage 4 — master (modes + signature chain)
    _p("master", 37)
    colored, master_info, notices = master.master(bus, s, reference_path=reference_path)

    # Stage 6 — loudness + true peak + dither
    _p("loudness", 82)
    final, loud_info = loudness.finalize(colored, s, target_lufs=target_lufs)

    # Stage 5 — report
    _p("report", 94)
    rep = report.measure(final, s)
    _p("done", 100)

    # Loudness-matched A/B previews
    ab_pre, _ = gain_to_lufs(bus.data, sr, AB_PREVIEW_LUFS)
    ab_mas, _ = gain_to_lufs(final.data, sr, AB_PREVIEW_LUFS)
    ab_pre = _safe(ab_pre)
    ab_mas = _safe(ab_mas)

    info = {**balance_info, **master_info, **loud_info, "working_sr": sr}
    return Result(
        master=final,
        premaster=bus,
        ab_premaster=Audio(ab_pre, sr),
        ab_master=Audio(ab_mas, sr),
        report=rep,
        info=info,
        notices=notices,
    )


def _safe(x: np.ndarray) -> np.ndarray:
    """Guard the preview against clipping after loudness matching."""
    x = dsp.to_stereo(x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > dsp.db_to_lin(-1.0):
        x = x * (dsp.db_to_lin(-1.0) / peak)
    return x.astype(np.float32)
