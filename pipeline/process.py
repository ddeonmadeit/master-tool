"""End-to-end orchestration.

Two entry points share the same master/loudness tail:
  - ``run``            mix two stems (vocal + instrumental), then master.
  - ``run_master_only``  master a single already-mixed stereo track.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import balance, dsp, ingest, loudness, master, mixbus, report, vocal
from .ingest import Audio
from .meters import gain_to_lufs, integrated_lufs
from .settings import Settings

# Equal-loudness level for the A/B preview (quiet enough to never clip in-browser).
AB_PREVIEW_LUFS = -16.0


@dataclass
class Result:
    master: Audio            # final 24-bit-ready master (float, dithered)
    premaster: Audio         # pre-master mix bus (raw, for reference)
    ab_premaster: Audio      # loudness-MATCHED preview of the pre-master mix
    ab_master: Audio         # loudness-MATCHED preview of the master
    ab_premaster_real: Audio # REAL-level preview of the mix (true loudness gap)
    ab_master_real: Audio    # REAL-level preview of the master (audibly louder)
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

    # Stages 4-6 — master + loudness + report (shared tail)
    return _finish_from_bus(bus, s, reference_path, target_lufs, progress_cb,
                            extra_info=balance_info, master_pct=37)


def run_master_only(track_path: str, s: Settings,
                    reference_path: str | None = None,
                    target_lufs: float | None = None,
                    progress_cb=None) -> Result:
    """Master a single, already-mixed stereo track (skips the mixing stages).

    The uploaded track is the "pre-master" — we give it consistent headroom and
    feed it straight into the master + loudness chain. No vocal/balance/mixbus
    processing happens; mix-only settings (vocal level, ducking, de-ess, offset)
    are ignored.
    """
    def _p(stage: str, pct: int):
        if progress_cb is not None:
            progress_cb(stage, pct)

    _p("ingest", 0)
    track = ingest.ingest_single(track_path)

    # Friendly heads-up if the upload is already loud/limited (little to gain).
    in_lufs = integrated_lufs(track.data, track.sr)
    pre_notices = _headroom_notice(track.data, in_lufs)

    # Normalise to the standard pre-master headroom so the master chain sees a
    # consistent level (no mix-bus glue comp — the track is already mixed).
    bus = _prep_track_bus(track, s)

    res = _finish_from_bus(bus, s, reference_path, target_lufs, progress_cb,
                           extra_info={"input_lufs": _round(in_lufs),
                                       "master_only": True},
                           master_pct=20)
    res.notices = pre_notices + res.notices
    return res


def _finish_from_bus(bus: Audio, s: Settings, reference_path: str | None,
                     target_lufs: float | None, progress_cb,
                     extra_info: dict | None = None, master_pct: int = 37) -> Result:
    """Shared tail: master (modes + signature chain) -> loudness -> report -> A/B."""
    sr = bus.sr

    def _p(stage: str, pct: int):
        if progress_cb is not None:
            progress_cb(stage, pct)

    # Stage 4 — master (modes + signature chain). Sub-progress fills the span up
    # to the loudness stage so the bar keeps moving through the longest stretch.
    _p("master", master_pct)
    master_span = 78 - master_pct
    colored, master_info, notices = master.master(
        bus, s, reference_path=reference_path,
        progress=lambda f: _p("master", master_pct + int(master_span * min(f, 1.0))))

    # Stage 6 — loudness + true peak + dither (iterative — also reports progress)
    _p("loudness", 80)
    final, loud_info = loudness.finalize(
        colored, s, target_lufs=target_lufs,
        progress=lambda f: _p("loudness", 80 + int(13 * min(f, 1.0))))

    # Stage 5 — report
    _p("report", 94)
    rep = report.measure(final, s)
    _p("done", 100)

    # Loudness-MATCHED previews: both at one loudness, to judge tone/width.
    ab_pre, _ = gain_to_lufs(bus.data, sr, AB_PREVIEW_LUFS)
    ab_mas, _ = gain_to_lufs(final.data, sr, AB_PREVIEW_LUFS)

    # REAL-level previews: ONE gain on BOTH, so you actually hear how much louder
    # the master is than the mix (scaled so the master sits just under clipping).
    mpeak = float(np.max(np.abs(final.data))) if final.data.size else 0.0
    g = (dsp.db_to_lin(-1.0) / mpeak) if mpeak > 0 else 1.0
    ab_pre_real = Audio(_safe(bus.data * g), sr)
    ab_mas_real = Audio(_safe(final.data * g), sr)

    info = {**(extra_info or {}), **master_info, **loud_info, "working_sr": sr}
    return Result(
        master=final,
        premaster=bus,
        ab_premaster=Audio(_safe(ab_pre), sr),
        ab_master=Audio(_safe(ab_mas), sr),
        ab_premaster_real=ab_pre_real,
        ab_master_real=ab_mas_real,
        report=rep,
        info=info,
        notices=notices,
    )


def _prep_track_bus(track: Audio, s: Settings) -> Audio:
    """Stereo-ise an uploaded track and set the standard pre-master headroom."""
    x = dsp.to_stereo(track.data)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x * (dsp.db_to_lin(s.premaster_headroom_db) / peak)
    return Audio(x.astype(np.float32), track.sr)


def _headroom_notice(x: np.ndarray, in_lufs: float) -> list:
    """Warn when an uploaded track is already loud/limited (mastering may hurt)."""
    notes: list[str] = []
    x = dsp.ensure_2d(x)
    peak_db = dsp.lin_to_db(float(np.max(np.abs(x)))) if x.size else -120.0
    if np.isfinite(in_lufs) and in_lufs > -10.0:
        notes.append(
            f"This track is already very loud (~{in_lufs:.1f} LUFS). Re-mastering "
            "an already-finished/limited file can add distortion — for the best "
            "result, upload a mix with headroom (peaks around -6 dBFS)."
        )
    elif peak_db > -1.5:
        notes.append(
            "This track peaks near 0 dBFS (little headroom). For the best result, "
            "master a mix that leaves headroom (peaks around -6 dBFS)."
        )
    return notes


def _round(v: float):
    return round(float(v), 2) if np.isfinite(v) else None


def _safe(x: np.ndarray) -> np.ndarray:
    """Guard the preview against clipping after loudness matching."""
    x = dsp.to_stereo(x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > dsp.db_to_lin(-1.0):
        x = x * (dsp.db_to_lin(-1.0) / peak)
    return x.astype(np.float32)
