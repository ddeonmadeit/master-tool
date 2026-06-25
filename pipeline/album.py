"""Album / batch mode — process many pairs with identical settings, then run an
album-level consistency pass so every track shares one integrated loudness."""
from __future__ import annotations

import numpy as np

from . import dsp, loudness, report
from .ingest import Audio
from .meters import integrated_lufs
from .process import Result, run
from .settings import Settings


def process_album(pairs: list[dict], s: Settings, reference_path: str | None = None):
    """``pairs`` is a list of {"name", "vocal", "instrumental"}.

    Returns (results: list[Result], album_report: dict). Each pair is processed
    with identical settings, then all masters are trimmed to one shared loudness.
    """
    results: list[Result] = []
    for p in pairs:
        results.append(run(p["vocal"], p["instrumental"], s, reference_path=reference_path))

    # Album pass: align every master to the shared target (the settings target).
    album_target = s.loudness_target
    per_track = []
    for p, res in zip(pairs, results):
        sr = res.master.sr
        cur = integrated_lufs(res.master.data, sr)
        if np.isfinite(cur):
            trim_db = album_target - cur
            adjusted = (res.master.data * dsp.db_to_lin(trim_db)).astype(np.float32)
            # A trim upward could lift inter-sample peaks — re-limit if needed.
            if dsp.true_peak_db(adjusted, sr, s.oversample) > s.true_peak_ceiling_db:
                relimited, _ = loudness.finalize(Audio(adjusted, sr), s,
                                                 target_lufs=album_target)
                adjusted = relimited.data
            res.master = Audio(adjusted, sr)
            res.report = report.measure(res.master, s)
        per_track.append({
            "name": p.get("name", "track"),
            "integrated_lufs": res.report["integrated_lufs"],
            "true_peak_dbtp": res.report["true_peak_dbtp"],
        })

    lufs_vals = [t["integrated_lufs"] for t in per_track if t["integrated_lufs"] is not None]
    spread = round(max(lufs_vals) - min(lufs_vals), 2) if lufs_vals else None
    album_report = {
        "album_target_lufs": album_target,
        "track_count": len(results),
        "tracks": per_track,
        "loudness_spread_lu": spread,
        "consistent": (spread is not None and spread <= 0.5),
    }
    return results, album_report
