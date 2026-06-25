"""Stage 5 — QA report: integrated LUFS, true peak (dBTP), LRA, correlation."""
from __future__ import annotations

import numpy as np

from . import dsp
from .ingest import Audio
from .meters import integrated_lufs, loudness_range, stereo_correlation
from .settings import Settings


def low_band_mono_ratio(x: np.ndarray, sr: int, crossover_hz: float) -> float:
    """Fraction of side (stereo-difference) energy that sits below the crossover.

    Near 0 means the lows are mono (the goal). Used to verify "mono < 120 Hz".
    """
    x = dsp.to_stereo(x)
    _, side = dsp.lr_to_ms(x)
    side2d = side[:, None]
    low = dsp.lowpass(side2d, sr, crossover_hz, order=2)
    total = float(np.sum(side2d.astype(np.float64) ** 2)) + dsp.EPS
    low_e = float(np.sum(low.astype(np.float64) ** 2))
    return low_e / total


def measure(master: Audio, s: Settings) -> dict:
    sr = master.sr
    x = dsp.to_stereo(master.data)
    lufs = integrated_lufs(x, sr)
    tp = dsp.true_peak_db(x, sr, s.oversample)
    lra = loudness_range(x, sr)
    corr = stereo_correlation(x)
    low_mono = low_band_mono_ratio(x, sr, s.mono_below_hz)
    sample_peak = dsp.peak_db(x)

    checks = {
        "loudness_on_target": abs(lufs - s.loudness_target) <= 0.5 if np.isfinite(lufs) else False,
        "true_peak_safe": tp <= s.true_peak_ceiling_db + 0.05,
        "no_clipping": sample_peak < 0.0,
        "correlation_non_negative": corr >= 0.0,
        "lows_mostly_mono": low_mono <= 0.02,
    }

    return {
        "integrated_lufs": _r(lufs),
        "true_peak_dbtp": _r(tp),
        "sample_peak_dbfs": _r(sample_peak),
        "loudness_range_lu": _r(lra),
        "stereo_correlation": _r(corr, 3),
        "low_band_side_energy": _r(low_mono, 4),
        "sample_rate": sr,
        "channels": x.shape[1],
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _r(v, n: int = 2):
    return round(float(v), n) if np.isfinite(v) else None
