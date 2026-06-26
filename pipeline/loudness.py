"""Stage 6 — Loudness + true peak (mandatory, always runs).

Push to the target LUFS (default -8), optionally add density with a soft-clipper, then brick-wall
at -1.0 dBTP with an oversampled true-peak limiter, and dither to 24-bit.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import dsp
from .ingest import Audio
from .meters import integrated_lufs
from .settings import Settings


def _oversampled_limit(x: np.ndarray, sr: int, ceiling_db: float, oversample: int) -> np.ndarray:
    """Brick-wall at ``ceiling_db`` dBTP using a two-stage oversampled limiter.

    Limits in the oversampled domain so inter-sample peaks are caught, then
    decimates back. Two gentle stages instead of one hard one — a slower stage
    rides sustained level (no pumping) and a faster stage catches the remaining
    transient tips — so the chain extracts more loudness *cleanly* before the
    safety clip. The ceiling is set a touch under target so post-decimation
    ripple still lands under the brick wall.
    """
    x = dsp.to_stereo(x)
    ceiling = dsp.db_to_lin(ceiling_db)
    sr_os = sr * oversample

    up = signal.resample_poly(x, oversample, 1, axis=0).astype(np.float32)
    # Stage 1: slow/long lookahead — handles sustained loudness without pumping.
    up = dsp.lookahead_limiter(up, sr_os, ceiling_db - 0.1,
                               lookahead_ms=2.0, release_ms=180.0)
    # Stage 2: fast/short lookahead — snaps the transient tips to the wall.
    up = dsp.lookahead_limiter(up, sr_os, ceiling_db - 0.1,
                               lookahead_ms=0.8, release_ms=40.0)
    np.clip(up, -ceiling, ceiling, out=up)

    down = signal.resample_poly(up, 1, oversample, axis=0).astype(np.float32)
    if down.shape[0] != x.shape[0]:
        down = dsp._fit_length(down, x.shape[0])
    # Decimation can reintroduce a hair of overshoot; clamp to the true ceiling.
    np.clip(down, -ceiling, ceiling, out=down)
    return down


def finalize(colored: Audio, s: Settings, target_lufs: float | None = None):
    """Reach the loudness target through a clipper -> true-peak limiter, dither.

    Returns (master_24: Audio, info: dict). ``master_24`` is float but already
    dithered & guaranteed under the true-peak ceiling; write it as PCM_24.

    Gain staging matters: we push level *into* a peak clipper + lookahead limiter
    rather than pre-amplifying the whole mix into a full-range waveshaper. The
    clipper only rounds transient tips toward the ceiling (the body of the mix is
    untouched), so the result is loud without the crushed, distorted, dull sound
    that pre-amplifying into a tanh saturator produces.
    """
    sr = colored.sr
    target = s.loudness_target if target_lufs is None else target_lufs
    x = dsp.to_stereo(colored.data)
    ceiling = s.true_peak_ceiling_db

    # Converge to target loudness. Each pass nudges gain toward target in gentle
    # steps, lightly shaves the sharpest tips with a *blended* 4x soft clipper
    # (most of the level work is left to the limiter, so clipping distortion stays
    # low), then brick-walls inter-sample peaks with the two-stage limiter. Small
    # steps + gentle clipping = loud but clean, not crushed.
    for _ in range(8):
        cur = integrated_lufs(x, sr)
        if not np.isfinite(cur):
            break
        diff = target - cur
        if abs(diff) <= 0.15:
            break
        x = (x * dsp.db_to_lin(float(np.clip(diff, -3.0, 2.0)))).astype(np.float32)
        # Clipper carries a fair share of the density (clipping percussive tips is
        # the clean way to get loud in hip-hop); the limiter does the rest.
        x = dsp.soft_clipper(x, sr, ceiling_db=ceiling, knee_db=3.0,
                             oversample=4, amount=0.65)
        x = _oversampled_limit(x, sr, ceiling, s.oversample)

    # Final safety: verify true peak; trim a hair if anything still pokes over.
    tp = dsp.true_peak_db(x, sr, s.oversample)
    if tp > ceiling:
        x = (x * dsp.db_to_lin(ceiling - tp - 0.1)).astype(np.float32)

    # 4. TPDF dither for the 24-bit render.
    x = dsp.tpdf_dither(x, bit_depth=24)
    np.clip(x, -1.0, 1.0, out=x)

    info = {
        "integrated_lufs": round(float(integrated_lufs(x, sr)), 2),
        "true_peak_db": round(float(dsp.true_peak_db(x, sr, s.oversample)), 2),
        "loudness_target": target,
    }
    return Audio(dsp.to_stereo(x), sr), info
