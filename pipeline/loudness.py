"""Stage 6 — Loudness + true peak (mandatory, always runs).

Push to ≈ -9 LUFS, optionally add density with a soft-clipper, then brick-wall
at -1.0 dBTP with an oversampled true-peak limiter, and dither to 24-bit.
"""
from __future__ import annotations

import numpy as np
import pedalboard as pb
from scipy import signal

from . import dsp
from .ingest import Audio
from .meters import integrated_lufs, gain_to_lufs
from .settings import Settings


def _oversampled_limit(x: np.ndarray, sr: int, ceiling_db: float, oversample: int) -> np.ndarray:
    """Brick-wall at ``ceiling_db`` dBTP using an oversampled lookahead limiter.

    Limits in the oversampled domain so inter-sample peaks are caught, then
    decimates back. A final hard safety clip guarantees the ceiling.
    """
    x = dsp.to_stereo(x)
    ceiling = dsp.db_to_lin(ceiling_db)

    up = signal.resample_poly(x, oversample, 1, axis=0).astype(np.float32)

    # JUCE limiter at the oversampled rate, ceiling set just under target so the
    # post-decimation ripple still lands under the brick wall.
    limiter = pb.Pedalboard([pb.Limiter(threshold_db=ceiling_db - 0.3, release_ms=80.0)])
    up = limiter(up, sr * oversample)

    # Guarantee the ceiling in the oversampled domain before decimating.
    np.clip(up, -ceiling, ceiling, out=up)

    down = signal.resample_poly(up, 1, oversample, axis=0).astype(np.float32)
    if down.shape[0] != x.shape[0]:
        down = dsp._fit_length(down, x.shape[0])
    return down


def finalize(colored: Audio, s: Settings, target_lufs: float | None = None):
    """Apply makeup to target LUFS, density soft-clip, true-peak limit, dither.

    Returns (master_24: Audio, info: dict). ``master_24`` is float but already
    dithered & guaranteed under the true-peak ceiling; write it as PCM_24.
    """
    sr = colored.sr
    target = s.loudness_target if target_lufs is None else target_lufs
    x = dsp.to_stereo(colored.data)

    # 1. Makeup gain toward the loudness target.
    x, _ = gain_to_lufs(x, sr, target)

    # 2. Gentle pre-limiter soft-clip for density (perceived loudness w/o pump).
    x = dsp.soft_clip_tanh(x, drive=0.25, sr=sr, oversample=s.oversample)

    # 3. True-peak limiter, brick-walled at the ceiling, oversampled. Iterate
    #    limit + loudness-match: density/limiting shifts loudness either way, so
    #    correct down (a safe clean attenuation) or up (then re-limit) to target.
    ceiling = s.true_peak_ceiling_db
    for _ in range(4):
        x = _oversampled_limit(x, sr, ceiling, s.oversample)
        cur = integrated_lufs(x, sr)
        if not np.isfinite(cur):
            break
        diff = target - cur
        if abs(diff) <= 0.3:
            break
        if diff < 0:
            # Too loud: clean gain reduction also lowers peaks — done.
            x = (x * dsp.db_to_lin(diff)).astype(np.float32)
            break
        # Too quiet: push up, then re-limit on the next pass.
        x = (x * dsp.db_to_lin(min(diff, 1.5))).astype(np.float32)

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
