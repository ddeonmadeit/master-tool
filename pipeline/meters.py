"""Loudness metering helpers shared across stages (ITU-R BS.1770 via pyloudnorm)."""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln

from . import dsp

# Meters are keyed by sample rate so we don't rebuild filters every call.
_METERS: dict[int, pyln.Meter] = {}


def _meter(sr: int) -> pyln.Meter:
    m = _METERS.get(sr)
    if m is None:
        m = pyln.Meter(sr)
        _METERS[sr] = m
    return m


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """Integrated (gated) loudness in LUFS. Returns -inf for silence."""
    x = dsp.ensure_2d(x)
    # pyloudnorm wants (samples,) for mono or (samples, channels)
    buf = x[:, 0] if x.shape[1] == 1 else x
    try:
        val = float(_meter(sr).integrated_loudness(buf))
    except Exception:
        return float("-inf")
    return val


def gain_to_lufs(x: np.ndarray, sr: int, target_lufs: float):
    """Return (gained_audio, applied_gain_db) to hit ``target_lufs`` integrated.

    Silent or near-silent input is returned unchanged.
    """
    x = dsp.ensure_2d(x)
    cur = integrated_lufs(x, sr)
    if not np.isfinite(cur):
        return x, 0.0
    gain_db = target_lufs - cur
    return (x * dsp.db_to_lin(gain_db)).astype(np.float32), gain_db


def loudness_range(x: np.ndarray, sr: int) -> float:
    """Loudness Range (LRA), LU, per EBU 3342: gated 10th-95th percentile spread
    of 3 s short-term loudness measured every 1 s."""
    x = dsp.ensure_2d(x)
    buf = x[:, 0] if x.shape[1] == 1 else x
    win = int(3.0 * sr)
    hop = int(1.0 * sr)
    if buf.shape[0] < win:
        return 0.0
    meter = _meter(sr)
    st = []
    for start in range(0, buf.shape[0] - win + 1, hop):
        seg = buf[start : start + win]
        try:
            lv = float(meter.integrated_loudness(seg))
        except Exception:
            continue
        if np.isfinite(lv):
            st.append(lv)
    if len(st) < 2:
        return 0.0
    st = np.array(st)
    # absolute gate at -70 LUFS, then relative gate at -20 LU below mean
    st = st[st >= -70.0]
    if st.size < 2:
        return 0.0
    rel_gate = st.mean() - 20.0
    st = st[st >= rel_gate]
    if st.size < 2:
        return 0.0
    lo, hi = np.percentile(st, 10), np.percentile(st, 95)
    return float(hi - lo)


def stereo_correlation(x: np.ndarray) -> float:
    """Pearson correlation between L and R. 1.0 = mono, 0 = decorrelated,
    <0 = out-of-phase. Mono input reports 1.0."""
    x = dsp.ensure_2d(x)
    if x.shape[1] == 1:
        return 1.0
    l, r = x[:, 0], x[:, 1]
    if np.allclose(l, r):
        return 1.0
    if np.std(l) < 1e-9 or np.std(r) < 1e-9:
        return 1.0
    return float(np.corrcoef(l, r)[0, 1])
