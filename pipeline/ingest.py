"""Stage 0 — Ingest: decode, resample, channel handling, manual offset nudge."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import signal

from . import dsp

DECODABLE = {".wav", ".aif", ".aiff", ".flac", ".ogg"}


@dataclass
class Audio:
    """A decoded buffer plus its sample rate. ``data`` is (n_samples, n_channels)."""

    data: np.ndarray
    sr: int

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    @property
    def n(self) -> int:
        return self.data.shape[0]


def _decode_mp3_via_ffmpeg(path: str) -> Audio:
    """Decode formats libsndfile can't (e.g. MP3) by piping through ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required to decode this file (e.g. MP3) but was not found on PATH."
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-f", "wav", "-acodec", "pcm_f32le", tmp.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        data, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
    finally:
        os.unlink(tmp.name)
    return Audio(dsp.ensure_2d(data), sr)


def decode(path: str) -> Audio:
    """Decode a file to float32 (n_samples, n_channels)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in DECODABLE:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return Audio(dsp.ensure_2d(data), sr)
        except Exception:
            return _decode_mp3_via_ffmpeg(path)
    return _decode_mp3_via_ffmpeg(path)


def resample(a: Audio, target_sr: int) -> Audio:
    if a.sr == target_sr:
        return a
    # rational resampling keeps phase clean
    from math import gcd

    g = gcd(a.sr, target_sr)
    up, down = target_sr // g, a.sr // g
    out = signal.resample_poly(a.data, up, down, axis=0).astype(np.float32)
    return Audio(out, target_sr)


def apply_offset(a: Audio, offset_ms: float) -> Audio:
    """Manual latency nudge on the vocal. Positive delays it; negative advances it.

    No automatic alignment is attempted (spec §4) — this is a user control only.
    """
    if abs(offset_ms) < 1e-6:
        return a
    shift = int(round(offset_ms * 1e-3 * a.sr))
    if shift > 0:
        out = np.pad(a.data, ((shift, 0), (0, 0)))
    else:
        out = a.data[-shift:]
        out = np.pad(out, ((0, -shift), (0, 0)))  # keep length stable
    return Audio(out.astype(np.float32), a.sr)


def ingest_pair(vocal_path: str, instrumental_path: str, offset_ms: float = 0.0):
    """Decode both files, resample to the higher common rate, fix channel layout.

    Vocal -> mono (center lane). Instrumental -> stereo. Returns
    (vocal: Audio mono, instrumental: Audio stereo, working_sr).
    """
    vocal = decode(vocal_path)
    instr = decode(instrumental_path)

    work_sr = max(vocal.sr, instr.sr)
    vocal = resample(vocal, work_sr)
    instr = resample(instr, work_sr)

    vocal = Audio(dsp.to_mono(vocal.data), work_sr)
    instr = Audio(dsp.to_stereo(instr.data), work_sr)

    vocal = apply_offset(vocal, offset_ms)
    return vocal, instr, work_sr


def ingest_single(track_path: str) -> Audio:
    """Decode a single already-mixed track to a stereo Audio at its native rate."""
    a = decode(track_path)
    return Audio(dsp.to_stereo(a.data), a.sr)


def write_wav(path: str, data: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    """Write a (n_samples, n_channels) float buffer to disk (24-bit by default)."""
    sf.write(path, dsp.ensure_2d(data), sr, subtype=subtype)
