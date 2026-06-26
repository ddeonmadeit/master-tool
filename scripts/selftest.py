"""Self-test: generate synthetic audio and assert the signature-sound targets.

    python scripts/selftest.py

Runs the full pipeline (Fully automatic mode — no reference needed) on a
synthetic vocal + instrumental and checks the acceptance criteria from the spec:
−9 LUFS ±0.5, true peak ≤ −1 dBTP, no clipping, correlation ≥ 0, lows mono, and
album-mode loudness consistency.
"""
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import Settings, process_album, run  # noqa: E402
from pipeline.ingest import write_wav  # noqa: E402


def _synth(out_dir, sr=44100, dur=8.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    rng = np.random.default_rng(0)

    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * (np.sin(2 * np.pi * 0.4 * t) > -0.3)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t) \
        + 0.15 * np.sin(2 * np.pi * 5000 * t)
    sib = 0.25 * rng.standard_normal(len(t)) * (np.sin(2 * np.pi * 2 * t) > 0.7)
    vocal = ((tone + sib) * env * 0.5).astype(np.float32)

    def instr(seed):
        r = np.random.default_rng(seed)
        sub = 0.6 * np.sin(2 * np.pi * 50 * t) * (0.4 + 0.6 * (np.sin(2 * np.pi * 2 * t) > 0.9))
        midL = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * r.standard_normal(len(t))
        midR = 0.3 * np.sin(2 * np.pi * 440 * t + 0.5) + 0.2 * r.standard_normal(len(t))
        hatsL = 0.2 * r.standard_normal(len(t)) * (np.sin(2 * np.pi * 8 * t) > 0.8)
        hatsR = 0.2 * r.standard_normal(len(t)) * (np.sin(2 * np.pi * 8 * t + 0.3) > 0.8)
        L = (sub + midL + hatsL) * 0.4
        R = (sub + midR + hatsR) * 0.4
        return np.stack([L, R], axis=1).astype(np.float32)

    vpath = os.path.join(out_dir, "vocal.wav")
    ipath = os.path.join(out_dir, "instr.wav")
    ipath2 = os.path.join(out_dir, "instr2.wav")
    sf.write(vpath, vocal, sr, subtype="FLOAT")
    sf.write(ipath, instr(1), sr, subtype="FLOAT")
    sf.write(ipath2, instr(5) * 0.7, sr, subtype="FLOAT")
    return vpath, ipath, ipath2, sr


def main():
    with tempfile.TemporaryDirectory() as td:
        vpath, ipath, ipath2, sr = _synth(td)

        print("Running single-track master (auto mode)…")
        s = Settings(mode="auto")
        res = run(vpath, ipath, s)
        rep = res.report
        print("  report:", {k: rep[k] for k in
              ("integrated_lufs", "true_peak_dbtp", "stereo_correlation", "low_band_side_energy")})

        # 24-bit / sample-rate round-trip
        mpath = os.path.join(td, "master.wav")
        write_wav(mpath, res.master.data, res.master.sr)
        info = sf.info(mpath)
        assert info.subtype == "PCM_24", f"expected PCM_24, got {info.subtype}"
        assert info.samplerate == sr, "sample rate not preserved"

        assert abs(rep["integrated_lufs"] - s.loudness_target) <= 0.5, rep["integrated_lufs"]
        assert rep["true_peak_dbtp"] <= -0.95, rep["true_peak_dbtp"]
        assert rep["sample_peak_dbfs"] < 0.0, "clipping detected"
        assert rep["stereo_correlation"] >= 0.0, rep["stereo_correlation"]
        assert rep["low_band_side_energy"] <= 0.02, rep["low_band_side_energy"]
        assert rep["all_passed"], rep["checks"]
        print("  single-track: ALL CHECKS PASSED")

        print("Running album mode (2 tracks)…")
        pairs = [
            {"name": "t1", "vocal": vpath, "instrumental": ipath},
            {"name": "t2", "vocal": vpath, "instrumental": ipath2},
        ]
        _, album = process_album(pairs, Settings(mode="auto"))
        assert album["consistent"], album
        print(f"  album spread {album['loudness_spread_lu']} LU: CONSISTENT")

    print("\nSELF-TEST PASSED ✓")


if __name__ == "__main__":
    main()
