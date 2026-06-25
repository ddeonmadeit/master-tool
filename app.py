"""Hip-Hop Mix & Master — local web app.

Run:  python app.py   (then open the printed http://127.0.0.1:8000 URL)

Drops in a processed vocal + an instrumental, returns a loud, streaming-safe
24-bit master plus a loudness/quality report, with batch/album mode.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
import webbrowser

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline import Settings, process_album, run
from pipeline.ingest import write_wav

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
JOBS_DIR = os.path.join(tempfile.gettempdir(), "mixmaster_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title="Hip-Hop Mix & Master")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _job_dir(job_id: str) -> str:
    d = os.path.join(JOBS_DIR, job_id)
    if not os.path.isdir(d) or os.path.dirname(os.path.realpath(d)) != os.path.realpath(JOBS_DIR):
        raise HTTPException(404, "Unknown job")
    return d


async def _save_upload(upload: UploadFile, dest_dir: str, name: str) -> str:
    path = os.path.join(dest_dir, name + os.path.splitext(upload.filename or "")[1])
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def _audio_urls(job_id: str) -> dict:
    base = f"/api/audio/{job_id}"
    return {
        "master_url": f"{base}/master.wav",
        "download_url": f"/api/download/{job_id}/master.wav",
        "ab_premaster_url": f"{base}/ab_premaster.wav",
        "ab_master_url": f"{base}/ab_master.wav",
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.post("/api/master")
async def api_master(
    vocal: UploadFile = File(...),
    instrumental: UploadFile = File(...),
    reference: UploadFile | None = File(None),
    settings: str = Form("{}"),
):
    s = Settings.from_dict(json.loads(settings or "{}"))
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    vpath = await _save_upload(vocal, job_dir, "vocal")
    ipath = await _save_upload(instrumental, job_dir, "instrumental")
    rpath = None
    if reference is not None and (reference.filename or ""):
        rpath = await _save_upload(reference, job_dir, "reference")

    try:
        result = run(vpath, ipath, s, reference_path=rpath)
    except Exception as exc:  # surface decode/processing errors to the UI
        raise HTTPException(400, f"Processing failed: {exc}")

    write_wav(os.path.join(job_dir, "master.wav"), result.master.data, result.master.sr)
    write_wav(os.path.join(job_dir, "ab_premaster.wav"), result.ab_premaster.data,
              result.ab_premaster.sr, subtype="PCM_16")
    write_wav(os.path.join(job_dir, "ab_master.wav"), result.ab_master.data,
              result.ab_master.sr, subtype="PCM_16")

    return JSONResponse({
        "job_id": job_id,
        "report": result.report,
        "info": result.info,
        "notices": result.notices,
        "settings": s.to_dict(),
        **_audio_urls(job_id),
    })


@app.post("/api/batch")
async def api_batch(
    files: list[UploadFile] = File(...),
    names: str = Form("[]"),
    settings: str = Form("{}"),
):
    """Album mode. ``files`` arrive as flat [vocal0, instr0, vocal1, instr1, ...];
    ``names`` is a JSON list of per-track display names."""
    s = Settings.from_dict(json.loads(settings or "{}"))
    track_names = json.loads(names or "[]")
    if len(files) % 2 != 0 or not files:
        raise HTTPException(400, "Provide an even number of files (vocal + instrumental per track).")

    batch_id = uuid.uuid4().hex
    batch_dir = os.path.join(JOBS_DIR, batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    pairs = []
    for i in range(0, len(files), 2):
        idx = i // 2
        vp = await _save_upload(files[i], batch_dir, f"t{idx}_vocal")
        ip = await _save_upload(files[i + 1], batch_dir, f"t{idx}_instrumental")
        nm = track_names[idx] if idx < len(track_names) else f"track{idx + 1}"
        pairs.append({"name": nm, "vocal": vp, "instrumental": ip})

    try:
        results, album_report = process_album(pairs, s)
    except Exception as exc:
        raise HTTPException(400, f"Batch processing failed: {exc}")

    tracks = []
    for idx, (pair, res) in enumerate(zip(pairs, results)):
        fname = f"master_{idx}.wav"
        write_wav(os.path.join(batch_dir, fname), res.master.data, res.master.sr)
        tracks.append({
            "name": pair["name"],
            "report": res.report,
            "notices": res.notices,
            "download_url": f"/api/download/{batch_id}/{fname}",
            "master_url": f"/api/audio/{batch_id}/{fname}",
        })

    return JSONResponse({
        "batch_id": batch_id,
        "album_report": album_report,
        "tracks": tracks,
        "settings": s.to_dict(),
    })


@app.get("/api/audio/{job_id}/{name}")
def api_audio(job_id: str, name: str):
    path = os.path.join(_job_dir(job_id), os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/download/{job_id}/{name}")
def api_download(job_id: str, name: str):
    path = os.path.join(_job_dir(job_id), os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav", filename=os.path.basename(name))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main():
    host, port = "127.0.0.1", 8000
    url = f"http://{host}:{port}"
    print(f"\n  Hip-Hop Mix & Master  →  {url}\n")
    threading.Timer(1.2, lambda: _try_open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


def _try_open(url: str):
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    main()
