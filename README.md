# Hip-Hop Mix & Master

A **local, drag-and-drop web app** that turns a processed **vocal** + a finished
**instrumental** into a loud, streaming-safe master with a warm, wide, smooth
signature sound — built for **hip-hop / trap / R&B**.

It is a **mix-then-master** tool, not a one-file mastering service: it lightly
mixes the vocal into the beat, sums to a stereo bus, then masters to a specific
tonal signature. Everything runs on your machine, in your browser.

## The signature sound

- **Loud** — integrated loudness ≈ **−9 LUFS**, as loud as possible while clean.
- **Streaming-safe** — true-peak ceiling **−1.0 dBTP**, ≥4× oversampled limiting,
  no audible clipping or inter-sample overs.
- **Wide, but only up top** — wide mids/highs, **mono below ~120 Hz** so 808s and
  kick stay centered and translate on phone speakers (stereo correlation stays ≥ 0).
- **Smooth on top** — sibilance/harshness tamed dynamically; gentle air, never brittle.
- **Warm / analog** — subtle harmonic saturation, not a clinical digital sound.

## Quick start

```bash
# 1. system deps
#   macOS:   brew install ffmpeg libsndfile
#   Ubuntu:  sudo apt-get install -y ffmpeg libsndfile1
#   Windows: install ffmpeg (https://ffmpeg.org) and add it to PATH;
#            libsndfile ships with the `soundfile` wheel.

# 2. python deps (Python 3.10–3.11 recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. run
python app.py        # opens / prints http://127.0.0.1:8000
```

Drop a **Vocal** and an **Instrumental**, pick a master mode, tweak the taste
sliders, and click **Master**. You get a loudness-matched A/B preview, a quality
report, and a **24-bit WAV** download.

## Two stems or one full track

Use the **Input** switch at the top of the Single tab:

- **Vocal + Beat** (default) — the full mix-then-master pipeline.
- **Full track** — already mixed your vocal and beat together? Upload the single
  finished file and it runs **master only** (tonal match → signature chain →
  loudness/true-peak), skipping the vocal/balance/mix stages. Mix-only controls
  (vocal level, ducking, de-esser, offset) hide automatically. For the best
  result the upload should still have **headroom** (peaks around −6 dBFS, not an
  already-slammed/streaming file) — you'll get a heads-up if it's too hot.

## Master modes

| Mode | What it does |
|---|---|
| **Genre reference** (default) | Pick a **Sound** (see below) and it voices your master to that style — no reference file needed. If a reference WAV sits in `references/hiphop/`, Matchering matches toward it instead. |
| **My reference** | Drop your own reference master; Matchering matches toward it. |
| **Fully automatic** | No reference — analyses your mix's spectrum and corrects it toward the selected Sound's target curve, then applies the signature chain. |

### Sounds (Genre selector)

In Genre/Automatic mode, choose a **Sound** — a voiced preset tuned to a style:

| Sound | Voiced from | Character |
|---|---|---|
| **Trap** | Young Thug · Future · Travis Scott | dark, 808-heavy, saturated, wide, loud |
| **Boom-bap** | Kanye · JID · Joey Bada$$ · Griselda · Larry June | warm, mid-forward, punchy, more dynamic |
| **Melodic / R&B** | untiljapan · Lancey Foux · Smino · Isiah Rashad | lush, smooth, widest, silky/airy top |

Every Sound sits on a **house character**: a primary **Kanye** layer (soulful
low-mid body, even-harmonic warmth, full and weighty) and a lighter **untiljapan**
layer (a smooth, open top + width baseline). No copyrighted audio is bundled —
each Sound is a *voicing* (target curve + dynamics/saturation/width), not a clip.
For an exact match to one specific record, use **My reference**.

> **Commercial references are user-supplied.** No copyrighted tracks are bundled.
> The built-in target curve means you don't *need* one — but adding a clean,
> same-genre reference WAV to `references/hiphop/` matches a specific record. See
> that folder's README.

## Taste controls

Vocal vs. beat (±6 dB) · Stereo width · Warmth · Loudness target (−12…−7, default
−9) · Vocal offset nudge (ms) · De-esser · Ducking · Mix-bus glue · Glue reverb
(off by default). All have pro defaults pre-set.

## Batch / Album mode

The **Batch / Album** tab queues multiple (vocal + instrumental) pairs, processes
them all with the current settings, then runs an album-level pass so every track
shares one integrated loudness (−9 LUFS) and stays tonally consistent.

## How it works (signal flow)

```
ingest → vocal conditioning → balance + ducking → sum to bus
       → master (mode + signature chain) → loudness + true-peak limit → 24-bit WAV
```

| Stage | Module | Summary |
|---|---|---|
| 0 Ingest | `pipeline/ingest.py` | decode (libsndfile / ffmpeg), resample to common rate, channel layout, manual offset nudge |
| 1 Vocal | `pipeline/vocal.py` | 90 Hz HPF, gentle de-esser, light leveling, optional glue reverb |
| 2 Balance | `pipeline/balance.py` | LUFS-match vocal to beat +1 LU, subtle 1–4 kHz ducking under the vocal |
| 3 Mix bus | `pipeline/mixbus.py` | sum to stereo, optional glue comp, −6 dBFS headroom |
| 4 Master | `pipeline/master.py` | 3 modes + signature chain: clarity EQ, warmth, dynamic harsh-tame, M/S width (lows mono), glue |
| 6 Loudness | `pipeline/loudness.py` | −9 LUFS makeup, density soft-clip, oversampled −1 dBTP true-peak limiter, TPDF dither |
| 5 Report | `pipeline/report.py` | integrated LUFS, true peak, LRA, correlation, mono-low check |
| Album | `pipeline/album.py` | batch + album loudness/tone consistency pass |

DSP primitives (mid/side, Linkwitz-Riley crossovers, oversampled tanh, dynamic
band processors, true-peak metering, dither) live in `pipeline/dsp.py`; loudness
metering in `pipeline/meters.py`.

## Tech stack

Python · FastAPI + uvicorn · numpy / scipy · soundfile (libsndfile) · ffmpeg
(MP3) · [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) (BS.1770 LUFS) ·
[pedalboard](https://github.com/spotify/pedalboard) (EQ / comp / limiter / reverb)
· [matchering](https://github.com/sergree/matchering) (reference matching).
Frontend is plain HTML/CSS/JS — no build step.

## Licensing

`matchering` is **GPLv3**. For **personal use** this is fine. If you ever
**distribute or sell** this tool, GPL obligations apply (the whole project must
be released under compatible terms) — verify the license of every dependency
before distributing. No copyrighted reference audio is included in this repo.
