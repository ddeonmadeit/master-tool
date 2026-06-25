# Getting a browser link

This app does heavy native-audio DSP (numpy/scipy/pedalboard/matchering) and
needs **ffmpeg** plus a long-running process and a writable disk. That rules out
serverless platforms — here's what works, fastest first.

## TL;DR — which to use

| Goal | Use | Why |
|---|---|---|
| A real public URL, always on | **Railway / Render / Fly.io** (Docker) | Real container, ffmpeg installs, process stays up |
| See it in a browser *right now* | **Local + Cloudflare Tunnel** | Instant public URL, no deploy, no account |
| ❌ Not this | **Vercel / Netlify** | Serverless: no ffmpeg, ~250 MB limit, 10–60 s timeouts, no disk — the build won't fit and renders time out |

## Option A — Railway (recommended public URL)

This repo already includes a `Dockerfile` and `railway.json`, so Railway builds
it as-is.

1. Push this branch to GitHub (you have it on `claude/new-session-j64z6q`).
2. Go to **railway.app** → **New Project** → **Deploy from GitHub repo** → pick
   this repo and branch.
3. Railway detects the `Dockerfile` and builds. No build config needed.
4. In the service’s **Settings → Networking**, click **Generate Domain**. That
   `https://<name>.up.railway.app` URL is your link — open it in any browser.

**Recommended settings (Variables tab):**
- `APP_PASSWORD` = something only you know → the site prompts for a password
  (any username) before anyone can use it. Leave unset for an open demo.
- Bump the service **memory to ≥ 1 GB** (Settings → Resources). A 3–4 min stereo
  song at 4× oversample needs headroom; 512 MB can OOM on long tracks.

CLI alternative:
```bash
npm i -g @railway/cli
railway login
railway init           # in this repo
railway up             # builds the Dockerfile and deploys
railway domain         # prints your public URL
```

Render and Fly.io work the same way from the `Dockerfile` (Render: New → Web
Service → Docker; Fly: `fly launch` then `fly deploy`).

## Option B — Local + Cloudflare Tunnel (fastest, no deploy)

Best when you just want to click a link today and you’ll run it on your own
machine.

```bash
# 1. run the app locally (see README for ffmpeg/libsndfile install)
pip install -r requirements.txt
python app.py                      # serves http://127.0.0.1:8000

# 2. in a second terminal, expose it (no account needed)
#    macOS: brew install cloudflared   |   else: see cloudflare docs
cloudflared tunnel --url http://localhost:8000
```
Cloudflare prints a temporary `https://<random>.trycloudflare.com` URL that works
in any browser, anywhere, while your machine and the tunnel stay running.
(`ngrok http 8000` does the same if you prefer ngrok.)

## Notes

- **Persistence:** outputs are written to a temp dir and aren’t permanent. Users
  download their master right after processing — fine for a tool like this. Don’t
  rely on the server keeping files between deploys/restarts.
- **One instance:** keep it to a single instance/worker — jobs are tracked
  in-process and written to that instance’s local disk.
- **References:** `references/hiphop/` is empty by design (no copyrighted audio is
  shipped). Genre mode uses the built-in target curve until you add a reference
  WAV. To host *with* a reference, drop a licensed WAV there before deploying
  (it’s gitignored, so add it explicitly or bake it into your own image).
