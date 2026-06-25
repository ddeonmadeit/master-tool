# Hip-Hop Mix & Master — container image for one-click hosting (Railway, Render, Fly…)
FROM python:3.11-slim

# System deps: ffmpeg (MP3 decode) + libsndfile (WAV/AIFF/FLAC via soundfile)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bind all interfaces; the host injects $PORT (app.py reads it).
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "app.py"]
