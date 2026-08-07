FROM python:3.11-slim

# build-essential covers native-extension wheels that don't ship prebuilt
# manylinux binaries for this Python/arch combination.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake NLTK's sentence tokenizer into the image so containers don't need
# network access on every cold start just to fetch it. Best-effort: nltk's
# dataset names have shifted between versions (punkt vs punkt_tab), and
# app.py already re-downloads at runtime if this didn't cover what it needs.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" || true

COPY . .

# Generated at runtime (uploads, corpus docs, the SQLite DB, downloaded
# model checkpoints) - created here so the app doesn't need to on first
# request. In docker-compose.yml the whole project directory is bind-mounted
# over this, so these persist on the host across container restarts/rebuilds.
RUN mkdir -p uploads corpus data models

EXPOSE 5000

# 1 worker: the translation/semantic models (NLLB-200 + IndicSBERT, several
# GB combined) are loaded once per worker process at import time, so each
# additional worker loads its own full copy - fine on a large instance,
# likely to OOM a small one. Raise --workers only if the host has the RAM to
# spare. --timeout is raised well above gunicorn's 30s default because
# CPU-only model loading and NMT inference can legitimately take that long.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
