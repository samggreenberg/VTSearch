# -----------------------------------------------------------------
# VTSearch — CPU Docker image
#
# Build (from repo root):
#   docker build -f docker/Dockerfile -t vtsearch .
# Run:
#   docker run -p 5000:5000 -v vtsearch-data:/app/data vtsearch
# -----------------------------------------------------------------

FROM python:3.10-slim AS base

# System packages needed by opencv, librosa, and other native deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- dependency layer ----------
# requirements/base.txt forwards to pyproject.toml via `-e .[dev]`, so the
# package source needs to be present when this layer builds. That means
# any Python source change busts this layer; full source copy happens here.
COPY requirements/ requirements/
COPY pyproject.toml ./
COPY vtsearch/ vtsearch/

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir \
        --prefer-binary \
        -r requirements/base.txt

# ---------- application layer ----------
COPY . .

# Bake the version into the image. The host computes it from git
# (e.g. `TZ=UTC git log -1 --format=%cd --date=format:%Y-%m-%dT%H:%M:%SZ`)
# and passes it as a build arg, since .git is excluded from the build context.
# Without this, vtsearch.__version__ falls back to "0.0.0-unknown".
ARG VTSEARCH_VERSION=
RUN if [ -n "$VTSEARCH_VERSION" ]; then echo "$VTSEARCH_VERSION" > vtsearch/_version.txt; fi

# Runtime data directory (models, embeddings, settings, media files).
# Mount a volume here to persist data across container restarts.
RUN mkdir -p /app/data
VOLUME /app/data

# Run as a non-root user so the image works under Kubernetes
# `runAsNonRoot: true`, OpenShift (arbitrary UID in the root group),
# and other security-hardened environments. The user is created with
# primary GID 0 and /app is made group-writable so any UID assigned at
# runtime can still read/write the app tree, the data volume, and the
# HuggingFace cache under $HOME.
RUN useradd --uid 1000 --gid 0 --create-home --home-dir /home/vtsearch vtsearch && \
    chgrp -R 0 /app /home/vtsearch && \
    chmod -R g=u /app /home/vtsearch

ENV HOME=/home/vtsearch
USER 1000

EXPOSE 5000

# OMP/MKL thread limits are already set in app.py; repeating here
# ensures they apply even if someone imports vtsearch as a library.
# VTSEARCH_SERVER_INIT=1 runs the model-load / preload / settings-sync
# sequence when gunicorn imports app.py (since the __main__ block is
# skipped under a WSGI server).
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    VTSEARCH_SERVER_INIT=1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
