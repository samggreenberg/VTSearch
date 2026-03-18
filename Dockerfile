# -----------------------------------------------------------------
# VTSearch — CPU Docker image
#
# Build:  docker build -t vtsearch .
# Run:    docker run -p 5000:5000 -v vtsearch-data:/app/data vtsearch
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

# ---------- dependency layer (cached unless requirements change) ----------
COPY requirements-cpu.txt requirements-importers.txt requirements-exporters.txt ./
COPY vtsearch/datasets/importers/pickle/requirements.txt  vtsearch/datasets/importers/pickle/requirements.txt
COPY vtsearch/datasets/importers/folder/requirements.txt  vtsearch/datasets/importers/folder/requirements.txt
COPY vtsearch/datasets/importers/http_zip/requirements.txt vtsearch/datasets/importers/http_zip/requirements.txt
COPY vtsearch/exporters/gui/requirements.txt  vtsearch/exporters/gui/requirements.txt
COPY vtsearch/exporters/email_smtp/requirements.txt vtsearch/exporters/email_smtp/requirements.txt
COPY vtsearch/exporters/webhook/requirements.txt vtsearch/exporters/webhook/requirements.txt
COPY vtsearch/datasets/importers/combine_datasets/requirements.txt vtsearch/datasets/importers/combine_datasets/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements-cpu.txt

# ---------- application layer ----------
COPY . .

# Runtime data directory (models, embeddings, settings, media files).
# Mount a volume here to persist data across container restarts.
VOLUME /app/data

EXPOSE 5000

# OMP/MKL thread limits are already set in app.py; repeating here
# ensures they apply even if someone imports vtsearch as a library.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
