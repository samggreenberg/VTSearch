"""Scalar runtime tunables: thread counts, size caps, training and UMAP knobs.

Every value here is a plain number read from the environment at import time (or,
for :func:`resolve_decode_workers`, at call time), with no torch import and no
dependency on any other module in this package.  What they mean is documented
beside each one; the summary table lives in ``vtscore/docs/packages/config.md``.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Runtime tunables
# ---------------------------------------------------------------------------

# Thread count for native math libraries (OpenMP / MKL) and ``torch``.  The
# default of 1 keeps memory overhead low in constrained environments; each
# additional thread allocates its own scratch buffers.  Override with
# ``VTSEARCH_TORCH_THREADS`` on bigger boxes where embedding throughput
# matters more than RSS.  Consumed by ``app.py`` (OMP/MKL env vars set
# before torch import) and ``vtscore.media.torch_setup.ensure_torch_configured``
# (``torch.set_num_threads``).
TORCH_THREADS = max(1, int(os.environ.get("VTSEARCH_TORCH_THREADS", "1")))

# Upper bound on the image-decode prefetch pool (see
# :func:`resolve_decode_workers`).  Decode throughput saturates well before a
# fat node's core count, and every extra worker holds one more decoded bitmap,
# so the pool is capped rather than scaled to whatever the box happens to have.
DEFAULT_DECODE_WORKER_CAP = 8


def allocated_cpus() -> int:
    """How many CPUs this process may actually run on.

    ``os.cpu_count()`` reports the *machine's* cores, which is the wrong number
    on a shared box: a SLURM job asking for ``--cpus-per-task=8`` on a 96-core
    node is entitled to 8, and sizing a pool from 96 would oversubscribe the
    allocation.  ``os.sched_getaffinity`` reflects the cgroup/affinity mask the
    scheduler actually imposed, so it is the allocation.  Falls back to
    ``cpu_count()`` on platforms without it (macOS, Windows).
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def resolve_decode_workers(cap: int = DEFAULT_DECODE_WORKER_CAP) -> int:
    """Threads used to decode images ahead of the GPU forward during bulk embed.

    Bulk image embedding used to decode a batch on the calling thread, run the
    forward, then decode the next batch, so nothing overlapped and the GPU sat
    idle through every decode.  On a small model that is most of the wall clock
    (measured: 82% idle for base SigLIP on a V100).  Pillow's decode releases
    the GIL in C, so a plain thread pool both parallelises a batch's decode and
    overlaps it with the previous batch's forward — see
    :mod:`vtscore.media.image._image_bulk`.

    Sized from :func:`allocated_cpus`, minus one for the calling thread (which
    runs the processor and tensor marshalling while the pool decodes), capped at
    *cap*.  Override with ``VTSEARCH_DECODE_WORKERS``; ``0`` disables the
    prefetch entirely and decodes inline on the calling thread.
    """
    raw = os.environ.get("VTSEARCH_DECODE_WORKERS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return max(1, min(cap, allocated_cpus() - 1))


# Server-side file access is governed by the active login provider, not a
# configurable allow-list.  In single-user / no-auth mode the lone trusted user
# may read from and write to any server-readable path (server-path importers,
# exporters, and the file browser are unrestricted).  In multi-user mode each
# user is confined to their own ``data/<username>/`` subtree.  See
# :func:`vtscore.security.path_validation.get_file_access_base_dir`.

# Maximum size (in megabytes) accepted for a single HTTP request body.  Wired
# into Flask's ``MAX_CONTENT_LENGTH`` config in ``app.py``.  Defaults to
# ``2048`` (2 GiB): generous enough for typical media archives and dataset
# pickles while still rejecting pathological uploads with HTTP 413 before they
# consume disk.  Set ``VTSEARCH_MAX_UPLOAD_MB`` to a different positive integer
# to raise or lower the cap, or to ``0`` to disable it entirely (Flask's
# out-of-the-box "no limit" behaviour) for genuinely large-archive uploads.
MAX_UPLOAD_MB = max(0, int(os.environ.get("VTSEARCH_MAX_UPLOAD_MB", "2048")))

# Pixel budget for a single image *decode*.  Pillow's own decompression-bomb
# ceiling is lifted at startup (see :mod:`vtscore.media.image.decode`) so a
# merely-large photo — a gigapixel panorama, a whole-slide scan — is imported
# rather than refused; this budget replaces it with the protection that
# actually matters, capping how big a bitmap any one decode may materialise.
# Sources above it are downsampled (aspect preserved) before the pixels reach
# a thumbnail, embedder, extractor or converter, all of which resize to a few
# hundred pixels anyway.  64 MP is ~192 MB as RGB — comfortably above any
# real photograph, so ordinary media is never touched.  Crop/clip paths
# deliberately bypass this and decode at native size.  Set
# ``VTSEARCH_MAX_DECODE_PIXELS=0`` to disable bounding entirely.
MAX_DECODE_PIXELS = max(0, int(os.environ.get("VTSEARCH_MAX_DECODE_PIXELS", str(64_000_000))))

# Training
#
# ``TRAIN_EPOCHS`` is an *upper bound*; :func:`vtscore.training.mlp.train_model`
# also short-circuits on a loss plateau (see ``TRAIN_PATIENCE``).  Override with
# ``VTSEARCH_TRAIN_EPOCHS`` for benchmarking or to disable early-stop entirely
# by pairing with ``VTSEARCH_TRAIN_PATIENCE=0``.
TRAIN_EPOCHS = int(os.environ.get("VTSEARCH_TRAIN_EPOCHS", "200"))
# Number of epochs the training loss must fail to improve before training
# stops early.  Set to 0 to disable early-stop and always run ``TRAIN_EPOCHS``.
TRAIN_PATIENCE = int(os.environ.get("VTSEARCH_TRAIN_PATIENCE", "10"))
# Default ``calibrate_count`` baked into ``data/settings.json`` on first run.
# Each unit adds one full fold-training pass per learned-sort; lower it to
# trade calibration quality for latency.  Min 1 (clamped in
# :mod:`vtsearch.settings`).  The default is 2: the Inclusion knob is a
# quantile rule over the *pooled* held-out fold scores (see
# ``vtscore.training.thresholds.conformal_threshold``), so its resolution is
# bounded by how many calibration scores the pool holds - at ~12 votes a
# single fold yields only ~4 positive scores, i.e. ~4 usable knob positions;
# a second fold doubles that for one extra fold fit.
DEFAULT_CALIBRATE_COUNT = max(1, int(os.environ.get("VTSEARCH_CALIBRATE_COUNT", "2")))
MLP_HIDDEN_MIN = 8
MLP_HIDDEN_MAX = 32
MLP_DROPOUT = 0.5
# Label-smoothing epsilon for MLP training targets (Good trains toward
# ``1 - eps``, Bad toward ``eps``).  Not a knob-mover: it exists as tie
# insurance for the conformal inclusion rule, which needs distinct calibration
# score values - smoothing bounds the optimal logit (~ +/-2.9 at 0.05), so a
# strongly-fit fold model can't collapse every score to exact 0.0/1.0 sigmoids
# where all quantiles degenerate to the same cut.
MLP_LABEL_SMOOTHING = 0.05

# Inverse regularisation strength of the production **linear SVM head**
# (``LINEAR_SVM_HEAD`` in ``vtscore.training.mlp``, fitted by
# ``vtscore.training.svm.fit_linear_svm_head``).  1.0 is sklearn's own default
# and the value the eval harness's ``svm_linear`` arm was measured at, so the
# shipped head and the measured arm are the same fit.  Lower it to regularise
# harder on very few votes; raise it to trust the labels more.
SVM_HEAD_C = float(os.environ.get("VTSEARCH_SVM_HEAD_C", "1.0"))

# --- Browse projection (UMAP Stage 1) ---------------------------------------
# Default UMAP knobs for the VTSBrowse 2-D projection.  Overridable per
# deployment via the ``projection_n_neighbors`` / ``projection_min_dist``
# server settings; these constants are the fallback and the values the
# ingest-time pre-build stamps.  The persisted projection is keyed on the
# effective values so changing a setting forces a recompute instead of
# serving a layout fit under the old params.  See
# docs/plans/vtsbrowse-empirical-tuning.md.
PROJECTION_N_NEIGHBORS = 15
PROJECTION_MIN_DIST = 0.1

# Per-embedder UMAP projection defaults, from the empirical sweep in
# ``docs/plans/vtsbrowse-empirical-tuning.md`` (§Results). Keyed off the dataset's
# *primary* embedder and consulted when the corresponding ``ServerSettings`` knob
# is left at the global default above (an explicit operator override still wins).
# Untuned embedders fall back to the globals. Image embedders peak at a smaller
# neighbourhood than the old global 15; large ``n_neighbors`` hurt every embedder.
PROJECTION_DEFAULTS_BY_EMBEDDER: dict[str, tuple[int, float]] = {
    "clap": (15, 0.10),  # audio: flat separability peak across 10-30
    "clap_general": (15, 0.10),  # same family, same flat audio peak
    "clip": (10, 0.05),  # image: the most n_neighbors-sensitive embedder
    "siglip": (10, 0.05),
    "siglip_l": (10, 0.05),
}

# Compaction (``compact_layout``) default. The sweep found compaction consistently
# costs ~2% taxonomy separability and ~5-6% neighbourhood structure (trustworthiness
# / continuity / recall) on every dataset and embedder, so it is off and not
# exposed as a user-facing setting; the ``compact`` param on ``fit_projection``
# remains for future experimentation.
PROJECTION_COMPACT_DEFAULT = False
