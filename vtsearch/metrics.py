"""Prometheus metrics for VTSearch.

Exposes a small set of counters, histograms, and gauges that cover the
signals called out in the 12.16 roadmap entry:

- ``vtsearch_votes_total`` — counter of votes recorded (labelled by
  ``vote`` and ``media_type``).
- ``vtsearch_embedding_seconds`` — histogram of single-item embedding
  latency (labelled by ``embedder`` and ``media_type``).
- ``vtsearch_training_seconds`` — histogram of MLP-training wall time
  (labelled by ``kind``: ``in_memory`` for the cached-training path,
  ``from_origins`` for the resolve+embed+train path).
- ``vtsearch_dataset_memory_bytes`` — per-dataset memory estimate
  (labelled by ``dataset_id`` and ``name``), computed at scrape time
  from the cached embedding matrix on each :class:`DatasetContext`.
- ``vtsearch_datasets_loaded`` / ``vtsearch_detectors_loaded`` — gauges
  reflecting in-memory registry counts.
- ``vtsearch_process_rss_bytes`` — best-effort resident-set size for
  the whole process (falls back silently when ``/proc/self/statm``
  isn't readable, e.g. on macOS).

The module uses a private :class:`~prometheus_client.CollectorRegistry`
so it's isolated from the default registry — important for tests, and
for any user who already exposes their own ``prometheus_client`` metrics
through the same process. The Flask blueprint in
``vtsearch.routes.metrics`` renders this registry at ``/metrics``.

Per-dataset memory is exposed as a custom collector so the snapshot is
fresh on every scrape rather than relying on counter updates at every
dataset mutation.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.metrics_core import Metric

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

# A private registry keeps VTSearch metrics independent of any
# application code that imports ``prometheus_client`` separately.
registry = CollectorRegistry(auto_describe=True)


# ---------------------------------------------------------------------------
# Static metric instances
# ---------------------------------------------------------------------------

votes_total = Counter(
    "vtsearch_votes_total",
    "Total votes recorded.",
    ("vote", "media_type"),
    registry=registry,
)

# Bucket boundaries chosen for typical single-item embedding latencies:
# stub embedders return in microseconds, real CLIP/CLAP forward passes are
# in the 50ms-2s range, document conversion + embedding can spike to 10s+.
_EMBEDDING_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

embedding_seconds = Histogram(
    "vtsearch_embedding_seconds",
    "Wall-clock latency of single-item embedding calls.",
    ("embedder", "media_type"),
    buckets=_EMBEDDING_BUCKETS,
    registry=registry,
)

# Training spans a much wider range — a few hundred ms for tiny vote sets,
# many seconds for thousands of labels, minutes if origins must be resolved
# and re-embedded.
_TRAINING_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)

training_seconds = Histogram(
    "vtsearch_training_seconds",
    "Wall-clock duration of detector MLP training.",
    ("kind",),
    buckets=_TRAINING_BUCKETS,
    registry=registry,
)

datasets_loaded = Gauge(
    "vtsearch_datasets_loaded",
    "Number of datasets currently held in memory.",
    registry=registry,
)

detectors_loaded = Gauge(
    "vtsearch_detectors_loaded",
    "Number of detectors currently held in memory.",
    registry=registry,
)

process_rss_bytes = Gauge(
    "vtsearch_process_rss_bytes",
    "Resident set size of the VTSearch process in bytes.",
    registry=registry,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def record_vote_event(vote: str, media_type: str = "") -> None:
    """Increment the vote counter.

    Called from the vote-handling paths so ``vtsearch_votes_total`` tracks
    every label transition the system observes. ``vote`` is one of
    ``"good"`` / ``"bad"`` / ``"unlabel"``; ``media_type`` is the active
    detector's media type when known, else the empty string.
    """
    try:
        votes_total.labels(vote=vote or "unknown", media_type=media_type or "unknown").inc()
    except Exception:
        # Metrics must never break the request path.
        pass


@contextmanager
def time_embedding(embedder_name: str, media_type: str = "") -> Iterator[None]:
    """Context manager that records the duration of an embedding call."""
    start = time.perf_counter()
    try:
        yield
    finally:
        try:
            embedding_seconds.labels(
                embedder=embedder_name or "unknown",
                media_type=media_type or "unknown",
            ).observe(time.perf_counter() - start)
        except Exception:
            pass


@contextmanager
def time_training(kind: str = "in_memory") -> Iterator[None]:
    """Context manager that records the duration of detector training."""
    start = time.perf_counter()
    try:
        yield
    finally:
        try:
            training_seconds.labels(kind=kind or "unknown").observe(time.perf_counter() - start)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-dataset memory collector (computed at scrape time)
# ---------------------------------------------------------------------------


def _read_rss_bytes() -> int | None:
    """Return the process resident-set size in bytes, or ``None``.

    Reads ``/proc/self/statm`` on Linux; returns ``None`` on platforms
    that don't expose it (we deliberately don't depend on ``psutil`` for
    a single gauge). The first field of ``statm`` is total program size
    in pages and the second is resident set size in pages.
    """
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            parts = fh.read().split()
        if len(parts) < 2:
            return None
        rss_pages = int(parts[1])
    except (OSError, ValueError):
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        page_size = 4096
    return rss_pages * page_size


def _estimate_dataset_bytes(ctx) -> int:
    """Estimate the in-memory footprint of a :class:`DatasetContext`.

    Dominated by the cached embedding matrix when present; falls back to
    ``num_medias * embedding_dim * 4 bytes`` (fp32) when the matrix
    hasn't been built yet. Other per-media metadata (path strings,
    origins, etc.) is small relative to the embedding bytes and is
    intentionally ignored — this metric is a useful trend, not a precise
    accounting.
    """
    matrix = getattr(ctx, "_emb_matrix", None)
    if matrix is not None and hasattr(matrix, "nbytes"):
        return int(matrix.nbytes)
    medias = getattr(ctx, "medias", None)
    if not medias:
        return 0
    sample = next(iter(medias.values()), None)
    if sample is None:
        return 0
    emb = sample.get("embedding") if isinstance(sample, dict) else None
    if emb is None or not hasattr(emb, "nbytes"):
        return 0
    return int(emb.nbytes) * len(medias)


class _DatasetMemoryCollector:
    """Prometheus collector that emits one sample per loaded dataset.

    Computing this at scrape time (rather than at every mutation) keeps
    the hot paths free of bookkeeping and guarantees the value reflects
    the current registry state without any update plumbing.
    """

    def collect(self) -> Iterator[Metric]:
        # Late import to avoid a circular import at module load.
        try:
            from vtsearch.state.core import _contexts, _state_lock  # type: ignore[attr-defined]
        except Exception:
            return

        family = GaugeMetricFamily(
            "vtsearch_dataset_memory_bytes",
            "Estimated in-memory size of each loaded dataset, in bytes.",
            labels=("dataset_id", "name"),
        )
        with _state_lock:
            for dataset_id, ctx in _contexts.items():
                name = getattr(ctx, "dataset_display_name", None) or dataset_id
                family.add_metric([dataset_id, name], float(_estimate_dataset_bytes(ctx)))
        yield family


registry.register(_DatasetMemoryCollector())


# ---------------------------------------------------------------------------
# Snapshot rendering
# ---------------------------------------------------------------------------


def _refresh_runtime_gauges() -> None:
    """Update gauges that mirror runtime state, called at scrape time."""
    try:
        from vtsearch.state.core import _contexts, _detector_contexts  # type: ignore[attr-defined]

        datasets_loaded.set(len(_contexts))
        detectors_loaded.set(len(_detector_contexts))
    except Exception:
        pass
    rss = _read_rss_bytes()
    if rss is not None:
        process_rss_bytes.set(rss)


def render() -> bytes:
    """Return the current Prometheus exposition payload as bytes."""
    _refresh_runtime_gauges()
    return generate_latest(registry)
