"""Env-gated per-phase load-timing recorder (library tier).

When ``VTSEARCH_PROFILE_LOAD=<path>`` is set, the dataset-load pipeline records
one JSONL row per phase boundary to ``<path>`` (append mode), e.g.::

    {"device": "cuda", "media_type": "image", "embedder": "siglip",
     "dataset_id": "caltech101_s", "n": 1240, "download_size_mb": 131.0,
     "phase": "embed", "seconds": 3.21, "cold_model": false,
     "cold_download": true}

This is the measurement instrument behind
``docs/plans/progress-weight-calibration.md``: the rows are fit to an affine
per-phase cost model ``T_phase ≈ a + b·n`` whose coefficients drive the
``n``-aware :func:`load_step_weights`. It is **off by default and has no
behaviour effect when off** — the pipeline only pays a single ``os.environ``
lookup per load and, when profiling, a lightweight tracker subscription.

Kept in ``vtscore`` (no Flask) so it works from a plain CLI/library load, not
just the app.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

# step number (see _common._STATUS_TO_STEP / _TOTAL_LOAD_STEPS) -> phase label.
_STEP_PHASE = {1: "download", 2: "model_load", 3: "embed", 4: "finalize"}

# Embedders whose model has already been loaded in THIS process. First load of a
# given embedder pays the cold model-load cost; later loads reuse the resident
# model (warm). Self-detecting this is more robust than a caller-supplied flag,
# and lets the driver measure cold+warm in one process.
_seen_embedders: set[str] = set()
_seen_lock = threading.Lock()

# Active profiler for the current worker thread, so ``FinalizeProgress.begin``
# can stamp finalize sub-slot boundaries without threading a profiler argument
# through the whole finalize block.
_active = threading.local()


def profiling_enabled() -> bool:
    """True when the ``VTSEARCH_PROFILE_LOAD`` recorder is armed."""
    return bool(os.environ.get("VTSEARCH_PROFILE_LOAD"))


def _active_profiler() -> Optional["LoadProfiler"]:
    return getattr(_active, "profiler", None)


def note_finalize_slot(slot: str) -> None:
    """Record a finalize sub-slot boundary on the active profiler (no-op when
    profiling is off). Called from ``FinalizeProgress.begin``."""
    prof = _active_profiler()
    if prof is not None:
        prof._mark_finalize_slot(slot)


def _resolve_device() -> str:
    try:
        from vtscore.config import resolve_device  # noqa: PLC0415

        return resolve_device()
    except Exception:
        return "unknown"


def _download_size_mb_for(dataset_id: str) -> Optional[float]:
    try:
        from vtscore.datasets.config import DEMO_DATASETS  # noqa: PLC0415

        info = DEMO_DATASETS.get(dataset_id)
        if info and info.get("download_size_mb") is not None:
            return float(info["download_size_mb"])
    except Exception:
        pass
    return None


class LoadProfiler:
    """Records phase-boundary timestamps for a single dataset load.

    Subscribe :meth:`on_update` to the load's :class:`ProgressTracker` before the
    first phase fires; call :meth:`bind_thread` at the start of the worker thread
    (so finalize sub-slots land on this profiler); call :meth:`finish` with the
    final item count when the load completes. Timings between consecutive phase
    starts (and start→finish for the last phase) are written to the JSONL path.
    """

    def __init__(self, tracker: Any, media_type: str, embedder: str) -> None:
        self._tracker = tracker
        self._media_type = media_type or ""
        self._embedder = embedder or ""
        self._path = os.environ.get("VTSEARCH_PROFILE_LOAD", "")
        self._lock = threading.Lock()
        self._phase_start: dict[str, float] = {}
        self._phase_order: list[str] = []
        self._last_step: Any = None
        # finalize sub-slots: ordered (slot, monotonic-start)
        self._slot_start: list[tuple[str, float]] = []
        self._t0 = time.monotonic()

    # -- capture ------------------------------------------------------------
    def on_update(self, snapshot: dict[str, Any]) -> None:
        """Tracker subscriber: stamp the first time each step becomes active."""
        step = snapshot.get("step")
        if step == self._last_step:
            return
        self._last_step = step
        phase = _STEP_PHASE.get(step) if isinstance(step, int) else None
        if phase is None:
            return
        now = time.monotonic()
        with self._lock:
            if phase not in self._phase_start:
                self._phase_start[phase] = now
                self._phase_order.append(phase)

    def _mark_finalize_slot(self, slot: str) -> None:
        with self._lock:
            self._slot_start.append((slot, time.monotonic()))

    def bind_thread(self) -> None:
        _active.profiler = self

    def _unbind_thread(self) -> None:
        if getattr(_active, "profiler", None) is self:
            _active.profiler = None

    # -- emit ---------------------------------------------------------------
    def finish(self, n: int, dataset_id: str = "") -> None:
        """Write one JSONL row per phase (and per finalize sub-slot), then
        release the thread binding."""
        try:
            self._emit(n, dataset_id)
        finally:
            self._unbind_thread()

    def _emit(self, n: int, dataset_id: str) -> None:
        end = time.monotonic()
        if not self._path:
            return
        dataset_id = dataset_id or os.environ.get("VTSEARCH_PROFILE_DATASET_ID", "")
        dl_mb_env = os.environ.get("VTSEARCH_PROFILE_DOWNLOAD_MB")
        download_size_mb = float(dl_mb_env) if dl_mb_env else _download_size_mb_for(dataset_id)

        with self._lock:
            order = list(self._phase_order)
            starts = dict(self._phase_start)
            slots = list(self._slot_start)

        # Main-phase durations: consecutive phase starts, last phase → end.
        durations: dict[str, float] = {}
        for i, phase in enumerate(order):
            t_start = starts[phase]
            t_end = starts[order[i + 1]] if i + 1 < len(order) else end
            durations[phase] = max(0.0, t_end - t_start)

        with _seen_lock:
            cold_model = self._embedder not in _seen_embedders
            _seen_embedders.add(self._embedder)
        # A download slice that actually ran (not a cached/absent archive).
        cold_download = durations.get("download", 0.0) > 1.0

        device = _resolve_device()
        base = {
            "device": device,
            "media_type": self._media_type,
            "embedder": self._embedder,
            "dataset_id": dataset_id,
            "n": int(n),
            "download_size_mb": download_size_mb,
            "cold_model": cold_model,
            "cold_download": cold_download,
        }
        rows = [{**base, "phase": phase, "seconds": round(secs, 4)} for phase, secs in durations.items()]

        # Finalize sub-slot durations (partition the finalize phase); recorded
        # as phase="finalize:<slot>" for the deferred sub-share calibration.
        fin_end = end
        for i, (slot, t_start) in enumerate(slots):
            t_end = slots[i + 1][1] if i + 1 < len(slots) else fin_end
            rows.append({**base, "phase": f"finalize:{slot}", "seconds": round(max(0.0, t_end - t_start), 4)})

        line = "".join(json.dumps(r) + "\n" for r in rows)
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            pass


class _NullProfiler:
    """No-op stand-in returned when profiling is off, so callers never branch on
    ``None`` (keeps the load pipeline's cyclomatic complexity down). Deliberately
    does NOT bind the thread-local, so :func:`note_finalize_slot` short-circuits
    and the off path pays nothing beyond a couple of no-op method calls."""

    def bind_thread(self) -> None:  # noqa: D401
        pass

    def finish(self, n: int, dataset_id: str = "") -> None:
        pass


_NULL_PROFILER = _NullProfiler()


def start_profiler(tracker: Any, media_type: str, embedder: str):
    """Create + subscribe a profiler when armed; else return a no-op stand-in.

    Always returns an object with ``bind_thread`` / ``finish`` so callers invoke
    it unconditionally.
    """
    if not profiling_enabled():
        return _NULL_PROFILER
    prof = LoadProfiler(tracker, media_type, embedder)
    try:
        tracker.subscribe(prof.on_update)
    except Exception:
        return _NULL_PROFILER
    return prof
