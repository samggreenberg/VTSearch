#!/usr/bin/env python3
"""Calibration driver for the dataset-load progress weights.

Iterates the ``device × media_type × embedder × size`` matrix from
``docs/plans/progress-weight-calibration.md``, invoking the real demo-load path
per cell with the env-gated per-phase recorder armed
(``vtscore.datasets.stages._load_profiler``, ``VTSEARCH_PROFILE_LOAD``). Each run
appends per-phase JSONL rows to ``--out``; the fitting step
(``fit_load_weights.py``) reads them and emits the affine cost-model table.

Device is fixed at process start (``CUDA_VISIBLE_DEVICES`` must be set before
torch imports), so run this **once per device**:

    python scripts/profiling/calibrate_load_weights.py --out cpu.jsonl               # CPU: CUDA_VISIBLE_DEVICES=""
    CUDA_VISIBLE_DEVICES=0 python scripts/profiling/calibrate_load_weights.py --out gpu.jsonl

Within one process the encoder is loaded once (first load per embedder is
``cold_model``, the rest warm) and the source archive is downloaded/extracted
once (first load per source is ``cold_download``, the rest warm) — the recorder
self-detects both. Between loads the demo **embeddings** cache is cleared so
embed always re-runs, giving fresh embed/finalize timings at each ``n``.

Isolation: a scratch ``VTSEARCH_DATA_DIR`` is used so nothing touches a real
data dir; ``VTSEARCH_MODELS_DIR`` should point at a warm model cache (else the
model-load phase measures a cold HuggingFace download instead of a load).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

# media_type -> demo source id stem (size suffix appended: _s/_m/_l/_a).
_SOURCE_BY_MEDIA = {
    "image": "caltech101",
    "audio": "esc50",
}
# Media types we deliberately do NOT calibrate in this run (logged as skipped).
_SKIPPED_MEDIA = ("video", "text", "document")


def _eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="JSONL output path (appended)")
    ap.add_argument("--media", default="image,audio", help="comma list of media types")
    ap.add_argument("--sizes", default="s,m,l,a", help="comma list of size suffixes")
    ap.add_argument("--reps", type=int, default=2, help="repetitions per cell (median later)")
    ap.add_argument(
        "--max-cpu-reps-size",
        default="m",
        help="on CPU, sizes larger than this get 1 rep (they are slow)",
    )
    args = ap.parse_args()

    # Arm the recorder + isolate the data dir BEFORE importing the app.
    os.environ["VTSEARCH_PROFILE_LOAD"] = os.path.abspath(args.out)
    scratch = tempfile.mkdtemp(prefix="calib_data_")
    os.environ["VTSEARCH_DATA_DIR"] = scratch
    os.environ.pop("VTSEARCH_SERVER_INIT", None)

    import app  # noqa: F401  wires Flask app + all plugins (media/importers/embedders)
    from vtscore.config import EMBEDDINGS_DIR, resolve_device  # noqa: PLC0415
    from vtscore.datasets.config import DEMO_DATASETS  # noqa: PLC0415
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415
    from vtscore.datasets.load_pipeline import (  # noqa: PLC0415
        _run_importer_in_background,
        loading_tasks,
    )
    from vtscore.media import embedders_for_type  # noqa: PLC0415

    device = resolve_device()
    _eprint(f"[calib] device={device} data_dir={scratch} out={args.out}")

    # Wait on completion via mark_finished (the success path never sets status
    # to 'idle'; it only calls mark_finished — see _load_profiler / scout notes).
    finished: dict[str, float] = {}
    _orig_mark = loading_tasks.mark_finished

    def _mark(tid, *a, **k):
        finished.setdefault(tid, time.monotonic())
        return _orig_mark(tid, *a, **k)

    loading_tasks.mark_finished = _mark  # type: ignore[method-assign]

    def _clear_embeddings():
        try:
            shutil.rmtree(EMBEDDINGS_DIR, ignore_errors=True)
        except OSError:
            pass

    def _load_and_wait(dataset_id: str, media_type: str, embedder: str) -> bool:
        info = DEMO_DATASETS.get(dataset_id)
        if info is None:
            _eprint(f"[calib] SKIP unknown dataset {dataset_id}")
            return False
        os.environ["VTSEARCH_PROFILE_DATASET_ID"] = dataset_id
        importer = get_importer("demo")
        fv = {"name": dataset_id, "media_type": media_type, "embedder": embedder}
        t0 = time.monotonic()
        tid = _run_importer_in_background(importer, fv)
        deadline = t0 + 3600
        while tid not in finished:
            if time.monotonic() > deadline:
                _eprint(f"[calib] TIMEOUT {dataset_id}")
                return False
            time.sleep(0.25)
        tr = loading_tasks.get_tracker(tid)
        snap = tr.get() if tr else {}
        if snap.get("error"):
            _eprint(f"[calib] FAILED {dataset_id}: {snap.get('error')}")
            return False
        _eprint(f"[calib] ok {dataset_id} ({time.monotonic() - t0:.1f}s)")
        return True

    media_types = [m.strip() for m in args.media.split(",") if m.strip()]
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    size_order = ["s", "m", "l", "a"]
    on_cpu = not device.startswith("cuda")
    cap_idx = size_order.index(args.max_cpu_reps_size) if args.max_cpu_reps_size in size_order else 1

    for m in _SKIPPED_MEDIA:
        _eprint(f"[calib] SKIPPED media={m} (out of scope for this run)")

    for media_type in media_types:
        stem = _SOURCE_BY_MEDIA.get(media_type)
        if stem is None:
            _eprint(f"[calib] SKIPPED media={media_type} (no source mapped)")
            continue
        embs = embedders_for_type(media_type)
        if not embs:
            _eprint(f"[calib] SKIPPED media={media_type} (no embedder)")
            continue
        embedder = embs[0].name  # is_default sorted first
        for other in embs[1:]:
            _eprint(f"[calib] SKIPPED embedder={other.name} media={media_type} (non-default)")
        for size in sizes:
            dataset_id = f"{stem}_{size}"
            reps = args.reps
            if on_cpu and size in size_order and size_order.index(size) > cap_idx:
                reps = 1  # large CPU cells are slow; one rep
            for r in range(reps):
                _clear_embeddings()  # force a fresh embed (keep the extracted source)
                _eprint(f"[calib] === {media_type}/{embedder}/{dataset_id} rep {r + 1}/{reps} ===")
                _load_and_wait(dataset_id, media_type, embedder)

    _eprint("[calib] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
