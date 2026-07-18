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

The cuML on/off split (it moves the coverage-atlas k-means and the UMAP
projection to the GPU, changing the finalize phase's cost) is a third device
variant: repeat the GPU run with ``--cuml off`` to measure the
CPU-clustering-on-a-GPU-host cells. The profiler stamps each row with the
active cuML state, and the fit keys cuML-on rows as device ``"cuda+cuml"``.

Within one in-process run the encoder is loaded once (first load per embedder
is ``cold_model``, the rest warm) and the source archive is
downloaded/extracted once (first load per source is ``cold_download``, the
rest warm) — the recorder self-detects both. Between loads the demo
**embeddings** cache is cleared so embed always re-runs, giving fresh
embed/finalize timings at each ``n``.

``--embedders all`` covers every registered embedder for each media type.
Each (media, embedder) cell then runs in its **own subprocess** so encoder
models don't accumulate in RAM/VRAM across cells (a full image sweep is ~11
resident models otherwise); the source archives are shared through a common
``--data-dir``, so only the first cell per source pays the download.

Isolation: a scratch ``VTSEARCH_DATA_DIR`` is used so nothing touches a real
data dir (pass ``--data-dir`` to reuse one across runs/subprocesses);
``VTSEARCH_MODELS_DIR`` should point at a warm model cache (else the
model-load phase measures a cold HuggingFace download instead of a load).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure the repo root (where app.py lives) is importable no matter the cwd:
# ``python scripts/profiling/x.py`` only puts the script's own dir on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# media_type -> demo source id stem (size suffix appended: _s/_m/_l/_a).
# Chosen for calibration economy: small archives with a good ``n`` spread.
# ``ucsf_documents`` only ships an ``_a`` variant (n=150), so the document fit
# has a single size point (constant terms, no slope).
_SOURCE_BY_MEDIA = {
    "image": "caltech101",
    "audio": "esc50",
    "video": "ucf101",
    "text": "20newsgroups",
    "document": "ucsf_documents",
}
# The face media type has no demo datasets, so it cannot be calibrated through
# the demo-load path this driver exercises; it is logged as skipped.
_NO_DEMO_MEDIA = ("face",)


def _eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="JSONL output path (appended)")
    ap.add_argument(
        "--media",
        default="image,audio,video,text,document",
        help="comma list of media types",
    )
    ap.add_argument(
        "--embedders",
        default="default",
        help="'default' (the media type's default embedder), 'all' (every "
        "registered embedder for the media type), or a comma list of names",
    )
    ap.add_argument("--sizes", default="s,m,l,a", help="comma list of size suffixes")
    ap.add_argument("--reps", type=int, default=2, help="repetitions per cell (median later)")
    ap.add_argument(
        "--max-cpu-reps-size",
        default="m",
        help="on CPU, sizes larger than this get 1 rep (they are slow)",
    )
    ap.add_argument(
        "--data-dir",
        default="",
        help="persistent scratch VTSEARCH_DATA_DIR (shared across subprocesses "
        "so each source archive downloads once); default: fresh temp dir",
    )
    ap.add_argument(
        "--cuml",
        choices=("auto", "off"),
        default="auto",
        help="'off' disables cuML (VTSEARCH_DISABLE_CUML=1) so the GPU run "
        "measures the CPU-clustering finalize variant",
    )
    return ap.parse_args()


def _resolve_cells(media_types: list[str], embedders_arg: str) -> list[tuple[str, str]]:
    """Resolve the (media_type, embedder_name) cells to measure.

    Requires the media registry (``import app`` must have run). The document
    media type has no embedder — it converts to another type on load — so its
    cell carries an empty embedder name, matching what the load pipeline
    records and what the runtime cost-model lookup uses for it.
    """
    from vtscore.media import embedders_for_type  # noqa: PLC0415

    wanted = [e.strip() for e in embedders_arg.split(",") if e.strip()]
    cells: list[tuple[str, str]] = []
    for media_type in media_types:
        if media_type in _NO_DEMO_MEDIA:
            _eprint(f"[calib] SKIPPED media={media_type} (no demo datasets)")
            continue
        if media_type not in _SOURCE_BY_MEDIA:
            _eprint(f"[calib] SKIPPED media={media_type} (no source mapped)")
            continue
        embs = embedders_for_type(media_type)
        if not embs:
            # Convert-out half types (document) load + embed via a converter
            # target; the pipeline records their rows with an empty embedder.
            cells.append((media_type, ""))
            continue
        if embedders_arg == "all":
            names = [e.name for e in embs]
        elif embedders_arg == "default":
            names = [embs[0].name]  # is_default sorted first
            for other in embs[1:]:
                _eprint(f"[calib] SKIPPED embedder={other.name} media={media_type} (non-default)")
        else:
            known = {e.name for e in embs}
            names = [n for n in wanted if n in known]
        cells.extend((media_type, n) for n in names)
    return cells


def _run_cells_in_process(args: argparse.Namespace, cells: list[tuple[str, str]]) -> int:
    """Measure *cells* (all sizes × reps) in this process. Assumes ``import
    app`` has already wired the registries and the recorder env is armed."""
    from vtscore.config import EMBEDDINGS_DIR, resolve_device  # noqa: PLC0415
    from vtscore.datasets.config import DEMO_DATASETS  # noqa: PLC0415
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415
    from vtscore.datasets.load_pipeline import (  # noqa: PLC0415
        _run_importer_in_background,
        loading_tasks,
    )

    device = resolve_device()
    _eprint(f"[calib] device={device} data_dir={os.environ['VTSEARCH_DATA_DIR']} out={args.out}")

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

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    size_order = ["s", "m", "l", "a"]
    on_cpu = not device.startswith("cuda")
    cap_idx = size_order.index(args.max_cpu_reps_size) if args.max_cpu_reps_size in size_order else 1

    for media_type, embedder in cells:
        stem = _SOURCE_BY_MEDIA[media_type]
        for size in sizes:
            dataset_id = f"{stem}_{size}"
            reps = args.reps
            if on_cpu and size in size_order and size_order.index(size) > cap_idx:
                reps = 1  # large CPU cells are slow; one rep
            for r in range(reps):
                _clear_embeddings()  # force a fresh embed (keep the extracted source)
                _eprint(f"[calib] === {media_type}/{embedder or '-'}/{dataset_id} rep {r + 1}/{reps} ===")
                _load_and_wait(dataset_id, media_type, embedder)
    return 0


def _spawn_per_cell(args: argparse.Namespace, cells: list[tuple[str, str]]) -> int:
    """Run each (media, embedder) cell as a child process of this script, so
    encoder models never accumulate across cells. The shared ``--data-dir``
    (already in the environment) keeps source downloads warm between children."""
    failures = 0
    for i, (media_type, embedder) in enumerate(cells, 1):
        _eprint(f"[calib] ---- cell {i}/{len(cells)}: {media_type}/{embedder or '-'} ----")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--out",
            args.out,
            "--media",
            media_type,
            "--embedders",
            embedder or "default",
            "--sizes",
            args.sizes,
            "--reps",
            str(args.reps),
            "--max-cpu-reps-size",
            args.max_cpu_reps_size,
            "--data-dir",
            os.environ["VTSEARCH_DATA_DIR"],
            "--cuml",
            args.cuml,
        ]
        rc = subprocess.call(cmd)  # noqa: S603 — re-invokes this same script via sys.executable
        if rc != 0:
            failures += 1
            _eprint(f"[calib] cell {media_type}/{embedder or '-'} exited rc={rc}")
    return 1 if failures else 0


def main() -> int:
    args = _parse_args()

    # Arm the recorder + isolate the data dir BEFORE importing the app.
    os.environ["VTSEARCH_PROFILE_LOAD"] = os.path.abspath(args.out)
    if args.data_dir:
        scratch = os.path.abspath(args.data_dir)
        Path(scratch).mkdir(parents=True, exist_ok=True)
    else:
        scratch = tempfile.mkdtemp(prefix="calib_data_")
    os.environ["VTSEARCH_DATA_DIR"] = scratch
    os.environ.pop("VTSEARCH_SERVER_INIT", None)
    if args.cuml == "off":
        os.environ["VTSEARCH_DISABLE_CUML"] = "1"

    import app  # noqa: F401  wires Flask app + all plugins (media/importers/embedders)

    media_types = [m.strip() for m in args.media.split(",") if m.strip()]
    cells = _resolve_cells(media_types, args.embedders)
    if not cells:
        _eprint("[calib] nothing to measure")
        return 0

    if len(cells) == 1:
        rc = _run_cells_in_process(args, cells)
    else:
        rc = _spawn_per_cell(args, cells)
    _eprint("[calib] DONE")
    # Skip interpreter teardown: with a live CUDA context (+ cuML/RMM) the
    # C++ destructor chain can abort ("terminate called without an active
    # exception") after everything of value has already been measured and
    # flushed, which would make the parent misread a good cell as failed.
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    raise SystemExit(main())
