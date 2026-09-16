#!/usr/bin/env python3
"""Serve the real app over a synthetic patch dataset, for reproducing #3853 offline.

A VG-scale slate cannot be loaded here: patch grids are deliberately never
pickled (they are re-derived from pixels by the patch embedder at load, which
needs the DINOv3 weights and a GPU-sized budget).  This script builds the
in-memory shape the app ends up with after such a load - N image medias each
carrying a ``dinov3_patch`` CLS vector and a ``(14, 14, 768)`` ``patch_grid``
- straight into a registered dataset context, then runs the ordinary Flask
dev server on it.  Every request path a labeling session exercises (vote,
labelset rewrite, learned sort with region flooding, labeling-status replay,
image fetch) is then the production code over production-shaped data; only
the vectors are random.

Run from the repo root::

    VTSEARCH_DATA_DIR=/tmp/stall-data VTSEARCH_LOG_FILE=/tmp/stall-data/app.log \\
    VTSEARCH_LOG_LEVEL=INFO VTSEARCH_SLOW_REQUEST_MS=400 VTSEARCH_SLOW_PHASE_MS=300 \\
    HF_HUB_OFFLINE=1 python scripts/experiments/stall_3853/serve_synthetic.py --medias 2000 --port 5077

then drive it with ``drive_labeling.py --create-detector knife --dataset-id auto``.

The vectors have structure so the SVM head has something to fit: ten class
centres, every image's CLS vector near one centre, its patches near the CLS
vector, and on "positive" images one patch pushed along a fixed object
direction (the thing a region vote would box).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Defaults that keep this off the real data dir and off the network.  Set
# before app.py is imported: it reads the data dir and thread count at import.
os.environ.setdefault("VTSEARCH_DATA_DIR", str(Path("/tmp/stall-3853-data")))  # noqa: S108 - scratch by design
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VTSEARCH_LOG_LEVEL", "INFO")


def _png(i: int) -> bytes:
    from PIL import Image  # noqa: PLC0415

    img = Image.new("RGB", (16, 16), ((i * 37) % 256, (i * 91) % 256, (i * 53) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_medias(n: int, *, grid: int, dim: int, seed: int, positive_frac: float) -> dict[int, dict]:
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((10, dim)).astype(np.float32)
    obj = rng.standard_normal(dim).astype(np.float32)
    obj /= np.linalg.norm(obj)
    medias: dict[int, dict] = {}
    for i in range(n):
        cls = centres[i % 10] + 0.6 * rng.standard_normal(dim).astype(np.float32)
        cls /= np.linalg.norm(cls)
        patches = cls[None, :] + 0.8 * rng.standard_normal((grid * grid, dim)).astype(np.float32)
        if rng.random() < positive_frac:
            k = int(rng.integers(0, grid * grid))
            patches[k] += 3.0 * obj
        patches /= np.linalg.norm(patches, axis=1, keepdims=True)
        png = _png(i)
        medias[i] = {
            "id": i,
            "media_type": "image",
            "embedder": "dinov3_patch",
            "embeddings": {"dinov3_patch": cls},
            "patch_grid": patches.reshape(grid, grid, dim).astype(np.float16),
            "media_bytes": png,
            "media_string": None,
            "media_path": None,
            "filename": f"synthetic_{i:05d}.png",
            "origin": None,
            "origin_name": f"synthetic_{i:05d}.png",
            "md5": hashlib.md5(png).hexdigest(),  # noqa: S324 - content id, not security
            "file_size": len(png),
            "duration": 0,
            "category": f"class_{i % 10}",
            "width": 16,
            "height": 16,
        }
    return medias


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--medias", type=int, default=2000)
    ap.add_argument("--grid", type=int, default=14, help="patch grid side (DINOv3 ViT-B/16 at 224px is 14)")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--positive-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=3853)
    ap.add_argument("--port", type=int, default=int(os.environ.get("VTSEARCH_PORT", "5077")))
    ap.add_argument("--no-atlas", action="store_true", help="skip the coverage atlas build")
    args = ap.parse_args()

    print(f"building {args.medias} synthetic patch medias ({args.grid}x{args.grid}x{args.dim})...", flush=True)
    medias = build_medias(args.medias, grid=args.grid, dim=args.dim, seed=args.seed, positive_frac=args.positive_frac)

    # Import the app after the environment is pinned.  ``initialize_server``
    # is deliberately not called: its embedder preload would try to fetch the
    # DINOv3 weights for the registered dataset.  Everything the request paths
    # need from it (torch thread config, the stall diagnostics) is done here.
    from app import app  # noqa: PLC0415

    from vtscore.concurrency.events import uncap_sse_connections  # noqa: PLC0415
    from vtscore.concurrency.stalls import (  # noqa: PLC0415
        freeze_gc_after_preload,
        start_stall_diagnostics_from_env,
    )
    from vtscore.datasets.registry import add_loaded_id, register_dataset  # noqa: PLC0415
    from vtscore.embedding import initialize_models  # noqa: PLC0415
    from vtscore.state import core  # noqa: PLC0415
    from vtscore.state.coverage import build_coverage_atlas  # noqa: PLC0415

    start_stall_diagnostics_from_env()
    initialize_models()
    # ``initialize_server`` freezes the GC once its preload is done (#3870);
    # do the same here or the offline reproduction measures a different
    # process shape from the deployed one.  ``VTSEARCH_GC_FREEZE=0`` skips it,
    # which is how the two arms are compared.
    frozen = freeze_gc_after_preload()
    if frozen is not None:
        print(f"froze {frozen[0]} objects for GC in {frozen[1]:.0f}ms", flush=True)

    entry = register_dataset(
        name=f"synthetic-patch-{args.medias}",
        media_type="image",
        num_items=len(medias),
        pkl_path=str(Path(os.environ["VTSEARCH_DATA_DIR"]) / "synthetic-never-written.pkl"),
        origin="synthetic",
        embedder="dinov3_patch",
        bound_embedders=["dinov3_patch"],
        readers=["*"],
    )
    ctx = core.DatasetContext(entry["id"])
    core.register_context(ctx)
    ctx.medias.update(medias)
    add_loaded_id(entry["id"])
    if not args.no_atlas:
        print("building the coverage atlas...", flush=True)
        core.set_thread_dataset_context(ctx)
        build_coverage_atlas()
    print(f"dataset registered: id={entry['id']} medias={len(ctx.medias)}", flush=True)
    print(f"serving on http://127.0.0.1:{args.port}  (data dir {os.environ['VTSEARCH_DATA_DIR']})", flush=True)
    uncap_sse_connections()
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
