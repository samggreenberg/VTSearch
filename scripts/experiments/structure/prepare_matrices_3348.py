#!/usr/bin/env python
"""Extract one (dataset, embedder) pile cell into a compact matrix (#3348).

    python prepare_matrices_3348.py --dataset coco_better --embedder dinov3_patch --out DIR

The patch cells are 1.5-7.5 GB because they carry every patch grid; the atlas
reads only the pooled vector.  Streaming the cell once and keeping
``(N, D) float32`` + ids + categories lets every later task load in a second,
and lets 15 tasks share the read instead of each paying it.

Vectors are written UNIT-NORMALISED -- the app normalises at ingest, and #4128
showed what an un-normalised cell does to an atlas -- with the raw norms
recorded so a cell that needed it is visible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))

import common  # noqa: E402

common.setup_env()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    from _cells_io import iter_medias  # noqa: PLC0415

    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415

    pkl = EMBEDDINGS_DIR / f"{args.dataset}__{args.embedder}.pkl"
    rows: list[tuple[int, np.ndarray, list[str]]] = []
    for cid, media in iter_medias(pkl):
        v = media.get("embeddings", {}).get(args.embedder)
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32).ravel()
        if arr.size == 0:
            continue
        # Same category fallback as structure_fits_3329.load_matrix.
        c = media.get("categories") or []
        if not c and media.get("category"):
            c = [media["category"]]
        rows.append((int(cid), arr, list(c)))
    if not rows:
        raise RuntimeError(f"no {args.embedder} vectors in {pkl}")
    rows.sort(key=lambda r: r[0])
    mat = np.stack([r[1] for r in rows])
    norms = np.linalg.norm(mat, axis=1)
    mat = mat / np.maximum(norms[:, None], 1e-12)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}__{args.embedder}"
    np.savez(out / f"{stem}.npz", vecs=mat.astype(np.float32), ids=np.array([r[0] for r in rows], dtype=np.int64))
    meta = {
        "dataset": args.dataset,
        "embedder": args.embedder,
        "source": str(pkl),
        "n": int(mat.shape[0]),
        "dim": int(mat.shape[1]),
        "raw_norm_min": float(norms.min()),
        "raw_norm_median": float(np.median(norms)),
        "raw_norm_max": float(norms.max()),
        "categories": [r[2] for r in rows],
    }
    (out / f"{stem}.json").write_text(json.dumps(meta))
    common.log(f"{stem}: {mat.shape}, raw norms {norms.min():.3f}..{norms.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
