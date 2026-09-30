"""Regenerate the cached tile projection that tiled Stage 1 needs (#3928).

The projection is a calculated asset, so it is never committed; this script is
how it is made (owner, 2026-09-30). It is deterministic:

* **pages:** every ``len/400``-th page of the tier, in sorted id order
  (:func:`~vtscore.media.structural_tiles.fit_sample_ids`);
* **features:** SIFT at the document budget, stored the way the app stores them
  (``StructuralFeatures.compact``), so the fit sees what the app will tile;
* **fit:** whitened PCA to :data:`~vtscore.media.structural_tiles.TILE_DIM`.

Fit on **mark-sparse** document pages. #3928's M2 found that a fit on pages
dense with the marks being searched costs those marks 0.17 of Stage-1 AP.
FullMarks' all-source tier ``s`` is mostly distractor pages, and it is the fit
that was measured.

Writes ``<models dir>/structural/tile_projection_v1.npz`` and its provenance
``.json``. The models dir is ``$VTSEARCH_MODELS_DIR``; on the GRID, source
``scripts/experiments/pile/pile_env.sh`` first so it lands in the pile's cache.

    OMP_NUM_THREADS=1 python fit_tile_projection.py --tier s [--workers 32] [--out <path>]
"""

from __future__ import annotations

import argparse
import os
import subprocess  # noqa: S404 -- fixed argv, no shell
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402


def _tiles(path: str) -> np.ndarray:
    from eval_splg_rank import _gray  # noqa: PLC0415
    from vtscore.media.structural import DOCUMENT_MAX_FEATURES, SiftMatcher  # noqa: PLC0415
    from vtscore.media.structural_tiles import raw_tiles  # noqa: PLC0415

    feats = SiftMatcher().detect_and_describe(_gray(path), max_features=DOCUMENT_MAX_FEATURES).compact()
    return raw_tiles(feats)[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vtscore.config import MAX_STRUCTURAL_DETECT_PIXELS  # noqa: PLC0415
    from vtscore.media.structural import DOCUMENT_MAX_FEATURES  # noqa: PLC0415
    from vtscore.media.structural_tiles import (  # noqa: PLC0415
        FIT_SAMPLE_PAGES,
        TILE_DIM,
        fit_projection,
        fit_sample_ids,
        projection_path,
        sample_digest,
        write_projection,
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--out", type=Path, help="default: the models cache (projection_path())")
    args = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        print("WARNING: OMP_NUM_THREADS is not 1; BLAS threads will oversubscribe the pool (#3928)", flush=True)

    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(args.corpus, args.tier)}
    sample = fit_sample_ids(list(pages), FIT_SAMPLE_PAGES)
    out = args.out or projection_path()
    print(f"tier {args.tier}: {len(pages)} pages, fitting on {len(sample)} -> {out}", flush=True)

    t0 = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(_tiles, [pages[p].path for p in sample], chunksize=4)
    stacked = np.concatenate(rows)
    print(f"  {stacked.shape[0]} tiles from {len(sample)} pages in {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    projection = fit_projection(stacked, TILE_DIM)
    print(f"  fitted {TILE_DIM} dims in {time.time() - t1:.0f}s", flush=True)

    try:
        commit = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    write_projection(
        projection,
        out,
        {
            "fit_on": f"FullMarks {cfg.CORPUS_VERSION} tier {args.tier}, all sources",
            "sample_pages": len(sample),
            "sample_digest": sample_digest(sample),
            "sample_tiles": int(stacked.shape[0]),
            "budget": DOCUMENT_MAX_FEATURES,
            "detect_pixels": MAX_STRUCTURAL_DETECT_PIXELS,
            "features": "compact (fp16 keypoints, uint8 descriptors)",
            "commit": commit,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    print(f"  wrote {out} and {out.with_suffix('.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
