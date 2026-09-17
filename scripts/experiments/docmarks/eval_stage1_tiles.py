"""Tiled VLAD as Stage 1 for DocMarks structural search (#3928, first probe).

#3911 found that SIFT verification ranks the roster well when every page is
verified (AP 0.78, tier s) but that a page-level VLAD vector built from the same
features ranks at chance (0.029): a mark that is ~1% of a page cannot move a
global average over its text.  The 2026-07-13 study measured max-over-tiles VLAD
multiplying Stage-1 AP 5.4x on screenshots.  This asks whether that shape works
on documents:

* each page's SIFT features (``--budget``) are split into overlapping tiles in
  normalised coordinates, one VLAD vector per tile with at least ``MIN_TILE_KP``
  keypoints;
* a page scores ``max`` over its tiles of the cosine with the query crop's VLAD;
* the resulting ranking is judged on its own, and as a shortlist whose top K are
  re-ordered by the SIFT inliers a previous ``eval_sift_rank.py`` run saved -- so
  no re-verification is needed and the Stage-2 numbers are paired with #3911's.

    python eval_stage1_tiles.py --tier s --budget 8192 \\
        --inliers <sift run>/inliers.json --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import eval_retrieval as ev  # noqa: E402
from eval_splg_rank import _gray  # noqa: E402
from sim_shortlist import shortlist_rank  # noqa: E402

#: Tile layouts as (width, height) in normalised page coordinates, stride half a
#: tile.  A DocMarks mark is ~0.1-0.3 of a page's width; the two layouts bracket it.
_ALL_LAYOUTS = {"t4": (0.25, 0.18), "t3": (0.34, 0.25)}
#: ``DOCMARKS_TILE_LAYOUTS`` (space-separated names) restricts the run -- tile
#: vectors for a 50,000-page tier are ~46 GB per layout.
LAYOUTS = {
    k: v
    for k, v in _ALL_LAYOUTS.items()
    if k in os.environ.get("DOCMARKS_TILE_LAYOUTS", " ".join(_ALL_LAYOUTS)).split()
}
MIN_TILE_KP = 20
KS = (100, 500, 1000)

_BUDGET = 0
#: Query VLADs, one row per class, set before the pool forks so workers inherit it.
#: Workers return per-class max-over-tiles scores, never the tile vectors -- at
#: tier m the vectors themselves would not fit in memory (~57 x 8,192 per page).
_QUERIES: Optional[np.ndarray] = None


def _init(budget: int) -> None:
    global _BUDGET
    _BUDGET = budget


def tile_windows(width: float, height: float) -> list[tuple[float, float, float, float]]:
    """Overlapping windows covering the unit square, stride half a tile."""
    xs = np.arange(0.0, max(1e-9, 1.0 - width) + 1e-9, width / 2)
    ys = np.arange(0.0, max(1e-9, 1.0 - height) + 1e-9, height / 2)
    return [(float(x), float(y), float(x + width), float(y + height)) for y in ys for x in xs]


def tile_vlads(
    keypoints: np.ndarray, descriptors: np.ndarray, codebook: np.ndarray, width: float, height: float
) -> np.ndarray:
    """One VLAD row per tile holding at least MIN_TILE_KP keypoints (float16)."""
    from vtscore.media.structural import aggregate_vlad  # noqa: PLC0415

    rows = []
    if keypoints.shape[0]:
        x, y = keypoints[:, 0], keypoints[:, 1]
        for x0, y0, x1, y1 in tile_windows(width, height):
            inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            if int(inside.sum()) >= MIN_TILE_KP:
                rows.append(aggregate_vlad(descriptors[inside], codebook))
    if not rows:
        rows.append(aggregate_vlad(descriptors, codebook))
    return np.asarray(rows, dtype=np.float16)


def _extract(item: tuple[str, str]) -> tuple[str, dict[str, np.ndarray], dict[str, int]]:
    from vtscore.media.structural import SiftMatcher, load_vlad_codebook  # noqa: PLC0415

    page_id, path = item
    feats = SiftMatcher().detect_and_describe(_gray(path), max_features=_BUDGET)
    kp, desc, cb = feats.keypoints_f32(), feats.descriptors_f32(), load_vlad_codebook()
    scores, counts = {}, {}
    for name, (w, h) in LAYOUTS.items():
        tv = tile_vlads(kp, desc, cb, w, h).astype(np.float32)
        scores[name] = (tv @ _QUERIES.T).max(axis=0).astype(np.float32)  # one max per class
        counts[name] = tv.shape[0]
    return page_id, scores, counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vtscore.media.structural import SiftMatcher, aggregate_vlad, load_vlad_codebook  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--inliers", type=Path, required=True, help="inliers.json from eval_sift_rank.py at the same tier")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        ap.error("--out is inside the corpus; an eval never writes there")
    args.out.mkdir(parents=True, exist_ok=True)

    inliers_all = json.loads(args.inliers.read_text(encoding="utf-8"))
    classes = {
        cid: meta
        for cid, meta in json.loads((args.corpus / "classes.json").read_text(encoding="utf-8")).items()
        if meta.get("on_roster") and meta.get("query_crop")
    }
    pages = embed_corpus.pages_for_tier(args.corpus, args.tier)
    pages_by_source: dict[str, list[str]] = {}
    industry_of: dict[str, Optional[str]] = {}
    for page in pages:
        pages_by_source.setdefault(page.source, []).append(page.page_id)
        industry_of[page.page_id] = page.meta.get("industry")

    global _QUERIES
    matcher, codebook = SiftMatcher(), load_vlad_codebook()
    order = sorted(classes)
    _QUERIES = np.stack(
        [
            aggregate_vlad(
                matcher.detect_and_describe(
                    _gray(classes[c]["query_crop"]), max_features=args.budget
                ).descriptors_f32(),
                codebook,
            )
            for c in order
        ]
    ).astype(np.float32)

    t0 = time.time()
    page_scores: dict[str, dict[str, np.ndarray]] = {}
    tile_counts: dict[str, list[int]] = {name: [] for name in LAYOUTS}
    with get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.budget,)) as pool:
        for i, (page_id, scores, counts) in enumerate(
            pool.imap_unordered(_extract, [(p.page_id, p.path) for p in pages], chunksize=8)
        ):
            page_scores[page_id] = scores
            for name, n in counts.items():
                tile_counts[name].append(n)
            if (i + 1) % 1000 == 0:
                print(f"  tiled {i + 1}/{len(pages)} in {time.time() - t0:.0f}s", flush=True)
    for name in LAYOUTS:
        print(
            f"  layout {name}: tiles/page median {np.median(tile_counts[name]):.0f}, max {max(tile_counts[name])}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    for ci, cid in enumerate(order):
        meta = classes[cid]
        pools = ev.class_pools(meta, pages_by_source, industry_of)
        pool, positives = pools[ev.HEADLINE_POOL], pools["positives"]
        inliers = inliers_all.get(cid, {})
        rankings: dict[str, list[str]] = {
            "sift": sorted(pool, key=lambda p: (-inliers.get(p, [0, 0])[0], -inliers.get(p, [0, 0])[1], p))
        }
        for name in LAYOUTS:
            scores = {p: float(page_scores[p][name][ci]) for p in pool}
            ranked = ev.rank_pool(scores, pool)
            rankings[f"vlad_{name}"] = ranked
            for k in KS:
                rankings[f"vlad_{name}{k}_sift"] = shortlist_rank(ranked, inliers, k)
        for method, ranked in rankings.items():
            k = next((kk for kk in KS if method.endswith(f"{kk}_sift")), None)
            stage1 = rankings.get(method[: method.index(str(k))] if k else method, ranked)
            rows.append(
                {
                    "tier": args.tier,
                    "class_id": cid,
                    "source": meta.get("source"),
                    "method": method,
                    "n_positive": len(positives),
                    "n_pool": len(pool),
                    "stage1_recall_at_k": ev.recall_at(stage1, positives, k) if k else "",
                    **ev.metrics(ranked, positives),
                }
            )
        ap_by = {r["method"]: r["ap"] for r in rows[-len(rankings) :]}
        print(
            f"  {cid}: "
            + " ".join(
                f"{m} {a:.2f}"
                for m, a in ap_by.items()
                if m == "sift" or m.startswith("vlad_") and (m.count("_") == 1 or "500_" in m)
            ),
            flush=True,
        )

    with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| method | mean AP | mean Stage-1 recall@K |", "|---|---:|---:|"]
    for method in sorted({r["method"] for r in rows}):
        sel = [r for r in rows if r["method"] == method]
        rec = [float(r["stage1_recall_at_k"]) for r in sel if r["stage1_recall_at_k"] != ""]
        lines.append(
            f"| {method} | {np.mean([r['ap'] for r in sel]):.3f} | {np.mean(rec):.2f} |"
            if rec
            else f"| {method} | {np.mean([r['ap'] for r in sel]):.3f} | |"
        )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
