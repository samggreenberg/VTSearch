"""End-to-end replay of the app's tiled structural path on FullMarks (#3928, build step 4).

Every ranking here comes from the app's own functions, not a copy (the eval
default arm is the app):

* **no votes:** ``maybe_structural_rerank_example`` with the class's query crop,
  which is example sort;
* **each vote:** ``maybe_structural_rerank`` with the Goods so far and their
  FullMarks boxes as RegionYes, on a real ``DetectorContext``, so the
  verification cache carries across votes as it does in the app.

Pages are ``sift_vlad_doc`` media: SIFT at 8,192 keypoints stored compact,
``tile_vectors`` from the cached projection. The matcher is the embedder's own.

Closed loop: each vote labels the top unlabelled page of the current ranking
from ground truth. Reported at each checkpoint: AP and P@10 on the unlabelled
remainder, positives found, and the wall-clock of that vote's retrain.

    python app_replay_tiled.py --tier s --matrix <votes-4162>/matrix-s --out <dir> [--workers 32] \
        [--k-policies fixed,adaptive,cap]
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

import fullmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import template_matrix as tm  # noqa: E402
import vote_curve as vc  # noqa: E402

CHECKPOINTS = (0, 3, 5, 10, 20)


def _extract(path: str) -> Any:
    from eval_splg_rank import _gray  # noqa: PLC0415
    from vtscore.media.structural import DOCUMENT_MAX_FEATURES, SiftMatcher  # noqa: PLC0415

    return SiftMatcher().detect_and_describe(_gray(path), max_features=DOCUMENT_MAX_FEATURES).compact()


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vtscore.media.structural_tiles import load_tile_projection, tile_vectors  # noqa: PLC0415
    from vtscore.state.core import DetectorContext  # noqa: PLC0415
    from vtscore.training.structural_similarity import (  # noqa: PLC0415
        maybe_structural_rerank,
        maybe_structural_rerank_example,
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--matrix", type=Path, required=True, help="#4162 matrix dir: pools and positives per class")
    ap.add_argument("--classes", default="")
    ap.add_argument("--max-v", type=int, default=20)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--k-policies", default="fixed", help="comma-separated #4391 arms: fixed, adaptive, cap")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    from vtscore.training import structural_stage1 as s1  # noqa: PLC0415

    policies = [p for p in args.k_policies.split(",") if p]
    unknown = set(policies) - {"fixed", "adaptive", "cap"}
    if unknown:
        ap.error(f"unknown K policies {sorted(unknown)}")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = [v for v in CHECKPOINTS if v <= args.max_v]

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(args.corpus, args.tier)}
    ids = sorted(pages)
    t0 = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        feats = dict(zip(ids, pool.map(_extract, [pages[p].path for p in ids], chunksize=8)))
    projection = load_tile_projection()
    snap_all = {
        pid: {"embedder": "sift_vlad_doc", "local_features": f, "tile_vectors": tile_vectors(f, projection)}
        for pid, f in feats.items()
    }
    print(f"tier {args.tier}: {len(ids)} pages featured + tiled in {time.time() - t0:.0f}s", flush=True)

    wanted = {c for c in args.classes.split(",") if c}
    rows: list[dict[str, Any]] = []
    for policy, f in [(p, f) for p in policies for f in sorted(args.matrix.glob("*.npz"))]:
        s1.K_POLICY = policy
        if f.name.startswith("vectors-"):
            continue
        cid = f.stem.replace("__", "/", 1)
        if wanted and cid not in wanted:
            continue
        z = np.load(f)
        pool_ids = [str(p) for p in z["pool_ids"]]
        positive = z["positives"].astype(bool)
        if not positive.any():
            continue
        col = {p: i for i, p in enumerate(pool_ids)}
        snap = {p: snap_all[p] for p in pool_ids}
        crop = _extract(classes[cid]["query_crop"])
        placeholder = [{"id": p, "score": 0.0} for p in pool_ids]
        det_ctx = DetectorContext(detector_id=f"replay-{f.stem}", media_type="image")
        goods: dict[str, None] = {}
        bads: dict[str, None] = {}
        boxes: dict[str, tuple[float, float, float, float]] = {}
        t_class = time.time()
        for v in range(args.max_v + 1):
            t1 = time.time()
            if goods:
                ranked, _thr = maybe_structural_rerank(placeholder, 0.5, snap, goods, boxes, det_ctx)
            else:
                ranked, _thr = maybe_structural_rerank_example(placeholder, 0.5, snap, crop)
            retrain_s = time.time() - t1
            order = np.array([col[e["id"]] for e in ranked])
            labelled = {col[p] for p in (*goods, *bads)}
            if v in checkpoints:
                rest = vc.remainder(order, labelled)
                rows.append(
                    {
                        "policy": policy,
                        "class_id": cid,
                        "v": v,
                        "found": len(goods),
                        "left": int(positive[rest].sum()),
                        "n_positive": int(positive.sum()),
                        "ap": vc.average_precision(rest, positive),
                        "p10": float(positive[rest[:10]].mean()) if len(rest) else float("nan"),
                        "retrain_s": round(retrain_s, 2),
                        "top_k": s1.LAST_TOP_K,
                        "cache_fits": len(det_ctx.structural_verification_cache or ()),
                    }
                )
            if v == args.max_v:
                break
            nxt = next((int(i) for i in order if int(i) not in labelled), None)
            if nxt is None:
                break
            pid = pool_ids[nxt]
            if positive[nxt]:
                goods[pid] = None
                box = tm.largest_box(pages[pid], cid)
                if box is not None:
                    boxes[pid] = box
            else:
                bads[pid] = None
        done = {r["v"]: r for r in rows if r["class_id"] == cid and r["policy"] == policy}
        print(
            f"  [{policy}] {cid}: {len(pool_ids)} pages, {int(positive.sum())} positives, "
            + ", ".join(f"v{v} AP {done[v]['ap']:.2f}" for v in checkpoints if v in done)
            + f"; {time.time() - t_class:.0f}s",
            flush=True,
        )
        with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
