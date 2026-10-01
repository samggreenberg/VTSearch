"""State of the App: Document Logo, the run (#4392).

The closed loop of ``app_replay_tiled.py`` through the app's own
``maybe_structural_rerank(_example)``, recording what the review reports at
**every** click (the skill's Document Logo decisions, owner 2026-10-01).

**Clicks and scores use different pages**, as in the photo reviews. Each
class's pool is split in half by a hash of the page id: the user clicks only
in the **click half**, and every readout is on the **test half**, which no
click ever touches. A metric on the unlabelled remainder would instead fall as
clicking used up the positives. (Round 1 did that: a 31-positive stamp class
"fell" to AP 0 once 30 were found.) The app still ranks the whole pool, as it
does for a user.

* ranking: AP on the test half, P@10, Goods found (in the click half);
* the returned set, as the app ships it: the inlier gate (score >= 0.5),
  with its precision, recall and F1 on the test half;
* the best F1 any cut of the same ranking reaches;
* floor-style cuts at P = 10 / 50 / 90%: the deepest cut of the ranking whose
  precision is >= P, with its recall and K. This is an oracle, since the
  structural path has no floor estimator;
* the retrain's wall clock and shortlist K.

``clicks.csv`` logs every click (page, label, test AP before and after) for
the per-image credit; every step is scored, so a click's credit is its own
step's change in test AP.

Features are cached per tier under ``--feature-cache`` (compact keypoints and
descriptors, ~170 KB a page), so a re-run skips the hour of SIFT at tier ``m``.

    python sota_documents.py --tier m --classes a,b,c --max-v 50 --out <run dir> \\
        --matrix <votes-4162>/matrix-m --feature-cache /expscratch/$USER/fullmarks/features
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
from app_replay_tiled import _extract  # noqa: E402

FLOORS = (0.1, 0.5, 0.9)


def in_test_half(page_id: str, salt: str = "sota-documents") -> bool:
    """A fixed, class-independent half of the pages: the review scores here and never clicks here."""
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(f"{salt}:{page_id}".encode()).digest()[0] % 2 == 1


# --------------------------------------------------------------------------
# Feature cache
# --------------------------------------------------------------------------


def load_or_extract(ids: list[str], paths: dict[str, str], cache: Optional[Path], tier: str, workers: int) -> dict:
    """``{page id: compact StructuralFeatures}``, read from *cache* when it holds every page."""
    from vtscore.media.structural import StructuralFeatures  # noqa: PLC0415

    f = cache / f"features-{tier}.npz" if cache else None
    if f is not None and f.exists():
        z = np.load(f, allow_pickle=False)
        cached = [str(p) for p in z["page_ids"]]
        if set(ids) <= set(cached):
            kp, desc, starts = z["keypoints"], z["descriptors"], np.append(z["starts"], len(z["keypoints"]))
            index = {p: i for i, p in enumerate(cached)}
            return {
                p: StructuralFeatures(
                    keypoints=kp[starts[index[p]] : starts[index[p] + 1]],
                    descriptors=desc[starts[index[p]] : starts[index[p] + 1]],
                )
                for p in ids
            }
    with get_context("fork").Pool(workers) as pool:
        feats = dict(zip(ids, pool.map(_extract, [paths[p] for p in ids], chunksize=8)))
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        counts = np.array([feats[p].count for p in ids], dtype=np.int64)
        tmp = f.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            page_ids=np.array(ids),
            starts=np.concatenate([[0], np.cumsum(counts)[:-1]]),
            keypoints=np.concatenate([feats[p].keypoints for p in ids]),
            descriptors=np.concatenate([feats[p].descriptors for p in ids]),
        )
        tmp.replace(f)
    return feats


# --------------------------------------------------------------------------
# Readouts on one ranking
# --------------------------------------------------------------------------


def set_metrics(accept: np.ndarray, positive: np.ndarray) -> tuple[float, float, float, int]:
    """Precision, recall, F1 and size of an accepted set on the scored pages."""
    k, tp, pos = int(accept.sum()), int((accept & positive).sum()), int(positive.sum())
    precision = tp / k if k else float("nan")
    recall = tp / pos if pos else float("nan")
    f1 = 2 * tp / (k + pos) if (k + pos) else float("nan")
    return precision, recall, f1, k


def cut_metrics(hits: np.ndarray) -> dict[str, float]:
    """Best-F1 cut, and the deepest cut with precision >= P, of a ranking given as hits."""
    pos = int(hits.sum())
    out: dict[str, float] = {}
    if pos == 0:
        out["best_f1"] = float("nan")
        for p in FLOORS:
            out[f"recall_at_p{int(p * 100)}"] = out[f"k_at_p{int(p * 100)}"] = float("nan")
        return out
    tp = np.cumsum(hits)
    k = np.arange(1, len(hits) + 1)
    precision = tp / k
    f1 = 2 * tp / (k + pos)
    out["best_f1"] = float(f1.max())
    for p in FLOORS:
        ok = np.flatnonzero(precision >= p)
        deepest = int(ok.max()) if ok.size else -1
        out[f"recall_at_p{int(p * 100)}"] = float(tp[deepest] / pos) if deepest >= 0 else 0.0
        out[f"k_at_p{int(p * 100)}"] = float(deepest + 1)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vtscore.media.structural_tiles import load_tile_projection, tile_vectors  # noqa: PLC0415
    from vtscore.state.core import DetectorContext  # noqa: PLC0415
    from vtscore.training import structural_stage1 as s1  # noqa: PLC0415
    from vtscore.training.structural_similarity import (  # noqa: PLC0415
        STRUCTURAL_DECISION_THRESHOLD,
        maybe_structural_rerank,
        maybe_structural_rerank_example,
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="m")
    ap.add_argument("--matrix", type=Path, required=True, help="#4162 matrix dir: pools and positives per class")
    ap.add_argument("--classes", default="", help="comma-separated roster subset (default: all)")
    ap.add_argument("--max-v", type=int, default=50)
    ap.add_argument("--feature-cache", type=Path)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(args.corpus, args.tier)}
    ids = sorted(pages)
    t0 = time.time()
    feats = load_or_extract(ids, {p: pages[p].path for p in ids}, args.feature_cache, args.tier, args.workers)
    projection = load_tile_projection()
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    with ThreadPoolExecutor(max_workers=args.workers) as pool:  # numpy releases the GIL while tiling
        tiles = dict(zip(ids, pool.map(lambda p: tile_vectors(feats[p], projection), ids)))
    snap_all = {
        pid: {"embedder": "sift_vlad_doc", "local_features": feats[pid], "tile_vectors": tiles[pid]} for pid in ids
    }
    print(f"tier {args.tier}: {len(ids)} pages featured + tiled in {time.time() - t0:.0f}s", flush=True)
    (args.out / "run.json").write_text(
        json.dumps(
            {
                "tier": args.tier,
                "corpus_version": cfg.CORPUS_VERSION,
                "max_v": args.max_v,
                "tiled_top_k": s1.TILED_TOP_K,
                "classes": args.classes,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    wanted = {c for c in args.classes.split(",") if c}
    steps: list[dict[str, Any]] = []
    clicks: list[dict[str, Any]] = []
    for f in sorted(args.matrix.glob("*.npz")):
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
        test = np.array([in_test_half(p) for p in pool_ids])
        if not (positive & test).any() or not (positive & ~test).any():
            print(f"  {cid}: skipped, no positive in one half", flush=True)
            continue
        snap = {p: snap_all[p] for p in pool_ids}
        crop = _extract(classes[cid]["query_crop"])
        placeholder = [{"id": p, "score": 0.0} for p in pool_ids]
        det_ctx = DetectorContext(detector_id=f"sota-{f.stem}", media_type="image")
        goods: dict[str, None] = {}
        bads: dict[str, None] = {}
        boxes: dict[str, tuple[float, float, float, float]] = {}
        prev_ap: Optional[float] = None
        t_class = time.time()
        for v in range(args.max_v + 1):
            t1 = time.time()
            if goods:
                ranked, _ = maybe_structural_rerank(placeholder, 0.5, snap, goods, boxes, det_ctx)
            else:
                ranked, _ = maybe_structural_rerank_example(placeholder, 0.5, snap, crop)
            retrain_s = time.time() - t1
            order = np.array([col[e["id"]] for e in ranked])
            score = np.array([float(e["score"]) for e in ranked])
            labelled = {col[p] for p in (*goods, *bads)}
            keep = test[order]
            rest, rest_score = order[keep], score[keep]
            hits = positive[rest]
            ap_now = vc.average_precision(rest, positive)
            g_prec, g_rec, g_f1, g_k = set_metrics(rest_score >= STRUCTURAL_DECISION_THRESHOLD, hits)
            steps.append(
                {
                    "class_id": cid,
                    "source": cid.split("/", 1)[0],
                    "v": v,
                    "found": len(goods),
                    "left": int((positive & ~test).sum()) - len(goods),
                    "n_positive": int(positive.sum()),
                    "n_test_positive": int(hits.sum()),
                    "n_pool": len(pool_ids),
                    "ap": ap_now,
                    "p10": float(hits[:10].mean()) if len(hits) else float("nan"),
                    "gate_precision": g_prec,
                    "gate_recall": g_rec,
                    "gate_f1": g_f1,
                    "gate_k": g_k,
                    **cut_metrics(hits),
                    "retrain_s": round(retrain_s, 2),
                    "top_k": s1.LAST_TOP_K,
                }
            )
            if prev_ap is not None and clicks and clicks[-1]["class_id"] == cid:
                clicks[-1]["ap_after"] = ap_now
                clicks[-1]["credit"] = ap_now - prev_ap
            prev_ap = ap_now
            if v == args.max_v:
                break
            nxt = next((int(i) for i in order if not test[int(i)] and int(i) not in labelled), None)
            if nxt is None:
                break
            pid = pool_ids[nxt]
            label = "good" if positive[nxt] else "bad"
            clicks.append(
                {
                    "class_id": cid,
                    "click": v + 1,
                    "page_id": pid,
                    "label": label,
                    "stage1_rank": int(np.flatnonzero(order == nxt)[0]),
                    "ap_before": ap_now,
                    "ap_after": float("nan"),
                    "credit": float("nan"),
                }
            )
            if label == "good":
                goods[pid] = None
                box = tm.largest_box(pages[pid], cid)
                if box is not None:
                    boxes[pid] = box
            else:
                bads[pid] = None
        last = steps[-1]
        print(
            f"  {cid}: {len(pool_ids)} pages, {int(positive.sum())} positives; click 0 AP {steps[-1 - last['v']]['ap']:.2f}, "
            f"final (v{last['v']}) AP {last['ap']:.2f}, found {last['found']}; {time.time() - t_class:.0f}s",
            flush=True,
        )
        for name, rows in (("steps.csv", steps), ("clicks.csv", clicks)):
            with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
