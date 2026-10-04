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
step's change in test AP. ``positives_final.csv`` places every test-half
positive at the final click: its rank, whether it was inside the verified
shortlist, and its gate score.

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
#: The balance's presets (#4413, #4472). Every run is scored at every beta: the returned set's
#: F-beta and the best F-beta any cut reaches. The structural line follows beta only from beta 2
#: up (#4458), so the default run serves beta 1/4 and 1, and a ``--beta 4`` run serves beta 4.
BETAS = (0.25, 1.0, 4.0)


def beta_tag(beta: float) -> str:
    """Column suffix for *beta*: 0.25 -> "025", 1.0 -> "1", 4.0 -> "4"."""
    return f"{beta:g}".replace(".", "")


def f_beta(tp: Any, k: Any, pos: int, beta: float) -> Any:
    """F-beta of a set of *k* pages holding *tp* of *pos* positives (scalars or arrays)."""
    b2 = beta * beta
    return (1 + b2) * tp / (b2 * pos + k)


def in_test_half(page_id: str, salt: str = "sota-documents") -> bool:
    """A fixed, class-independent half of the pages: the review scores here and never clicks here."""
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(f"{salt}:{page_id}".encode()).digest()[0] % 2 == 1


# --------------------------------------------------------------------------
# Feature cache
# --------------------------------------------------------------------------


def _from_cache_files(ids: list[str], cache: Path) -> dict:
    """Every page of *ids* any ``features-*.npz`` under *cache* holds (a larger tier can reuse a smaller one's)."""
    from vtscore.media.structural import StructuralFeatures  # noqa: PLC0415

    want, found = set(ids), {}
    for f in sorted(cache.glob("features-*.npz")):
        if not want - set(found):
            break
        z = np.load(f, allow_pickle=False)
        cached = [str(p) for p in z["page_ids"]]
        need = [(i, p) for i, p in enumerate(cached) if p in want and p not in found]
        if not need:
            continue
        kp, desc, starts = z["keypoints"], z["descriptors"], np.append(z["starts"], len(z["keypoints"]))
        for i, p in need:
            found[p] = StructuralFeatures(
                keypoints=kp[starts[i] : starts[i + 1]], descriptors=desc[starts[i] : starts[i + 1]]
            )
    return found


def _tile_cache_path(cache: Path, tier: str, st: Any) -> Path:
    layers = "_".join(f"{w:g}x{h:g}" for w, h in st.TILE_LAYERS)
    return cache / f"tiles-{tier}-{st.PROJECTION_NAME}-{layers}-k{st.MIN_TILE_KP}.npz"


def _cached_tiles(ids: list[str], cache: Path, tier: str, st: Any) -> dict:
    """The tiles a previous run saved for this tier, projection and tile layout, if it covers *ids*."""
    f = _tile_cache_path(cache, tier, st)
    if not f.exists():
        return {}
    z = np.load(f, allow_pickle=False)
    cached = [str(p) for p in z["page_ids"]]
    if not set(ids) <= set(cached):
        return {}
    starts = np.append(z["starts"], len(z["vectors"]))
    vec, box = z["vectors"], z["boxes"]
    out = {}
    for i, p in enumerate(cached):
        a, b = starts[i], starts[i + 1]
        out[p] = st.TileVectors(vec[a:b], box[a:b])
    return out


def _save_tiles(tiles: dict, ids: list[str], cache: Path, tier: str, st: Any) -> None:
    counts = np.array([tiles[p].vectors.shape[0] for p in ids], dtype=np.int64)
    f = _tile_cache_path(cache, tier, st)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        page_ids=np.array(ids),
        starts=np.concatenate([[0], np.cumsum(counts)[:-1]]),
        vectors=np.concatenate([tiles[p].vectors for p in ids]),
        boxes=np.concatenate([tiles[p].boxes for p in ids]),
    )
    tmp.replace(f)


def thin_pool(cid: str, pool_ids: list[str], positive: np.ndarray, keep: float) -> tuple[list[str], np.ndarray]:
    """The pool with only a seeded fraction *keep* of its positives (at least one); the rest leave the pool.

    The kept positives are the first by ``sha256("thin:<class>:<page>")``, so a run is reproducible and a
    smaller fraction keeps a subset of a larger one's.
    """
    import hashlib  # noqa: PLC0415
    import math  # noqa: PLC0415

    pos = [p for p, y in zip(pool_ids, positive) if y]
    ranked = sorted(pos, key=lambda p: hashlib.sha256(f"thin:{cid}:{p}".encode()).hexdigest())
    drop = set(ranked[max(1, math.ceil(keep * len(pos))) :])
    mask = np.array([p not in drop for p in pool_ids])
    return [p for p, m in zip(pool_ids, mask) if m], positive[mask]


def load_or_extract(ids: list[str], paths: dict[str, str], cache: Optional[Path], tier: str, workers: int) -> dict:
    """``{page id: compact StructuralFeatures}``: what the cache files hold, plus the missing pages extracted.

    A page found in any ``features-*.npz`` under *cache* is read from it, so a larger tier reuses the
    smaller tiers' files. Pages no file holds are extracted and saved as their own
    ``features-<tier>-part-<n>.npz``, so no existing file is ever rewritten.
    """
    found = _from_cache_files(ids, cache) if cache is not None else {}
    missing = [p for p in ids if p not in found]
    if not missing:
        return found
    with get_context("fork").Pool(workers) as pool:
        extracted = dict(zip(missing, pool.map(_extract, [paths[p] for p in missing], chunksize=8)))
    if cache is not None:
        save_features(extracted, missing, cache / f"features-{tier}-part-{len(list(cache.glob('features-*.npz')))}.npz")
    return {**found, **extracted}


def save_features(feats: dict, ids: list[str], f: Path) -> None:
    """Write *ids*' features to *f* in the cache format (atomically)."""
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
        for beta in BETAS:
            out[f"best_fb{beta_tag(beta)}"] = float("nan")
        for p in FLOORS:
            out[f"recall_at_p{int(p * 100)}"] = out[f"k_at_p{int(p * 100)}"] = float("nan")
        return out
    tp = np.cumsum(hits)
    k = np.arange(1, len(hits) + 1)
    precision = tp / k
    f1 = 2 * tp / (k + pos)
    out["best_f1"] = float(f1.max())
    for beta in BETAS:
        out[f"best_fb{beta_tag(beta)}"] = float(f_beta(tp, k, pos, beta).max())
    for p in FLOORS:
        ok = np.flatnonzero(precision >= p)
        deepest = int(ok.max()) if ok.size else -1
        out[f"recall_at_p{int(p * 100)}"] = float(tp[deepest] / pos) if deepest >= 0 else 0.0
        out[f"k_at_p{int(p * 100)}"] = float(deepest + 1)
    return out


def save_frame(
    out: Path,
    cid: str,
    v: int,
    pool_ids: list[str],
    positive: np.ndarray,
    test: np.ndarray,
    snap: dict,
    goods: dict,
    bads: dict,
    boxes: dict,
    det_ctx: Any,
) -> None:
    """Everything an accept rule may read at click *v*, per pool page and per vote (#4367).

    Per page: the Stage-1 score against the current queries, whether the page is in the
    verified shortlist, and its best inliers over the current templates (from the app's
    own verification cache). Per Good: its leave-one-out inliers (its own template
    excluded) and leave-one-out Stage-1 score (its own query excluded). Per Bad: its best
    inliers.
    """
    from vtscore.training import structural_stage1 as s1  # noqa: PLC0415

    cache = det_ctx.structural_verification_cache
    col = {p: i for i, p in enumerate(pool_ids)}
    # The app's template keys (structural_similarity): media, box, features, prune tag. Frames are only
    # recorded without the stop-list, so the tag is None.
    keys = [(g, boxes.get(g), id(snap[g].get("local_features")), None) for g in goods]

    def best_fit(page: str, exclude: Optional[str] = None) -> Any:
        """The page's strongest cached fit over the current templates (the app's ``best_match_stats`` key)."""
        fits = [cache.fit(k, page) for k in keys if k[0] != exclude]
        fits = [f for f in fits if f is not None]
        return max(fits, key=lambda f: (f.model_ok, f.inlier_count, f.inlier_ratio)) if fits else None

    def best(page: str, exclude: Optional[str] = None) -> float:
        f = best_fit(page, exclude)
        return float("nan") if f is None else float(f.inlier_count if f.model_ok else 0)

    def geometry(page: str, exclude: Optional[str] = None) -> tuple[float, float]:
        """``(inlier ratio, median reprojection error)`` of the page's best fit, for #4434's rules."""
        f = best_fit(page, exclude)
        if f is None or not f.model_ok:
            return float("nan"), float("nan")
        return float(f.inlier_ratio), float(f.median_reproj_error)

    def stage1(queries_from: dict) -> np.ndarray:
        q = s1.vote_queries(queries_from, snap, boxes)
        scores = np.zeros(len(pool_ids), dtype=np.float32)
        if q is None:
            return scores
        for e in s1.tiled_stage1(snap, q):
            scores[col[e["id"]]] = e["score"]
        return scores

    s1_all = stage1(goods)
    order = np.argsort(-s1_all, kind="stable")
    shortlisted = np.zeros(len(pool_ids), dtype=bool)
    shortlisted[order[: s1.LAST_TOP_K]] = True
    inliers = np.array([best(p) if shortlisted[i] else np.nan for i, p in enumerate(pool_ids)], dtype=np.float32)
    geo = np.array(
        [geometry(p) if shortlisted[i] else (np.nan, np.nan) for i, p in enumerate(pool_ids)], dtype=np.float32
    )
    good_ids = list(goods)
    good_loo_inl = np.array([best(g, exclude=g) for g in good_ids], dtype=np.float32)
    good_loo_geo = np.array([geometry(g, exclude=g) for g in good_ids], dtype=np.float32).reshape(-1, 2)
    good_loo_s1 = np.array(
        [stage1({o: None for o in goods if o != g})[col[g]] if len(goods) > 1 else np.nan for g in good_ids],
        dtype=np.float32,
    )
    bad_ids = list(bads)
    bad_inl = np.array([best(b) for b in bad_ids], dtype=np.float32)
    bad_geo = np.array([geometry(b) for b in bad_ids], dtype=np.float32).reshape(-1, 2)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{cid.replace('/', '__')}__v{v:03d}.npz",
        positive=positive,
        test=test,
        stage1=s1_all,
        shortlisted=shortlisted,
        inliers=inliers,
        ratio=geo[:, 0],
        reproj=geo[:, 1],
        good_ids=np.array(good_ids),
        good_loo_ratio=good_loo_geo[:, 0],
        good_loo_reproj=good_loo_geo[:, 1],
        good_loo_inliers=good_loo_inl,
        good_loo_stage1=good_loo_s1,
        bad_ids=np.array(bad_ids),
        bad_inliers=bad_inl,
        bad_ratio=bad_geo[:, 0],
        bad_reproj=bad_geo[:, 1],
    )


def save_opening_frame(
    out: Path,
    cid: str,
    pool_ids: list[str],
    positive: np.ndarray,
    test: np.ndarray,
    snap: dict,
    crop: Any,
    ranked: list[dict],
) -> None:
    """Click 0's frame (#4458): the example sort's shortlist, and each shortlisted page's fit to the crop.

    The example sort keeps no verification cache, so the crop is verified again against its
    shortlist (the same matcher and template, so the same fits). There are no votes yet.
    """
    from vtscore.training import structural_stage1 as s1  # noqa: PLC0415
    from vtscore.training.structural_similarity import _resolve_matcher  # noqa: PLC0415

    col = {p: i for i, p in enumerate(pool_ids)}
    head = [e["id"] for e in ranked[: s1.LAST_TOP_K]]
    shortlisted = np.zeros(len(pool_ids), dtype=bool)
    shortlisted[[col[p] for p in head]] = True
    stage1 = np.zeros(len(pool_ids), dtype=np.float32)  # the example sort's order, as a decreasing score
    for r, e in enumerate(ranked):
        stage1[col[e["id"]]] = 1.0 - r / max(1, len(ranked))
    matcher = _resolve_matcher(snap)
    fits = matcher.verify_many(crop, [snap[p]["local_features"] for p in head]) if matcher else []
    inliers = np.full(len(pool_ids), np.nan, dtype=np.float32)
    ratio = np.full(len(pool_ids), np.nan, dtype=np.float32)
    reproj = np.full(len(pool_ids), np.nan, dtype=np.float32)
    for p, f in zip(head, fits):
        i = col[p]
        inliers[i] = float(f.inlier_count if f.model_ok else 0)
        if f.model_ok:
            ratio[i], reproj[i] = float(f.inlier_ratio), float(f.median_reproj_error)
    empty = np.array([], dtype=np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{cid.replace('/', '__')}__v000.npz",
        positive=positive,
        test=test,
        stage1=stage1,
        shortlisted=shortlisted,
        inliers=inliers,
        ratio=ratio,
        reproj=reproj,
        good_ids=np.array([], dtype=str),
        good_loo_ratio=empty,
        good_loo_reproj=empty,
        good_loo_inliers=empty,
        good_loo_stage1=empty,
        bad_ids=np.array([], dtype=str),
        bad_inliers=empty,
        bad_ratio=empty,
        bad_reproj=empty,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vtscore.media.structural_tiles import load_tile_projection, tile_vectors  # noqa: PLC0415
    from vtscore.state.core import DetectorContext  # noqa: PLC0415
    from vtscore.training import structural_stage1 as s1  # noqa: PLC0415
    from vtscore.training.structural_similarity import (  # noqa: PLC0415
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
    ap.add_argument(
        "--frames",
        default="",
        help="clicks at which to save per-page frames for accept-rule studies (#4367), e.g. 10,25,50",
    )
    ap.add_argument(
        "--tile-layers",
        default="",
        help="#4415 arms: 'coarse' cuts only the 0.25 x 0.18 layer (default: the shipped layers)",
    )
    ap.add_argument("--projection", default="", help="#4415 arms: a cached projection name, e.g. tile_projection_v1")
    ap.add_argument("--stoplist", default="", help="#4170/#4180 arms: 'all' or 'gated' (default: the shipped 'off')")
    ap.add_argument(
        "--swap-halves",
        action="store_true",
        help="click in the test half and score on the click half: a second replicate per class (#4432)",
    )
    ap.add_argument(
        "--seed-crop",
        choices=("tiles", "whole"),
        help="#4170 candidate: the query crop stays a Good vote (no box) all session, its Stage-1 query either "
        "its tiles or its whole VLAD",
    )
    ap.add_argument("--beta", type=float, default=None, help="the balance passed to the app's structural line (#4458)")
    ap.add_argument(
        "--thin",
        type=float,
        default=1.0,
        help="#4488: keep this seeded fraction of each class's positives in the pool (prevalence knob)",
    )
    ap.add_argument(
        "--stop-log",
        action="store_true",
        help="#4488: write stop_log.jsonl - per click the line and the unlabelled pages above it",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import vtscore.media.structural_tiles as st  # noqa: PLC0415

    if args.tile_layers == "coarse":
        st.TILE_LAYERS = ((st.TILE_W, st.TILE_H),)
    elif args.tile_layers:
        ap.error(f"unknown --tile-layers {args.tile_layers!r}")
    if args.projection:
        st.PROJECTION_NAME = args.projection
    if args.stoplist:
        import vtscore.training.structural_similarity as ss  # noqa: PLC0415

        if args.stoplist not in ("all", "gated"):
            ap.error(f"unknown --stoplist {args.stoplist!r}")
        ss.STOPLIST_POLICY = args.stoplist
    args.out.mkdir(parents=True, exist_ok=True)
    frame_at = {int(x) for x in args.frames.split(",") if x}

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(args.corpus, args.tier)}
    ids = sorted(pages)
    t0 = time.time()
    feats = load_or_extract(ids, {p: pages[p].path for p in ids}, args.feature_cache, args.tier, args.workers)
    projection = load_tile_projection()
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    tiles = _cached_tiles(ids, args.feature_cache, args.tier, st) if args.feature_cache else {}
    if len(tiles) < len(ids):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:  # numpy releases the GIL while tiling
            tiles = dict(zip(ids, pool.map(lambda p: tile_vectors(feats[p], projection), ids)))
        if args.feature_cache:
            _save_tiles(tiles, ids, args.feature_cache, args.tier, st)
    snap_all = {
        pid: {"embedder": "sift_vlad_doc", "local_features": feats[pid], "tile_vectors": tiles[pid]} for pid in ids
    }
    n_tiles = sum(t.count for t in tiles.values())
    print(
        f"tier {args.tier}: {len(ids)} pages featured + tiled in {time.time() - t0:.0f}s; "
        f"{n_tiles / len(ids):.1f} tiles/page, {sum(t.vectors.nbytes for t in tiles.values()) / 2**30:.2f} GiB of tiles",
        flush=True,
    )
    (args.out / "run.json").write_text(
        json.dumps(
            {
                "tier": args.tier,
                "corpus_version": cfg.CORPUS_VERSION,
                "max_v": args.max_v,
                "tiled_top_k": s1.TILED_TOP_K,
                "tile_layers": [list(layer) for layer in st.TILE_LAYERS],
                "projection": st.PROJECTION_NAME,
                "stoplist": args.stoplist or "off",
                "swap_halves": args.swap_halves,
                "beta": args.beta,
                "thin": args.thin,
                "seed_crop": args.seed_crop,
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
    misses: list[dict[str, Any]] = []
    for f in sorted(args.matrix.glob("*.npz")):
        if f.name.startswith("vectors-"):
            continue
        cid = f.stem.replace("__", "/", 1)
        if wanted and cid not in wanted:
            continue
        z = np.load(f)
        pool_ids = [str(p) for p in z["pool_ids"]]
        positive = z["positives"].astype(bool)
        if args.thin < 1.0:
            pool_ids, positive = thin_pool(cid, pool_ids, positive, args.thin)
        if not positive.any():
            continue
        col = {p: i for i, p in enumerate(pool_ids)}
        test = np.array([in_test_half(p) != args.swap_halves for p in pool_ids])
        if not (positive & test).any() or not (positive & ~test).any():
            print(f"  {cid}: skipped, no positive in one half", flush=True)
            continue
        snap = {p: snap_all[p] for p in pool_ids}
        crop = _extract(classes[cid]["query_crop"])
        # #4170 candidate: the crop stays a Good vote all session. (The app's seeded examples carry no
        # ``local_features``, so today the structural path drops the crop at the first page Good.)
        seeded: dict[str, None] = {}
        if args.seed_crop:
            crop_id = f"crop:{cid}"
            snap[crop_id] = {
                "embedder": "sift_vlad_doc",
                "local_features": crop,
                "tile_vectors": tile_vectors(crop, projection),
                "seeded_example": args.seed_crop == "whole",
            }
            seeded[crop_id] = None
        placeholder = [{"id": p, "score": 0.0} for p in pool_ids]
        det_ctx = DetectorContext(detector_id=f"sota-{f.stem}", media_type="image")
        goods: dict[str, None] = {}
        bads: dict[str, None] = {}
        boxes: dict[str, tuple[float, float, float, float]] = {}
        prev_ap: Optional[float] = None
        t_class = time.time()
        for v in range(args.max_v + 1):
            t1 = time.time()
            if goods or seeded:
                ranked, line = maybe_structural_rerank(
                    placeholder, 0.5, snap, {**seeded, **goods}, boxes, det_ctx, bad_votes=bads, beta=args.beta
                )
                ranked = [e for e in ranked if e["id"] in col]  # the seeded crop is not a pool page
            else:
                ranked, line = maybe_structural_rerank_example(placeholder, 0.5, snap, crop, beta=args.beta)
            retrain_s = time.time() - t1
            order = np.array([col[e["id"]] for e in ranked])
            score = np.array([float(e["score"]) for e in ranked])
            labelled = {col[p] for p in (*goods, *bads)}
            if args.stop_log:
                above = [int(i) for i, sc in zip(order, score) if sc >= line and int(i) not in labelled]
                with (args.out / "stop_log.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"class_id": cid, "v": v, "line": float(line), "above": above}) + "\n")
            keep = test[order]
            rest, rest_score = order[keep], score[keep]
            hits = positive[rest]
            ap_now = vc.average_precision(rest, positive)
            # The returned set is what the app returns: scores at or above the line it hands back.
            accept = rest_score >= line
            g_prec, g_rec, g_f1, g_k = set_metrics(accept, hits)
            g_tp, g_pos = int((accept & hits).sum()), int(hits.sum())
            gate_fb = {
                f"gate_fb{beta_tag(b)}": float(f_beta(g_tp, g_k, g_pos, b)) if (g_k + g_pos) else float("nan")
                for b in BETAS
            }
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
                    **gate_fb,
                    "line": line,
                    **cut_metrics(hits),
                    "retrain_s": round(retrain_s, 2),
                    "top_k": s1.LAST_TOP_K,
                }
            )
            if prev_ap is not None and clicks and clicks[-1]["class_id"] == cid:
                clicks[-1]["ap_after"] = ap_now
                clicks[-1]["credit"] = ap_now - prev_ap
            prev_ap = ap_now
            if v in frame_at and goods:
                save_frame(args.out / "frames", cid, v, pool_ids, positive, test, snap, goods, bads, boxes, det_ctx)
            elif v in frame_at and not seeded:
                save_opening_frame(args.out / "frames", cid, pool_ids, positive, test, snap, crop, ranked)
            if v == args.max_v:
                # Where each test-half positive sits at the end: inside the verified shortlist or
                # beyond it, and its gate score, so a weak class's misses can be told apart.
                position = {int(i): r for r, i in enumerate(order)}
                for i in np.flatnonzero(positive & test):
                    r = position[int(i)]
                    misses.append(
                        {
                            "class_id": cid,
                            "page_id": pool_ids[i],
                            "rank_in_pool": r,
                            "rank_in_test": int(np.flatnonzero(rest == i)[0]),
                            "in_shortlist": r < s1.LAST_TOP_K,
                            "gate_score": float(score[r]),
                        }
                    )
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
                    # The detector's call on this page before the click (#4488's Smart).
                    "score_before": float(score[int(np.flatnonzero(order == nxt)[0])]),
                    "line_before": float(line),
                    "page_index": int(nxt),
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
        for name, rows in (("steps.csv", steps), ("clicks.csv", clicks), ("positives_final.csv", misses)):
            if not rows:
                continue
            with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
