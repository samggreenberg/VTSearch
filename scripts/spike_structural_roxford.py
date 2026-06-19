"""Pre-impl spike: tune the structural re-rank on Revisited Oxford (ROxford5k).

This is the experiment the structural-embedder design
(``docs/plans/structural-embedder.md``) defers under "K / threshold /
live-vs-on-demand spike".  It runs the real SIFT/VLAD pipeline over the
canonical instance-retrieval benchmark and measures how the Stage-2 geometric
re-rank moves retrieval quality and latency as a function of the shortlist size
*K* and the decision threshold.

It is a research script, not product code: it lives under ``scripts/`` (not
imported by the app), caches its per-image features under ``scratch/`` (gitignored),
and prints a results table.  Rebuild the shipped codebook first
(``python scripts/build_vlad_codebook.py --images <corpus>``) so the spike runs
against a real vocabulary, then::

    python scripts/spike_structural_roxford.py            # full 5063-image run
    python scripts/spike_structural_roxford.py --max-db 800   # quick smoke pass

Ground truth + images come from ``download_roxford5k`` (see the downloader).
mAP follows the revisitop protocol (Medium: easy+hard positives, junk ignored;
Hard: hard positives, easy+junk ignored).
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from vtscore.datasets.downloader import download_roxford5k
from vtscore.media import get_embedder
from vtscore.media.structural import StructuralFeatures, aggregate_vlad
from vtscore.training.structural_similarity import (
    VerificationScorer,
    filter_features_to_box,
    structural_rerank,
)

_SCRATCH = Path(__file__).resolve().parent.parent / "scratch"
_CACHE = _SCRATCH / "roxford_features.pkl"


def _load_gray(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def _embed_corpus(jpg_dir: Path, names: list[str], embedder, max_features: int):
    """One SIFT pass per image → (VLAD matrix, list[StructuralFeatures])."""
    matcher = embedder.structural_matcher
    codebook = embedder._codebook  # noqa: SLF001 - spike reaches into the loaded embedder
    vlads: list[np.ndarray] = []
    feats: list[StructuralFeatures] = []
    t0 = time.time()
    for i, name in enumerate(names):
        gray = _load_gray(jpg_dir / f"{name}.jpg")
        feat = matcher.detect_and_describe(gray, max_features=max_features)
        vlads.append(aggregate_vlad(feat.descriptors, codebook))
        feats.append(feat.compact())
        if (i + 1) % 250 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  embedded {i + 1}/{len(names)} ({rate:.1f} img/s)")
    return np.stack(vlads, axis=0), feats


def _compute_ap(ranked_ids: list[int], positives: set[int], junk: set[int]) -> float:
    """Average precision over *ranked_ids*, dropping *junk* from the ranking."""
    if not positives:
        return float("nan")
    hits = 0
    ap = 0.0
    rank = 0
    for cid in ranked_ids:
        if cid in junk:
            continue
        rank += 1
        if cid in positives:
            hits += 1
            ap += hits / rank
    return ap / len(positives)


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-db", type=int, default=0, help="Subsample the db to N images (0 = all).")
    parser.add_argument("--max-features", type=int, default=1024, help="SIFT keypoint cap per image.")
    parser.add_argument("--ks", type=int, nargs="+", default=[25, 50, 100, 200], help="Shortlist sizes to sweep.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore the scratch feature cache.")
    args = parser.parse_args()

    _SCRATCH.mkdir(exist_ok=True)
    roxford_dir = download_roxford5k(on_progress=lambda *a: None)
    jpg_dir = roxford_dir / "jpg"
    with open(roxford_dir / "gnd_roxford5k.pkl", "rb") as f:
        gnd = pickle.load(f)  # noqa: S301 - trusted local asset
    imlist: list[str] = gnd["imlist"]
    qimlist: list[str] = gnd["qimlist"]
    if args.max_db:
        imlist = imlist[: args.max_db]
    print(f"ROxford: {len(imlist)} db images, {len(qimlist)} queries, max_features={args.max_features}")

    embedder = get_embedder("sift_vlad")
    embedder.load_models()
    codebook = embedder._codebook  # noqa: SLF001
    print(f"Codebook: {codebook.shape}  VLAD dim={codebook.shape[0] * codebook.shape[1]}")

    # --- embed db (cached) --------------------------------------------------
    blob = None
    cache_ok = _CACHE.exists() and not args.no_cache
    if cache_ok:
        with open(_CACHE, "rb") as f:
            blob = pickle.load(f)  # noqa: S301
        cache_ok = blob.get("imlist") == imlist and blob.get("max_features") == args.max_features
    if cache_ok and blob is not None:
        print("Loading cached db features...")
        db_vlad = blob["db_vlad"]
        db_feats = blob["db_feats"]
    else:
        print("Embedding db (one SIFT pass per image)...")
        db_vlad, db_feats = _embed_corpus(jpg_dir, imlist, embedder, args.max_features)
        with open(_CACHE, "wb") as f:
            pickle.dump(
                {"imlist": imlist, "max_features": args.max_features, "db_vlad": db_vlad, "db_feats": db_feats}, f
            )
    # L2-normalise db VLAD for cosine (aggregate_vlad already L2-norms, but be safe).
    db_norm = db_vlad / (np.linalg.norm(db_vlad, axis=1, keepdims=True) + 1e-12)

    snap = {i: {"local_features": db_feats[i], "embedder": "sift_vlad"} for i in range(len(imlist))}
    matcher = embedder.structural_matcher

    # --- per-query templates (crop to the gnd bounding box) -----------------
    print("Embedding queries (cropped to gnd bbx)...")
    from PIL import Image

    query_vlad: list[np.ndarray] = []
    query_templates: list[StructuralFeatures] = []
    for qi, qname in enumerate(qimlist):
        with Image.open(jpg_dir / f"{qname}.jpg") as img:
            w, h = img.size
            x0, y0, x1, y1 = gnd["gnd"][qi]["bbx"]
            crop = img.crop((x0, y0, x1, y1)).convert("L")
        gray = np.asarray(crop, dtype=np.uint8)
        feat = matcher.detect_and_describe(gray, max_features=args.max_features)
        query_vlad.append(aggregate_vlad(feat.descriptors, codebook))
        # The crop already restricts the template, so no further box filtering.
        query_templates.append(filter_features_to_box(feat, None))
    qv = np.stack(query_vlad, axis=0)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)

    # --- ground-truth sets per protocol -------------------------------------
    def sets_for(protocol: str):
        out = []
        for qi in range(len(qimlist)):
            g = gnd["gnd"][qi]
            if protocol == "medium":
                pos = set(g["easy"]) | set(g["hard"])
                junk = set(g["junk"])
            else:  # hard
                pos = set(g["hard"])
                junk = set(g["junk"]) | set(g["easy"])
            if args.max_db:
                pos = {p for p in pos if p < args.max_db}
                junk = {j for j in junk if j < args.max_db}
            out.append((pos, junk))
        return out

    gt = {"medium": sets_for("medium"), "hard": sets_for("hard")}

    # --- Stage-1 cosine ranking (shared across K) ---------------------------
    sims = qv @ db_norm.T  # (Q, N)
    stage1_order = np.argsort(-sims, axis=1)  # (Q, N) db-index ranking per query

    def stage1_map(protocol: str) -> float:
        aps = []
        for qi in range(len(qimlist)):
            pos, junk = gt[protocol][qi]
            ap = _compute_ap(stage1_order[qi].tolist(), pos, junk)
            if not np.isnan(ap):
                aps.append(ap)
        return float(np.mean(aps))

    print("\n=== Stage-1 (VLAD cosine) baseline ===")
    print(f"  mAP medium={stage1_map('medium'):.4f}  hard={stage1_map('hard'):.4f}")

    # --- Stage-2 re-rank sweep over K ---------------------------------------
    print("\n=== Stage-1 + Stage-2 geometric re-rank ===")
    print(f"{'K':>5} {'mAP-med':>9} {'mAP-hard':>9} {'rerank ms/q':>13}")
    scorer = VerificationScorer()  # cold-start gate (example-sort regime: no votes)
    for k in args.ks:
        med_aps, hard_aps = [], []
        total_ms = 0.0
        for qi in range(len(qimlist)):
            results = [{"id": int(cid), "score": float(sims[qi, cid])} for cid in stage1_order[qi]]
            t0 = time.time()
            reranked = structural_rerank(results, snap, [query_templates[qi]], scorer, matcher, top_k=k)
            total_ms += (time.time() - t0) * 1000.0
            ranked_ids = [e["id"] for e in reranked]
            for protocol, bucket in (("medium", med_aps), ("hard", hard_aps)):
                pos, junk = gt[protocol][qi]
                ap = _compute_ap(ranked_ids, pos, junk)
                if not np.isnan(ap):
                    bucket.append(ap)
        print(f"{k:>5} {np.mean(med_aps):>9.4f} {np.mean(hard_aps):>9.4f} {total_ms / len(qimlist):>13.1f}")

    # --- threshold behaviour at a representative K --------------------------
    k_ref = args.ks[len(args.ks) // 2]
    print(f"\n=== Verification-score threshold sweep (K={k_ref}, medium GT) ===")
    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'F1':>7}")
    # Collect (score, is_relevant) over each query's verified shortlist.
    scored_pairs: list[tuple[float, bool]] = []
    rel_total = 0
    for qi in range(len(qimlist)):
        pos, junk = gt["medium"][qi]
        rel_total += len(pos)
        results = [{"id": int(cid), "score": float(sims[qi, cid])} for cid in stage1_order[qi]]
        reranked = structural_rerank(results, snap, [query_templates[qi]], scorer, matcher, top_k=k_ref)
        for e in reranked[:k_ref]:
            if e["id"] in junk:
                continue
            scored_pairs.append((float(e["score"]), e["id"] in pos))
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        sel = [rel for s, rel in scored_pairs if s >= thr]
        tp = sum(1 for r in sel if r)
        fp = len(sel) - tp
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / rel_total if rel_total else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and not np.isnan(prec)) else float("nan")
        print(f"{thr:>7.2f} {prec:>10.4f} {rec:>8.4f} {f1:>7.4f}")


if __name__ == "__main__":
    main()
