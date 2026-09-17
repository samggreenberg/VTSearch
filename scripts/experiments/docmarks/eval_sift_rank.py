"""SIFT at a larger keypoint budget as a ranker on a DocMarks tier (#3911).

The inlier diagnostic says the shipped SIFT matcher separates most roster classes
once a page keeps more than 1,024 keypoints and the scale floor is 0.03 (#3912).
This asks the question that matters -- does it *rank* -- over the same pools and
with the same ordering rules as ``eval_splg_rank.py``, so the two backends are
paired class for class.  CPU only: pages are extracted and verified in a process
pool, and features are compacted to uint8 exactly as a cell would store them.

    python eval_sift_rank.py --tier s --budget 16384 --out <dir>

Writes ``rows.csv``, ``inliers.json`` (for ``sim_shortlist.py``) and ``summary.md``.
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
from eval_splg_rank import _gray, rank_splg, rank_splg_siglip  # noqa: E402
from sim_shortlist import shortlist_rank  # noqa: E402

#: Shortlist sizes replayed from the exhaustive inliers: Stage 1 (VLAD or SigLIP)
#: keeps its top K, SIFT re-orders them, the rest stay in Stage-1 order.
SHORTLIST_KS = (100, 500, 1000)

_FEATURES: dict[str, Any] = {}
_CROP: Any = None
_MATCHER: Any = None
_BUDGET = 0
_CODEBOOK: Any = None


def _matcher():
    global _MATCHER
    if _MATCHER is None:
        from vtscore.media.structural import SiftMatcher  # noqa: PLC0415

        _MATCHER = SiftMatcher()
    return _MATCHER


def _codebook():
    global _CODEBOOK
    if _CODEBOOK is None:
        from vtscore.media.structural import load_vlad_codebook  # noqa: PLC0415

        _CODEBOOK = load_vlad_codebook()
    return _CODEBOOK


def _vlad(feats: Any) -> np.ndarray:
    from vtscore.media.structural import aggregate_vlad  # noqa: PLC0415

    return aggregate_vlad(feats.descriptors, _codebook())


def _extract(item: tuple[str, str]) -> tuple[str, Any, int, np.ndarray]:
    page_id, path = item
    feats = _matcher().detect_and_describe(_gray(path), max_features=_BUDGET)
    # VLAD from the float descriptors, before compaction -- as the embedder does.
    return page_id, feats.compact(), feats.count, _vlad(feats)


def _verify(page_id: str) -> tuple[str, int, int]:
    stats = _matcher().verify(_CROP, _FEATURES[page_id])
    return page_id, (stats.inlier_count if stats.model_ok else 0), stats.tentative_count


def _init_budget(budget: int) -> None:
    global _BUDGET
    _BUDGET = budget


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _CROP
    from vtscore.media import get_embedder  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--floor", type=int, default=24)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        ap.error("--out is inside the corpus; an eval never writes there")
    args.out.mkdir(parents=True, exist_ok=True)

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

    ids, matrix = ev.read_vectors(embed_corpus.cell_path(args.tier, "siglip"), "siglip")
    siglip_emb = get_embedder("siglip")
    siglip_emb.load_models()
    siglip = {}
    for cid, meta in classes.items():
        vec, _ = ev.embed_query(siglip_emb, Path(meta["query_crop"]))
        siglip[cid] = dict(zip(ids, (matrix @ np.asarray(vec, dtype=np.float32)).tolist()))

    print(
        f"=== tier {args.tier}: {len(pages)} pages, {len(classes)} classes, SIFT budget {args.budget}, {args.workers} workers",
        flush=True,
    )
    t0 = time.time()
    counts = []
    vlads: dict[str, np.ndarray] = {}
    with get_context("fork").Pool(args.workers, initializer=_init_budget, initargs=(args.budget,)) as pool:
        for i, (page_id, feats, n, vlad) in enumerate(
            pool.imap_unordered(_extract, [(p.page_id, p.path) for p in pages], chunksize=8)
        ):
            _FEATURES[page_id] = feats
            vlads[page_id] = vlad
            counts.append(n)
            if (i + 1) % 1000 == 0:
                print(f"  extracted {i + 1}/{len(pages)} in {time.time() - t0:.0f}s", flush=True)
    kb = np.median(counts) * (128 + 16) / 1024
    print(
        f"  keypoints/page median {np.median(counts):.0f} (p10 {np.percentile(counts, 10):.0f}, p90 {np.percentile(counts, 90):.0f}), ~{kb:.0f} KB/page",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    all_inliers: dict[str, dict[str, list[int]]] = {}
    for cid, meta in sorted(classes.items()):
        t1 = time.time()
        _init_budget(args.budget)
        crop_full = _matcher().detect_and_describe(_gray(meta["query_crop"]), max_features=args.budget)
        crop_vlad = _vlad(crop_full)
        _CROP = crop_full.compact()
        pools = ev.class_pools(meta, pages_by_source, industry_of)
        pool_ids, positives = pools[ev.HEADLINE_POOL], pools["positives"]
        # Fork after the crop and features are set, so workers inherit both.
        with get_context("fork").Pool(args.workers) as pool:
            results = pool.map(_verify, sorted(pool_ids), chunksize=64)
        inliers = {p: (n, t) for p, n, t in results}
        all_inliers[cid] = {p: list(v) for p, v in inliers.items() if v[0] or v[1]}
        vlad_scores = {p: float(vlads[p] @ crop_vlad) for p in pool_ids}
        siglip_ranked = ev.rank_pool(siglip[cid], pool_ids)
        vlad_ranked = ev.rank_pool(vlad_scores, pool_ids)
        inl_lists = {p: list(v) for p, v in inliers.items()}
        rankings = {
            "siglip": siglip_ranked,
            "splg": rank_splg(inliers, pool_ids),
            "splg_siglip": rank_splg_siglip(inliers, siglip[cid], pool_ids, args.floor),
            "vlad": vlad_ranked,
        }
        for k in SHORTLIST_KS:
            rankings[f"vlad{k}_splg"] = shortlist_rank(vlad_ranked, inl_lists, k)
            rankings[f"siglip{k}_splg"] = shortlist_rank(siglip_ranked, inl_lists, k)
        pos_inl = sorted((inliers[p][0] for p in positives), reverse=True)
        neg_inl = sorted((inliers[p][0] for p in pool_ids - positives), reverse=True)
        for method, ranked in rankings.items():
            k = next((kk for kk in SHORTLIST_KS if method.endswith(f"{kk}_splg")), None)
            stage1 = vlad_ranked if method.startswith("vlad") else siglip_ranked
            rows.append(
                {
                    "tier": args.tier,
                    "class_id": cid,
                    "source": meta.get("source"),
                    "kind": meta.get("kind"),
                    "n_positive": len(positives),
                    "pool": ev.HEADLINE_POOL,
                    "n_pool": len(pool_ids),
                    # Same method names as eval_splg_rank.py so the two are paired;
                    # here "splg" means "rank by this backend's inliers".
                    "method": method.replace("splg", "sift"),
                    "budget": args.budget,
                    "floor": args.floor,
                    "pos_inliers_median": float(np.median(pos_inl)) if pos_inl else 0.0,
                    "neg_inliers_max": neg_inl[0] if neg_inl else 0,
                    "neg_over_floor": sum(1 for n in neg_inl if n >= args.floor),
                    "pos_over_floor": sum(1 for n in pos_inl if n >= args.floor),
                    "stage1_recall_at_k": ev.recall_at(stage1, positives, k) if k else "",
                    **ev.metrics(ranked, positives),
                }
            )
        ap_by = {r["method"]: r["ap"] for r in rows[-len(rankings) :]}
        print(
            f"  {cid}: {len(pool_ids)} pages in {time.time() - t1:.0f}s; AP siglip {ap_by['siglip']:.2f} "
            f"sift {ap_by['sift']:.2f} vlad {ap_by['vlad']:.2f} vlad500 {ap_by['vlad500_sift']:.2f} siglip500 {ap_by['siglip500_sift']:.2f}",
            flush=True,
        )

    with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "inliers.json").write_text(json.dumps(all_inliers) + "\n", encoding="utf-8")
    lines = ["| source | method | classes | mean AP | mean r@10 |", "|---|---|---:|---:|---:|"]
    for source in ("all", *sorted({r["source"] for r in rows})):
        for method in sorted({r["method"] for r in rows}):
            sel = [r for r in rows if r["method"] == method and source in ("all", r["source"])]
            lines.append(
                f"| {source} | {method} | {len(sel)} | {np.mean([r['ap'] for r in sel]):.3f} | {np.mean([r['r@10'] for r in sel]):.2f} |"
            )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
