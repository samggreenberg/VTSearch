"""SuperPoint + LightGlue as a ranker on a DocMarks tier (#3911).

``eval_retrieval.py`` scores what is embedded in the cells; no cell exists for
SuperPoint + LightGlue, the backend the 2026-07-13 study found beats SigLIP on
documents.  This extracts SuperPoint features for every page of a tier on the
GPU, verifies each roster class's query crop against *every* page in its
headline pool (``own_verified``, #3913), and ranks by the fitted inlier count.
No shortlist: it measures what the backend can do before anyone designs a
Stage 1 for it.

Three rankings per class, over the same pool as ``eval_retrieval.py``:

* ``siglip``        -- the cosine baseline, recomputed here so it is paired;
* ``splg``          -- inlier count (model fits only), tentative matches breaking ties;
* ``splg_siglip``   -- pages at or above ``--floor`` inliers first, by inliers,
                       then everything else by SigLIP: the verify-then-fall-back
                       shape an app would use.

    python eval_splg_rank.py --tier s --out <dir>     # GPU

Writes ``rows.csv`` (one row per class and method), ``inliers.json`` (every
page's inlier count per class, so the floor can be re-swept without re-running)
and ``summary.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import eval_retrieval as ev  # noqa: E402

METHODS = ("siglip", "splg", "splg_siglip")


def _gray(path: str) -> np.ndarray:
    from vtscore.media.image._image_bulk import _load_pil  # noqa: PLC0415

    img = _load_pil(Path(path))
    return np.asarray(img.convert("L"), dtype=np.uint8)


def extract_pages(matcher: Any, pages: Sequence[Any], budget: int, workers: int, log=print) -> dict[str, Any]:
    """SuperPoint features for every page; decode on CPU threads, extract on the GPU."""
    from vtscore.media.structural import StructuralFeatures  # noqa: PLC0415

    features: dict[str, Any] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (page, gray) in enumerate(zip(pages, pool.map(lambda p: _gray(p.path), pages))):
            f = matcher.detect_and_describe(gray, max_features=budget)
            # fp16 in memory: the float SuperPoint descriptors survive it, uint8 would not.
            features[page.page_id] = StructuralFeatures(
                keypoints=f.keypoints_f32(), descriptors=f.descriptors_f32().astype(np.float16)
            )
            if (i + 1) % 500 == 0:
                log(f"  extracted {i + 1}/{len(pages)} pages in {time.time() - t0:.0f}s")
    return features


def rank_splg(inliers: dict[str, tuple[int, int]], pool: set[str]) -> list[str]:
    return sorted(pool, key=lambda p: (-inliers.get(p, (0, 0))[0], -inliers.get(p, (0, 0))[1], p))


def rank_splg_siglip(
    inliers: dict[str, tuple[int, int]], siglip: dict[str, float], pool: set[str], floor: int
) -> list[str]:
    def key(p: str):
        n = inliers.get(p, (0, 0))[0]
        verified = n >= floor
        return (not verified, -n if verified else 0, -siglip.get(p, float("-inf")), p)

    return sorted(pool, key=key)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vtscore.media import get_embedder  # noqa: PLC0415
    from vtscore.media.structural_splg import SplgMatcher  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--long-side", type=int, default=1536)
    ap.add_argument(
        "--floor", type=int, default=24, help="inlier floor for splg_siglip (the plan's LightGlue working point)"
    )
    ap.add_argument("--workers", type=int, default=8)
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
    pools = {cid: ev.class_pools(meta, pages_by_source, industry_of) for cid, meta in classes.items()}

    ids, matrix = ev.read_vectors(embed_corpus.cell_path(args.tier, "siglip"), "siglip")
    siglip_emb = get_embedder("siglip")
    siglip_emb.load_models()
    siglip = {}
    for cid, meta in classes.items():
        vec, _ = ev.embed_query(siglip_emb, Path(meta["query_crop"]))
        siglip[cid] = dict(zip(ids, (matrix @ np.asarray(vec, dtype=np.float32)).tolist()))

    matcher = SplgMatcher(extract_long_side=args.long_side)
    print(
        f"=== tier {args.tier}: {len(pages)} pages, {len(classes)} classes, budget {args.budget}, device {matcher.device}"
    )
    features = extract_pages(matcher, pages, args.budget, args.workers)

    rows: list[dict[str, Any]] = []
    all_inliers: dict[str, dict[str, list[int]]] = {}
    for cid, meta in sorted(classes.items()):
        crop = matcher.detect_and_describe(_gray(meta["query_crop"]), max_features=args.budget)
        pool = pools[cid][ev.HEADLINE_POOL]
        positives = pools[cid]["positives"]
        t0 = time.time()
        inliers: dict[str, tuple[int, int]] = {}
        for page_id in pool:
            stats = matcher.verify(crop, features[page_id])
            inliers[page_id] = (stats.inlier_count if stats.model_ok else 0, stats.tentative_count)
        all_inliers[cid] = {p: list(v) for p, v in inliers.items() if v[0] or v[1]}
        rankings = {
            "siglip": ev.rank_pool(siglip[cid], pool),
            "splg": rank_splg(inliers, pool),
            "splg_siglip": rank_splg_siglip(inliers, siglip[cid], pool, args.floor),
        }
        pos_inl = sorted((inliers[p][0] for p in positives), reverse=True)
        neg_inl = sorted((inliers[p][0] for p in pool - positives), reverse=True)
        for method, ranked in rankings.items():
            rows.append(
                {
                    "tier": args.tier,
                    "class_id": cid,
                    "source": meta.get("source"),
                    "kind": meta.get("kind"),
                    "n_positive": len(positives),
                    "pool": ev.HEADLINE_POOL,
                    "n_pool": len(pool),
                    "method": method,
                    "budget": args.budget,
                    "floor": args.floor,
                    "pos_inliers_median": float(np.median(pos_inl)) if pos_inl else 0.0,
                    "neg_inliers_max": neg_inl[0] if neg_inl else 0,
                    "neg_over_floor": sum(1 for n in neg_inl if n >= args.floor),
                    "pos_over_floor": sum(1 for n in pos_inl if n >= args.floor),
                    **ev.metrics(ranked, positives),
                }
            )
        ap_by = {r["method"]: r["ap"] for r in rows[-3:]}
        print(
            f"  {cid}: {len(pool)} pages in {time.time() - t0:.0f}s; AP siglip {ap_by['siglip']:.2f} "
            f"splg {ap_by['splg']:.2f} splg_siglip {ap_by['splg_siglip']:.2f}; positives over floor "
            f"{rows[-1]['pos_over_floor']}/{len(positives)}, negatives over floor {rows[-1]['neg_over_floor']}",
            flush=True,
        )

    with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "inliers.json").write_text(json.dumps(all_inliers) + "\n", encoding="utf-8")
    lines = ["| source | method | classes | mean AP | mean r@10 |", "|---|---|---:|---:|---:|"]
    for source in ("all", *sorted({r["source"] for r in rows})):
        for method in METHODS:
            sel = [r for r in rows if r["method"] == method and source in ("all", r["source"])]
            lines.append(
                f"| {source} | {method} | {len(sel)} | {np.mean([r['ap'] for r in sel]):.3f} | {np.mean([r['r@10'] for r in sel]):.2f} |"
            )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
