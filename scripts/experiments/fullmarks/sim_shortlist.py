"""What SuperPoint + LightGlue keeps when it only verifies SigLIP's shortlist (#3911).

``eval_splg_rank.py`` verifies every page, which is the backend's ceiling and not
something an app can afford past a few thousand pages.  The natural Stage 1 is
the SigLIP vector every cell already has.  This replays that design from the
full run's saved ``inliers.json`` -- no GPU, no re-matching: for each shortlist
size K, SigLIP's top K are re-ordered by inlier count and everything below K
keeps SigLIP's order.

    python sim_shortlist.py --run <eval_splg_rank out dir> --out <dir>

Writes ``shortlist.csv`` (class x K) and prints mean AP and SigLIP's recall@K.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import eval_retrieval as ev  # noqa: E402

KS = (50, 100, 200, 500, 1000, 2000)


def shortlist_rank(siglip_ranked: list[str], inliers: dict[str, list[int]], k: int) -> list[str]:
    head = siglip_ranked[:k]
    order = {p: i for i, p in enumerate(head)}
    head = sorted(head, key=lambda p: (-inliers.get(p, [0, 0])[0], -inliers.get(p, [0, 0])[1], order[p]))
    return head + siglip_ranked[k:]


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vtscore.media import get_embedder  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    inliers_all = json.loads((args.run / "inliers.json").read_text(encoding="utf-8"))
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
    emb = get_embedder("siglip")
    emb.load_models()

    rows = []
    for cid, meta in sorted(classes.items()):
        pools = ev.class_pools(meta, pages_by_source, industry_of)
        pool, positives = pools[ev.HEADLINE_POOL], pools["positives"]
        vec, _ = ev.embed_query(emb, Path(meta["query_crop"]))
        scores = dict(zip(ids, (matrix @ np.asarray(vec, dtype=np.float32)).tolist()))
        siglip_ranked = ev.rank_pool(scores, pool)
        inliers = inliers_all.get(cid, {})
        for k in (0, *KS, len(pool)):
            ranked = siglip_ranked if k == 0 else shortlist_rank(siglip_ranked, inliers, k)
            rows.append(
                {
                    "class_id": cid,
                    "source": meta.get("source"),
                    "k": k,
                    "ap": ev.average_precision(ranked, positives),
                    "siglip_recall_at_k": ev.recall_at(siglip_ranked, positives, k) if k else 0.0,
                    "r@10": ev.recall_at(ranked, positives, 10),
                }
            )
    with (args.out / "shortlist.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("| K | mean AP | SigLIP recall@K | mean r@10 |")
    print("|---:|---:|---:|---:|")
    for k in sorted({r["k"] for r in rows if r["k"] <= max(KS)} | {-1}):
        sel = [r for r in rows if (r["k"] == k if k >= 0 else r["k"] > max(KS))]
        label = "all" if k < 0 else ("0 (SigLIP)" if k == 0 else str(k))
        print(
            f"| {label} | {np.mean([r['ap'] for r in sel]):.3f} | {np.mean([r['siglip_recall_at_k'] for r in sel]):.2f} | {np.mean([r['r@10'] for r in sel]):.2f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
