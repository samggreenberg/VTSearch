"""SigLIP shortlist, SuperPoint + LightGlue verification, on a real tier (#3911).

The deployable shape of the backend ``eval_splg_rank.py`` measured exhaustively:
SigLIP ranks the class's headline pool (Stage 1, the vector every cell already
has), SuperPoint + LightGlue verifies only the top ``--k`` pages, and the
shortlist is re-ordered by inlier count with everything below it left in
SigLIP's order.  Pages are extracted per class and dropped afterwards, so memory
is one shortlist, not the tier.

    python eval_splg_shortlist.py --tier m --k 2000 --out <dir>     # GPU

Writes ``rows.csv`` (class x {siglip, shortlist}) and ``summary.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import eval_retrieval as ev  # noqa: E402
from eval_splg_rank import _gray, extract_pages  # noqa: E402
from sim_shortlist import shortlist_rank  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vtscore.media import get_embedder  # noqa: PLC0415
    from vtscore.media.structural_splg import SplgMatcher  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="m")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--long-side", type=int, default=1536)
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
    by_id = {p.page_id: p for p in pages}
    pages_by_source: dict[str, list[str]] = {}
    industry_of: dict[str, Optional[str]] = {}
    for page in pages:
        pages_by_source.setdefault(page.source, []).append(page.page_id)
        industry_of[page.page_id] = page.meta.get("industry")
    ids, matrix = ev.read_vectors(embed_corpus.cell_path(args.tier, "siglip"), "siglip")
    siglip_emb = get_embedder("siglip")
    siglip_emb.load_models()
    matcher = SplgMatcher(extract_long_side=args.long_side)
    print(
        f"=== tier {args.tier}: {len(pages)} pages, {len(classes)} classes, K {args.k}, device {matcher.device}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for cid, meta in sorted(classes.items()):
        t0 = time.time()
        pools = ev.class_pools(meta, pages_by_source, industry_of)
        pool, positives = pools[ev.HEADLINE_POOL], pools["positives"]
        vec, _ = ev.embed_query(siglip_emb, Path(meta["query_crop"]))
        scores = dict(zip(ids, (matrix @ np.asarray(vec, dtype=np.float32)).tolist()))
        siglip_ranked = ev.rank_pool(scores, pool)
        head = siglip_ranked[: args.k]
        features = extract_pages(matcher, [by_id[p] for p in head], args.budget, args.workers, log=lambda *_: None)
        crop = matcher.detect_and_describe(_gray(meta["query_crop"]), max_features=args.budget)
        inliers: dict[str, list[int]] = {}
        for page_id in head:
            stats = matcher.verify(crop, features[page_id])
            inliers[page_id] = [stats.inlier_count if stats.model_ok else 0, stats.tentative_count]
        del features
        for method, ranked in (
            ("siglip", siglip_ranked),
            ("shortlist", shortlist_rank(siglip_ranked, inliers, args.k)),
        ):
            rows.append(
                {
                    "tier": args.tier,
                    "class_id": cid,
                    "source": meta.get("source"),
                    "method": method,
                    "k": args.k,
                    "n_positive": len(positives),
                    "n_pool": len(pool),
                    "siglip_recall_at_k": ev.recall_at(siglip_ranked, positives, args.k),
                    **ev.metrics(ranked, positives),
                }
            )
        print(
            f"  {cid}: {time.time() - t0:.0f}s; AP siglip {rows[-2]['ap']:.3f} shortlist {rows[-1]['ap']:.3f}; "
            f"SigLIP recall@{args.k} {rows[-1]['siglip_recall_at_k']:.2f}",
            flush=True,
        )

    with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| source | method | classes | mean AP | mean r@10 | SigLIP recall@K |", "|---|---|---:|---:|---:|---:|"]
    for source in ("all", *sorted({r["source"] for r in rows})):
        for method in ("siglip", "shortlist"):
            sel = [r for r in rows if r["method"] == method and source in ("all", r["source"])]
            lines.append(
                f"| {source} | {method} | {len(sel)} | {np.mean([r['ap'] for r in sel]):.3f} "
                f"| {np.mean([r['r@10'] for r in sel]):.2f} | {np.mean([r['siglip_recall_at_k'] for r in sel]):.2f} |"
            )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
