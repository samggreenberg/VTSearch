"""Write each roster class's pool and positives for a tier, without the template matrix (#4488).

``sota_documents.py`` reads only ``pool_ids`` and ``positives`` from a class's matrix file, so a tier
the template matrix was never built for (tier ``l``) needs only those: the same ``own_verified`` pool
``template_matrix.py`` uses (the query page excluded), from ``eval_retrieval.class_pools``.

    python class_pools_tier.py --tier l --out <dir>/matrix-l-pools
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import embed_corpus  # noqa: E402
import eval_retrieval as ev  # noqa: E402
import fullmarks_config as cfg  # noqa: E402
from template_matrix import slug  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
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
    for cid, meta in sorted(classes.items()):
        pools = ev.class_pools(meta, pages_by_source, industry_of)
        pool_ids = sorted(p for p in pools[ev.HEADLINE_POOL] if p != meta["query_page_id"])
        positives = pools["positives"]
        np.savez_compressed(
            args.out / f"{slug(cid)}.npz",
            pool_ids=np.array(pool_ids),
            positives=np.array([p in positives for p in pool_ids]),
        )
        print(f"  {cid}: {len(pool_ids)} pages, {sum(p in positives for p in pool_ids)} positives", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
