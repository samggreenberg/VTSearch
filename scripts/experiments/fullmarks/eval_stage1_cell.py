"""What the cached Stage 1 costs and keeps (#3928).

``stage1_cell.py`` stores tile VLADs projected to a chosen width.  This prices
that choice on the roster: for each cell width, the Stage-1 recall@K, the AP
once SIFT re-orders the top K, and the wall-clock of a query against the
exhaustive verification it replaces.

Stage 2 is not re-run.  Each shortlist is re-ordered by the inliers
``eval_sift_rank.py`` already saved at the same tier and budget, which is
exactly what verifying that shortlist would compute -- so every number here is
paired class for class with #3911's and with #3928's probe.

    python eval_stage1_cell.py --tier s --cells <dir>/raw,<dir>/d256 \\
        --inliers <sift run>/inliers.json --out <dir>
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
import stage1_cell as s1  # noqa: E402
from sim_shortlist import shortlist_rank  # noqa: E402

KS = (100, 500, 1000, 2000)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--cells", required=True, help="comma-separated cell dirs")
    ap.add_argument(
        "--inliers",
        type=Path,
        help="inliers.json from eval_sift_rank.py at this tier; omit it to report Stage 1 alone, "
        "which is the only option at a tier nobody can verify exhaustively",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        ap.error("--out is inside the corpus; an eval never writes there")
    args.out.mkdir(parents=True, exist_ok=True)

    inliers_all = json.loads(args.inliers.read_text(encoding="utf-8")) if args.inliers else {}
    classes = {
        cid: meta
        for cid, meta in json.loads((args.corpus / "classes.json").read_text(encoding="utf-8")).items()
        if meta.get("on_roster") and meta.get("query_crop")
    }
    order = sorted(classes)
    pages = embed_corpus.pages_for_tier(args.corpus, args.tier)
    pages_by_source: dict[str, list[str]] = {}
    industry_of: dict[str, Optional[str]] = {}
    for page in pages:
        pages_by_source.setdefault(page.source, []).append(page.page_id)
        industry_of[page.page_id] = page.meta.get("industry")

    rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for cell_path in [Path(p) for p in args.cells.split(",")]:
        t_load = time.time()
        cell = s1.Cell(cell_path)
        load_s = time.time() - t_load
        name = f"d{cell.dim}" if cell.projection is not None else "raw"
        print(f"\n{name}: {cell.dim} dims, {cell.nbytes / 2**30:.2f} GiB, loaded in {load_s:.0f}s", flush=True)

        raw_queries = np.stack([s1.query_vlad(classes[c]["query_crop"], cell.meta["budget"]) for c in order])
        t_q = time.time()
        prepared = [cell.prepare_query(raw_queries[i]) for i in range(len(order))]
        per_query = []
        scores_by_class = {}
        for ci, cid in enumerate(order):
            t0 = time.time()
            scores = cell.scores(prepared[ci])
            _ = s1.top_k(cell.page_ids, scores, max(KS))
            per_query.append(time.time() - t0)
            scores_by_class[cid] = scores
        print(f"  search {np.mean(per_query) * 1000:.0f} ms/query (median {np.median(per_query) * 1000:.0f} ms)")
        timings.append(
            {
                "cell": name,
                "dim": cell.dim,
                "gib": round(cell.nbytes / 2**30, 3),
                "bytes_per_page": cell.meta.get("bytes_per_page"),
                "load_s": round(load_s, 1),
                "prepare_all_s": round(time.time() - t_q - sum(per_query), 2),
                "search_ms_mean": round(float(np.mean(per_query)) * 1000, 1),
                "search_ms_p50": round(float(np.median(per_query)) * 1000, 1),
            }
        )

        page_index = {p: i for i, p in enumerate(cell.page_ids)}
        for ci, cid in enumerate(order):
            meta = classes[cid]
            pools = ev.class_pools(meta, pages_by_source, industry_of)
            pool, positives = pools[ev.HEADLINE_POOL], pools["positives"]
            inliers = inliers_all.get(cid, {})
            scores = scores_by_class[cid]
            stage1_ranked = ev.rank_pool({p: float(scores[page_index[p]]) for p in pool if p in page_index}, pool)
            rankings = {f"{name}_alone": stage1_ranked}
            if inliers_all:
                rankings["sift_exhaustive"] = sorted(
                    pool, key=lambda p: (-inliers.get(p, [0, 0])[0], -inliers.get(p, [0, 0])[1], p)
                )
                for k in KS:
                    rankings[f"{name}_{k}_sift"] = shortlist_rank(stage1_ranked, inliers, k)
            else:
                # No Stage 2 to re-order with: recall@K is what a shortlist can
                # still be judged on, and it is the number that says whether
                # verifying K pages would find the class at all.
                for k in KS:
                    rankings[f"{name}_{k}_recall_only"] = stage1_ranked
            for method, ranked in rankings.items():
                k = next(
                    (kk for kk in KS if method.endswith(f"_{kk}_sift") or method.endswith(f"_{kk}_recall_only")), None
                )
                rows.append(
                    {
                        "tier": args.tier,
                        "cell": name,
                        "class_id": cid,
                        "source": meta.get("source"),
                        "method": method,
                        "k": k or "",
                        "n_positive": len(positives),
                        "n_pool": len(pool),
                        "stage1_recall_at_k": ev.recall_at(stage1_ranked, positives, k) if k else "",
                        **ev.metrics(ranked, positives),
                    }
                )
        del cell

    with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "timings.json").write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")

    lines = ["| cell | method | K | mean AP | mean recall@K |", "|---|---|---:|---:|---:|"]
    for cell_name in dict.fromkeys(r["cell"] for r in rows):
        for method in dict.fromkeys(r["method"] for r in rows if r["cell"] == cell_name):
            sel = [r for r in rows if r["cell"] == cell_name and r["method"] == method]
            rec = [float(r["stage1_recall_at_k"]) for r in sel if r["stage1_recall_at_k"] != ""]
            lines.append(
                f"| {cell_name} | {method} | {sel[0]['k'] or ''} | {np.mean([r['ap'] for r in sel]):.3f} | "
                + (f"{np.mean(rec):.2f} |" if rec else " |")
            )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
