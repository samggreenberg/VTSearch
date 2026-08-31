#!/usr/bin/env python
"""Why does enrichment cost the text default so much? (#3127, mechanism probe)

The main run says every wrapper hurts `e5` and `bge` by ~0.09-0.13 AP, on 90 of
90 category-slices, while the same wrappers are worth +0.014 on `clap_general`.
The candidate explanation is in the text embedders' own code: E5 is an
**asymmetric retrieval model**.  It encodes a query as ``query: <text>`` and a
document as ``passage: <text>`` (`vtscore/media/text/embedder_e5.py`), trained so
that the two sides land in *different* regions and match across the gap.  Every
text wrapper ("a document about {text}", "a text passage about {text}") rewrites
the query to describe a document -- i.e. pushes it toward the side of the gap it
is supposed to be searching from.

This probe measures the size of that gap directly, on the same datasets and the
same metric, by ranking with the *passage* encoder instead of the query encoder.
If wrapping costs about what crossing the gap costs, the wrappers are paying the
asymmetry; if it costs far less, they are not, and the story is something else.

    python probe_text_prompt.py --exp /expscratch/$USER/enrich-3127 \\
        --datasets 20newsgroups_s 20newsgroups_m 20newsgroups_l
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

FIELDS = ["dataset", "embedder", "arm", "category", "query", "ap", "p10", "n_relevant", "n_pool"]


def rank_and_score(medias, query, vec, k_values):
    """One ranking, scored exactly the way `eval_text_sort` scores its own."""
    import numpy as np

    from vtscore.embedding.media_vectors import media_embedding
    from vtscore.eval.labels import evaluable_pool, media_is_positive
    from vtscore.eval.metrics import compute_metrics

    pool = evaluable_pool(medias, query.target_category)
    sims = []
    for media_id, media in pool.items():
        m = media_embedding(media)
        denom = np.linalg.norm(m) * np.linalg.norm(vec)
        sims.append((media_id, 0.0 if denom == 0 else float(np.dot(m, vec) / denom)))
    sims.sort(key=lambda t: t[1], reverse=True)
    ranked = [i for i, _ in sims]
    relevant = {cid for cid, c in pool.items() if media_is_positive(c, query.target_category)}
    return compute_metrics(ranked, relevant, query.text, query.target_category, k_values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--embedder", default="e5")
    args = ap.parse_args()

    from vtscore.embedding import initialize_models

    initialize_models()

    from vtscore.datasets.loader import load_demo_dataset
    from vtscore.eval.config import EVAL_DATASETS
    from vtscore.media import get_embedder

    out = Path(args.exp) / "results_probe"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds_id in args.datasets:
        cfg = EVAL_DATASETS[ds_id]
        medias: dict[int, dict] = {}
        t0 = time.monotonic()
        load_demo_dataset(cfg["demo_dataset"], medias, embedder_name=args.embedder)
        emb = get_embedder(args.embedder)
        print(f"[{ds_id}] {len(medias)} medias, load={time.monotonic() - t0:.0f}s", flush=True)

        if not hasattr(emb, "embed_text_passage"):
            print(f"{args.embedder} has no passage encoder; nothing to probe", file=sys.stderr)
            return 2

        arms = {
            "query": emb.embed_text,  # what the app does today
            "passage": emb.embed_text_passage,  # the other side of the asymmetry
        }
        for arm, encode in arms.items():
            aps = []
            for q in cfg["queries"]:
                vec = encode(q.text)
                m = rank_and_score(medias, q, vec, [5, 10, 20])
                aps.append(m.average_precision)
                rows.append(
                    {
                        "dataset": ds_id,
                        "embedder": args.embedder,
                        "arm": arm,
                        "category": q.target_category,
                        "query": q.text,
                        "ap": f"{m.average_precision:.6f}",
                        "p10": f"{m.precision_at_k.get(10, 0.0):.6f}",
                        "n_relevant": m.num_relevant,
                        "n_pool": m.num_total,
                    }
                )
            print(f"    {arm:9s} mAP={sum(aps) / len(aps):.4f}", flush=True)

    path = out / f"probe_prompt__{args.embedder}.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
