#!/usr/bin/env python3
"""#4216: how far down the typed query's sort a cell's positives sit, per embedder, offline.

    source gridenv.sh && source scripts/experiments/pile/pile_env.sh
    python reach.py --results <a SotA run's results dir> --embedders siglip,siglip2_l,clip_l,clip,embeddinggemma2 \\
        --seeds 24 --seeds-other 5 --out <dir> [--examples bowl@small,chair@small,knife@small]

Autopilot's Good phase walks the top of the typed query's sort over the voting half until it
holds 3 Goods.  A run that never gets there votes exactly the sort's top 150, so the positives
in that top 150 *are* what it finds.  This rebuilds the sort the harness opens on, with its
own pieces (``text_baseline.py``'s: the same query text, ``evaluable_pool``, the
``whole_image`` scorer and the seed's split), and writes per (embedder, category, seed):

* ``pos_top150``: positives among the voting half's top 150 (``< 3`` predicts a starved run
  under a pure Good-phase walk);
* ``rank1`` / ``rank3``: 1-based rank of the first and third positive in the voting half;
* ``n_pos_vote`` and the haystack size.

``--examples`` also writes ``examples.json``: for seed 0 and the first embedder, the top of
the sort (what the Good phase shows first) and the best-ranked positives with their boxes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAL = Path(__file__).resolve().parents[1] / "calibration"
sys.path.insert(0, str(CAL))


def split(ids: list, sim_fraction: float, seed: int) -> list:
    """The voting half: ``text_baseline._split`` (byte-identical to the harness's split)."""
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(sorted(ids)).tolist()
    return shuffled[: max(1, int(len(shuffled) * sim_fraction))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--results", type=Path, required=True, help="a run's results dir (prepare_info.json names the cells)"
    )
    ap.add_argument("--embedders", default="siglip")
    ap.add_argument("--seeds", type=int, default=24, help="seeds for the first embedder")
    ap.add_argument("--seeds-other", type=int, default=5, help="seeds for the rest")
    ap.add_argument("--examples", default="", help="categories to dump seed-0 examples for")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import common

    common.setup_env()  # the harness's environment, as text_baseline.py sets it
    import experiment_config as cfg
    from _cells_io import load_medias
    from vtscore.config import EMBEDDINGS_DIR
    from vtscore.embedding import embed_text_query
    from vtscore.eval.labels import evaluable_pool, media_is_positive
    from vtscore.eval.patch_styles import resolve_style

    prepare = json.loads((a.results / "prepare_info.json").read_text())
    ds = "coco_better"
    cats = sorted(next(iter(prepare["datasets"][ds].values()))["category_counts"])
    style = resolve_style("whole_image")
    a.out.mkdir(parents=True, exist_ok=True)
    rows, examples = [], {}
    for n_emb, emb in enumerate(a.embedders.split(",")):
        text_emb = cfg.text_embedder(emb)
        medias = load_medias(EMBEDDINGS_DIR / cfg.text_pickle_name(ds, emb))
        seeds = range(a.seeds if n_emb == 0 else a.seeds_other)
        print(f"{emb}: {len(medias)} medias, text tower {text_emb}, {len(cats)} cells x {len(seeds)} seeds", flush=True)
        for cat in cats:
            query = cfg.seed_query_text(ds, cat) or cat
            tvec = embed_text_query(query, "image", enrich=cfg.SEED_ENRICH, embedder_name=text_emb)
            if tvec is None:
                print(f"  {emb}: no text tower")
                break
            pool = evaluable_pool(medias, cat)
            sims = style.exemplar_sims(pool, np.asarray(tvec, dtype=np.float32))
            for seed in seeds:
                vote = split(list(pool), cfg.SIM_FRACTION, seed)
                sc = np.asarray([float(sims[i]) for i in vote])
                order = np.argsort(-sc, kind="stable")
                ranked = [vote[i] for i in order]
                pos = np.asarray([media_is_positive(pool[i], cat) for i in ranked])
                where = np.flatnonzero(pos) + 1
                rows.append(
                    dict(
                        embedder=emb,
                        category=cat,
                        seed=seed,
                        query=query,
                        n_vote=len(vote),
                        n_pos_vote=int(pos.sum()),
                        pos_top150=int(pos[:150].sum()),
                        pos_top300=int(pos[:300].sum()),
                        rank1=int(where[0]) if len(where) else -1,
                        rank3=int(where[2]) if len(where) > 2 else -1,
                    )
                )
                if n_emb == 0 and seed == 0 and cat in a.examples.split(","):
                    n = a.n_examples

                    def rec(i, r):
                        m = pool[i]
                        boxes = [g["box"] for g in (m.get("regions") or []) if g.get("label") == cat]
                        return dict(
                            id=int(i),
                            rank=int(r),
                            origin=m.get("origin_name"),
                            categories=m.get("categories"),
                            boxes=boxes,
                        )

                    examples[cat] = dict(
                        query=query,
                        top=[rec(ranked[k], k + 1) for k in range(n)],
                        positives=[rec(ranked[k - 1], k) for k in where[:n]],
                    )
    R = pd.DataFrame(rows)
    R.to_csv(a.out / "reach.csv", index=False)
    if examples:
        (a.out / "examples.json").write_text(json.dumps(examples, indent=1))
    sm = R[R.category.str.endswith("@small")]
    print(
        sm.assign(pred_starved=sm.pos_top150 < 3)
        .groupby("embedder")
        .agg(runs=("seed", "size"), pred_starved=("pred_starved", "mean"), median_rank3=("rank3", "median"))
        .round(3)
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
