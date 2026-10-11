#!/usr/bin/env python
"""Click 0 of an example opening (#4779): the example sort's line with and without the examples in its count.

An example sort ranks every media of the collection by cosine to the starting photos' centroid, and the app
drew its display line over all of them, so the starting photos - at the top of their own sort - counted as
what stands out. This script rebuilds each cell's click 0 with the harness's own pieces (``evaluable_pool``,
``stratified_split`` on ``RandomState(seed)``, ``choose_examples``, ``example_sort``, as
``example_baseline.py`` does) and draws the app's line (:data:`~vtscore.detectors.centroid_head.CENTROID_LINE_RULE`)
two ways at each balance:

- ``pool``: over every media of the cell, the examples included (the app before #4779);
- ``unvoted``: over every media but the examples (the fix).

Each line is scored on the withheld half: F-beta at the balance, how many of the withheld half it keeps, and
whether it keeps none of them. Every dataset in ``--results``'s prepared grid is read, whole-image embedders
only: FHIBE's four cells and COCO Better's Binary cells, opened on K example photos instead of a typed query.

Writes ``cells.csv`` (one row per cell x K x corpus x beta; FHIBE rows name people, so owner-only) and
``summary.csv`` / ``summary.md`` (per dataset x K x beta: each corpus's F-beta and keeps-nothing share, and
``unvoted`` minus ``pool`` paired, SE on the category).

Usage (in a run's environment, e.g. via ``fhibe/launch.sh``'s env)::

    python centroid_unvoted_4779.py --results <run>/results --ks 1,4 --out <dir> [--datasets a,b]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))
import common  # noqa: E402

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiment_config as cfg  # noqa: E402

BETAS = (0.25, 1.0, 4.0)


def fbeta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    return 0.0 if not (p > 0 or r > 0) else (1 + b2) * p * r / (b2 * p + r)


def price_cell(pool: dict, cat: str, seed: int, k: int) -> list[dict]:
    from vtscore.detectors.centroid_head import centroid_cut
    from vtscore.eval.example_opening import choose_examples, example_sort, stratified_split
    from vtscore.eval.labels import media_is_positive

    rng = np.random.RandomState(seed)
    sim_ids, test_ids = stratified_split(pool, cat, cfg.SIM_FRACTION, rng)
    try:
        examples = choose_examples(sim_ids, pool, cat, k, seed)
    except ValueError:
        return []
    sims = example_sort(pool, examples)
    ids = sorted(pool)
    scores = np.asarray([float(sims[i]) for i in ids], dtype=np.float64)
    voted = set(examples)
    unvoted = np.asarray([i not in voted for i in ids])
    tset = set(test_ids)
    in_test = np.asarray([i in tset for i in ids])
    y = np.asarray([1 if media_is_positive(pool[i], cat) else 0 for i in ids])
    yt, st = y[in_test], scores[in_test]
    npos = int(yt.sum())
    if npos == 0:
        return []
    rows = []
    for corpus, mask in (("pool", np.ones_like(unvoted)), ("unvoted", unvoted)):
        for beta in BETAS:
            line = float(centroid_cut(scores[mask], beta=beta))
            kept = st >= line
            tp, kk = int((kept & (yt == 1)).sum()), int(kept.sum())
            p = tp / kk if kk else 0.0
            r = tp / npos
            rows.append({"category": cat, "seed": seed, "k": k, "corpus": corpus, "beta": beta,
                         "kept_test": kk, "precision": p, "recall": r, "fbeta": fbeta(p, r, beta)})  # fmt: skip
    return rows


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset", "k", "category", "seed", "beta"]
    piv = cells.pivot_table(index=key, columns="corpus", values="fbeta").reset_index()
    out = []
    for (ds, k, beta), h in piv.groupby(["dataset", "k", "beta"]):
        d = h["unvoted"] - h["pool"]
        per = d.groupby(h["category"]).mean()
        c = cells[(cells["dataset"] == ds) & (cells["k"] == k) & (cells["beta"] == beta)]
        nothing = c.assign(none=c["kept_test"] == 0).groupby("corpus")["none"].mean()
        out.append({"dataset": ds, "k": k, "beta": beta, "cells": len(h),
                    "fbeta_pool": h["pool"].mean(), "fbeta_unvoted": h["unvoted"].mean(),
                    "d_unvoted": d.mean(), "d_se": per.std(ddof=1) / math.sqrt(len(per)) if len(per) > 1 else float("nan"),
                    "keeps_none_pool": nothing.get("pool"), "keeps_none_unvoted": nothing.get("unvoted")})  # fmt: skip
    return pd.DataFrame(out).round(4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--ks", default="1,4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", default="", help="only these datasets of the grid (comma-separated)")
    args = ap.parse_args()

    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.eval.labels import evaluable_pool  # noqa: PLC0415

    ks = [int(k) for k in args.ks.split(",") if k]
    only = {d for d in args.datasets.split(",") if d}
    prepare = json.loads((args.results / "prepare_info.json").read_text())
    rows = []
    for ds, by_emb in prepare.get("datasets", {}).items():
        if only and ds not in only:
            continue
        for emb, info in by_emb.items():
            if cfg.is_patch_embedder(emb):
                continue  # the region path pools patches, which a whole-image example sort does not reproduce
            cats = info.get("selected_categories") or []
            medias = load_medias(EMBEDDINGS_DIR / cfg.pickle_name(ds, emb), repair=True)
            common.log(f"{ds} x {emb}: {len(medias)} medias, {len(cats)} cells")
            for cat in cats:
                pool = evaluable_pool(medias, cat)
                for seed in cfg.SEEDS:
                    for k in ks:
                        for r in price_cell(pool, cat, seed, k):
                            rows.append({"dataset": ds, "embedder": emb, **r})
    cells = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out / "cells.csv", index=False)
    summary = summarize(cells)
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "summary.md").write_text(summary.to_string(index=False) + "\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
