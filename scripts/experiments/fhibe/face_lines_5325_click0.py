#!/usr/bin/env python
"""Click 0 of the face review on imagekit's cosines (#5325): where the example sort's line cuts, and why.

#5321 read click 0 from ``example_baseline.py``, which draws the example sort's line over every media of the
cell, the starting photos included: the app's line before #4779. Since #4779 the app counts the unvoted media
only (:func:`vtscore.training.query_sort.cosine_sort_cuts`). This script rebuilds each cell's click 0 with the
harness's own pieces (``evaluable_pool``, ``stratified_split`` on ``RandomState(seed)``, ``choose_examples``,
``example_sort``, as ``centroid_unvoted_4779.py`` does) and

- ``dump``: keeps each cell's cosines, which of them are starting photos, withheld and the person's, in one
  pickle per dataset (``<out>/click0_<dataset>.pkl``; FHIBE ids, so owner-only);
- ``price``: draws candidate lines on them and scores each on the withheld half at every balance:
  the corpus the line counts (``pool`` / ``unvoted``), the rule (the app's
  :data:`~vtscore.detectors.centroid_head.CENTROID_LINE_RULE`, its count alone, its drop alone, its count read at a
  higher bar or against a mirrored bulk, :data:`BARS`), and a scale on the count (the app's is 1). The app's rule at scale 1 is drawn by the library's
  :func:`~vtscore.detectors.centroid_head.centroid_cut` too, and the two must agree on every cell.

Writes ``<out>/cells.csv`` (one row per cell x K x line x beta) and ``<out>/summary.csv``.

Usage (in a run's environment)::

    python face_lines_5325_click0.py dump --results <run>/results --out <dir>
    python face_lines_5325_click0.py price --out <dir>
"""

from __future__ import annotations

import argparse
import json
import pickle
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
KS = (1, 4)
#: The app's rule with its count read at another bar (z bulk spreads) or bulk (``mirror``: the bulk's share above
#: the bar taken as its own count below ``median - z * spread``), against a heavier upper tail than a Gaussian's.
BARS = {"app_z5": (5.0, "gauss"), "app_z5.5": (5.5, "gauss"), "app_z6": (6.0, "gauss"), "app_z7": (7.0, "gauss"),
        "app_mirror4": (4.0, "mirror")}  # fmt: skip
#: Replicas the library draws too (:data:`~vtscore.detectors.centroid_head.CENTROID_GATED`): they must agree.
LIBRARY = {"app_z5": "count_gap_floor_skip25_b1_z5", "app_z6": "count_gap_floor_skip25_b1_z6"}
#: Scales on the count's multiplier (the app's is 1).
SCALES = (0.25, 0.35, 0.5, 0.71, 1.0, 1.4, 2.0, 2.8, 4.0)


def fbeta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    return 0.0 if not (p > 0 or r > 0) else (1 + b2) * p * r / (b2 * p + r)


def dump_cell(pool: dict, cat: str, seed: int, k: int) -> dict | None:
    from vtscore.eval.example_opening import choose_examples, example_sort, stratified_split
    from vtscore.eval.labels import media_is_positive

    rng = np.random.RandomState(seed)
    sim_ids, test_ids = stratified_split(pool, cat, cfg.SIM_FRACTION, rng)
    try:
        examples = choose_examples(sim_ids, pool, cat, k, seed)
    except ValueError:
        return None
    sims = example_sort(pool, examples)
    ids = sorted(pool)
    voted, tset = set(examples), set(test_ids)
    return {
        "category": cat,
        "seed": seed,
        "k": k,
        "scores": np.asarray([float(sims[i]) for i in ids], dtype=np.float64),
        "example": np.asarray([i in voted for i in ids]),
        "test": np.asarray([i in tset for i in ids]),
        "pos": np.asarray([bool(media_is_positive(pool[i], cat)) for i in ids]),
    }


def dump(args: argparse.Namespace) -> int:
    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.eval.labels import evaluable_pool  # noqa: PLC0415

    prepare = json.loads((args.results / "prepare_info.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for ds, by_emb in prepare.get("datasets", {}).items():
        for emb, info in by_emb.items():
            cats = info.get("selected_categories") or []
            medias = load_medias(EMBEDDINGS_DIR / cfg.pickle_name(ds, emb), repair=True)
            common.log(f"{ds} x {emb}: {len(medias)} medias, {len(cats)} cells")
            cells = []
            for cat in cats:
                pool = evaluable_pool(medias, cat)
                for seed in cfg.SEEDS:
                    for k in KS:
                        c = dump_cell(pool, cat, seed, k)
                        if c is not None:
                            cells.append({"dataset": ds, "embedder": emb, **c})
            with open(args.out / f"click0_{ds}.pkl", "wb") as fh:
                pickle.dump(cells, fh)
            common.log(f"{ds}: {len(cells)} cells -> {args.out / f'click0_{ds}.pkl'}")
    return 0


def keep_count(x: np.ndarray, beta: float, rule: str, scale: float) -> int:
    """How many of *x* the centroid's count line keeps at *beta*, its multiplier times *scale*.

    The app's rule (``count_gap_floor_skip25_b1``) is ``round(scale * beta ** 0.708 * max(count, drop))``, never
    fewer than the drop; ``count`` drops the drop, ``drop`` keeps the drop alone.
    """
    from vtscore.detectors.centroid_head import _top_gap_count
    from vtscore.training.thresholds.gmm import TEXT_SORT_COUNT_EXPONENT, _count_estimate

    z, bulk = BARS.get(rule, (None, "gauss"))
    counted = _count_estimate(x, z=z, bulk=bulk)
    gap = _top_gap_count(x, drop="skip25" if beta >= 1.0 else "widest")
    if rule == "drop":
        return gap
    if counted is None:
        return -1  # the guarded line: priced by the library only
    n_hat = counted if rule == "count" else max(counted, float(gap))
    keep = min(max(1, round(scale * beta**TEXT_SORT_COUNT_EXPONENT * n_hat)), x.size)
    return keep if rule == "count" else max(keep, gap)


def price_cell(c: dict) -> list[dict]:
    from vtscore.detectors.centroid_head import centroid_cut

    s, ex, te, pos = c["scores"], c["example"], c["test"], c["pos"]
    yt, st = pos[te], s[te]
    npos = int(yt.sum())
    if npos == 0:
        return []
    rows = []
    base = {k: c[k] for k in ("dataset", "category", "seed", "k")}
    for corpus, mask in (("pool", np.ones_like(ex)), ("unvoted", ~ex)):
        x = s[mask]
        srt = -np.sort(-x)
        for beta in BETAS:
            lib = float(centroid_cut(x, beta=beta))
            lines = {("app", 1.0): lib}
            for rule in ("app", "count", "drop", *BARS):
                for scale in SCALES if rule in ("app", "count") else (1.0,):
                    kk = keep_count(x, beta, rule, scale)
                    if kk < 0:
                        continue
                    line = float(srt[kk - 1])
                    if rule in LIBRARY:
                        lib_z = float(centroid_cut(x, rule=LIBRARY[rule], beta=beta))
                        if abs(line - lib_z) > 1e-12:
                            raise SystemExit(f"replica != library: {base} {corpus} {rule} {beta}: {line} vs {lib_z}")
                    if rule == "app" and scale == 1.0:
                        if abs(line - lib) > 1e-12:
                            raise SystemExit(f"replica != library: {base} {corpus} {beta}: {line} vs {lib}")
                        continue
                    lines[(rule, scale)] = line
            for (rule, scale), line in lines.items():
                kept = st >= line
                tp, kk = int((kept & yt).sum()), int(kept.sum())
                p = tp / kk if kk else 0.0
                r = tp / npos
                rows.append({**base, "corpus": corpus, "rule": rule, "scale": scale, "beta": beta,
                             "n_test_pos": npos, "kept_test": kk, "tp": tp, "precision": p, "recall": r,
                             "fbeta": fbeta(p, r, beta)})  # fmt: skip
    return rows


def price(args: argparse.Namespace) -> int:
    rows = []
    for f in sorted(args.out.glob("click0_*.pkl")):
        with open(f, "rb") as fh:
            cells = pickle.load(fh)  # noqa: S301 - this script's own dump
        common.log(f"{f.name}: {len(cells)} cells")
        for c in cells:
            rows.extend(price_cell(c))
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "cells.csv.gz", index=False)
    key = ["dataset", "k", "corpus", "rule", "scale", "beta"]
    g = out.groupby(key)
    summ = g.agg(cells=("fbeta", "size"), fbeta=("fbeta", "mean"), precision=("precision", "mean"),
                 recall=("recall", "mean"), kept=("kept_test", "mean"), pos=("n_test_pos", "mean"),
                 keeps_none=("kept_test", lambda v: float((v == 0).mean()))).reset_index()  # fmt: skip
    summ["se"] = g["fbeta"].std(ddof=1).values / np.sqrt(summ["cells"].values)
    summ = summ.round(4)
    summ.to_csv(args.out / "summary.csv", index=False)
    app = summ[(summ["rule"] == "app") & (summ["scale"] == 1.0)]
    print(app.to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--results", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("price")
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    return dump(args) if args.cmd == "dump" else price(args)


if __name__ == "__main__":
    raise SystemExit(main())
