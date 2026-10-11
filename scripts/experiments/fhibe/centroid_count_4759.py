#!/usr/bin/env python
"""The centroid's count, explored on the K example photos' own centroid (#4759).

Since #4732 the Goods' centroid's line is the typed query's count line (#4603): it keeps
``beta ** 0.708 * n_hat``, where ``n_hat`` is the count above ``median + 4 * 1.4826 * MAD`` beyond what a
Gaussian bulk puts there. On FHIBE's face crops a person has about three photos withheld, the centroid
ranks them first, and the line still keeps one. This script looks at why, cheaply, before any session
runs: it rebuilds each cell's opening centroid (the K example photos, the first K votes of the run)
and its cosines on the withheld half, which is the corpus Test cuts it on, and prices count estimators
offline on them.

``dump`` (in a FHIBE run's environment, as ``example_baseline.py`` runs) writes one ``.npz`` per dataset
and K: every cell's withheld-half cosines and labels, concatenated, with offsets. The cell is rebuilt
with the harness's own pieces (``evaluable_pool``, ``stratified_split`` on ``RandomState(seed)``,
``choose_examples``, ``example_sort``), so it is the centroid the run's vote-K row cuts.

``price`` reads those files and, for each estimator and balance, the set the line keeps on each cell:
its size, precision, recall and F-beta, beside the oracle count (the true number withheld) and the
best cut on the labels. Writes ``cells.csv`` and ``summary.md``. The cosines name people, so both
outputs stay owner-only beside the runs.

Usage::

    python centroid_count_4759.py dump --results <run>/results --ks 1,4 --out <dir>
    python centroid_count_4759.py price --dumps <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BETAS = (0.25, 1.0, 4.0)
#: c(beta) = beta ** EXPONENT, the shipped count line's multiple of n_hat (#4603).
EXPONENT = math.log(3 / 8) / math.log(1 / 4)


def dump(results: Path, ks: list[int], out: Path) -> None:
    sys.path.insert(0, str(HERE.parent / "calibration"))
    import common  # noqa: PLC0415

    common.setup_env()
    import experiment_config as cfg  # noqa: PLC0415
    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.eval.example_opening import choose_examples, example_sort, stratified_split  # noqa: PLC0415
    from vtscore.eval.labels import evaluable_pool, media_is_positive  # noqa: PLC0415

    prepare = json.loads((results / "prepare_info.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    for ds in cfg.DATASETS:
        for emb, info in prepare.get("datasets", {}).get(ds, {}).items():
            cats = info.get("selected_categories") or []
            medias = load_medias(EMBEDDINGS_DIR / cfg.pickle_name(ds, emb), repair=True)
            common.log(f"{ds} x {emb}: {len(medias)} medias, {len(cats)} people")
            for k in ks:
                scores, labels, offsets, names, seeds = [], [], [0], [], []
                for cat in cats:
                    pool = evaluable_pool(medias, cat)
                    for seed in cfg.SEEDS:
                        rng = np.random.RandomState(seed)
                        sim_ids, test_ids = stratified_split(pool, cat, cfg.SIM_FRACTION, rng)
                        try:
                            examples = choose_examples(sim_ids, pool, cat, k, seed)
                        except ValueError:
                            continue
                        sims = example_sort(pool, examples)
                        s = np.asarray([float(sims[i]) for i in test_ids], dtype=np.float32)
                        y = np.asarray([1 if media_is_positive(pool[i], cat) else 0 for i in test_ids], dtype=np.uint8)
                        if not y.any():
                            continue
                        scores.append(s)
                        labels.append(y)
                        offsets.append(offsets[-1] + len(s))
                        names.append(cat)
                        seeds.append(seed)
                path = out / f"cosines_{ds}_k{k}.npz"
                np.savez_compressed(
                    path,
                    scores=np.concatenate(scores),
                    labels=np.concatenate(labels),
                    offsets=np.asarray(offsets, dtype=np.int64),
                    category=np.asarray(names),
                    seed=np.asarray(seeds, dtype=np.int32),
                    dataset=ds,
                    embedder=emb,
                    k=k,
                )
                common.log(f"K={k}: {len(names)} cells -> {path}")


# ---------------------------------------------------------------- pricing


def bulk(scores: np.ndarray) -> tuple[float, float]:
    """The bulk's median and robust spread, as #4603's count line reads them."""
    med = float(np.median(scores))
    return med, 1.4826 * float(np.median(np.abs(scores - med)))


def n_hat(scores: np.ndarray, family: str, z: float) -> float | None:
    """The estimated count of matches; ``None`` where the bulk has no spread (the guarded line stays)."""
    n = scores.size
    med, sigma = bulk(scores)
    if n < 50 or not sigma > 0:
        return None
    if family == "gap":
        # The largest drop between neighbours among the top int(z) scores: everything above it.
        top = -np.sort(-scores)[: int(z) + 1]
        return float(np.argmax(top[:-1] - top[1:]) + 1)
    if family.startswith("gapmax"):
        # The shipped count, or the top 10's largest drop if that keeps more.
        top = -np.sort(-scores)[:11]
        return max(float(np.argmax(top[:-1] - top[1:]) + 1), n_hat(scores, "gauss", 4.0) or 1.0)
    if family.startswith("gapsig"):
        # The top 10's largest drop when it is at least z bulk sigmas wide, else the shipped count.
        top = -np.sort(-scores)[:11]
        gaps = top[:-1] - top[1:]
        i = int(np.argmax(gaps))
        return float(i + 1) if gaps[i] >= z * sigma else n_hat(scores, "gauss", 4.0)
    if family == "gauss_up":
        # The bulk's spread read off its upper half only.
        sigma = 1.4826 * float(np.median(scores[scores > med] - med))
    above = float(np.count_nonzero(scores > med + z * sigma))
    if family in ("gauss", "gauss_up"):
        excess = above - n * 0.5 * math.erfc(z / math.sqrt(2.0))
    elif family == "mirror":
        excess = above - float(np.count_nonzero(scores < med - z * sigma))
    else:
        raise ValueError(family)
    return min(max(excess, 1.0), float(n))


def fbeta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    return 0.0 if not (p > 0 or r > 0) else (1 + b2) * p * r / (b2 * p + r)


def at_top(y_sorted: np.ndarray, keep: int, beta: float) -> tuple[float, float, float]:
    """Precision, recall and F-beta of the top *keep* of a ranking whose labels are *y_sorted* (best first)."""
    tp = float(y_sorted[:keep].sum())
    p = tp / keep if keep else 0.0
    r = tp / float(y_sorted.sum())
    return p, r, fbeta(p, r, beta)


ESTIMATORS = (
    [(f"gauss_z{z:g}", "gauss", z, 1.0) for z in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)]
    + [(f"mirror_z{z:g}", "mirror", z, 1.0) for z in (2.0, 2.5, 3.0, 3.5, 4.0)]
    + [(f"gauss_up_z{z:g}", "gauss_up", z, 1.0) for z in (3.0, 3.5, 4.0)]
    + [(f"gap_top{m:g}", "gap", m, 1.0) for m in (10, 30, 100)]
    + [("gapmax_top10", "gapmax", 0.0, 1.0)]
    + [(f"gapsig{t:g}_top10", "gapsig", t, 1.0) for t in (0.25, 0.5, 1.0)]
    + [(f"gauss_z4_x{m:g}", "gauss", 4.0, m) for m in (2.0, 3.0)]
    + [("gapfloor_top10", "gapfloor", 10, 1.0), ("gapflat_top10", "gapflat", 10, 1.0)]
)
#: Price only these estimators (and the reference), when set: ``--only a,b``.
ONLY: set[str] = set()


def price_cell(s: np.ndarray, y: np.ndarray) -> list[dict]:
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    n, npos = s.size, int(y.sum())
    med, sigma = bulk(s)
    zpos = (s[y == 1] - med) / sigma if sigma > 0 else np.full(npos, np.nan)
    base = {
        "n_test": n,
        "n_pos": npos,
        "pos_z_min": float(np.min(zpos)),
        "pos_z_median": float(np.median(zpos)),
        "pos_z_max": float(np.max(zpos)),
        # The bulk's own tail against a Gaussian's, negatives only: observed / expected above z.
        **{
            f"neg_tail_ratio_z{z:g}": float(np.count_nonzero(s[y == 0] > med + z * sigma))
            / max((n - npos) * 0.5 * math.erfc(z / math.sqrt(2.0)), 1e-9)
            for z in (2.0, 3.0, 4.0)
        },
    }
    rows = []
    for beta in BETAS:
        c = beta**EXPONENT
        # The best cut on the labels: the best top-k for any k.
        tp = np.cumsum(ys)
        k = np.arange(1, n + 1)
        p, r = tp / k, tp / npos
        b2 = beta * beta
        f = np.where((p > 0) | (r > 0), (1 + b2) * p * r / np.maximum(b2 * p + r, 1e-12), 0.0)
        best = int(np.argmax(f))
        rows.append({**base, "beta": beta, "estimator": "oracle_best", "n_hat": np.nan, "keep": best + 1,
                     "precision": p[best], "recall": r[best], "fbeta": f[best]})  # fmt: skip
        keep = min(max(1, round(c * npos)), n)
        pp, rr, ff = at_top(ys, keep, beta)
        rows.append({**base, "beta": beta, "estimator": "oracle_count", "n_hat": float(npos), "keep": keep,
                     "precision": pp, "recall": rr, "fbeta": ff})  # fmt: skip
        for name, family, z, mult in ESTIMATORS:
            if ONLY and name not in ONLY and name != "gauss_z4":
                continue
            if family in ("gapfloor", "gapflat"):
                # The gap as a line, not a count: keep everything above the top 10's widest drop at
                # every balance ("flat"), or at least that when the shipped count keeps less ("floor").
                nh = n_hat(s, "gap", 10)
                counted = n_hat(s, "gauss", 4.0)
                if counted is None:
                    continue
                keep = int(nh) if family == "gapflat" else max(int(nh), round(c * counted))
                keep = min(max(1, keep), n)
                pp, rr, ff = at_top(ys, keep, beta)
                rows.append({**base, "beta": beta, "estimator": name, "n_hat": nh, "keep": keep,
                             "precision": pp, "recall": rr, "fbeta": ff})  # fmt: skip
                continue
            nh = n_hat(s, family, z)
            if nh is None:
                continue
            keep = min(max(1, round(mult * c * nh)), n)
            pp, rr, ff = at_top(ys, keep, beta)
            rows.append({**base, "beta": beta, "estimator": name, "n_hat": nh, "keep": keep,
                         "precision": pp, "recall": rr, "fbeta": ff})  # fmt: skip
    return rows


def price(dumps: Path, out: Path) -> None:
    import pandas as pd  # noqa: PLC0415

    frames = []
    for path in sorted(dumps.glob("cosines_*_k*.npz")):
        z = np.load(path)
        off = z["offsets"]
        rows = []
        for i in range(len(off) - 1):
            s = z["scores"][off[i] : off[i + 1]].astype(np.float64)
            y = z["labels"][off[i] : off[i + 1]].astype(np.int64)
            for r in price_cell(s, y):
                r.update(
                    dataset=str(z["dataset"]), k=int(z["k"]), category=str(z["category"][i]), seed=int(z["seed"][i])
                )
                rows.append(r)
        frames.append(pd.DataFrame(rows))
        print(f"{path.name}: {len(off) - 1} cells", file=sys.stderr)
    cells = pd.concat(frames, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "cells.csv", index=False)
    aggregate(cells).to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(summarize(cells))
    print((out / "summary.md").read_text())


def aggregate(cells):
    """Per dataset x K x estimator x beta, no person named: what the line keeps and the paired difference."""
    import pandas as pd  # noqa: PLC0415

    key = ["dataset", "k", "category", "seed", "beta"]
    ref = cells[cells["estimator"] == "gauss_z4"].set_index(key)["fbeta"]
    rows = []
    for (ds, k, est, beta), h in cells.groupby(["dataset", "k", "estimator", "beta"], sort=False):
        d = (h.set_index(key)["fbeta"] - ref).dropna()
        per = d.groupby(level="category").mean()
        rows.append(
            {
                "dataset": ds,
                "k": k,
                "estimator": est,
                "beta": beta,
                "cells": len(h),
                "n_hat_over_n_median": float((h["n_hat"] / h["n_pos"]).median()),
                "kept_median": float(h["keep"].median()),
                "precision": float(h["precision"].mean()),
                "recall": float(h["recall"].mean()),
                "fbeta": float(h["fbeta"].mean()),
                "d_vs_count": float(d.mean()),
                "d_vs_count_se": float(per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).round(4)


def summarize(cells) -> str:
    import pandas as pd  # noqa: PLC0415

    lines = ["# #4759 exploration: count estimators on the K examples' centroid, withheld half", ""]
    one = cells.drop_duplicates(["dataset", "k", "category", "seed"])
    lines.append("## Where a person's withheld photos sit, in bulk sigmas above the median")
    lines.append("")
    lines.append("| dataset | K | cells | positives (median) | lowest positive z (median) | median positive z (median) "
                 "| negatives above z 2 / 3 / 4, over a Gaussian's (median) |")  # fmt: skip
    lines.append("|---|---|---|---|---|---|---|")
    for (ds, k), g in one.groupby(["dataset", "k"]):
        lines.append(
            f"| {ds} | {k} | {len(g)} | {g['n_pos'].median():.0f} | {g['pos_z_min'].median():.1f} "
            f"| {g['pos_z_median'].median():.1f} | {g['neg_tail_ratio_z2'].median():.1f} / "
            f"{g['neg_tail_ratio_z3'].median():.1f} / {g['neg_tail_ratio_z4'].median():.1f} |"
        )
    lines.append("")
    ref = cells[cells["estimator"] == "gauss_z4"].set_index(["dataset", "k", "category", "seed", "beta"])["fbeta"]
    for (ds, k), g in cells.groupby(["dataset", "k"]):
        lines.append(f"## {ds}, K = {k}")
        lines.append("")
        lines.append("| estimator | beta | n_hat / n (median) | kept (median) | precision | recall | F-beta "
                     "| vs gauss_z4 (paired, SE on the person) |")  # fmt: skip
        lines.append("|---|---|---|---|---|---|---|---|")
        for (est, beta), h in g.groupby(["estimator", "beta"], sort=False):
            d = h.set_index(["dataset", "k", "category", "seed", "beta"])["fbeta"] - ref
            d = d.dropna()
            per = d.groupby(level="category").mean()
            se = per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else np.nan
            ratio = (h["n_hat"] / h["n_pos"]).median()
            lines.append(
                f"| {est} | {beta:g} | {ratio:.2f} | {h['keep'].median():.0f} | {h['precision'].mean():.2f} "
                f"| {h['recall'].mean():.2f} | {h['fbeta'].mean():.3f} | {d.mean():+.3f} ± {se:.3f} |"
            )
        lines.append("")
    _ = pd
    return "\n".join(lines) + "\n"


def examples(dumps: Path, dataset: str, k: int, out: Path, n_each: int = 3, top: int = 12) -> None:
    """Literal examples from one dump: each person's top cosines (in bulk sigmas), labelled, with each line.

    Three kinds, ``n_each`` of each, chosen by the seed-0 draw of the rows that qualify:
    the gap keeps more of the person than the count does; the gap does worse than the count;
    and both miss a photo the ranking put in its top 12.
    """
    z = np.load(dumps / f"cosines_{dataset}_k{k}.npz")
    off = z["offsets"]
    kinds: dict[str, list[str]] = {"gap wins": [], "gap loses": [], "both miss": []}
    rng = np.random.default_rng(0)
    order = rng.permutation(len(off) - 1)
    for i in order:
        s = z["scores"][off[i] : off[i + 1]].astype(np.float64)
        y = z["labels"][off[i] : off[i + 1]].astype(np.int64)
        med, sigma = bulk(s)
        idx = np.argsort(-s, kind="stable")[:top]
        c = 1.0
        kept_count = min(max(1, round(c * n_hat(s, "gauss", 4.0))), s.size)
        gap = int(n_hat(s, "gap", 10))
        floor = max(gap, kept_count)
        f_count = at_top(y[np.argsort(-s, kind="stable")], kept_count, 1.0)[2]
        f_floor = at_top(y[np.argsort(-s, kind="stable")], floor, 1.0)[2]
        npos, top_pos = int(y.sum()), int(y[idx].sum())
        kind = (
            "gap wins" if f_floor > f_count + 0.2
            else "gap loses" if f_floor < f_count - 0.05
            else "both miss" if max(kept_count, floor) < top_pos and top_pos == npos
            else None
        )  # fmt: skip
        if kind is None or len(kinds[kind]) >= n_each:
            continue
        cells = " ".join(f"{(s[j] - med) / sigma:.2f}{'*' if y[j] else ''}" for j in idx)
        kinds[kind].append(
            f"| {z['category'][i]} | {npos} | {cells} | {kept_count} | {gap} | {f_count:.2f} | {f_floor:.2f} |"
        )
        if all(len(v) >= n_each for v in kinds.values()):
            break
    lines = [f"# #4759 examples: {dataset}, K = {k}, beta 1", ""]
    lines.append("Top 12 of the withheld half, in bulk sigmas above the median; * marks the person's photos.")
    lines.append("")
    for kind, rows in kinds.items():
        lines += [f"## {kind}", "", "| person | photos | top 12 (z) | count keeps | gap keeps | F count | F gap floor |",
                  "|---|---|---|---|---|---|---|", *rows, ""]  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--results", type=Path, required=True)
    d.add_argument("--ks", default="1,4")
    d.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("price")
    p.add_argument("--dumps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", default="", help="comma-separated estimators to price (plus gauss_z4)")
    e = sub.add_parser("examples")
    e.add_argument("--dumps", type=Path, required=True)
    e.add_argument("--dataset", required=True)
    e.add_argument("--k", type=int, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ONLY.update(x for x in getattr(args, "only", "").split(",") if x)
    if args.cmd == "dump":
        dump(args.results, [int(k) for k in args.ks.split(",") if k], args.out)
    elif args.cmd == "examples":
        examples(args.dumps, args.dataset, args.k, args.out)
    else:
        price(args.dumps, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
