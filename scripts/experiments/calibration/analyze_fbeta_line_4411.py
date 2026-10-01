#!/usr/bin/env python3
"""Could the line be drawn at an F-beta optimum instead of a precision floor? (#4411)

The owner's question (2026-10-01): the precision floor is one way to record a
preference; "what balance of precision and recall should we maximize?" is
another.  As a metric F-beta is clean: one number per preference, peaking at
the preference by construction, the best F-beta of any cut as the oracle.  The
difference is in drawing the line without ground truth.  The floor keeps the
deepest *k* with precision(k) >= P: one estimate, and uniform audits estimate
precision along the ranking without bias.  The F-beta optimum is

    argmax_k (1 + b^2) tp(k) / (b^2 n_pos + k)

so it also needs **n_pos**, the corpus's total positives, which is what the
app estimates worst (#3827: the mixture over-counts them 10x at 0.1%, 5-8x at
0.44%).  This prices F-beta line rules on the #4383 frames, the same worlds,
sizes and draws, so the floor's rules and these are read on the same corpora.

Rules, at each ``BETAS`` (0.5 precision-leaning, 1 balanced, 2 recall-leaning):

* ``oracle`` - the best F-beta of any cut (truth; the share's denominator).
* ``fb-gmm`` - the vote-anchored mixture's posterior: tp(k) is its cumulative
  sum, n_pos its total, and k the argmax of the estimated F-beta.  No audits.
* ``fb-walk`` - the band walk (bands 8, 8, 16, 32, ...; 5 uniform picks a
  band) with an F-beta stop: tp at each band edge from the audited bands
  (band-stratified), n_pos from the mixture; start at the mixture's argmax
  band, deeper while the estimated F-beta rises, else shallower while it
  rises; the edge with the highest estimate.
* ``fb-bands`` - every band audited (5 picks each, ~50 votes on 11k items),
  tp and n_pos both from the audits (n_pos = sum over all bands of share x
  size): the walk with the tail audited, no mixture.
* ``floor-walk`` / ``floor-novote`` - the shipped mechanism at the preset the
  preference maps to (``BETA_TO_P``: beta 0.5 -> P 90%, 1 -> 50%, 2 -> 10%):
  the band walk at P, and the no-vote line min(schedule count, mixture's
  crossing).  What the floor returns for the same user, scored at beta.

Metrics per world x t x size x beta x rule (``summary.csv``): the returned
set's ``fbeta`` and ``fb_share`` (over the oracle's), ``precision``,
``recall``, ``k`` beside ``oracle_k``, and ``votes``; cluster (category, seed)
SEs; ``paired_vs_floor.csv`` pairs each rule against ``floor-walk`` on the
same corpus.

    python analyze_fbeta_line_4411.py --world 0.44%=DIR --world 0.1%=DIR \\
        --world 5%=DIR --out OUT [--jobs N] [--steps 50,150] [--draws 2] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _cells_paths import main_frame_files, pframe_files  # noqa: E402
from analyze_line_estimate_4383 import (  # noqa: E402
    BASE_FINE,
    EPS,
    SIZES,
    THIN_TO,
    Audits,
    cluster_se,
    corpus_draw,
    draw_audits,
    frame_arrays,
    gmm_posterior,
    kept_count,
    rule_grow,
    sha_dir,
)

BETAS: tuple[float, ...] = (0.5, 1.0, 2.0)
#: The floor preset a balance preference maps to, for the shipped-mechanism rows.
BETA_TO_P = {0.5: 0.9, 1.0: 0.5, 2.0: 0.1}
RULES: tuple[str, ...] = ("fb-gmm", "fb-walk", "fb-bands", "floor-walk", "floor-novote")
HEADLINE = ("fbeta", "fb_share", "precision", "recall", "k", "votes")


def fbeta(tp: np.ndarray | float, k: np.ndarray | float, n_pos: float, beta: float) -> np.ndarray | float:
    return (1.0 + beta * beta) * tp / (beta * beta * n_pos + k)


def oracle_fbeta(labels: np.ndarray, beta: float) -> tuple[int, float]:
    """The cut with the best F-beta: checked just after each positive, where F-beta can only peak."""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0, float("nan")
    pos = np.flatnonzero(labels == 1)
    tp = np.arange(1, n_pos + 1)
    k = pos + 1
    f = fbeta(tp, k, n_pos, beta)
    i = int(np.argmax(f))
    return int(k[i]), float(f[i])


def score_set(labels: np.ndarray, k: int, beta: float, best: float) -> dict[str, float]:
    n_pos = int(labels.sum())
    k = int(max(1, min(k, len(labels))))
    right = int(labels[:k].sum())
    f = float(fbeta(right, k, n_pos, beta)) if n_pos else float("nan")
    return {
        "k": k,
        "precision": right / k,
        "recall": right / n_pos if n_pos else float("nan"),
        "fbeta": f,
        "fb_share": f / best if (n_pos and best > 0) else float("nan"),
    }


# ------------------------------------------------------------------ rules


def rule_fb_gmm(p: np.ndarray | None, beta: float, n: int, fallback: int) -> tuple[int, int, float]:
    if p is None:
        return fallback, 0, float("nan")
    tp = np.cumsum(p)
    est = fbeta(tp, np.arange(1, n + 1), float(tp[-1]), beta)
    i = int(np.argmax(est))
    return i + 1, 0, float(est[i])


def _edge_tp(audits: Audits, bands: int) -> float:
    """Estimated positives in the top ``edges[bands]``: each band's audited share times its size."""
    sizes = np.diff(audits.edges[: bands + 1])
    return float(sum(audits.share(b) * sz for b, sz in enumerate(sizes)))


def rule_fb_walk(audits: Audits, beta: float, n_pos_est: float | None, start_k: int) -> tuple[int, int, float]:
    """The band walk with an F-beta stop; n_pos from the caller (the mixture's), else the audits' own."""
    start = int(np.searchsorted(audits.edges, max(1, start_k), side="left"))
    bands = max(1, min(start, audits.n_bands))
    touched = {bands}

    def est(b: int) -> float:
        tp = _edge_tp(audits, b)
        n_pos = n_pos_est if n_pos_est is not None else max(tp, 1e-9)
        return float(fbeta(tp, float(audits.edges[b]), max(n_pos, 1e-9), beta))

    cur = est(bands)
    best_b, best_f = bands, cur
    direction = 0
    for _ in range(2 * audits.n_bands + 2):
        if direction >= 0 and bands < audits.n_bands:
            nxt = est(bands + 1)
            touched.add(bands + 1)
            if nxt >= cur - EPS:
                bands, cur, direction = bands + 1, nxt, 1
                if nxt > best_f:
                    best_b, best_f = bands, nxt
                continue
            if direction == 1:
                break
            direction = -1
        if direction <= 0 and bands > 1:
            prv = est(bands - 1)
            touched.add(bands - 1)
            if prv > cur + EPS:
                bands, cur, direction = bands - 1, prv, -1
                if prv > best_f:
                    best_b, best_f = bands, prv
                continue
        break
    votes = sum(len(audits.labels[b]) for b in range(max(touched)))
    return int(audits.edges[best_b]), votes, best_f


def rule_fb_bands(audits: Audits, beta: float) -> tuple[int, int, float]:
    """Every band audited: tp and n_pos both from the picks; the band edge with the best estimated F-beta."""
    n_pos = max(_edge_tp(audits, audits.n_bands), 1e-9)
    ests = [fbeta(_edge_tp(audits, b), float(audits.edges[b]), n_pos, beta) for b in range(1, audits.n_bands + 1)]
    i = int(np.argmax(ests))
    votes = sum(len(y) for y in audits.labels)
    return int(audits.edges[i + 1]), votes, float(ests[i])


# ------------------------------------------------------------------ one cell


def cell_rows(job: tuple) -> list[dict]:
    world, npz_path, main_path, steps, draws, seed_base = job
    first = pd.read_csv(main_path, nrows=1)
    if first.empty:
        return []
    cat, seed = str(first["category"].iloc[0]), int(first["seed"].iloc[0])
    z = np.load(npz_path)
    have = {k.split("/")[0] for k in z.files}
    rows: list[dict] = []
    for t in steps:
        if f"t{t}" not in have:
            continue
        fr = frame_arrays(z, f"t{t}")
        for size in SIZES:
            for d in range(draws):
                # The same draws as #4383's, so the corpora are the same.
                rng = np.random.default_rng([seed_base, seed, t, SIZES.index(size), d, len(cat)])
                s, y = corpus_draw(fr["test_s"], fr["test_y"], size, THIN_TO.get(world), rng)
                n = len(y)
                if n == 0:
                    continue
                _coarse = draw_audits(y, rng)  # drawn to keep the RNG stream aligned with #4383
                fine = draw_audits(y, np.random.default_rng(rng.integers(2**31)), BASE_FINE)
                p_gmm = gmm_posterior(s, fr["vote_s"], fr["vote_y"])
                n_pos_mix = float(np.sum(p_gmm)) if p_gmm is not None else None
                ident = {
                    "world": world,
                    "category": cat,
                    "band": cat.split("@")[-1],
                    "seed": seed,
                    "t": t,
                    "size": size,
                    "n_corpus": n,
                    "n_pos": int(y.sum()),
                    "n_pos_mixture": n_pos_mix if n_pos_mix is not None else float("nan"),
                    "n_votes": len(fr["vote_y"]),
                    "draw": d,
                }
                for beta in BETAS:
                    ok, best = oracle_fbeta(y, beta)
                    floor = BETA_TO_P[beta]
                    fixed_k = kept_count(floor, n)
                    gmm_k, _v, _e = rule_fb_gmm(p_gmm, beta, n, fixed_k)
                    outcomes: dict[str, tuple[int, int, float]] = {
                        "fb-gmm": (gmm_k, 0, _e),
                        "fb-walk": rule_fb_walk(fine, beta, n_pos_mix, gmm_k),
                        "fb-bands": rule_fb_bands(fine, beta),
                        "floor-walk": rule_grow(fine, floor, n, lb=False),
                    }
                    # The no-vote line: the smaller of the schedule's count and the mixture's P crossing.
                    if p_gmm is not None:
                        cum = np.cumsum(p_gmm) / np.arange(1, n + 1)
                        hit = np.flatnonzero(cum >= floor - EPS)
                        cross = int(hit.max()) + 1 if hit.size else int(np.argmax(cum)) + 1
                        outcomes["floor-novote"] = (min(fixed_k, cross), 0, float("nan"))
                    else:
                        outcomes["floor-novote"] = (fixed_k, 0, float("nan"))
                    for rule, (k, votes, est) in outcomes.items():
                        m = score_set(y, k, beta, best)
                        rows.append(
                            {
                                **ident,
                                "beta": beta,
                                "floor": floor,
                                "rule": rule,
                                **m,
                                "votes": votes,
                                "est": est,
                                "oracle_k": ok,
                                "oracle_fbeta": best,
                            }
                        )
    return rows


# ------------------------------------------------------------------ summaries


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["world", "t", "size", "beta", "rule"]
    out = []
    for key, g in rows.groupby(keys, sort=False):
        cells = (g["category"] + "|" + g["seed"].astype(str)).to_numpy()
        rec = {k: v for k, v in zip(keys, key, strict=True)}
        rec["n_corpus"] = float(g["n_corpus"].mean())
        rec["n_pos"] = float(g["n_pos"].mean())
        rec["n_pos_mixture"] = float(g["n_pos_mixture"].mean())
        rec["runs"] = int(len(g))
        for m in (*HEADLINE, "oracle_k", "oracle_fbeta"):
            rec[m] = float(g[m].mean())
        for m in HEADLINE:
            rec[f"se_{m}"] = cluster_se(g[m].to_numpy(dtype=float), cells)
        out.append(rec)
    return pd.DataFrame(out)


def paired(rows: pd.DataFrame, ref: str = "floor-walk") -> pd.DataFrame:
    keys = ["world", "t", "size", "beta", "category", "seed", "draw"]
    base = rows[rows["rule"] == ref].set_index(keys)[["fbeta", "fb_share"]]
    out = []
    for (world, t, size, beta, rule), g in rows.groupby(["world", "t", "size", "beta", "rule"], sort=False):
        if rule == ref:
            continue
        j = g.set_index(keys)[["fbeta", "fb_share"]].join(base, rsuffix="_ref", how="inner").dropna()
        if j.empty:
            continue
        cells = np.array(["|".join(map(str, i[4:6])) for i in j.index])
        d = (j["fbeta"] - j["fbeta_ref"]).to_numpy(dtype=float)
        ds = (j["fb_share"] - j["fb_share_ref"]).to_numpy(dtype=float)
        out.append(
            {
                "world": world,
                "t": t,
                "size": size,
                "beta": beta,
                "rule": rule,
                "n": len(j),
                "d_fbeta": float(d.mean()),
                "se_d_fbeta": cluster_se(d, cells),
                "d_fb_share": float(ds.mean()),
                "se_d_fb_share": cluster_se(ds, cells),
            }
        )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--world", action="append", required=True, help="label=results_dir (holding cells/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--steps", default="50,150")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4383, help="#4383's, so the corpora are its draws")
    ap.add_argument("--limit", type=int, default=0, help="cells per world (0 = all); for a smoke run")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    steps = tuple(int(x) for x in args.steps.split(","))
    jobs, prov = [], {"inputs": {}, "steps": steps, "draws": args.draws, "seed": args.seed, "betas": BETAS}
    for spec in args.world:
        world, _, d = spec.partition("=")
        cells = Path(d) / "cells"
        mains = {p.name.split(".")[0]: p for p in main_frame_files(cells)}
        pf = [p for p in pframe_files(cells) if p.name.split("__")[0] in mains]
        if args.limit:
            pf = pf[: args.limit]
        prov["inputs"][world] = {"dir": str(cells), "cells": len(pf), "sha": sha_dir(pf)}
        jobs += [(world, str(p), str(mains[p.name.split("__")[0]]), steps, args.draws, args.seed) for p in pf]
    rows: list[dict] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for r in ex.map(cell_rows, jobs, chunksize=4):
                rows += r
    else:
        for j in jobs:
            rows += cell_rows(j)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "rows.csv.gz", index=False)
    summarise(df).to_csv(args.out / "summary.csv", index=False)
    paired(df).to_csv(args.out / "paired_vs_floor.csv", index=False)
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"{len(df)} rows over {len(jobs)} cells -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
