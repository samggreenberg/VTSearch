#!/usr/bin/env python
"""The training-domain summary's power against the full domain-shift report, on one (embedder, seed) (#5568).

    python run_summary_power_5568.py --embedder dinov3_patch --seed 0 \
        --datasets a,b,c --mats DIR --out DIR

The #3348 bench, unchanged in its split and its pairs: each dataset's build
80 % is the reference, its held-out 20 % the in-domain null, and every ordered
(reference, query) pair is scored, the diagonal being that holdout.  Every arm
scores the SAME query rows, so the arms are paired item for item:

``full``
    the shipped :func:`~vtscore.coverage.domain_shift.typicality_pvalues`
    (the report's p-values), against the reference's own vectors.
``sum{N}``
    the shipped :func:`~vtscore.coverage.domain_summary.summarize_reference`
    at ``N`` anchors, read back by ``summary_pvalues``.  The anchors are the
    reference's own vectors, as a reader that re-embeds them in the same
    embedder gets them.
``rand128``
    128 fit-half items drawn at random as anchors, calibrated the same way:
    does choosing anchors by k-means earn its keep?
``radius128``
    the 128 shipped anchors, the score divided by a per-anchor radius
    (the 90th percentile of its fit-half members' cosine distance): the
    negative control #5567 measured no better than the raw cosine.

Every arm is split-conformal over the same fit/calibration halves, so every
arm is valid; only power differs.  Queries are capped at 4 000 rows as in
#3348, so the reports' own query subsample is the identity here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))

import common  # noqa: E402

common.setup_env()

import node_models_3348 as nm  # noqa: E402
from run_node_models_3348 import MAX_QUERY, load  # noqa: E402
from structure_fits_3329 import HOLDOUT_FRACTION, _pit_stats, _rows  # noqa: E402

from vtscore.coverage.domain_shift import (  # noqa: E402
    _SEED,
    DOMAIN_SHIFT_ALPHA,
    DOMAIN_SHIFT_MAX_FIT,
    _normalize,
    _topk_mean_cosine,
    typicality_pvalues,
)
from vtscore.coverage.domain_summary import (  # noqa: E402
    SCORE_DECIMALS,
    SUMMARY_MAX_CAL,
    _nearest_anchor,
    summarize_reference,
    summary_pvalues,
)

#: Anchor counts priced for the shipped summary (the issue's 64/128/256, plus
#: 32 and 512 to show the curve's ends).
ANCHOR_COUNTS = (32, 64, 128, 256, 512)

#: Anchor count the two controls use (the shipped ``SUMMARY_ANCHORS``).
CONTROL_ANCHORS = 128

#: Percentile of a cluster's member distances taken as its anchor's radius.
RADIUS_Q = 0.90

#: Share of a partial-shift query drawn from the other dataset; the rest is the
#: reference's own holdout.  Whole-dataset swaps mostly saturate every arm, and
#: #5567's synthetic bench measured a 30 % shift.
MIX_SHARE = 0.3


def _halves(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The fit and calibration rows ``summarize_reference`` uses, for the controls."""
    perm = np.random.default_rng(_SEED).permutation(n)
    half = n // 2
    return perm[:half][:DOMAIN_SHIFT_MAX_FIT], perm[half:][:SUMMARY_MAX_CAL]


class Arms:
    """Every arm fitted on one reference (unit rows)."""

    def __init__(self, ref: np.ndarray, add):
        self.ref = _normalize(ref)
        self.summaries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for n in ANCHOR_COUNTS:
            t0 = time.time()
            rows, cal = summarize_reference(self.ref, n_anchors=n)
            self.summaries[f"sum{n}"] = (self.ref[rows], cal)
            add("fit", f"sum{n}", "seconds", time.time() - t0)
            add("fit", f"sum{n}", "anchors", rows.size)
        fit_rows, cal_rows = _halves(self.ref.shape[0])
        # rand128: random fit-half anchors, same calibration rule.
        pick = np.sort(fit_rows[np.random.default_rng(_SEED + 5).choice(fit_rows.size, CONTROL_ANCHORS, replace=False)])
        anchors = self.ref[pick]
        cal = np.round(np.sort(_topk_mean_cosine(self.ref[cal_rows], anchors, 1)), SCORE_DECIMALS)
        self.summaries["rand128"] = (anchors, cal)
        add("fit", "rand128", "anchors", pick.size)
        # radius128: the shipped anchors, distance over a per-anchor radius.
        anchors = self.summaries[f"sum{CONTROL_ANCHORS}"][0]
        fit = self.ref[fit_rows]
        assign = _nearest_anchor(fit, anchors)
        dist = 1.0 - np.einsum("ij,ij->i", fit, anchors[assign])
        radius = np.full(anchors.shape[0], np.nan)
        for j in range(anchors.shape[0]):
            m = dist[assign == j]
            if m.size:
                radius[j] = np.quantile(m, RADIUS_Q)
        fallback = float(np.nanmedian(radius)) if np.isfinite(radius).any() else 1.0
        radius = np.where(np.isfinite(radius) & (radius > 1e-6), radius, max(fallback, 1e-6))
        self.radius = (anchors, radius)
        self.radius_cal = np.sort(self._radius_score(self.ref[cal_rows]))
        add("fit", "radius128", "radius_median", float(np.median(radius)), radius.size)

    def _radius_score(self, q: np.ndarray) -> np.ndarray:
        anchors, radius = self.radius
        sims = q @ anchors.T
        return -((1.0 - sims) / radius[None, :]).min(axis=1)

    def pvalues(self, x: np.ndarray, timing: dict[str, float]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        t0 = time.time()
        out["full"] = typicality_pvalues(self.ref, x)
        timing["full"] = timing.get("full", 0.0) + time.time() - t0
        for name, (anchors, cal) in self.summaries.items():
            t0 = time.time()
            out[name] = summary_pvalues(anchors, cal, x)
            timing[name] = timing.get(name, 0.0) + time.time() - t0
        out["radius128"] = nm.conformal_p(self.radius_cal, self._radius_score(_normalize(x)))
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--datasets", required=True, help="comma-separated")
    ap.add_argument("--mats", required=True, help="prepare_matrices_3348 output dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    import pandas as pd  # noqa: PLC0415

    t_start = time.time()
    alpha = DOMAIN_SHIFT_ALPHA
    names = [d for d in args.datasets.split(",") if d]
    data = {d: load(Path(args.mats), d, args.embedder)[0] for d in names}
    pit_rows: list[dict] = []
    arms: dict[str, Arms] = {}
    holdouts: dict[str, np.ndarray] = {}
    store: dict[str, np.ndarray] = {}
    for d in names:
        vecs = data[d]
        rows, add = _rows(embedder=args.embedder, seed=args.seed, dataset=d)
        # The #3348 split, so every reference and holdout is that run's.
        perm = np.random.default_rng(args.seed).permutation(vecs.shape[0])
        n_hold = max(1, int(round(HOLDOUT_FRACTION * vecs.shape[0])))
        holdouts[d] = vecs[perm[:n_hold]]
        arms[d] = Arms(vecs[perm[n_hold:]], add)
        pit_rows += rows
        common.log(f"{d}: n={vecs.shape[0]} reference {vecs.shape[0] - n_hold}")

    shift_rows: list[dict] = []
    timing: dict[str, float] = {}
    pairs = [(b, q, 0.0) for b in names for q in names]
    pairs += [(b, q, MIX_SHARE) for b in names for q in names if q != b]
    for b, q, mix in pairs:
        if mix:
            # The reference's holdout, topped up with the other dataset to MIX_SHARE of the total.
            own = holdouts[b][: int(MAX_QUERY * (1 - mix))]
            n_q = min(data[q].shape[0], int(round(own.shape[0] * mix / (1 - mix))))
            other = data[q][np.random.default_rng(args.seed + 9).choice(data[q].shape[0], n_q, replace=False)]
            x = np.concatenate([own, other])
        else:
            x = holdouts[q] if q == b else data[q]
        if x.shape[0] > MAX_QUERY:
            x = x[np.random.default_rng(args.seed + 7).choice(x.shape[0], MAX_QUERY, replace=False)]
        pv = arms[b].pvalues(x, timing)
        full_flag = pv["full"] < alpha
        if q == b and not mix:
            rows, add = _rows(embedder=args.embedder, seed=args.seed, dataset=b)
            for var, p in pv.items():
                _pit_stats(add, var, "holdout", p)
                store[f"pit|{b}|{var}"] = p.astype(np.float32)
            pit_rows += rows
        for var, p in pv.items():
            flag = p < alpha
            shift_rows.append(
                {
                    "embedder": args.embedder,
                    "seed": args.seed,
                    "build_dataset": b,
                    "query_dataset": q,
                    "is_self": b == q,
                    "mix": mix,
                    "variant": var,
                    **nm.shift_verdict(p, alpha),
                    # Of the items the full report flags, the share this arm also flags.
                    "n_full_flagged": int(full_flag.sum()),
                    "recall_of_full": float(flag[full_flag].mean()) if full_flag.any() else np.nan,
                    # Of the items this arm flags, the share the full report does not.
                    "only_this": float((flag & ~full_flag).sum() / max(1, flag.sum())),
                }
            )
            if q != b:
                store[f"shift|{b}|{q}|{mix:g}|{var}"] = p.astype(np.float16)
        common.log(
            f"  {b} <- {q} mix {mix:g}: n={x.shape[0]} full={np.mean(pv['full'] < alpha):.3f} "
            + " ".join(f"{v}={np.mean(pv[v] < alpha):.3f}" for v in pv if v != "full")
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.embedder}__s{args.seed}"
    pd.DataFrame(pit_rows).to_csv(out / f"pit_{stem}.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(out / f"shift_{stem}.csv", index=False)
    np.savez_compressed(out / f"pvals_{stem}.npz", **store)
    (out / f"timing_{stem}.json").write_text(
        json.dumps(
            {
                "embedder": args.embedder,
                "seed": args.seed,
                "query_seconds": timing,
                "total_seconds": time.time() - t_start,
            },
            indent=2,
        )
    )
    common.log(f"wrote {len(pit_rows)} pit rows, {len(shift_rows)} shift rows in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
