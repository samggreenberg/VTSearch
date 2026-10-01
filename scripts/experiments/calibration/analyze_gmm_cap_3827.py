#!/usr/bin/env python3
"""What does ``_GMM_MAX_SAMPLES`` change, and save, at the sizes it binds? (#3827)

Every score mixture the app fits sees at most ``_GMM_MAX_SAMPLES`` (50,000)
scores: above that, ``gmm_fit_array`` takes a deterministic seed-42 subsample.
No study has read the cap where it binds - every bench sort is under 50k - and
a GUI Find reaches ~250k, a CLI Find 2M+.  Lowering it would make each fit
cheaper (per-iteration cost is linear in the sample); it would also change
what the fit *sees*, a statistical change, not an implementation detail.

This prices it on the two places a capped fit reaches the user:

* **The no-vote line** (#4389, ``mixture_count``): the vote-anchored mixture
  on the corpus's scores, the deepest top *k* whose mean high-component
  posterior meets the floor (``gmm``), and the line the app keeps before any
  check, the smaller of that and the schedule's count (``min-fixed-gmm``).
* **The fold-anchored cut** (``fit_fold_anchored_cut``): one anchored mixture
  per calibration fold on that fold model's haystack, carried to the final
  model's scores through a *subsampled* final haystack.  It draws the line
  with no floor, and Autopilot's acquisition cut is read off it
  (``ACQUISITION_INCLUSION_OFFSET`` steps from the line).  Recorded at
  inclusion 0 (``cut0``) and at the offset (``cut_acq``).

Inputs are the #4220 precision frames of the #4383 study (0.1 / 0.44 / 5%;
``task_NNNN__pframes.npz``: the test half's scores and labels, the session
pool's scores, the votes, the folds' haystacks and held-out evidence).

**What is measured is a bootstrap.**  The bench's halves hold ~11.6k items
(the 5% world's ~1k), so each corpus here is the test half (thinned to the
world's prevalence) resampled with replacement to ``SIZES`` items, and each
fold haystack and the final haystack are the session pool's resampled to the
same size.  A capped fit of a bootstrap corpus is a fit on a smaller bootstrap
of the same half; what the arms compare is therefore the sampling noise a
cap-sized fit adds on the bench's score distribution, against a fit on every
score.  A real 2M-item corpus could have a different shape; the cap's effect
on a fit of a given shape is what this reads.

**The cap is also an anchor-mass knob.**  Each vote counts ``anchor_weight``
haystack points in the M-step (10 for ``mixture_count``'s fit, 0.3 for the
fold cut's, ``FOLD_ANCHOR_WEIGHT``), against the *fitted sample*, so the votes'
share of a fit is ``w n / (min(N, cap) + w n)``: above the cap it is fixed by
the cap, below it it falls as the corpus grows.  Lowering the cap therefore
strengthens the votes on a large corpus, and lifting it drowns them; the
uncapped fit is one more arm, not the truth.  Each row records both shares.

Arms (``CAPS``): the cap set to 5k, 10k, 25k, 50k (shipped), 100k, or no cap
(``0``: every score).  Each arm runs the app's own code with the module's cap
patched (``vtscore.training.thresholds.gmm._GMM_MAX_SAMPLES``), and asserts
the fit array came out the size the arm says.  ``SIZES`` adds the bench half
itself (``0``: no resample, ~11.6k items, the size every threshold study
tuned at, where the shipped cap does not bind) as the reference.

Per world x t x size x cap (``summary_lines.csv``, ``summary_cuts.csv``):
each arm's own returned set (count, precision, shortfall, F1, beside the
oracle's best F1), and paired against the **shipped cap** on the same corpus,
with cluster (category, seed) standard errors: the share of runs whose line
moved, the change in F1 and in shortfall below the floor; for the cut, where
it sits as a share of the corpus at inclusion 0 and at the acquisition offset,
how far it moved from the shipped cap's (``dq``, in items per million), and the
F1 of the set it returns.  ``timing.csv``: seconds per fit by size and cap
(single-threaded; subsample draw, EM and, for the cut, the final haystack's
sort included), the saving the cap buys.

    python analyze_gmm_cap_3827.py --world 0.44%=DIR --world 0.1%=DIR \\
        --world 5%=DIR --out OUT [--jobs N] [--steps 50,150] [--shard K/N]
    python analyze_gmm_cap_3827.py --merge OUT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _cells_paths import main_frame_files, pframe_files  # noqa: E402
from analyze_line_estimate_4383 import (  # noqa: E402
    THIN_TO,
    cluster_se,
    frame_arrays,
    k_from_curve,
    kept_count,
    oracle,
    score_set,
    sha_dir,
)

#: The cap per arm; 0 is no cap (every score).  50,000 is shipped.
CAPS: tuple[int, ...] = (5_000, 10_000, 25_000, 50_000, 100_000, 0)
SHIPPED = 50_000
UNCAPPED = 0
#: Corpus sizes: the bench half as it is (0, the reference), then sizes above the
#: shipped cap: a large GUI Find, ~250k, and a CLI Find.
SIZES: tuple[int, ...] = (0, 100_000, 250_000, 2_000_000)
FLOORS: tuple[float, ...] = (0.1, 0.5, 0.9)
LINE_RULES = ("gmm", "min-fixed-gmm")
#: The floor the cut's returned set is scored against (meets / shortfall only).
CUT_FLOOR = 0.5


def set_cap(cap: int) -> None:
    """Point the app's mixture fits at *cap* scores (0: every score)."""
    import vtscore.training.thresholds.gmm as gmm  # noqa: PLC0415

    gmm._GMM_MAX_SAMPLES = cap if cap else 1 << 62


def fit_size(n: int, cap: int) -> int:
    return n if not cap else min(n, cap)


def thinned(test_s: np.ndarray, test_y: np.ndarray, thin_to: float | None, rng) -> tuple[np.ndarray, np.ndarray]:
    """The test half at the world's prevalence (``corpus_draw``'s thinning, before any resample)."""
    pos, neg = np.flatnonzero(test_y == 1), np.flatnonzero(test_y == 0)
    if thin_to is not None and len(pos):
        keep_n = int(round(len(pos) * (1 - thin_to) / thin_to))
        neg = rng.choice(neg, size=min(keep_n, len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    return test_s[idx], test_y[idx]


def bootstrap(scores: np.ndarray, n: int, rng, labels: np.ndarray | None = None):
    idx = rng.integers(0, len(scores), size=n)
    if labels is None:
        return scores[idx]
    s, y = scores[idx], labels[idx]
    order = np.argsort(-s, kind="stable")
    return s[order].astype(np.float64), y[order].astype(np.int64)


def mixture_posterior(
    scores: np.ndarray, vote_s: np.ndarray, vote_y: np.ndarray
) -> tuple[np.ndarray | None, float, dict]:
    """``mixture_count``'s posterior over *scores*, the fit's seconds, and its parameters."""
    from scipy.stats import norm  # noqa: PLC0415

    from vtscore.training.thresholds.gmm import anchored_gmm_fit  # noqa: PLC0415

    t0 = time.perf_counter()
    fit, _prov = anchored_gmm_fit(scores, vote_s, vote_y)
    seconds = time.perf_counter() - t0
    if fit is None or fit.var_hi <= 0 or fit.var_lo <= 0:
        return None, seconds, {}
    hi = fit.w_hi * norm.pdf(scores, fit.mu_hi, np.sqrt(fit.var_hi))
    lo = fit.w_lo * norm.pdf(scores, fit.mu_lo, np.sqrt(fit.var_lo))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.nan_to_num(hi / (hi + lo), nan=0.5)
    params = {"w_hi": fit.w_hi, "mu_hi": fit.mu_hi, "sd_hi": float(np.sqrt(fit.var_hi)), "mu_lo": fit.mu_lo}
    return p, seconds, params


def cell_rows(job: tuple) -> tuple[list[dict], list[dict]]:
    from vtscore.training.thresholds import ACQUISITION_INCLUSION_OFFSET  # noqa: PLC0415
    from vtscore.training.thresholds.anchored import FOLD_ANCHOR_WEIGHT, fit_fold_anchored_cut  # noqa: PLC0415
    from vtscore.training.thresholds.gmm import ANCHOR_WEIGHT_DEFAULT, gmm_fit_array  # noqa: PLC0415

    world, npz_path, main_path, steps, seed_base = job
    first = pd.read_csv(main_path, nrows=1)
    if first.empty:
        return [], []
    cat, seed = str(first["category"].iloc[0]), int(first["seed"].iloc[0])
    z = np.load(npz_path)
    have = {k.split("/")[0] for k in z.files}
    lines: list[dict] = []
    cuts: list[dict] = []
    for t in steps:
        if f"t{t}" not in have:
            continue
        fr = frame_arrays(z, f"t{t}")
        rng0 = np.random.default_rng([seed_base, seed, t, len(cat)])
        half_s, half_y = thinned(fr["test_s"], fr["test_y"], THIN_TO.get(world), rng0)
        if len(half_y) == 0 or half_y.sum() == 0:
            continue
        # A frame whose folds fell back (too few votes, or one class) has no fold
        # haystacks: its no-vote line is still read, its cut is not.
        has_folds = bool(fr["haystacks"]) and min(len(h) for h in fr["haystacks"]) > 0
        hay_len = min(len(h) for h in fr["haystacks"]) if has_folds else 0
        orderings = [(list(map(float, a)), list(map(float, b))) for a, b in fr["orderings"]]
        n_cal = min(len(a) for a, _b in orderings) if orderings else 0
        for n in SIZES:
            rng = np.random.default_rng([seed_base, seed, t, n, len(cat)])
            if n == 0:  # the bench half as it is: the reference, where the shipped cap does not bind
                order = np.argsort(-half_s, kind="stable")
                s, y = half_s[order].astype(np.float64), half_y[order].astype(np.int64)
                hays = [np.asarray(h[:hay_len], dtype=np.float64) for h in fr["haystacks"]] if has_folds else []
                final = fr["pool_ref"]
            else:
                s, y = bootstrap(half_s, n, rng, half_y)
                hays = []
                if has_folds:
                    hay_idx = rng.integers(0, hay_len, size=n)  # one draw for every fold: the folds score one pool
                    hays = [np.asarray(h[:hay_len])[hay_idx] for h in fr["haystacks"]]
                final = bootstrap(fr["pool_ref"], n, rng)
            n_corpus, n_hay, n_votes = len(y), (len(hays[0]) if hays else 0), len(fr["vote_y"])
            orc = {f: oracle(y, f) for f in FLOORS}
            ident = {
                "world": world,
                "category": cat,
                "band": cat.split("@")[-1],
                "seed": seed,
                "t": t,
                "size": n,
                "n_corpus": n_corpus,
                "n_pos": int(y.sum()),
                "n_votes": n_votes,
            }
            for cap in CAPS:
                set_cap(cap)
                assert len(gmm_fit_array(s)) == fit_size(n_corpus, cap), (cap, n_corpus)
                mass = ANCHOR_WEIGHT_DEFAULT * n_votes
                share_mix = mass / (fit_size(n_corpus, cap) + mass)
                p, mix_s, params = mixture_posterior(s, fr["vote_s"], fr["vote_y"])
                for floor in FLOORS:
                    fixed_k = kept_count(floor, n_corpus)
                    gmm_k = fixed_k if p is None else k_from_curve(p, floor)[0]
                    for rule, k in (("gmm", gmm_k), ("min-fixed-gmm", min(fixed_k, gmm_k))):
                        lines.append(
                            {
                                **ident,
                                "cap": cap,
                                "floor": floor,
                                "rule": rule,
                                "fit_ok": p is not None,
                                "anchor_share": share_mix,
                                **score_set(y, k, floor),
                                "oracle_k": orc[floor][0],
                                "oracle_f1": orc[floor][2],
                                "fit_seconds": mix_s,
                            }
                        )
                if not hays:
                    cuts.append(
                        {
                            **ident,
                            "cap": cap,
                            "anchor_share_mix": share_mix,
                            "mix_seconds": mix_s,
                            "cut_ok": False,
                            "cut_skip": "no_folds",
                            **params,
                        }
                    )
                    continue
                t0 = time.perf_counter()
                cut = fit_fold_anchored_cut(hays, orderings, final)
                cut_s = time.perf_counter() - t0
                cmass = FOLD_ANCHOR_WEIGHT * n_cal
                rec = {
                    **ident,
                    "cap": cap,
                    "anchor_share_mix": share_mix,
                    "anchor_share_cut": cmass / (fit_size(n_hay, cap) + cmass),
                    "oracle_f1": orc[CUT_FLOOR][2],
                    "cut_seconds": cut_s,
                    "mix_seconds": mix_s,
                    **params,
                }
                if cut is None:
                    cuts.append({**rec, "cut_ok": False})
                    continue
                for name, inc in (("cut0", 0), ("cut_acq", ACQUISITION_INCLUSION_OFFSET)):
                    thr = float(cut.threshold_at(inc))
                    # Where the cut sits in the WHOLE final haystack (not the fit's subsample),
                    # and the set it returns on the corpus.
                    rec[f"{name}_thr"] = thr
                    rec[f"{name}_q"] = float(np.mean(final >= thr))
                    k = int(np.count_nonzero(s >= thr))
                    m = score_set(y, k, CUT_FLOOR) if k else {"k": 0, "precision": np.nan, "f1": 0.0, "recall": 0.0}
                    rec.update({f"{name}_{key}": m[key] for key in ("k", "precision", "recall", "f1")})
                rec["cut_ok"] = True
                rec["n_unconverged"] = int(cut.n_unconverged)
                cuts.append(rec)
    set_cap(SHIPPED)
    return lines, cuts


# ------------------------------------------------------------------ summaries


def _cells(g: pd.DataFrame) -> np.ndarray:
    return (g["world"] + "|" + g["category"] + "|" + g["seed"].astype(str)).to_numpy()


def summarise_lines(lines: pd.DataFrame, ref: int = SHIPPED) -> pd.DataFrame:
    """Each arm's own line, and paired against the *ref* cap (the shipped one) on the same corpus."""
    keys = ["world", "t", "size", "floor", "rule", "seed", "category"]
    base = lines[lines["cap"] == ref].set_index(keys)[["k", "f1", "shortfall", "precision"]]
    out = []
    for (world, t, n, floor, rule, cap), g in lines.groupby(["world", "t", "size", "floor", "rule", "cap"], sort=True):
        j = g.set_index(keys)[["k", "f1", "shortfall", "precision"]].join(base, rsuffix="_ref", how="inner")
        cells = np.array(["|".join(map(str, i)) for i in j.index.droplevel(["world", "t", "size", "floor", "rule"])])
        d_f1 = (j["f1"] - j["f1_ref"]).to_numpy(dtype=float)
        d_sf = (j["shortfall"] - j["shortfall_ref"]).to_numpy(dtype=float)
        rel_k = (j["k"] / j["k_ref"] - 1.0).to_numpy(dtype=float)
        out.append(
            {
                "world": world,
                "t": t,
                "size": n,
                "floor": floor,
                "rule": rule,
                "cap": cap,
                "runs": len(j),
                "n_corpus": float(g["n_corpus"].mean()),
                "anchor_share": float(g["anchor_share"].mean()),
                "k": float(j["k"].mean()),
                "k_shipped": float(j["k_ref"].mean()),
                "oracle_k": float(g["oracle_k"].mean()),
                "precision": float(j["precision"].mean()),
                "shortfall": float(j["shortfall"].mean()),
                "meets": float(g["meets"].mean()),
                "moved": float(np.mean(j["k"].to_numpy() != j["k_ref"].to_numpy())),
                "rel_k": float(np.mean(rel_k)),
                "abs_rel_k_p90": float(np.percentile(np.abs(rel_k), 90)),
                "f1": float(j["f1"].mean()),
                "oracle_f1": float(g["oracle_f1"].mean()),
                "d_f1": float(np.mean(d_f1)),
                "se_d_f1": cluster_se(d_f1, cells),
                "d_shortfall": float(np.mean(d_sf)),
                "se_d_shortfall": cluster_se(d_sf, cells),
            }
        )
    return pd.DataFrame(out)


def summarise_cuts(cuts: pd.DataFrame, ref: int = SHIPPED) -> pd.DataFrame:
    """Where each arm's cut sits and what it returns, and how far it moved from the *ref* cap's."""
    keys = ["world", "t", "size", "seed", "category"]
    ok = cuts[cuts["cut_ok"].astype(bool)]
    cols = ["cut0_q", "cut_acq_q", "cut0_k", "cut0_f1", "cut_acq_k"]
    base = ok[ok["cap"] == ref].set_index(keys)[cols]
    out = []
    for (world, t, n, cap), g in ok.groupby(["world", "t", "size", "cap"], sort=True):
        j = g.set_index(keys)[cols].join(base, rsuffix="_ref", how="inner")
        cells = np.array(["|".join(map(str, i)) for i in j.index.droplevel(["world", "t", "size"])])
        rec = {
            "world": world,
            "t": t,
            "size": n,
            "cap": cap,
            "runs": len(j),
            "n_corpus": float(g["n_corpus"].mean()),
            "anchor_share_cut": float(g["anchor_share_cut"].mean()),
            "oracle_f1": float(g["oracle_f1"].mean()),
        }
        for name in ("cut0", "cut_acq"):
            dq = (j[f"{name}_q"] - j[f"{name}_q_ref"]).abs().to_numpy(dtype=float) * 1e6
            rec[f"{name}_dq_ppm"] = float(np.mean(dq))
            rec[f"{name}_dq_ppm_p90"] = float(np.percentile(dq, 90))
            rec[f"{name}_dq_ppm_max"] = float(np.max(dq))
            rec[f"{name}_q"] = float(j[f"{name}_q"].mean())
        d_f1 = (j["cut0_f1"] - j["cut0_f1_ref"]).to_numpy(dtype=float)
        rec["cut0_k"] = float(j["cut0_k"].mean())
        rec["cut0_precision"] = float(g["cut0_precision"].mean())
        rec["cut0_f1"] = float(j["cut0_f1"].mean())
        rec["cut0_d_f1"] = float(np.nanmean(d_f1))
        rec["cut0_se_d_f1"] = cluster_se(d_f1, cells)
        rec["cut0_rel_k_p90"] = float(
            np.nanpercentile(np.abs(j["cut0_k"] / j["cut0_k_ref"].replace(0, np.nan) - 1.0), 90)
        )
        out.append(rec)
    return pd.DataFrame(out)


def timing(cuts: pd.DataFrame) -> pd.DataFrame:
    g = cuts.groupby(["size", "cap"])
    return pd.DataFrame(
        {
            "fits": g.size(),
            "mix_median_s": g["mix_seconds"].median(),
            "mix_p90_s": g["mix_seconds"].quantile(0.9),
            "cut_median_s": g["cut_seconds"].median(),
            "cut_p90_s": g["cut_seconds"].quantile(0.9),
        }
    ).reset_index()


def merge(out: Path) -> int:
    lines = pd.concat([pd.read_csv(p) for p in sorted((out / "shards").glob("lines_*.csv.gz"))], ignore_index=True)
    cuts = pd.concat([pd.read_csv(p) for p in sorted((out / "shards").glob("cuts_*.csv.gz"))], ignore_index=True)
    lines.to_csv(out / "lines.csv.gz", index=False)
    cuts.to_csv(out / "cuts.csv.gz", index=False)
    summarise_lines(lines).to_csv(out / "summary_lines.csv", index=False)
    summarise_cuts(cuts).to_csv(out / "summary_cuts.csv", index=False)
    timing(cuts).to_csv(out / "timing.csv", index=False)
    print(f"{len(lines)} line rows, {len(cuts)} cut rows -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--world", action="append", default=[], help="label=results_dir (holding cells/)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--steps", default="50,150")
    ap.add_argument("--seed", type=int, default=3827)
    ap.add_argument("--limit", type=int, default=0, help="cells per world (0 = all); for a smoke run")
    ap.add_argument("--shard", default="0/1", help="K/N: this task's slice of the cells")
    ap.add_argument("--merge", type=Path, help="combine OUT/shards/ into the summaries, and stop")
    args = ap.parse_args(argv)
    if args.merge:
        return merge(args.merge)
    if not args.world or args.out is None:
        ap.error("--world and --out are required unless --merge")
    (args.out / "shards").mkdir(parents=True, exist_ok=True)
    steps = tuple(int(x) for x in args.steps.split(","))
    k_shard, n_shard = (int(x) for x in args.shard.split("/"))
    jobs, prov = [], {"inputs": {}, "steps": steps, "seed": args.seed, "caps": CAPS, "sizes": SIZES}
    for spec in args.world:
        world, _, d = spec.partition("=")
        cells = Path(d) / "cells"
        mains = {p.name.split(".")[0]: p for p in main_frame_files(cells)}
        pf = [p for p in pframe_files(cells) if p.name.split("__")[0] in mains]
        if args.limit:
            pf = pf[: args.limit]
        prov["inputs"][world] = {"dir": str(cells), "cells": len(pf), "sha": sha_dir(pf)}
        jobs += [(world, str(p), str(mains[p.name.split("__")[0]]), steps, args.seed) for p in pf]
    mine = jobs[k_shard::n_shard]
    lines: list[dict] = []
    cuts: list[dict] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for li, cu in ex.map(cell_rows, mine, chunksize=1):
                lines += li
                cuts += cu
    else:
        for j in mine:
            li, cu = cell_rows(j)
            lines += li
            cuts += cu
    tag = f"{k_shard:04d}"
    pd.DataFrame(lines).to_csv(args.out / "shards" / f"lines_{tag}.csv.gz", index=False)
    pd.DataFrame(cuts).to_csv(args.out / "shards" / f"cuts_{tag}.csv.gz", index=False)
    if k_shard == 0:
        (args.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"shard {args.shard}: {len(mine)} cells, {len(lines)} line rows, {len(cuts)} cut rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
