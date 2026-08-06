"""Anchored-κ estimator study on the known-truth region model (#2864 follow-up).

The #2864 anchor-mass sweep found, empirically, that the fused threshold is a
clear win on **region voting** and a wash-to-loss on **binary voting**, with a
small interior κ* on region environments and a flat, meaningless κ curve on
binary ones.  This bench tests the proposed *mechanism* on the closed-form
generative model :mod:`theory_bench` built for #2836, where the truth is exact:

    An image's score is the max over its ``m`` regions.  The max operator is an
    **overlap generator**: it translates the negative bulk up by ~sqrt(2 ln m)
    and compresses it, censors the positive lower tail at the background max,
    and amplifies FPR sensitivity m-fold.  All three raise the curvature
    ``L''(tau*)`` of the rate loss at the optimum.  Since regret ~
    0.5 * L'' * (tau_hat - tau*)^2, curvature converts threshold *imprecision*
    into cost: at m = 1 the optimum sits in a flat valley and every estimator
    ties; at m = 24 precise cut *location* dominates, so the low-variance
    population estimator wins and the labels' biased, high-variance authority
    must be kept small - hence a small κ*.

Where :mod:`theory_bench` swept cut *rules* on a fixed fit, this bench sweeps
the *estimators*, reproducing the production stack in 1-D miniature.  Each
replicate trains a regularized 1-D logistic head on the votes (the trained
score *scale* production fits its mixtures on) plus one head per calibration
fold, so the three structural deficits of cross-calibration (#2790/#2799:
conformal sample size, fold-to-final scale transfer, per-retrain redraw) all
exist here and can be priced against the exact optimum.  A first cut of this
bench skipped the trained heads and scored a conformal quantile of honest
votes on the *raw* score scale - and that arm beat every mixture even at
m = 24, which is itself a finding: the conformal rule's real-data losses are
not a bare sample-size effect but come from the scale deficits the fold
machinery exists to repair, amplified by the max operator's curvature.  The
``conformal_true_scale`` arm keeps that decomposition measurable.

Arms (all production code paths from :mod:`vtscore.training.thresholds`):

* ``xcal_only`` - :func:`threshold_from_fold_orderings` on the fold heads'
  held-out vote scores: the pure cross-calibration estimator, fold-scale
  deficit included.
* ``conformal_true_scale`` - the same conformal rule on the *final* head's
  vote scores: sample-size deficit only (not shippable - a diagnostic bound).
* ``rank_transfer`` - the x-cal cut carried to the final scale as a quantile:
  repairs scale transfer only.
* ``unanchored_{mid,rate}`` - :func:`fit_score_gmm` on the final head's sim
  scores: the κ -> 0 population-only limit.
* ``blend_{cap50,slow_cap50}`` - :func:`blend_gmm_threshold` on the shipped
  per-mode schedules: the pre-#2861 production blends the real data ranks
  fusion against (``cap50`` beats fusion on binary voting there).
* ``fold_w{κ}_{mid,rate}`` - :func:`fold_anchored_gmm_threshold`, the shipped
  fusion estimator, per anchor mass κ.
* ``label_w{κ}_{mid,rate}`` - :func:`anchored_gmm_fit` anchoring on the final
  head's scores of the voted items (train-set anchors, the #2852 label family).

Every cut is inverted through the final head and scored on the **closed-form
true loss**, so excess is measured against the exact optimum with no held-out
sample noise.

Pre-registered predictions (checked and printed by :func:`check_predictions`):

* **P1** - the fusion advantage (``xcal_only`` excess minus best fold-anchored
  excess) grows with ``m`` and is smallest at ``m = 1``.
* **P2** - the κ curve is flat at ``m = 1``; at ``m >= 6`` it has an interior
  optimum whose argmin falls as the vote count grows.
* **P3** - across configurations, the fusion advantage tracks the true-loss
  curvature ``L''(tau*)`` (curvature as the mediating variable, also ordering
  configurations *within* each ``m``).
* **P4** - threshold-adjacent ("hard") vote acquisition shifts κ* down
  relative to random votes at high ``m``.

Usage: ``python theory_kappa_bench.py [--reps 25] [--procs 4] [--smoke]``
"""

from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vtscore.training.blend_schedules import BlendContext  # noqa: E402
from vtscore.training.thresholds import (  # noqa: E402
    anchored_gmm_fit,
    blend_gmm_threshold,
    conformal_threshold,
    fit_score_gmm,
    fold_anchored_gmm_threshold,
    gmm_cut_from_fit,
    rank_transfer,
    threshold_from_fold_orderings,
)

#: Region counts. 1 = the whole-image (binary-voting) control; 24 ~ the
#: production patch grid's usable region count (matches theory_bench).
M_VALUES: tuple[int, ...] = (1, 6, 24)
#: Class separation in per-region logit units.  2-3 are overlap-heavy (the
#: region-voting regime); 4-5 are near-separable - the analog of the real
#: *binary* environments, whose x-cal FNR runs 0.04-0.06.  The real #2864
#: design confounds pooling with achieved separation (its region environments
#: are also its overlappy ones); crossing m with separation is what lets this
#: bench split them.
SEPARATIONS: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
#: Positive-class prevalence of the sim population.
PREVALENCES: tuple[float, ...] = (0.02, 0.05)
#: Sim-set (haystack) size - the real fit populations were 419-2476.
SIM_N: int = 2000
#: Vote counts, spanning the #2864 windows.
VOTE_COUNTS: tuple[int, ...] = (20, 50, 100, 300)
#: Fraction of votes that are positive (~ the deep-regime environments'
#: 7-24 positives per ~176 votes).
POS_VOTE_FRAC: float = 0.1
#: Vote acquisition modes: iid class samples vs Autopilot-like mixed
#: acquisition (half iid, half nearest-tau* from a 5x pool - the Hard phase is
#: only one of Autopilot's phases; an all-hard vote set makes ~half the 1-D
#: heads degenerate at 20 votes, a selection bias run 1 of this bench hit).
VOTE_MODES: tuple[str, ...] = ("random", "hard")
HARD_POOL_FACTOR: int = 5
HARD_FRACTION: float = 0.5
#: Anchor masses. Spans #2864's grid, extended one decade up (the
#: label-anchored trap) and one down (at 300 votes the argmin fell to run 1's
#: bottom edge).
KAPPAS: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
RULES: tuple[str, ...] = ("mid", "rate")
#: Calibration folds, matching production's ``calibrate_count``.
N_FOLDS: int = 2

MU_BAD = 0.0
SD_BAD = 1.0
SD_GOOD = 1.0
#: Inclusion 0 cost weights - the setting every #2864 arm was scored at.
FPR_WEIGHT = 1.0
FNR_WEIGHT = 1.0
INCLUSION = 0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return float(np.log(p) - np.log1p(-p))


def _normal_cdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    from scipy.special import ndtr  # noqa: PLC0415

    return ndtr((x - mu) / sd)


def true_loss(t_logit: np.ndarray, m: int, mu_good: float) -> np.ndarray:
    """Exact rate loss at each raw-logit threshold (theory_bench's model)."""
    phi_bad = _normal_cdf(t_logit, MU_BAD, SD_BAD)
    phi_good = _normal_cdf(t_logit, mu_good, SD_GOOD)
    fpr = 1.0 - phi_bad**m
    fnr = phi_good * phi_bad ** (m - 1)
    return FPR_WEIGHT * fpr + FNR_WEIGHT * fnr


def population_optimum(m: int, mu_good: float) -> tuple[float, float, float]:
    """``(tau*_logit, min_loss, curvature L'')`` on a fine grid."""
    grid = np.linspace(MU_BAD - 6.0 * SD_BAD, mu_good + 6.0 * SD_GOOD, 20_001)
    loss = true_loss(grid, m, mu_good)
    i = int(np.argmin(loss))
    tau, lmin = float(grid[i]), float(loss[i])
    h = 0.05
    lpp = float(
        (true_loss(np.array([tau + h]), m, mu_good)[0] - 2.0 * lmin + true_loss(np.array([tau - h]), m, mu_good)[0])
        / h**2
    )
    return tau, lmin, lpp


def _class_logits(rng: np.random.Generator, n: int, m: int, mu_good: float, positive: bool) -> np.ndarray:
    """*n* raw image logits (max over regions) drawn from one class."""
    bad = rng.normal(MU_BAD, SD_BAD, size=(n, m))
    logits = bad.max(axis=1)
    if positive:
        obj = rng.normal(mu_good, SD_GOOD, size=n)
        others = bad[:, 1:].max(axis=1) if m > 1 else np.full(n, -np.inf)
        logits = np.maximum(obj, others)
    return logits


def sample_population(
    rng: np.random.Generator, n: int, m: int, prevalence: float, mu_good: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(raw logits, labels)`` for a mixed-population sample of *n* images."""
    labels = (rng.random(n) < prevalence).astype(np.float64)
    n_pos = int(labels.sum())
    logits = _class_logits(rng, n, m, mu_good, positive=False)
    if n_pos:
        logits[labels == 1.0] = _class_logits(rng, n_pos, m, mu_good, positive=True)
    return logits, labels


def sample_votes(
    rng: np.random.Generator, n_votes: int, m: int, mu_good: float, tau_star_logit: float, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(raw vote logits, labels)`` under an acquisition *mode*.

    ``random`` draws each class iid.  ``hard`` draws :data:`HARD_FRACTION` of
    each class as the nearest-to-tau* picks from a :data:`HARD_POOL_FACTOR`
    times larger pool (Autopilot's Hard phase) and the rest iid (its other
    phases) - a stateless stand-in for threshold-adjacent acquisition.
    """
    n_pos = max(2, round(POS_VOTE_FRAC * n_votes))
    n_neg = n_votes - n_pos

    def draw(n: int, positive: bool) -> np.ndarray:
        if mode == "random":
            return _class_logits(rng, n, m, mu_good, positive)
        if mode != "hard":
            raise ValueError(f"unknown vote mode {mode!r}")
        n_hard = round(HARD_FRACTION * n)
        pool = _class_logits(rng, max(n_hard, 1) * HARD_POOL_FACTOR, m, mu_good, positive)
        hard = pool[np.argsort(np.abs(pool - tau_star_logit))[:n_hard]]
        return np.concatenate([hard, _class_logits(rng, n - n_hard, m, mu_good, positive)])

    pos, neg = draw(n_pos, positive=True), draw(n_neg, positive=False)
    return np.concatenate([pos, neg]), np.concatenate([np.ones(n_pos), np.zeros(n_neg)])


def _fit_head(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Regularized 1-D logistic head ``score = sigmoid(a*x + c)``; ``None`` if degenerate.

    The miniature of production's trained head: it cannot change the ranking
    (1-D, monotone for ``a > 0``) but it *learns the score scale* the mixture
    estimators fit on and the conformal rule cuts on - which is exactly the
    part of the system the threshold estimators are sensitive to.  ``a <= 0``
    (possible under heavy overlap with few votes) is reported as degenerate
    rather than silently flipping the ranking.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    if len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(C=1.0)
    clf.fit(x.reshape(-1, 1), y)
    a = float(clf.coef_[0][0])
    c = float(clf.intercept_[0])
    if not (np.isfinite(a) and np.isfinite(c)) or a <= 0.0:
        return None
    return a, c


def _head_scores(head: tuple[float, float], x: np.ndarray) -> np.ndarray:
    a, c = head
    return _sigmoid(a * x + c)


def _fold_split(labels: np.ndarray) -> list[np.ndarray]:
    """Deterministic 2-fold split, both classes represented in each fold."""
    folds: list[list[int]] = [[] for _ in range(N_FOLDS)]
    for cls in (1.0, 0.0):
        for j, idx in enumerate(np.flatnonzero(labels == cls)):
            folds[j % N_FOLDS].append(int(idx))
    return [np.asarray(f, dtype=np.int64) for f in folds]


def evaluate_config(
    rng: np.random.Generator, m: int, separation: float, prevalence: float, n_votes: int, vote_mode: str
) -> dict:
    """One replicate: every estimator on one (sim sample, vote set, trained heads)."""
    mu_good = MU_BAD + separation
    tau_star, best_loss, curvature = population_optimum(m, mu_good)

    sim_logits, _sim_labels = sample_population(rng, SIM_N, m, prevalence, mu_good)
    vote_logits, vote_labels = sample_votes(rng, n_votes, m, mu_good, tau_star, vote_mode)

    row: dict = {
        "m": m,
        "separation": separation,
        "prevalence": prevalence,
        "n_votes": n_votes,
        "n_pos": int(vote_labels.sum()),
        "vote_mode": vote_mode,
        "tau_star_logit": tau_star,
        "best_loss": best_loss,
        "curvature": curvature,
        "degenerate_head": 0,
    }

    final = _fit_head(vote_logits, vote_labels)
    fold_idx = _fold_split(vote_labels)
    fold_heads = []
    for k in range(N_FOLDS):
        train = np.concatenate([fold_idx[j] for j in range(N_FOLDS) if j != k])
        fold_heads.append(_fit_head(vote_logits[train], vote_labels[train]))
    if final is None or any(h is None for h in fold_heads):
        row["degenerate_head"] = 1
        return row

    a, c = final
    sim_final = _head_scores(final, sim_logits)
    fold_haystacks = [np.asarray(_head_scores(h, sim_logits), dtype=np.float64) for h in fold_heads]
    fold_orderings = [
        (_head_scores(h, vote_logits[fold_idx[k]]).tolist(), vote_labels[fold_idx[k]].tolist())
        for k, h in enumerate(fold_heads)
    ]

    def score(name: str, threshold_sigmoid: float) -> None:
        """Excess true loss of cutting the *final* head's scores at this threshold."""
        if not np.isfinite(threshold_sigmoid) or not 0.0 < threshold_sigmoid < 1.0:
            row[f"excess_{name}"] = float("nan")
            return
        t_raw = (_logit(float(threshold_sigmoid)) - c) / a
        row[f"excess_{name}"] = float(true_loss(np.array([t_raw]), m, mu_good)[0] - best_loss)

    xcal = threshold_from_fold_orderings(fold_orderings, INCLUSION)
    score("xcal_only", xcal)
    score("rank_transfer", rank_transfer(xcal, np.concatenate(fold_haystacks), sim_final))
    score(
        "conformal_true_scale",
        conformal_threshold(_head_scores(final, vote_logits).tolist(), vote_labels.tolist(), INCLUSION),
    )

    fit = fit_score_gmm(sim_final)
    for rule in RULES:
        score(f"unanchored_{rule}", gmm_cut_from_fit(fit, rule, FPR_WEIGHT, FNR_WEIGHT)[0] if fit else float("nan"))

    # The shipped pre-fusion blends (#2841/#2849): conformal x-cal averaged
    # with the unanchored mid cut on the per-mode schedule.  ``cap50`` is what
    # binary voting had before #2861 - the arm the real data says fusion loses
    # to there - and ``slow_cap50`` its region counterpart.
    if fit is not None:
        ctx = BlendContext(n_labels=n_votes, n_good=int(vote_labels.sum()), n_bad=int((1.0 - vote_labels).sum()))
        mid_cut = gmm_cut_from_fit(fit, "mid", FPR_WEIGHT, FNR_WEIGHT)[0]
        for schedule in ("cap50", "slow_cap50"):
            score(f"blend_{schedule}", blend_gmm_threshold(xcal, mid_cut, ctx, schedule=schedule))
    else:
        row["excess_blend_cap50"] = float("nan")
        row["excess_blend_slow_cap50"] = float("nan")

    anchor_scores = _head_scores(final, vote_logits).tolist()
    for kappa in KAPPAS:
        for rule in RULES:
            thr, provenance = fold_anchored_gmm_threshold(
                fold_haystacks, fold_orderings, sim_final, INCLUSION, anchor_weight=kappa, cut_rule=rule
            )
            if rule == RULES[0]:
                row[f"fold_fallback_w{kappa:g}"] = 0 if provenance.startswith("fold_anchored") else 1
            score(f"fold_w{kappa:g}_{rule}", thr)
        lfit, lprov = anchored_gmm_fit(sim_final, anchor_scores, vote_labels.tolist(), anchor_weight=kappa)
        for rule in RULES:
            score(
                f"label_w{kappa:g}_{rule}",
                gmm_cut_from_fit(lfit, rule, FPR_WEIGHT, FNR_WEIGHT)[0] if lfit else float("nan"),
            )
    return row


def selfcheck() -> None:
    """The sampled scores must reproduce the closed-form rates (planted truth)."""
    rng = np.random.default_rng(7)
    m, sep = 6, 3.0
    n = 200_000
    pos = _class_logits(rng, n, m, sep, positive=True)
    neg = _class_logits(rng, n, m, sep, positive=False)
    for t in (1.0, 2.0, 2.5):
        fpr_emp, fnr_emp = float((neg >= t).mean()), float((pos < t).mean())
        phi_bad = _normal_cdf(np.array([t]), MU_BAD, SD_BAD)[0]
        phi_good = _normal_cdf(np.array([t]), sep, SD_GOOD)[0]
        fpr_true, fnr_true = 1.0 - phi_bad**m, phi_good * phi_bad ** (m - 1)
        if abs(fpr_emp - fpr_true) > 0.01 or abs(fnr_emp - fnr_true) > 0.01:
            raise AssertionError(
                f"selfcheck failed at t={t}: empirical ({fpr_emp:.4f}, {fnr_emp:.4f}) "
                f"vs closed-form ({fpr_true:.4f}, {fnr_true:.4f})"
            )
    common.log("selfcheck passed: sampled rates match the closed form")


def _cell_path(outdir: Path, cfg_i: int) -> Path:
    return outdir / "cells" / f"cfg_{cfg_i:04d}.csv"


def _run_one_config(task: tuple[int, tuple[int, float, float, int, str], int, str]) -> str:
    """All reps for one configuration, written to its own cell file.

    Each configuration checkpoints itself so a killed run resumes instead of
    losing everything: the sweep is ~30 minutes and a mid-run interruption used
    to discard 184 of 192 completed configurations.  The cell is written to a
    temp path and renamed, so a kill during the write cannot leave a truncated
    file that resume would count as done.
    """
    cfg_i, (m, sep, prev, n_votes, mode), reps, out = task
    path = _cell_path(Path(out), cfg_i)
    rng = np.random.default_rng(np.random.SeedSequence(entropy=22864, spawn_key=(cfg_i,)))
    rows = [evaluate_config(rng, m, sep, prev, n_votes, mode) for _ in range(reps)]
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.rename(path)
    return str(path)


def run_sweep(reps: int, procs: int, outdir: Path, smoke: bool = False, resume: bool = True) -> pd.DataFrame:
    grid = list(itertools.product(M_VALUES, SEPARATIONS, PREVALENCES, VOTE_COUNTS, VOTE_MODES))
    if smoke:
        grid = [g for g in grid if g[1] == 3.0 and g[2] == 0.02 and g[3] == 100]
    (outdir / "cells").mkdir(parents=True, exist_ok=True)

    # A zero-byte cell is debris from a kill mid-write, not a completed cell.
    done = {
        i
        for i in range(len(grid))
        if resume and _cell_path(outdir, i).exists() and _cell_path(outdir, i).stat().st_size > 0
    }
    for i in range(len(grid)):
        p = _cell_path(outdir, i)
        if p.exists() and i not in done:
            p.unlink()
    tasks = [(i, cfg, reps, str(outdir)) for i, cfg in enumerate(grid) if i not in done]
    common.log(
        f"kappa bench: {len(grid)} configurations x {reps} reps = {len(grid) * reps} replicates, "
        f"{procs} procs ({len(done)} cells already done, {len(tasks)} to run)"
    )

    if procs <= 1:
        for i, task in enumerate(tasks):
            _run_one_config(task)
            common.log(f"  {i + 1}/{len(tasks)} configurations")
    else:
        with Pool(procs) as pool:
            for i, _ in enumerate(pool.imap_unordered(_run_one_config, tasks)):
                if (i + 1) % 8 == 0:
                    common.log(f"  {i + 1}/{len(tasks)} configurations")

    cells = sorted((outdir / "cells").glob("cfg_*.csv"))
    missing = len(grid) - len(cells)
    if missing:
        common.log(f"WARNING: {missing} of {len(grid)} cells missing; analysing the {len(cells)} present")
    return pd.concat([pd.read_csv(c) for c in cells], ignore_index=True)


def _kappa_curve(df: pd.DataFrame, family: str, rule: str) -> pd.DataFrame:
    """Mean excess per (m, n_votes, vote_mode, κ) for one estimator family+rule."""
    records = []
    for (m, n_votes, mode), sub in df.groupby(["m", "n_votes", "vote_mode"]):
        for kappa in KAPPAS:
            records.append(
                {
                    "m": m,
                    "n_votes": n_votes,
                    "vote_mode": mode,
                    "kappa": kappa,
                    "excess": float(sub[f"excess_{family}_w{kappa:g}_{rule}"].mean()),
                }
            )
    return pd.DataFrame(records)


def check_predictions(df: pd.DataFrame, outdir: Path) -> dict:
    """Evaluate P1-P4 and write the tidy aggregates they are read from."""
    n_total = len(df)
    df = df[df["degenerate_head"] == 0].copy()
    dropped = n_total - len(df)

    fold_cols = [f"excess_fold_w{k:g}_{r}" for k in KAPPAS for r in RULES]
    base_cols = [
        "excess_xcal_only",
        "excess_rank_transfer",
        "excess_conformal_true_scale",
        "excess_unanchored_mid",
        "excess_unanchored_rate",
        "excess_blend_cap50",
        "excess_blend_slow_cap50",
    ]
    label_cols = [f"excess_label_w{k:g}_{r}" for k in KAPPAS for r in RULES]

    keys = ["m", "separation", "prevalence", "n_votes", "vote_mode"]
    by_config = df.groupby(keys)[[*base_cols, *fold_cols, *label_cols, "curvature"]].mean().reset_index()
    by_config["best_fold"] = by_config[fold_cols].min(axis=1)
    by_config["advantage"] = by_config["excess_xcal_only"] - by_config["best_fold"]
    by_config.to_csv(outdir / "kappa_by_config.csv", index=False)

    curve = _kappa_curve(df, "fold", "mid")
    curve.to_csv(outdir / "kappa_curve_fold_mid.csv", index=False)
    _kappa_curve(df, "label", "mid").to_csv(outdir / "kappa_curve_label_mid.csv", index=False)

    # P1: advantage vs m (random votes).
    rnd_cfg = by_config[by_config["vote_mode"] == "random"]
    p1 = {str(int(m)): float(rnd_cfg[rnd_cfg["m"] == m]["advantage"].mean()) for m in M_VALUES}

    # P2: κ-curve spread by m, and argmin κ by n at the largest m (fold mid,
    # random votes - the recommended family).
    rnd = curve[curve["vote_mode"] == "random"]
    p2_spread = {}
    for m in M_VALUES:
        sub = rnd[rnd["m"] == m].groupby("kappa")["excess"].mean()
        p2_spread[str(int(m))] = float(sub.max() - sub.min())
    p2_argmin = {}
    for n_votes, sub in rnd[rnd["m"] == max(M_VALUES)].groupby("n_votes"):
        p2_argmin[str(int(n_votes))] = float(sub.set_index("kappa")["excess"].idxmin())

    # P3: does curvature explain the advantage across all configs, and within m?
    finite = by_config.dropna(subset=["advantage", "curvature"])
    p3_corr = float(np.corrcoef(finite["curvature"], finite["advantage"])[0, 1]) if len(finite) > 2 else float("nan")
    p3_within = {}
    for m in M_VALUES:
        sub = finite[finite["m"] == m]
        if len(sub) > 2 and sub["curvature"].std() > 0:
            p3_within[str(int(m))] = float(np.corrcoef(sub["curvature"].rank(), sub["advantage"].rank())[0, 1])

    # P4: argmin κ under hard vs random votes at each m (fold mid, pooled n).
    p4 = {}
    for m in M_VALUES:
        entry = {}
        for mode in VOTE_MODES:
            s = curve[(curve["m"] == m) & (curve["vote_mode"] == mode)].groupby("kappa")["excess"].mean()
            entry[mode] = float(s.idxmin())
        p4[str(int(m))] = entry

    # Deficit decomposition at the biggest m (random votes): what each repair buys.
    big = rnd_cfg[rnd_cfg["m"] == max(M_VALUES)]
    decomposition = {
        col.removeprefix("excess_"): float(big[col].mean())
        for col in [*base_cols, "excess_fold_w0.3_mid", "best_fold"]
        if col in big
    }

    # The real-data mode split, reproduced or not: fusion-vs-blend by
    # separation (random votes, pooled m/prev/n).  The #2864 ranking to match:
    # blend ahead of fusion on near-separable ("binary") configs, fusion ahead
    # where classes overlap ("region").
    ranking_by_separation = {}
    for sep in SEPARATIONS:
        sub = rnd_cfg[rnd_cfg["separation"] == sep]
        ranking_by_separation[f"{sep:g}"] = {
            "xcal_only": float(sub["excess_xcal_only"].mean()),
            "blend_cap50": float(sub["excess_blend_cap50"].mean()),
            "fold_w0.3_mid": float(sub["excess_fold_w0.3_mid"].mean()),
            "best_fold": float(sub["best_fold"].mean()),
        }

    fallback_rate = {f"{k:g}": float(df[f"fold_fallback_w{k:g}"].mean()) for k in KAPPAS}
    return {
        "n_replicates": int(n_total),
        "n_degenerate_heads_dropped": int(dropped),
        "p1_advantage_by_m": p1,
        "p2_kappa_spread_by_m": p2_spread,
        "p2_argmin_kappa_by_n_at_max_m": p2_argmin,
        "p3_curvature_advantage_corr": p3_corr,
        "p3_within_m_rank_corr": p3_within,
        "p4_argmin_kappa_by_mode": p4,
        "ranking_by_separation": ranking_by_separation,
        "decomposition_at_max_m_random": decomposition,
        "fold_fallback_rate_by_kappa": fallback_rate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=25, help="replicates per configuration")
    parser.add_argument("--procs", type=int, default=4, help="worker processes")
    parser.add_argument("--smoke", action="store_true", help="tiny grid, for sizing")
    parser.add_argument("--no-resume", action="store_true", help="re-run every cell instead of reusing cells/")
    parser.add_argument("--out", default=str(common.RESULTS / "theory_kappa"))
    args = parser.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    selfcheck()
    df = run_sweep(args.reps, args.procs, outdir, smoke=args.smoke, resume=not args.no_resume)
    df.to_csv(outdir / "kappa_raw.csv", index=False)
    summary = check_predictions(df, outdir)
    (outdir / "kappa_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    common.log(f"kappa bench complete -> {outdir}")
    common.log(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
