#!/usr/bin/env python3
"""Where should the precision floor's line go on a corpus of ANY size? (#4383)

Since #4272 the line keeps a **fixed count**: the top 128 unvoted at P = 10%
and the top 32 at 50% and above (``floor_schedule``), whatever the corpus
holds. A Find set of 30 items is returned whole; a corpus of 30M gets 32. The
owner asked for a line that estimates where *this* corpus crosses P. This
prices candidate rules for it, offline, on the #4220 precision frames
(``task_NNNN__pframes.npz``: the test half's scores and labels, the session
pool's scores, the votes, and the calibration folds' held-out evidence).

Every rule may read only what the app has at Find time:

* the corpus's own **unlabeled scores** (the final model's);
* the session's **votes** (Autopilot's picks, a biased sample: #4221, #4256);
* a small budget of **audit picks**, drawn uniformly at random within a band of
  the corpus's ranking, whose labels are unbiased by construction (#4257).

The test half stands in for the Find corpus. It is resampled to several sizes
at each arm's prevalence (``SIZES``: 32, 320, 3,200, the whole half, and a x10
bootstrap of it, the last flagged as a resample), so the same rule is read on a
30-item Find set and a corpus ten times the bench.

Rules (``RULES``), each returning a count ``k >= 1`` (best effort, never empty:
the owner's #4267 framing), the audit votes it spent, and its own estimate of
the returned set's precision:

* ``fixed`` - today's unchecked line: the top ``K(P)`` (128 / 32 / 32).
* ``check`` - today's Train-time spot check run on this corpus: 5 picks
  uniformly from the top ``K(P)``, a Clopper-Pearson lower bound at alpha / R,
  halving to 32 on a failed round; ``short`` keeps the top 32.
* ``grow`` / ``grow-lb`` - audits by **band** (5 picks in each of the top 32,
  the next 32, the next 64, ... doubling to the corpus's end), searched
  adaptively: start at ``K(P)``'s bands, go one band deeper while the
  band-stratified estimate of the union's precision is >= P, one band
  shallower while it is not; return the deepest union that met P (the top 32
  when none did). ``grow-lb`` replaces each band's share by its one-sided
  Clopper-Pearson lower bound at alpha.
* ``bands-iso`` / ``bands-iso-lb`` - audit **every** band, fit a monotone
  (isotonic, non-increasing in log rank) precision curve to the picks, and
  return the deepest k whose cumulative estimate is >= P. ``-lb`` reads the
  10th percentile of ``N_BOOT`` bootstrap refits of the picks.
* ``gmm`` - the owner's partially-labelled mixture: ``fit_anchored_score_gmm``
  on the corpus's scores with the session's votes as anchors (the shipped
  fallback to the unanchored fit on a degeneracy), the high component's
  posterior at each score, and the deepest k whose mean posterior is >= P.
  No audits.
* ``gmm+shift5`` / ``gmm+shift`` - the same posterior, recalibrated by one
  logit offset fitted by maximum likelihood on the audit labels: the top
  band's 5 picks, or every band's.
* ``post`` - the shipped estimator's shape (#4220: fold-rank evidence, an
  unpenalized logistic posterior, Saerens EM to this corpus's prior), read at
  its point estimate on this corpus, ungated. No audits. This is #4267's
  best-effort option F.
* ``post+shift5`` / ``post+shift`` - that posterior with the same logit
  offset fitted on the audits.
* ``grow-fine`` - ``grow`` over **finer bands** at the top (8, 16, 32, 64, ...),
  so the search can return fewer than 32 on a small or sparse corpus.
* ``gmm-grow`` - the band search over the fine bands, **started at the band the
  mixture proposes** (``gmm``'s k) instead of at ``K(P)``: the model sizes the
  first guess, the audits correct it in either direction.
* ``min-fixed-gmm`` - **the smaller of ``fixed`` and ``gmm``** (the owner's
  ruling on #4383 for the line with no votes, #4389): the mixture where it is
  right-sized (small corpora, 5%), today's count where the mixture over-returns
  (large sparse corpora). No audits.

Where a rule's curve never reaches P, it returns the k with the highest
estimate (>= 1). ``oracle`` is the deepest k whose true precision is >= P.

Metrics per world x t x size x floor x rule (``summary.csv``), means over
cells x draws with cluster (category, seed) standard errors for the headline
ones: the returned set's ``precision``, ``meets`` (share >= P), ``shortfall``
(mean max(0, P - precision)), ``recall`` and ``recall_share`` (mean recall over
mean oracle recall), ``f1`` and ``f1_share`` (over the best F1 any cut
reaches), ``votes``, and ``est_gap`` (mean |estimate - truth|, how honest the
rule's own reading is).

    python analyze_line_estimate_4383.py --world 0.44%=DIR --world 0.1%=DIR \\
        --world 5%=DIR --out OUT [--jobs N] [--steps 50,150] [--draws 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _cells_paths import main_frame_files, pframe_files  # noqa: E402
from analyze_random_verification import lower_bound  # noqa: E402

FLOORS: tuple[float, ...] = (0.1, 0.5, 0.9)
#: Corpus sizes: item counts, the whole test half, and a x10 bootstrap of it.
SIZES: tuple[str, ...] = ("32", "320", "3200", "full", "x10")
#: Worlds whose test half is thinned to a prevalence (the 5% arm's sim pool was
#: thinned, its test half was not: #4220's "same" scenario).
THIN_TO = {"5%": 0.05}
ALPHA = 0.05
#: Picks per audit band, and the first band's size (the app's smallest line).
M_PICKS = 5
BASE = 32
N_BOOT = 30
LB_PCT = 10.0
EPS = 1e-9
RULES: tuple[str, ...] = (
    "fixed",
    "check",
    "grow",
    "grow-lb",
    "bands-iso",
    "bands-iso-lb",
    "gmm",
    "gmm+shift5",
    "gmm+shift",
    "post",
    "post+shift5",
    "post+shift",
    "grow-fine",
    "gmm-grow",
    "min-fixed-gmm",
)
#: The fine bands' first edge: the smallest set ``grow-fine`` / ``gmm-grow`` can return.
BASE_FINE = 8
HEADLINE = ("precision", "meets", "shortfall", "recall", "f1", "votes")


# ------------------------------------------------------------------ corpora


def band_edges(n: int, base: int = BASE) -> np.ndarray:
    """``[0, 32, 64, 128, ..., n]``: the audit bands of a corpus of *n* items (``[0, 8, 16, ...]`` at *base* 8)."""
    edges = [0]
    e = base
    while e < n:
        edges.append(e)
        e *= 2
    edges.append(n)
    return np.unique(np.asarray(edges, dtype=np.int64))


def floor_schedule(floor: float):
    """The precision floor's schedule (#4267): its starting candidate, the bands that hold it, the picks a band.

    ``32 * 2**max(0, floor(log2(0.5 / P)))`` - the top 128 at 10%, 64 at 25%,
    32 at 50% and above - with each band censused at ``P >= 1``.  The library
    carried it as ``check_schedule`` until the floor was removed (#4421); the
    balance keeps its 50% and 10% counts as its caps (``balance_schedule``).
    """
    import math  # noqa: PLC0415

    from vtscore.training.thresholds import (  # noqa: PLC0415
        BAND_BASE,
        CHECK_BASE_CANDIDATE,
        CHECK_MIN_PICKS,
        CheckSchedule,
        rounds_for,
    )

    if not 0.0 < floor <= 1.0:
        raise ValueError(f"precision floor must be in (0, 1], got {floor!r}")
    candidate = CHECK_BASE_CANDIDATE * 2 ** max(0, math.floor(math.log2(0.5 / floor) + EPS))
    return CheckSchedule(candidate, rounds_for(candidate), BAND_BASE if floor >= 1.0 - EPS else CHECK_MIN_PICKS)


def kept_count(floor: float, n: int) -> int:
    return int(min(floor_schedule(floor).candidate, n))


def corpus_draw(test_s: np.ndarray, test_y: np.ndarray, size: str, thin_to: float | None, rng) -> tuple:
    """``(scores, labels)`` sorted best first: the test half resampled to *size* at its prevalence."""
    pos, neg = np.flatnonzero(test_y == 1), np.flatnonzero(test_y == 0)
    if thin_to is not None and len(pos):
        keep_n = int(round(len(pos) * (1 - thin_to) / thin_to))
        neg = rng.choice(neg, size=min(keep_n, len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    if size == "x10":
        idx = rng.choice(idx, size=10 * len(idx), replace=True)
    elif size != "full":
        idx = rng.choice(idx, size=min(int(size), len(idx)), replace=False)
    s, y = test_s[idx], test_y[idx]
    order = np.argsort(-s, kind="stable")
    return s[order].astype(np.float64), y[order].astype(np.int64)


# ------------------------------------------------------------------ truth


def oracle(labels: np.ndarray, floor: float) -> tuple[int, float, float]:
    """``(k, recall, best_f1)``: the deepest cut at precision >= floor, and the best F1 of any cut."""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0, float("nan"), float("nan")
    cum = np.cumsum(labels)
    k = np.arange(1, len(labels) + 1)
    ok = np.flatnonzero(cum / k >= floor - EPS)
    ok_k = int(ok.max()) + 1 if ok.size else 0
    recall = float(cum[ok_k - 1] / n_pos) if ok_k else 0.0
    return ok_k, recall, float(np.max(2.0 * cum / (k + n_pos)))


def score_set(labels: np.ndarray, k: int, floor: float) -> dict[str, float]:
    n_pos = int(labels.sum())
    k = int(max(1, min(k, len(labels))))
    right = int(labels[:k].sum())
    precision = right / k
    return {
        "k": k,
        "precision": precision,
        "meets": float(precision >= floor - EPS),
        "shortfall": max(0.0, floor - precision),
        "recall": right / n_pos if n_pos else float("nan"),
        "f1": 2.0 * right / (k + n_pos) if n_pos else float("nan"),
    }


def k_from_curve(est: np.ndarray, floor: float) -> tuple[int, float]:
    """The deepest k whose cumulative estimate is >= floor, else the k with the highest; and that estimate."""
    cum = np.cumsum(est) / np.arange(1, len(est) + 1)
    ok = np.flatnonzero(cum >= floor - EPS)
    k = int(ok.max()) + 1 if ok.size else int(np.argmax(cum)) + 1
    return k, float(cum[k - 1])


# ------------------------------------------------------------------ audits


@dataclass
class Audits:
    """Uniform picks per band: ``ranks[b]`` and ``labels[b]`` for band *b* of ``edges``."""

    edges: np.ndarray
    ranks: list[np.ndarray]
    labels: list[np.ndarray]

    @property
    def n_bands(self) -> int:
        return len(self.edges) - 1

    def upto(self, bands: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.concatenate(self.ranks[:bands]) if bands else np.zeros(0, dtype=np.int64)
        y = np.concatenate(self.labels[:bands]) if bands else np.zeros(0, dtype=np.int64)
        return r, y

    def share(self, b: int, lb: bool = False) -> float:
        y = self.labels[b]
        if lb:
            return float(lower_bound(np.asarray([int(y.sum())]), np.asarray([len(y)]), ALPHA)[0])
        return float(y.mean())

    def union_estimate(self, bands: int, lb: bool = False) -> float:
        """Band-stratified precision of the top ``edges[bands]``: each band's share, weighted by its size."""
        top = int(self.edges[bands])
        sizes = np.diff(self.edges[: bands + 1])
        return float(sum(self.share(b, lb) * sz for b, sz in enumerate(sizes)) / top)


def draw_audits(labels: np.ndarray, rng, base: int = BASE) -> Audits:
    edges = band_edges(len(labels), base)
    ranks, labs = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        pick = np.sort(rng.choice(np.arange(lo, hi), size=min(M_PICKS, hi - lo), replace=False))
        ranks.append(pick)
        labs.append(labels[pick])
    return Audits(edges, ranks, labs)


# ------------------------------------------------------------------ rules


def rule_fixed(labels: np.ndarray, floor: float) -> tuple[int, int, float]:
    return kept_count(floor, len(labels)), 0, float("nan")


def rule_check(labels: np.ndarray, floor: float, rng) -> tuple[int, int, float]:
    """Today's spot check (#4272) on this corpus: uniform picks from the candidate, halving to 32."""
    n = len(labels)
    sched = floor_schedule(floor)
    k = min(sched.candidate, n)
    level = ALPHA / sched.rounds
    known: dict[int, int] = {}
    votes = 0
    for _ in range(sched.rounds):
        unlabelled = np.asarray([i for i in range(k) if i not in known], dtype=np.int64)
        fresh = rng.choice(unlabelled, size=min(sched.picks, len(unlabelled)), replace=False)
        for i in fresh:
            known[int(i)] = int(labels[i])
        votes += len(fresh)
        inside = [v for i, v in known.items() if i < k]
        n_k, s_k = len(inside), sum(inside)
        if n_k >= k:
            passed = s_k >= floor * k - EPS
        else:
            passed = float(lower_bound(np.asarray([s_k]), np.asarray([n_k]), level)[0]) >= floor - EPS
        if passed:
            return k, votes, s_k / max(n_k, 1)
        if k <= BASE:
            break
        k = max(BASE, k // 2)
    k = min(BASE, n)
    inside = [v for i, v in known.items() if i < k]
    return k, votes, (sum(inside) / len(inside)) if inside else float("nan")


def rule_grow(audits: Audits, floor: float, n: int, lb: bool, start_k: int | None = None) -> tuple[int, int, float]:
    """Adaptive band search: deeper while the union meets P, shallower while it does not.

    Starts at the band holding *start_k* (today's ``K(P)`` by default; a
    model's proposal for ``gmm-grow``).  Returns the deepest band edge whose
    union met P, else the first edge (32, or 8 over the fine bands).
    """
    k0 = kept_count(floor, n) if start_k is None else int(max(1, min(start_k, n)))
    start = int(np.searchsorted(audits.edges, k0, side="left"))
    bands = max(1, min(start, audits.n_bands))
    touched = {bands}
    best: int | None = None
    best_est = float("nan")
    for _ in range(2 * audits.n_bands + 2):
        est = audits.union_estimate(bands, lb)
        if est >= floor - EPS:
            best, best_est = bands, est
            if bands >= audits.n_bands:
                break
            bands += 1
        else:
            if best is not None or bands <= 1:
                break
            bands -= 1
        touched.add(bands)
    votes = sum(len(audits.labels[b]) for b in range(max(touched)))
    if best is None:
        return int(audits.edges[1]), votes, audits.union_estimate(1, lb)
    return int(audits.edges[best]), votes, best_est


def _isotonic(ranks: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.log2(ranks + 1.0), labels.astype(np.float64))
    return iso.predict(np.log2(np.arange(n) + 1.0))


def rule_bands_iso(audits: Audits, floor: float, n: int, lb: bool, rng) -> tuple[int, int, float]:
    ranks, labels = audits.upto(audits.n_bands)
    votes = len(labels)
    if labels.min() == labels.max():
        est = np.full(n, float(labels.mean()))
    elif not lb:
        est = _isotonic(ranks, labels, n)
    else:
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(labels), len(labels))
            if labels[idx].min() == labels[idx].max():
                boots.append(np.full(n, float(labels[idx].mean())))
            else:
                boots.append(_isotonic(ranks[idx], labels[idx], n))
        est = np.percentile(np.stack(boots), LB_PCT, axis=0)
    k, e = k_from_curve(est, floor)
    return k, votes, e


def gmm_posterior(scores: np.ndarray, vote_scores: np.ndarray, vote_labels: np.ndarray) -> np.ndarray | None:
    """The high component's posterior at each score, from the vote-anchored mixture on this corpus."""
    from scipy.stats import norm  # noqa: PLC0415

    from vtscore.training.thresholds.gmm import anchored_gmm_fit  # noqa: PLC0415

    fit, _prov = anchored_gmm_fit(scores, vote_scores, vote_labels)
    if fit is None:
        return None
    hi = fit.w_hi * norm.pdf(scores, fit.mu_hi, np.sqrt(fit.var_hi))
    lo = fit.w_lo * norm.pdf(scores, fit.mu_lo, np.sqrt(fit.var_lo))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = hi / (hi + lo)
    return np.nan_to_num(p, nan=0.5)


def shipped_posterior(scores: np.ndarray, pool_ref: np.ndarray, orderings, haystacks) -> np.ndarray | None:
    """#4220's shape at its point estimate: fold-rank logistic, EM-shifted to this corpus's prior."""
    from vtscore.training.thresholds.precision_floor import (  # noqa: PLC0415
        em_prior_shift,
        fit_posterior,
        fold_rank_evidence,
        percentile_in,
    )

    x, y = fold_rank_evidence(orderings, haystacks)
    post = fit_posterior(x, y, "logistic")
    if post is None or pool_ref.size == 0:
        return None
    pool_sorted = np.sort(pool_ref)
    prior_fitted = float(post(percentile_in(pool_sorted, pool_sorted)).mean())
    return em_prior_shift(post(percentile_in(pool_sorted, scores)), prior_fitted)


def fit_shift(p: np.ndarray, ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """*p* recalibrated by one logit offset, the maximum-likelihood fit to the audit labels."""
    from scipy.optimize import minimize_scalar  # noqa: PLC0415

    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    za, y = z[ranks], labels.astype(np.float64)
    # Laplace: one pseudo-positive and one pseudo-negative at the audits' mean
    # logit, so five all-positive picks cannot push the offset to its bound and
    # call the whole corpus positive.
    za = np.concatenate([za, [za.mean(), za.mean()]])
    y = np.concatenate([y, [1.0, 0.0]])

    def nll(b: float) -> float:
        q = 1.0 / (1.0 + np.exp(-(za + b)))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).sum())

    b = float(minimize_scalar(nll, bounds=(-12.0, 12.0), method="bounded").x)
    return 1.0 / (1.0 + np.exp(-(z + b)))


def rule_curve(p: np.ndarray | None, floor: float, n: int, fallback: int) -> tuple[int, float]:
    if p is None:
        return fallback, float("nan")
    return k_from_curve(p, floor)


# ------------------------------------------------------------------ one cell


def frame_arrays(z, st: str) -> dict:
    g = lambda k: z[f"{st}/{k}"]  # noqa: E731
    pool = g("pool_scores").astype(np.float64)
    votes = g("vote_scores").astype(np.float64)
    keep = np.ones(len(pool), dtype=bool)
    for v in votes:  # the consistent reference pool: one entry per voted item removed (#4221)
        hit = np.flatnonzero(keep & (np.abs(pool - v) <= 1e-7))
        if len(hit):
            keep[hit[0]] = False
    cf, hf = g("fold_cal_fold"), g("fold_hay_fold")
    folds = sorted(set(cf.tolist()))
    return {
        "t": int(g("t")),
        "test_s": g("test_scores").astype(np.float64),
        "test_y": g("test_labels").astype(np.int64),
        "pool_ref": pool[keep],
        "vote_s": votes,
        "vote_y": g("vote_labels").astype(np.float64),
        "orderings": [(g("fold_cal_scores")[cf == k], g("fold_cal_labels")[cf == k]) for k in folds],
        "haystacks": [g("fold_hay_scores")[hf == k] for k in folds],
    }


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
                rng = np.random.default_rng([seed_base, seed, t, SIZES.index(size), d, len(cat)])
                s, y = corpus_draw(fr["test_s"], fr["test_y"], size, THIN_TO.get(world), rng)
                n = len(y)
                if n == 0:
                    continue
                audits = draw_audits(y, rng)
                fine = draw_audits(y, np.random.default_rng(rng.integers(2**31)), BASE_FINE)
                p_gmm = gmm_posterior(s, fr["vote_s"], fr["vote_y"])
                p_post = shipped_posterior(s, fr["pool_ref"], fr["orderings"], fr["haystacks"])
                r5, y5 = audits.upto(1)
                r_all, y_all = audits.upto(audits.n_bands)
                shifted = {}
                for name, p in (("gmm", p_gmm), ("post", p_post)):
                    shifted[f"{name}+shift5"] = None if p is None else fit_shift(p, r5, y5)
                    shifted[f"{name}+shift"] = None if p is None else fit_shift(p, r_all, y_all)
                ident = {
                    "world": world,
                    "category": cat,
                    "band": cat.split("@")[-1],
                    "seed": seed,
                    "t": t,
                    "size": size,
                    "n_corpus": n,
                    "n_pos": int(y.sum()),
                    "n_votes": len(fr["vote_y"]),
                    "n_vote_pos": int(fr["vote_y"].sum()),
                    "draw": d,
                }
                for floor in FLOORS:
                    ok, orc_recall, orc_f1 = oracle(y, floor)
                    fixed_k = kept_count(floor, n)
                    outcomes: dict[str, tuple[int, int, float]] = {
                        "fixed": rule_fixed(y, floor),
                        "check": rule_check(y, floor, np.random.default_rng(rng.integers(2**31))),
                        "grow": rule_grow(audits, floor, n, lb=False),
                        "grow-lb": rule_grow(audits, floor, n, lb=True),
                        "bands-iso": rule_bands_iso(audits, floor, n, False, rng),
                        "bands-iso-lb": rule_bands_iso(
                            audits, floor, n, True, np.random.default_rng(rng.integers(2**31))
                        ),
                    }
                    k, e = rule_curve(p_gmm, floor, n, fixed_k)
                    outcomes["gmm"] = (k, 0, e)
                    k, e = rule_curve(p_post, floor, n, fixed_k)
                    outcomes["post"] = (k, 0, e)
                    for name in ("gmm+shift5", "post+shift5"):
                        k, e = rule_curve(shifted[name], floor, n, fixed_k)
                        outcomes[name] = (k, len(y5), e)
                    for name in ("gmm+shift", "post+shift"):
                        k, e = rule_curve(shifted[name], floor, n, fixed_k)
                        outcomes[name] = (k, len(y_all), e)
                    gmm_k_raw, gmm_e = outcomes["gmm"][0], outcomes["gmm"][2]
                    outcomes["min-fixed-gmm"] = (
                        min(fixed_k, gmm_k_raw),
                        0,
                        gmm_e if (p_gmm is not None and gmm_k_raw < fixed_k) else float("nan"),
                    )
                    outcomes["grow-fine"] = rule_grow(fine, floor, n, lb=False)
                    gmm_k = outcomes["gmm"][0] if p_gmm is not None else None
                    outcomes["gmm-grow"] = rule_grow(fine, floor, n, lb=False, start_k=gmm_k)
                    for rule, (k, votes, est) in outcomes.items():
                        m = score_set(y, k, floor)
                        rows.append(
                            {
                                **ident,
                                "floor": floor,
                                "rule": rule,
                                **m,
                                "votes": votes,
                                "est": est,
                                "est_gap": abs(est - m["precision"]) if np.isfinite(est) else float("nan"),
                                "oracle_k": ok,
                                "oracle_recall": orc_recall,
                                "oracle_f1": orc_f1,
                            }
                        )
    return rows


# ------------------------------------------------------------------ summaries


def cluster_se(x: np.ndarray, cells: np.ndarray) -> float:
    """SE of a mean with (category, seed) clusters: the cluster means' SE."""
    df = pd.DataFrame({"x": x, "c": cells}).dropna()
    if df.empty:
        return float("nan")
    means = df.groupby("c")["x"].mean()
    return float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else float("nan")


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["world", "t", "size", "floor", "rule"]
    out = []
    for key, g in rows.groupby(keys, sort=False):
        cells = (g["category"] + "|" + g["seed"].astype(str)).to_numpy()
        rec = {k: v for k, v in zip(keys, key, strict=True)}
        rec["n_corpus"] = float(g["n_corpus"].mean())
        rec["n_pos"] = float(g["n_pos"].mean())
        rec["runs"] = int(len(g))
        for m in (*HEADLINE, "est_gap", "k", "oracle_k", "oracle_recall", "oracle_f1"):
            rec[m] = float(g[m].mean())
        for m in HEADLINE:
            rec[f"se_{m}"] = cluster_se(g[m].to_numpy(dtype=float), cells)
        rec["recall_share"] = rec["recall"] / rec["oracle_recall"] if rec["oracle_recall"] > 0 else float("nan")
        rec["f1_share"] = rec["f1"] / rec["oracle_f1"] if rec["oracle_f1"] > 0 else float("nan")
        out.append(rec)
    return pd.DataFrame(out)


def paired_vs_fixed(rows: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Each rule minus ``fixed`` on the same corpus draw, per world x t x size x floor, with cluster SE."""
    idx = ["world", "category", "seed", "t", "size", "draw", "floor"]
    base = rows[rows["rule"] == "fixed"].set_index(idx)[metric]
    out = []
    for rule, g in rows[rows["rule"] != "fixed"].groupby("rule"):
        gi = g.set_index(idx)
        d = (gi[metric] - base.reindex(gi.index)).dropna()
        if d.empty:
            continue
        df = d.reset_index()
        for key, gg in df.groupby(["world", "t", "size", "floor"]):
            cells = (gg["category"] + "|" + gg["seed"].astype(str)).to_numpy()
            out.append(
                {
                    "rule": rule,
                    **dict(zip(["world", "t", "size", "floor"], key, strict=True)),
                    f"d_{metric}": float(gg[metric].mean()),
                    f"se_d_{metric}": cluster_se(gg[metric].to_numpy(dtype=float), cells),
                    "n": int(len(gg)),
                }
            )
    return pd.DataFrame(out)


def sha_dir(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--world", action="append", required=True, help="label=results_dir (holding cells/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--steps", default="50,150")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4383)
    ap.add_argument("--limit", type=int, default=0, help="cells per world (0 = all); for a smoke run")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    steps = tuple(int(x) for x in args.steps.split(","))
    jobs, prov = [], {"inputs": {}, "steps": steps, "draws": args.draws, "seed": args.seed, "alpha": ALPHA}
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
    paired_vs_fixed(df, "f1").merge(
        paired_vs_fixed(df, "shortfall"), on=["rule", "world", "t", "size", "floor", "n"]
    ).to_csv(args.out / "paired_vs_fixed.csv", index=False)
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"{len(df)} rows over {len(jobs)} cells -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
