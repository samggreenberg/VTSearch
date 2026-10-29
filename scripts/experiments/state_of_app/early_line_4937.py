#!/usr/bin/env python3
"""#4937: why the labels line keeps too few early in a session, read on the withheld half.

Reads the snapshots of #4936's instrument (``fold_scale_4936.sh``, ``CALIB_SAVE_FOLD_SCALES``): at each listed
click, the withheld half under the final model and under every live fold model, the sim set with its truth and
votes, and the folds' held-out scores. Nothing here re-runs a session.

Find's line on the withheld half (``fit_labels_line(...).on_corpus(withheld scores)``, what the harness draws) is
rebuilt here piece by piece, and each piece can be swapped for what the withheld half's own truth says it should be:

* **the Good component** (mean, spread, shape): today's (the held-out Goods, pooled over the folds, one spread with
  the Bads, floored on the corpus); its mean with selection taken out (the withheld positives' mean on the fold
  models' scale); its mean set to the withheld positives' own on the final scale; its spread too; or the withheld
  positives' own distribution (a kernel density) in place of the normal;
* **the corpus fit**: the prevalence (2-part EM, capped at twice the counted bound), and each item's chance of being
  a positive (3-part EM): the bound removed, the true total, the true chances (isotonic on the truth), the 2-part
  chances;
* **thin evidence**: the corpus spread floor (#4492) removed, and the head's count line (#4786) replaced by the class
  model's cut.

Each swap is read as Find's line on the same head: its prevalence estimate, set size, expected F-beta and F-beta.
The Good component's error is split exactly into selection and scale (mean) and into floor, pooling, selection and
scale (spread). Every swap of the class model goes through the same code path as today's line, and ``today``
reproduces the library's line on every snapshot (``chk_*`` columns).

  early_line_4937.py measure --results DIR [--results DIR ...] --out DIR [--procs N]
  early_line_4937.py report --out DIR [--docs DIR] [--figs DIR]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fold_scale_4936 import PATHS, Snapshot, _ident, _logit, click_of, load_snapshots  # noqa: E402

#: Find's line, rebuilt with one piece swapped (see the module docstring).  ``today`` is the run's own line.
VARIANTS = (
    "today",
    "sel_off",  # Good mean: the withheld positives' mean on the fold models' scale (selection out, scale as today)
    "mean_true",  # Good mean: the withheld positives' own on the final scale
    "sd_true",  # Good spread: the withheld positives' own (the Bad component keeps today's)
    "good_true",  # Good mean and spread: the withheld positives' own
    "shape_true",  # Good component: the withheld positives' own kernel density
    "total_true",  # today's chances, the true number of positives
    "neg_true",  # the chances with the negatives as the withheld negatives' own density (Good today), today's total
    "shapes_true",  # the chances with both classes as the withheld half's own densities, today's total
    "post_true",  # the true chances (isotonic on the truth), today's total
    "both_true",  # the true chances and the true total: the best cut, smoothed
    "no_bound",  # the prevalence uncapped by the counted bound
    "post_2part",  # the chances from the 2-part fit (Good against the bulk, no Bad component)
    "no_floor",  # no corpus spread floor (#4492): the pooled spread as the folds give it
    "no_gate",  # the class model's cut even where the head's count line serves (#4786)
    "no_gate_good_true",  # the same, with the Good component's mean and spread the truth
)


# --- Find's line, piece by piece ---------------------------------------------------------------------------------


class Normal:
    def __init__(self, mu: float, sigma: float):
        self.mu, self.sigma = float(mu), float(sigma)

    def pdf(self, x):
        from scipy.stats import norm

        return norm.pdf(x, self.mu, self.sigma)

    def sf(self, x):
        from scipy.stats import norm

        return norm.sf((np.asarray(x) - self.mu) / self.sigma)


class Kde:
    """A Gaussian kernel density of the withheld positives' logits (Silverman's bandwidth)."""

    def __init__(self, pts):
        self.pts = np.asarray(pts, dtype=np.float64)
        n = self.pts.size
        sd = float(np.std(self.pts, ddof=1)) if n > 1 else 0.25
        iqr = float(np.subtract(*np.percentile(self.pts, [75, 25]))) / 1.349 if n > 3 else sd
        self.h = max(0.9 * (min(sd, iqr) if iqr > 0 else sd) * n ** (-0.2), 0.01)
        self.mu = float(self.pts.mean())

    def pdf(self, x):
        from scipy.stats import norm

        x = np.asarray(x, dtype=np.float64)
        return norm.pdf((x[:, None] - self.pts[None, :]) / self.h).mean(axis=1) / self.h

    def sf(self, x):
        from scipy.stats import norm

        x = np.asarray(x, dtype=np.float64)
        return norm.sf((x[:, None] - self.pts[None, :]) / self.h).mean(axis=1)


def em2(x, good, floor: float, mu_cap: float, iterations: int = 500, tol: float = 1e-9):
    """:func:`~vtscore.training.thresholds.labels_line.fit_corpus` with any Good component: ``(EM positives, r)``."""
    from scipy.stats import norm

    from vtscore.training.thresholds.labels_line import PREVALENCE_MAX, PREVALENCE_MIN

    mu0 = float(np.median(x))
    s0 = max(1.4826 * float(np.median(np.abs(x - mu0))), floor)
    pi = 0.01
    g = good.pdf(x)
    r = np.zeros_like(x)
    for _ in range(iterations):
        f1 = pi * g
        f0 = (1.0 - pi) * norm.pdf(x, mu0, s0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nan_to_num(f1 / (f1 + f0), nan=0.0)
        new_pi = min(max(float(r.mean()), PREVALENCE_MIN), PREVALENCE_MAX)
        w = 1.0 - r
        sw = float(w.sum())
        if sw <= 0:
            break
        new_mu0 = min(float((w * x).sum()) / sw, mu_cap)
        new_s0 = max(math.sqrt(float((w * (x - new_mu0) ** 2).sum()) / sw), floor)
        done = abs(new_pi - pi) < tol and abs(new_mu0 - mu0) < 1e-7 and abs(new_s0 - s0) < 1e-7
        pi, mu0, s0 = new_pi, new_mu0, new_s0
        if done:
            break
    return float(r.sum()), r


def em3(x, good, bad, floor: float, mu_cap: float, iterations: int = 300):
    """:func:`~vtscore.training.thresholds.labels_line.corpus_posteriors` with any Good and Bad components."""
    from scipy.stats import norm

    from vtscore.training.thresholds.labels_line import PREVALENCE_MAX, PREVALENCE_MIN

    mu0 = float(np.median(x))
    s0 = max(1.4826 * float(np.median(np.abs(x - mu0))), floor)
    w1, wb = 0.01, 0.05
    g, b = good.pdf(x), bad.pdf(x)
    r1 = np.zeros_like(x)
    for _ in range(iterations):
        a1 = w1 * g
        ab = wb * b
        a0 = max(1.0 - w1 - wb, 1e-9) * norm.pdf(x, mu0, s0)
        tot = a1 + ab + a0
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = np.nan_to_num(a1 / tot, nan=0.0)
            rb = np.nan_to_num(ab / tot, nan=0.0)
        r0 = np.clip(1.0 - r1 - rb, 0.0, 1.0)
        nw1 = min(max(float(r1.mean()), PREVALENCE_MIN), PREVALENCE_MAX)
        nwb = min(max(float(rb.mean()), PREVALENCE_MIN), 0.9)
        s_r0 = float(r0.sum())
        if s_r0 > 1e-12:
            mu0 = min(float((r0 * x).sum()) / s_r0, mu_cap)
            s0 = max(math.sqrt(max(float((r0 * (x - mu0) ** 2).sum()) / s_r0, 0.0)), floor)
        done = abs(nw1 - w1) < 1e-8 and abs(nwb - wb) < 1e-7
        w1, wb = nw1, nwb
        if done:
            break
    return r1, w1, wb


def counted_bound(good, x_desc) -> float:
    """:func:`~vtscore.training.thresholds.labels_line._counted_share_bound` with any Good component."""
    from vtscore.training.thresholds.labels_line import _BOUND_MIN_S1, PREVALENCE_MAX

    n = x_desc.size
    s1 = good.sf(x_desc)
    ok = s1 >= _BOUND_MIN_S1
    if n == 0 or not ok.any():
        return PREVALENCE_MAX
    r = np.arange(1, n + 1, dtype=np.float64)
    return float(np.min(r[ok] / s1[ok]) / n)


def _kde_cache(pts):
    """A :class:`Kde` whose ``pdf`` and ``sf`` are evaluated once per array (the EM calls them on the same x)."""
    k = Kde(pts)
    memo: dict = {}

    def cached(name):
        fn = getattr(Kde, name)

        def call(x):
            key = (name, id(x), x.size)
            if key not in memo:
                memo[key] = fn(k, x)
            return memo[key]

        return call

    k.pdf, k.sf = cached("pdf"), cached("sf")  # type: ignore[method-assign]
    return k


class FindLine:
    """Find's line on one corpus with no votes in it (the withheld half), as ``LabelsLine.on_corpus`` draws it."""

    def __init__(self, model, scores):
        from vtscore.training.thresholds.class_model import _finite_unit
        from vtscore.training.thresholds.labels_line import corpus_sigma_floor

        self.model = model
        s = np.asarray(scores, dtype=np.float64)
        self.unvoted = _finite_unit(s)
        self.n = self.unvoted.size
        self.floor = corpus_sigma_floor(self.unvoted)
        self.x = _logit(self.unvoted)
        order = np.argsort(-self.unvoted, kind="stable")
        self.order = order
        self.s_desc = self.unvoted[order]
        self.x_desc = self.x[order]

    def draw(
        self,
        beta: float,
        *,
        good=None,
        bad=None,
        floor: float | None = None,
        bound: bool = True,
        total: float | None = None,
        post=None,
        post_kind: str = "3part",
        gate: bool = True,
        mu_pos_cap: float | None = None,
    ) -> dict:
        """The line with the given pieces swapped in; every piece left ``None`` is today's."""
        from vtscore.training.thresholds.labels_line import (
            _BOUND_MARGIN,
            PREVALENCE_MAX,
            PREVALENCE_MIN,
            _bads_are_a_random_sample,
            corpus_cut,
            corpus_expected_fbeta,
            head_count_cut,
            head_count_serves,
        )

        m = self.model
        fl = self.floor if floor is None else floor
        sigma = max(m.sigma if m.sigma_raw is None else m.sigma_raw, fl)
        good = Normal(m.mu_pos, sigma) if good is None else good
        bad = Normal(m.mu_neg, sigma) if bad is None else bad
        if _bads_are_a_random_sample(m, self.x):  # never for a session's Bads; flagged rather than modelled
            return {"bads_shape": True}
        em_pos, r2 = em2(self.x, good, fl, good.mu if mu_pos_cap is None else mu_pos_cap)
        em_share = em_pos / self.n if self.n else 0.0
        bnd = counted_bound(good, self.x_desc)
        share = min(em_share, _BOUND_MARGIN * bnd) if bound else em_share
        w1 = wb = math.nan
        if post is None:
            if post_kind == "2part":
                post = r2
            else:
                post, w1, wb = em3(self.x, good, bad, fl, m.mu_neg)
        tot = share * self.n if total is None else float(total)
        post_desc = np.asarray(post, dtype=np.float64)[self.order]
        if gate and head_count_serves(m):
            thr = head_count_cut(self.s_desc, beta)
        else:
            thr = corpus_cut(post_desc, self.s_desc, tot, beta)
        efb = corpus_expected_fbeta(post_desc, tot, beta)
        prevalence = min(max(share, PREVALENCE_MIN), PREVALENCE_MAX)
        return {
            "thr": float(thr),
            "prev": prevalence,
            "em_share": em_share,
            "bound2": _BOUND_MARGIN * bnd,
            "total": tot,
            "post_sum": float(np.sum(post)),
            "efb": efb,
            "post_desc": post_desc,
            "sigma": sigma,
            "w1": w1,
            "wb": wb,
        }


def monotone(x, post) -> np.ndarray:
    """*post* made non-decreasing in *x* (a running maximum from the lowest score up), as a single cut needs.

    The kernel densities' ratio wiggles where either class is thin and turns down past the top positive, where the
    two tails are extrapolated; the line's own chances are monotone by construction (one spread).
    """
    o = np.argsort(x, kind="stable")
    out = np.empty_like(post)
    out[o] = np.maximum.accumulate(np.asarray(post)[o])
    return out


def em_fixed(f1, f0, iterations: int = 500) -> np.ndarray:
    """Each item's chance of being a positive with both classes' densities fixed: only the share is fitted (EM)."""
    from vtscore.training.thresholds.labels_line import PREVALENCE_MAX, PREVALENCE_MIN

    pi, r = 0.01, np.zeros_like(f1)
    for _ in range(iterations):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nan_to_num(pi * f1 / (pi * f1 + (1.0 - pi) * f0), nan=0.0)
        new = min(max(float(r.mean()), PREVALENCE_MIN), PREVALENCE_MAX)
        done = abs(new - pi) < 1e-10
        pi = new
        if done:
            break
    return r


def isotonic_post(x, y):
    """Each item's chance of being a positive, the truth's isotonic fit on its logit (the true chances, smoothed)."""
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y).predict(x)


def fbeta_at(scores, y, thr: float, beta: float) -> tuple[float, int, int]:
    """``(F-beta, kept, true positives kept)`` of keeping every score at or above *thr*."""
    kept = scores >= thr
    k = int(kept.sum())
    tp = int(y[kept].sum())
    p = int(y.sum())
    b2 = beta * beta
    f = (1 + b2) * tp / (b2 * p + k) if (b2 * p + k) > 0 else 0.0
    return float(f), k, tp


# --- one snapshot ------------------------------------------------------------------------------------------------


def _pooled_sd(parts) -> float:
    xs = [np.asarray(p, dtype=np.float64) for p in parts if len(p)]
    if not xs:
        return math.nan
    n = sum(x.size for x in xs)
    dof = n - len(xs)
    if dof <= 0:
        return math.nan
    return math.sqrt(sum(float(((x - x.mean()) ** 2).sum()) for x in xs) / dof)


def measure(snap: dict, patch_space: bool = False) -> dict:
    """One row: today's Find line against the truth, its Good component's error split, and every swap."""
    from vtscore.detectors.label_quota import query_holds
    from vtscore.training.thresholds.labels_line import HEAD_COUNT_GOODS, fit_labels_line, head_count_serves

    s = Snapshot(snap)
    beta = s.beta
    y = s.y.astype(np.int64)
    n_pos = int(y.sum())
    row: dict = {
        "t": s.t,
        "beta": beta,
        "n_folds": s.k,
        "n_test": int(y.size),
        "n_pos_test": n_pos,
        "prev_true": n_pos / y.size if y.size else math.nan,
        "n_good": int((s.sim_vote == 1).sum()),
        "n_bad": int((s.sim_vote == 0).sum()),
        "query_hold": bool(query_holds(int((s.sim_vote >= 0).sum()), 0, patch_space=patch_space)),
        "mu_true": float(s.final[s.pos].mean()) if s.pos.any() else math.nan,
        "sd_true": float(s.final[s.pos].std(ddof=1)) if s.pos.sum() > 1 else math.nan,
    }
    # The best cut on the same ranking: what any line could reach.
    tp = np.cumsum(y[np.argsort(-s.scores, kind="stable")])
    k = np.arange(1, y.size + 1)
    b2 = beta * beta
    fb = (1 + b2) * tp / (b2 * n_pos + k)
    row["best_fbeta"] = float(fb.max())
    row["best_kept"] = int(k[int(np.argmax(fb))])
    if not s.pos.any():
        row["line_src"] = "no_pos"
        return row

    labels = s.labels_dict()
    line = fit_labels_line(s.orderings() or None, s.sim_scores, s.sim_ids.tolist(), labels)
    if line is None:
        row["line_src"] = "none"
        return row
    model = line.model
    folds_model = s.k > 0 and line.held_out is None and any((f.held_labels == 1).any() for f in s.folds)
    row["line_src"] = (
        "folds" if folds_model and model.n_pos == sum(int((f.held_labels == 1).sum()) for f in s.folds) else "in_sample"
    )
    row["head_count"] = bool(head_count_serves(model))
    row["model_n_pos"], row["model_n_neg"] = model.n_pos, model.n_neg
    row["mu_pos"], row["mu_neg"] = model.mu_pos, model.mu_neg
    row["sigma_raw"] = model.sigma_raw if model.sigma_raw is not None else model.sigma
    on_test = line.on_corpus(s.scores)
    lib_thr = float(on_test.threshold(beta))

    fl = FindLine(model, s.scores)
    row["floor"] = fl.floor
    row["sigma"] = max(row["sigma_raw"], fl.floor)
    row["separation"] = (model.mu_pos - model.mu_neg) / row["sigma"]

    # --- the Good component's error, split exactly ---
    if row["line_src"] == "folds":
        w = np.array([float((f.held_labels == 1).sum()) for f in s.folds])
        w = w / w.sum()
        pos_fold = [f.test[s.pos] for f in s.folds]
        row["gap_mean"] = model.mu_pos - row["mu_true"]
        row["gap_selection"] = float(
            sum(
                wk * (np.mean(_logit(f.held_scores[f.held_labels == 1])) - np.mean(pf))
                for wk, f, pf in zip(w, s.folds, pos_fold)
                if wk > 0
            )
        )
        row["gap_scale"] = float(sum(wk * (np.mean(pf) - row["mu_true"]) for wk, pf in zip(w, pos_fold)))
        # Spread: sigma - sd_true = floor + pooling (with the Bads) + selection + scale.
        goods = [_logit(f.held_scores[f.held_labels == 1]) for f in s.folds]
        sd_goods = _pooled_sd([np.concatenate(goods)])  # the held-out Goods pooled raw, as the class model pools them
        sd_pos_fold = float(sum(wk * np.std(pf, ddof=1) for wk, pf in zip(w, pos_fold))) if n_pos > 1 else math.nan
        row["sd_goods"] = sd_goods
        row["sd_pos_fold"] = sd_pos_fold
        row["sd_bads"] = _pooled_sd([np.concatenate([_logit(f.held_scores[f.held_labels == 0]) for f in s.folds])])
        row["gap_sd"] = row["sigma"] - row["sd_true"]
        row["gap_sd_floor"] = row["sigma"] - row["sigma_raw"]
        row["gap_sd_pool"] = row["sigma_raw"] - sd_goods if np.isfinite(sd_goods) else math.nan
        row["gap_sd_selection"] = sd_goods - sd_pos_fold
        row["gap_sd_scale"] = sd_pos_fold - row["sd_true"]
        sel_mu = model.mu_pos - row["gap_selection"]
    else:
        sel_mu = math.nan

    # --- every swap ---
    iso = isotonic_post(fl.x, y.astype(np.float64)) if y.size == fl.n else None
    sig = row["sigma"]
    # The withheld half's own class densities (binned kernel densities of its logits).
    from vtscore.training.thresholds.labels_line import _bads_density

    neg_post = shapes_post = None
    if y.size == fl.n and n_pos > 1:
        f0 = _bads_density(np.sort(fl.x[y == 0]))(fl.x)
        f1_today = Normal(model.mu_pos, sig).pdf(fl.x)
        f1_true = _bads_density(np.sort(fl.x[y == 1]))(fl.x)
        neg_post = monotone(fl.x, em_fixed(f1_today, f0))
        shapes_post = monotone(fl.x, em_fixed(f1_true, f0))
    swaps: dict[str, dict] = {
        "today": {},
        "mean_true": {"good": Normal(row["mu_true"], sig)},
        "sd_true": {"good": Normal(model.mu_pos, row["sd_true"])} if np.isfinite(row["sd_true"]) else None,
        "good_true": {"good": Normal(row["mu_true"], row["sd_true"])} if np.isfinite(row["sd_true"]) else None,
        "shape_true": {"good": _kde_cache(s.final[s.pos])} if n_pos > 1 else None,
        "sel_off": {"good": Normal(sel_mu, sig)} if np.isfinite(sel_mu) else None,
        "total_true": {"total": float(n_pos)},
        "neg_true": {"post": neg_post} if neg_post is not None else None,
        "shapes_true": {"post": shapes_post} if shapes_post is not None else None,
        "post_true": {"post": iso} if iso is not None else None,
        "both_true": {"post": iso, "total": float(n_pos)} if iso is not None else None,
        "no_bound": {"bound": False},
        "post_2part": {"post_kind": "2part"},
        "no_floor": {"floor": 1e-3},
        "no_gate": {"gate": False},
        "no_gate_good_true": {"gate": False, "good": Normal(row["mu_true"], row["sd_true"])}
        if np.isfinite(row["sd_true"])
        else None,
    }
    for name in VARIANTS:
        kw = swaps.get(name)
        if kw is None:
            continue
        d = fl.draw(beta, **kw)
        if d.get("bads_shape"):
            row["bads_shape"] = True
            break
        f, kept, tpk = fbeta_at(s.scores, y, d["thr"], beta)
        row[f"{name}_fbeta"] = f
        row[f"{name}_kept"] = kept
        row[f"{name}_prev"] = d["prev"]
        row[f"{name}_efb"] = d["efb"]
        if name == "today":
            row["em_share"] = d["em_share"]
            row["bound2"] = d["bound2"]
            row["bound_binds"] = bool(d["bound2"] < d["em_share"])
            row["total"] = d["total"]
            row["post_sum"] = d["post_sum"]
            row["tp_kept"] = tpk
            # Its expected F-beta at its own set, split: the total (P-hat against P) and the chances (their sum
            # over the kept items against the positives kept).
            kk = min(kept, d["post_desc"].size)
            sp = float(d["post_desc"][:kk].sum())
            p_hat = max(d["total"], d["post_sum"])
            row["post_kept"] = sp
            row["w1_3part"], row["wb_3part"] = d["w1"], d["wb"]
            # The chances at the best cut: their sum over the best set against its true positives, and the chance
            # of the items around the best set's edge against the truth's (isotonic) there.
            kb = row["best_kept"]
            row["post_at_best"] = float(d["post_desc"][:kb].sum())
            row["tp_at_best"] = int(tp[kb - 1])
            lo, hi = max(0, kb - 5), min(d["post_desc"].size, kb + 5)
            row["post_edge"] = float(d["post_desc"][lo:hi].mean())
            if iso is not None:
                row["iso_edge"] = float(np.asarray(iso)[fl.order][lo:hi].mean())
            row["efb_own"] = (1 + b2) * sp / (b2 * p_hat + kk) if kk else 0.0
            row["efb_own_trueP"] = (1 + b2) * sp / (b2 * n_pos + kk) if kk else 0.0
            # The replay must be the library's own line.
            row["chk_thr"] = abs(d["thr"] - lib_thr)
            row["chk_prev"] = abs(d["prev"] - on_test.prevalence)
            row["chk_efb"] = abs(d["efb"] - on_test.expected_fbeta(beta))
    row["model_n_pos_lt_gate"] = bool(model.n_pos < HEAD_COUNT_GOODS)
    return row


# --- many cells --------------------------------------------------------------------------------------------------


def _cell_rows(args) -> list[dict]:
    npz, ident = args
    try:
        snaps = load_snapshots(npz)
    except Exception as exc:  # a truncated file: report, do not stop the pool
        return [{**ident, "error": f"{type(exc).__name__}: {exc}"}]
    rows = []
    from vtscore.eval.live_threshold_rules import head_count_recall, labels_chances

    for prefix, path in PATHS.items():
        mine = {k: v for k, v in snaps.items() if k.startswith(prefix + "/")}
        listed = sorted(int(k.rsplit("/t", 1)[1]) for k in mine)
        by_step: dict[int, dict] = {}
        for snap in mine.values():
            by_step.setdefault(int(snap["t"]), snap)
        for t, snap in sorted(by_step.items()):
            try:
                # The line this study measured: the one before #5040 changed the chances and the bound, and
                # before #5039 changed the head's count above beta 1.
                with labels_chances("labels", 2.0), head_count_recall("off", "off"):
                    measure_row = measure(snap, path == "region")
                rows.append({**ident, "path": path, "click": click_of(t, listed), **measure_row})
            except Exception as exc:
                rows.append({**ident, "path": path, "t": t, "error": f"{type(exc).__name__}: {exc}"})
    return rows


def cmd_measure(a) -> None:
    jobs = []
    for res in a.results:
        cells = os.path.join(res, "cells")
        run = os.path.basename(os.path.dirname(res.rstrip("/")))
        for npz in sorted(glob.glob(os.path.join(cells, "task_*__testscores.npz"))):
            jobs.append((npz, {"run": run, **_ident(cells, npz)}))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} cells", flush=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        for i, out in enumerate(ex.map(_cell_rows, jobs, chunksize=1)):
            rows.extend(out)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "snapshots.csv.gz"), index=False)
    bad = df.get("error")
    print(f"wrote {len(df)} snapshot rows; errors: {0 if bad is None else int(bad.notna().sum())}")
    for col in ("chk_thr", "chk_prev", "chk_efb"):
        if col in df:
            print(f"{col}: max {df[col].max():.2e}, n>1e-9 {int((df[col] > 1e-9).sum())}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--results", action="append", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--procs", type=int, default=8)
    m.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--docs")
    r.add_argument("--figs")
    e = sub.add_parser("examples")
    e.add_argument("--results", action="append", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--path", default="binary")
    e.add_argument("--beta", type=float, default=1.0)
    e.add_argument("--click", type=int, default=40)
    e.add_argument("--docs")
    e.add_argument("--figs")
    a = ap.parse_args(argv)
    from vtscore.eval.live_threshold_rules import head_count_recall, labels_chances

    with (
        labels_chances("labels", 2.0),
        head_count_recall("off", "off"),
    ):  # the line this study measured: before #5040, #5039
        {"measure": cmd_measure, "report": cmd_report, "examples": cmd_examples}[a.cmd](a)
    return 0


# --- report ------------------------------------------------------------------------------------------------------

#: The swaps the report reads, in the order of its tables, and their labels.
LABEL = {
    "today": "today",
    "sel_off": "Good mean: selection out",
    "mean_true": "Good mean: the truth",
    "sd_true": "Good spread: the truth",
    "good_true": "Good mean and spread: the truth",
    "shape_true": "Good component: the positives' own shape",
    "neg_true": "negatives: their own shape",
    "shapes_true": "both classes: their own shapes",
    "post_true": "the true chances",
    "both_true": "the true chances and total",
    "total_true": "the true total",
    "no_bound": "no counted bound",
    "post_2part": "2-part chances",
    "no_floor": "no spread floor",
    "no_gate": "no head-count line",
    "no_gate_good_true": "no head-count line, Good component true",
}
COLOR = {
    "today": "#333333",
    "sel_off": "#aec7e8",
    "mean_true": "#1f77b4",
    "sd_true": "#ff9896",
    "good_true": "#2ca02c",
    "shape_true": "#98df8a",
    "neg_true": "#ff7f0e",
    "shapes_true": "#8c564b",
    "post_true": "#9467bd",
    "both_true": "#c5b0d5",
    "total_true": "#17becf",
    "no_bound": "#d62728",
    "post_2part": "#7f7f7f",
    "no_floor": "#bcbd22",
    "no_gate": "#e377c2",
    "no_gate_good_true": "#f7b6d2",
}
PRESETS = (0.25, 1.0, 4.0)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """The rows the report reads: a line the head drew (folds or the in-sample stand-in), off the typed query's hold."""
    df = df[df["error"].isna()] if "error" in df else df
    df = df[df["line_src"].isin(["folds", "in_sample"]) & ~df["query_hold"].astype(bool)].copy()
    lk = lambda c: np.log2((df[c] + 1) / (df["best_kept"] + 1))  # noqa: E731
    for v in VARIANTS:
        if f"{v}_kept" in df:
            df[f"{v}_lk"] = lk(f"{v}_kept")
            df[f"{v}_alk"] = df[f"{v}_lk"].abs()
            df[f"{v}_dF"] = df[f"{v}_fbeta"] - df["today_fbeta"]
            df[f"{v}_lp"] = np.log2(df[f"{v}_prev"] / df["prev_true"])
    df["efb_err"] = df["today_efb"] - df["today_fbeta"]
    df["gapF"] = df["best_fbeta"] - df["today_fbeta"]
    df["lem"] = np.log2(df["em_share"] / df["prev_true"])
    df["postsum_ratio"] = df["post_sum"] / df["n_pos_test"]
    df["efb_argmax"] = df["today_efb"] - df["efb_own"]
    df["efb_total"] = df["efb_own"] - df["efb_own_trueP"]
    df["efb_chances"] = df["efb_own_trueP"] - df["today_fbeta"]
    df["head_count"] = df["head_count"].astype(bool)
    df["bound_binds"] = df["bound_binds"].astype(bool)
    return df


def _by_class(frame: pd.DataFrame, col: str) -> tuple[float, float]:
    """Mean and SE over classes (a class's bands and seeds averaged first: they are not independent)."""
    per = frame.groupby("class")[col].mean().dropna()
    if per.empty:
        return math.nan, math.nan
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per))) if len(per) > 1 else math.nan


def _f(m: float, se: float, digits: int) -> str:
    if not np.isfinite(m):
        return "-"
    return f"{m:+.{digits}f} (± {se:.{digits}f})" if np.isfinite(se) else f"{m:+.{digits}f}"


def _ratio(frame: pd.DataFrame, col: str) -> float:
    """The median of a log2 ratio, read back as a ratio."""
    v = frame[col].replace([np.inf, -np.inf], np.nan).dropna()
    return float(2 ** v.median()) if len(v) else math.nan


def report_tables(df: pd.DataFrame, path: str) -> dict[str, pd.DataFrame]:
    """Every table the report reads, for one path, one row per preset and click."""
    out: dict[str, list[dict]] = {
        k: [] for k in ("symptom", "good", "ladder", "sizes", "prevalence", "chances", "head_count")
    }
    for (beta, c), g_all in df.groupby(["beta", "click"]):
        base = {"beta": beta, "click": int(c), "snapshots": len(g_all)}
        hc = g_all[g_all["head_count"]]
        g = g_all[~g_all["head_count"]]  # the class model's own cut: what every swap but the gate's can move
        out["symptom"].append(
            {
                **base,
                "head-count line": f"{100 * g_all['head_count'].mean():.0f}%",
                "set / best set (median)": f"{_ratio(g_all, 'today_lk'):.2f}",
                "class model's cut": f"{_ratio(g, 'today_lk'):.2f}",
                "head-count line ": f"{_ratio(hc, 'today_lk'):.2f}",
                "prevalence / truth (median)": f"{_ratio(g_all, 'today_lp'):.2f}",
                "expected F - F": _f(*_by_class(g_all, "efb_err"), 2),
                "best cut's F - F": _f(*_by_class(g_all, "gapF"), 2),
                "F": f"{_by_class(g_all, 'today_fbeta')[0]:.2f}",
                "best cut's F": f"{_by_class(g_all, 'best_fbeta')[0]:.2f}",
            }
        )
        out["head_count"].append(
            {
                **base,
                "head-count snapshots": len(hc),
                "set / best set (median)": f"{_ratio(hc, 'today_lk'):.2f}",
                "best cut's F - F": _f(*_by_class(hc, "gapF"), 2) if len(hc) else "-",
                "class model's cut instead: gain": _f(*_by_class(hc, "no_gate_dF"), 3) if len(hc) else "-",
                "its set / best": f"{_ratio(hc, 'no_gate_lk'):.2f}" if len(hc) else "-",
                "with the Good component true: gain": _f(*_by_class(hc, "no_gate_good_true_dF"), 3)
                if len(hc) and "no_gate_good_true_dF" in hc
                else "-",
                "its set / best ": f"{_ratio(hc, 'no_gate_good_true_lk'):.2f}"
                if len(hc) and "no_gate_good_true_lk" in hc
                else "-",
                "held-out Goods (median)": f"{hc['model_n_pos'].median():.0f}" if len(hc) else "-",
            }
        )
        base = {**base, "class-model snapshots": len(g)}
        f = g[g["line_src"] == "folds"]
        out["good"].append(
            {
                **base,
                "folds snapshots": len(f),
                "mean: Good - truth": _f(*_by_class(f, "gap_mean"), 2),
                "selection": _f(*_by_class(f, "gap_selection"), 2),
                "scale": _f(*_by_class(f, "gap_scale"), 2),
                "spread: Good - truth": _f(*_by_class(f, "gap_sd"), 2),
                "pooling with the Bads": _f(*_by_class(f, "gap_sd_pool"), 2),
                "selection ": _f(*_by_class(f, "gap_sd_selection"), 2),
                "scale ": _f(*_by_class(f, "gap_sd_scale"), 2),
                "floor": _f(*_by_class(f, "gap_sd_floor"), 2),
                "Good spread": f"{_by_class(f, 'sigma')[0]:.2f}",
                "positives' spread": f"{_by_class(f, 'sd_true')[0]:.2f}",
            }
        )
        lad = dict(base)
        sizes_row = dict(base)
        for v in VARIANTS:
            if f"{v}_dF" not in g or v == "today":
                continue
            lad[LABEL[v]] = _f(*_by_class(g, f"{v}_dF"), 3)
        for v in VARIANTS:
            if f"{v}_lk" in g:
                sizes_row[LABEL[v]] = f"{_ratio(g, f'{v}_lk'):.2f} / {g[f'{v}_alk'].mean():.2f}"
        out["ladder"].append(lad)
        out["sizes"].append(sizes_row)
        # The prevalence the class model estimates whichever line serves (the issue's reading: every snapshot).
        out["prevalence"].append(
            {
                **{k: v for k, v in base.items() if k != "class-model snapshots"},
                "today / truth (median)": f"{_ratio(g_all, 'today_lp'):.2f}",
                "EM alone / truth": f"{_ratio(g_all, 'lem'):.2f}",
                "bound binds": f"{100 * g_all['bound_binds'].mean():.0f}%",
                "no bound / truth": f"{_ratio(g_all, 'no_bound_lp'):.2f}",
                "Good mean and spread true / truth": f"{_ratio(g_all, 'good_true_lp'):.2f}",
                "positives' own shape / truth": f"{_ratio(g_all, 'shape_true_lp'):.2f}",
            }
        )
        out["chances"].append(
            {
                **base,
                "chances' sum / positives (median)": f"{g['postsum_ratio'].median():.2f}",
                "chance at the best set's edge: line": f"{g['post_edge'].median():.2f}",
                "the truth": f"{g['iso_edge'].median():.2f}",
                "expected F - F": _f(*_by_class(g, "efb_err"), 2),
                "of which the total": _f(*_by_class(g, "efb_total"), 2),
                "of which the chances": _f(*_by_class(g, "efb_chances"), 2),
            }
        )
    band = []
    for (beta, c, b), g in df[~df["head_count"]].groupby(["beta", "click", "band"]):
        f = g[g["line_src"] == "folds"]
        band.append(
            {
                "beta": beta,
                "click": int(c),
                "band": b,
                "class-model snapshots": len(g),
                "set / best set (median)": f"{_ratio(g, 'today_lk'):.2f}",
                "Good mean and spread true": f"{_ratio(g, 'good_true_lk'):.2f}",
                "negatives' own shape": f"{_ratio(g, 'neg_true_lk'):.2f}" if "neg_true_lk" in g else "-",
                "selection (mean)": _f(*_by_class(f, "gap_selection"), 2),
                "selection (spread)": _f(*_by_class(f, "gap_sd_selection"), 2),
                "true chances' gain": _f(*_by_class(g, "post_true_dF"), 3),
            }
        )
    out["by_band"] = band
    return {k: pd.DataFrame(v) for k, v in out.items()}


def report_figures(df: pd.DataFrame, path: str, figdir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    betas = [b for b in PRESETS if b in set(df["beta"])]
    # A click most sessions read (a sparse one, e.g. Binary's 25 under the typed query's hold, is noise).
    counts = df.groupby(["beta", "click"]).size()
    keep = counts[counts >= 0.1 * counts.groupby(level=0).transform("max")].index
    df = df.set_index(["beta", "click"]).loc[keep].reset_index()

    def curve(frame, col, clicks):
        m, se = zip(*[_by_class(frame[frame["click"] == c], col) for c in clicks])
        return np.array(m), np.array(se)

    def ratio_axis(ax):
        ax.set_yscale("log", base=2)
        ticks = [0.06, 0.12, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.7, 2, 4]
        lo, hi = ax.get_ylim()
        ticks = [t for t in ticks if lo <= t <= hi] or [1]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()

    def med(frame, col, clicks):
        return np.array([_ratio(frame[frame["click"] == c], col) for c in clicks])

    # 1. The symptom, at each preset.
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
    for b, color in zip(betas, ("#1f77b4", "#333333", "#d62728")):
        g = df[df["beta"] == b]
        clicks = sorted(g["click"].unique())
        x = np.array(clicks, dtype=float)
        lab = f"beta {b:g}"
        axes[0].plot(x, med(g, "today_lk", clicks), "-o", ms=3, color=color, label=lab)
        axes[0].plot(x, med(g[~g["head_count"]], "today_lk", clicks), ":", color=color, lw=1)
        axes[4].plot(x, [100 * g[g["click"] == c]["head_count"].mean() for c in clicks], "-o", ms=3, color=color)
        axes[1].plot(x, med(g, "today_lp", clicks), "-o", ms=3, color=color, label=lab)
        for ax, col in ((axes[2], "efb_err"), (axes[3], "gapF")):
            m, se = curve(g, col, clicks)
            ax.plot(x, m, "-o", ms=3, color=color, label=lab)
            ax.fill_between(x, m - se, m + se, color=color, alpha=0.15, lw=0)
    for ax, t in zip(
        axes,
        (
            "set size / the best set's (median; dotted: class model's cut)",
            "prevalence estimate / truth (median)",
            "its expected F-beta minus its F-beta",
            "best cut's F-beta minus its F-beta",
            "% of snapshots on the head's count line",
        ),
    ):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("vote")
        ax.grid(alpha=0.3)
    for ax in axes[:2]:
        ax.axhline(1, color="k", lw=0.8)
        ratio_axis(ax)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{path}: today's Find line on the withheld half", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "symptom.png", dpi=110)
    plt.close(fig)

    # 2. The Good component's error, split, at each preset.
    fig, axes = plt.subplots(2, len(betas), figsize=(5 * len(betas), 6.4), squeeze=False)
    for j, b in enumerate(betas):
        g = df[(df["beta"] == b) & (df["line_src"] == "folds")]
        clicks = sorted(g["click"].unique())
        x = np.array(clicks, dtype=float)
        for i, parts in enumerate(
            (
                (
                    ("gap_mean", "Good mean - truth", "#333333"),
                    ("gap_selection", "selection", "#d62728"),
                    ("gap_scale", "scale", "#1f77b4"),
                ),
                (
                    ("gap_sd", "Good spread - truth", "#333333"),
                    ("gap_sd_selection", "selection", "#d62728"),
                    ("gap_sd_scale", "scale", "#1f77b4"),
                    ("gap_sd_pool", "pooling with the Bads", "#2ca02c"),
                    ("gap_sd_floor", "floor", "#bcbd22"),
                ),
            )
        ):
            ax = axes[i][j]
            for col, lab, color in parts:
                m, se = curve(g, col, clicks)
                ax.plot(x, m, "-o", ms=3, color=color, label=lab, lw=2 if col in ("gap_mean", "gap_sd") else 1.2)
                ax.fill_between(x, m - se, m + se, color=color, alpha=0.15, lw=0)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"beta {b:g}: the Good component's {'mean' if i == 0 else 'spread'} (logit)", fontsize=9)
            ax.set_xlabel("vote")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "good_component.png", dpi=110)
    plt.close(fig)

    # 3. The swaps: F-beta gain and set size, at each preset.
    show = ("mean_true", "good_true", "shape_true", "neg_true", "shapes_true", "post_true", "total_true", "no_bound")
    fig, axes = plt.subplots(2, len(betas), figsize=(5 * len(betas), 7), squeeze=False)
    for j, b in enumerate(betas):
        g = df[(df["beta"] == b) & ~df["head_count"]]
        clicks = sorted(g["click"].unique())
        x = np.array(clicks, dtype=float)
        ax = axes[0][j]
        for v in show:
            if f"{v}_dF" not in g:
                continue
            m, se = curve(g, f"{v}_dF", clicks)
            ax.plot(x, m, "-o", ms=3, color=COLOR[v], label=LABEL[v])
            ax.fill_between(x, m - se, m + se, color=COLOR[v], alpha=0.12, lw=0)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"beta {b:g}: F-beta gain over today's line, same heads (class model's cut)", fontsize=9)
        ax = axes[1][j]
        for v in ("today",) + show:
            if f"{v}_lk" in g:
                ax.plot(x, med(g, f"{v}_lk", clicks), "-o", ms=3, color=COLOR[v], label=LABEL[v])
        ax.axhline(1, color="k", lw=0.8)
        ratio_axis(ax)
        ax.set_title(f"beta {b:g}: set size / the best set's (median, class model's cut)", fontsize=9)
        for ax in axes[:, j]:
            ax.set_xlabel("vote")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    axes[1][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "swaps.png", dpi=110)
    plt.close(fig)

    # 4. Per session: today's set size against the best set's, and the true chances', thin lines per session.
    fig, axes = plt.subplots(1, len(betas), figsize=(5 * len(betas), 3.6), squeeze=False)
    for j, b in enumerate(betas):
        ax = axes[0][j]
        g = df[df["beta"] == b]
        for _, s in g.groupby(["cell", "seed"]):
            s = s.sort_values("click")
            ax.plot(s["click"], s["today_lk"], color="#333333", alpha=0.05, lw=0.7)
        clicks = sorted(g["click"].unique())
        for v in ("today", "good_true", "post_true"):
            ax.plot(
                clicks, np.log2(med(g, f"{v}_lk", clicks)), "-o", ms=3, color=COLOR[v], label=LABEL[v] + " (median)"
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-4, 4)
        ax.set_title(f"beta {b:g}: log2(set size / best set's), one line per session", fontsize=9)
        ax.set_xlabel("vote")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "per_session.png", dpi=110)
    plt.close(fig)


def pick_examples(df: pd.DataFrame, beta: float, click: int) -> list[tuple[str, pd.Series]]:
    """Sessions picked by rule at one preset and click: the median shortfall, the worst, the true chances' case."""
    g = df[(df["beta"] == beta) & (df["click"] == click) & ~df["head_count"] & (df["best_kept"] >= 10)]
    short = g[g["today_lk"] < 0]
    picks = []
    if len(short):
        i = (short["today_lk"] - short["today_lk"].median()).abs().idxmin()
        picks.append(("median shortfall", short.loc[i]))
        picks.append(("largest shortfall", short.loc[short["today_lk"].idxmin()]))
    rest = g.drop([p[1].name for p in picks], errors="ignore")
    if len(rest):
        gap = rest["post_true_dF"] - rest["good_true_dF"]
        picks.append(("true Good component leaves the most", rest.loc[gap.idxmax()]))
    rest = rest.drop([p[1].name for p in picks], errors="ignore")
    big = rest[rest["band"] == "large"]
    if len(big):
        i = (big["today_lk"]).abs().idxmin()
        picks.append(("large band, set about right", big.loc[i]))
    return picks


def _find_snapshot(results: list[str], run: str, idx: int, t: int, path: str) -> dict:
    prefix = {v: k for k, v in PATHS.items()}[path]
    for res in results:
        if os.path.basename(os.path.dirname(res.rstrip("/"))) != run:
            continue
        npz = os.path.join(res, "cells", f"task_{idx:04d}__testscores.npz")
        for key, snap in load_snapshots(npz).items():
            if key.startswith(prefix + "/") and int(snap["t"]) == t:
                return snap
    raise KeyError(f"{run} task {idx} t {t}")


def cmd_examples(a) -> None:
    """Literal examples: each picked session's withheld half, the line's Good component and chances against the truth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    from vtscore.training.thresholds.labels_line import _bads_density, fit_labels_line

    df = _prep(pd.read_csv(os.path.join(a.out, "snapshots.csv.gz"), low_memory=False))
    df = df[df["path"] == a.path]
    picks = pick_examples(df, a.beta, a.click)
    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 3.4 * len(picks)), squeeze=False)
    rows = []
    for i, (why, r) in enumerate(picks):
        snap = _find_snapshot(a.results, r["run"], int(r["idx"]), int(r["t"]), a.path)
        s = Snapshot(snap)
        line = fit_labels_line(s.orderings() or None, s.sim_scores, s.sim_ids.tolist(), s.labels_dict())
        fl = FindLine(line.model, s.scores)
        d = fl.draw(s.beta)
        y = s.y
        x = fl.x
        sig = d["sigma"]
        thr_x, best_x = float(_logit([d["thr"]])[0]), float(np.sort(x)[::-1][int(r["best_kept"]) - 1])
        lo = min(thr_x, best_x, float(np.quantile(x[y == 1], 0.1))) - 0.5
        hi = float(x.max()) + 0.2
        ax = axes[i][0]
        bins = np.linspace(lo, hi, 50)
        ax.hist(x[(y == 0) & (x >= lo)], bins=bins, color="#999999", alpha=0.6, label="withheld negatives")
        ax.hist(x[(y == 1) & (x >= lo)], bins=bins, color="#2ca02c", alpha=0.8, label="withheld positives")
        ax.set_yscale("log")
        ax.set_ylabel("images")
        grid = np.linspace(lo, hi, 400)
        w = bins[1] - bins[0]
        ax.plot(
            grid,
            Normal(line.model.mu_pos, sig).pdf(grid) * y.sum() * w,
            color="#d62728",
            label="the line's Good component (x positives)",
        )
        ax.plot(
            grid,
            Normal(r["mu_true"], r["sd_true"]).pdf(grid) * y.sum() * w,
            color="#2ca02c",
            ls="--",
            label="the positives' own normal",
        )
        ax.set_ylim(0.5, None)
        ax.axvline(thr_x, color="k", lw=1.5, label=f"today's cut ({int(r['today_kept'])} kept)")
        ax.axvline(best_x, color="#9467bd", lw=1.5, ls="--", label=f"best cut ({int(r['best_kept'])} kept)")
        ax.set_title(f"{r['cell']} seed {int(r['seed'])}, vote {int(r['t'])}: {why}", fontsize=9)
        ax.set_xlabel("final model's logit")
        ax.legend(fontsize=6)
        ax = axes[i][1]
        o = np.argsort(x)
        sel = x[o] >= lo
        iso = isotonic_post(x, y.astype(np.float64))
        f0 = _bads_density(np.sort(x[y == 0]))(x)
        f1 = _bads_density(np.sort(x[y == 1]))(x)
        shapes = monotone(x, em_fixed(f1, f0))
        post = np.empty_like(x)
        post[fl.order] = d["post_desc"]
        ax.plot(x[o][sel], post[o][sel], color="#d62728", label="the line's chances (3-part fit)")
        ax.plot(x[o][sel], shapes[o][sel], color="#8c564b", label="both classes' own shapes")
        ax.plot(x[o][sel], iso[o][sel], color="#9467bd", label="the truth (isotonic)")
        ax.axvline(thr_x, color="k", lw=1.5)
        ax.axvline(best_x, color="#9467bd", lw=1.5, ls="--")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("final model's logit")
        ax.set_ylabel("chance of a positive")
        ax.legend(fontsize=6)
        rows.append(
            {
                "session": r["cell"],
                "seed": int(r["seed"]),
                "vote": int(r["t"]),
                "picked as": why,
                "kept / best": f"{int(r['today_kept'])} / {int(r['best_kept'])}",
                "F today / best": f"{r['today_fbeta']:.2f} / {r['best_fbeta']:.2f}",
                "Good mean - truth": f"{r['gap_mean']:+.2f}",
                "selection": f"{r['gap_selection']:+.2f}",
                "Good spread / truth": f"{sig:.2f} / {r['sd_true']:.2f}",
                "prevalence / truth": f"{r['today_prev'] / r['prev_true']:.2f}",
                "chance at edge: line / truth": f"{r['post_edge']:.2f} / {r['iso_edge']:.2f}",
                "F: good_true / shapes_true / post_true": f"{r['good_true_fbeta']:.2f} / {r.get('shapes_true_fbeta', math.nan):.2f} / {r['post_true_fbeta']:.2f}",
            }
        )
    fig.tight_layout()
    figdir = Path(a.figs or os.path.join(a.out, "figures")) / a.path
    figdir.mkdir(parents=True, exist_ok=True)
    tag = {0.25: "025", 1.0: "1", 4.0: "4"}[float(a.beta)]
    fig.savefig(figdir / f"examples_b{tag}_vote{a.click}.png", dpi=110)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    if a.docs:
        t.to_csv(Path(a.docs) / a.path / f"examples_b{tag}_vote{a.click}.csv", index=False)


def head_count_examples(df: pd.DataFrame, click: int) -> pd.DataFrame:
    """At each preset and one click, the head-count sessions at the median, the 10th and the 90th percentile of the set ratio."""
    rows = []
    for beta, g in df[(df["click"] == click) & df["head_count"]].groupby("beta"):
        g = g.sort_values("today_lk").reset_index(drop=True)
        for q in (0.1, 0.5, 0.9):
            r = g.iloc[min(len(g) - 1, int(q * len(g)))]
            rows.append(
                {
                    "beta": beta,
                    "percentile": int(q * 100),
                    "session": r["cell"],
                    "seed": int(r["seed"]),
                    "vote": int(r["t"]),
                    "held-out Goods": int(r["model_n_pos"]),
                    "kept / best": f"{int(r['today_kept'])} / {int(r['best_kept'])}",
                    "withheld positives": int(r["n_pos_test"]),
                    "F / best": f"{r['today_fbeta']:.2f} / {r['best_fbeta']:.2f}",
                    "class model's cut instead: kept, F": f"{int(r['no_gate_kept'])}, {r['no_gate_fbeta']:.2f}",
                }
            )
    return pd.DataFrame(rows)


def cmd_report(a) -> None:
    from pathlib import Path

    df = _prep(pd.read_csv(os.path.join(a.out, "snapshots.csv.gz"), low_memory=False))
    for path, g in df.groupby("path"):
        tabs = Path(a.docs or os.path.join(a.out, "tables")) / path
        tabs.mkdir(parents=True, exist_ok=True)
        hc = head_count_examples(g, 40 if path == "binary" else 25)
        hc.to_csv(tabs / "head_count_examples.csv", index=False)
        print(f"\n== {path}: head-count examples")
        print(hc.to_string(index=False))
        for name, t in report_tables(g, path).items():
            t.to_csv(tabs / f"{name}.csv", index=False)
            print(f"\n== {path}: {name}")
            print(t.to_string(index=False))
        report_figures(g, path, Path(a.figs or os.path.join(a.out, "figures")) / path)


if __name__ == "__main__":
    sys.exit(main())
