#!/usr/bin/env python3
"""#5040: price deployable fixes for the labels line's chances on #4937's snapshots, same heads, offline.

#4937 found the labels line's chances (the 3-part fit) underrate the positives at the line: the true chances would
gain +0.04 to +0.13 F-beta on the class model's cut. It split the error into a Good side (the held-out Goods are the
session's easy positives: the Good component sits too high and too narrow) and a negatives' side (the 3-part fit's
Bad component and bulk put too much negative weight at the line). This script redraws Find's line on the withheld half
of every snapshot with one deployable candidate swapped in, and reads it as #4937 read its truth swaps: F-beta at the
session's preset, set size against the best set's, the prevalence estimate and the expected F-beta's error.

Every candidate reads only what the app has: the labels' held-out fold scores (Goods and Bads, each vote held out
once), their surfacing provenance (a spot check's picks are tagged ``flow: check``), and the scores of the corpus
being decided. None reads the withheld half's truth.

* **Good side.** ``good_own_sd``: the Good component's spread is the held-out Goods' own, not pooled with the Bads.
  ``good_ssl``: the Good component's mean and spread are fitted on the corpus by EM, with the held-out Goods counted
  as labelled points of that component (semi-supervised: each M-step pools the Goods with the corpus's soft
  positives), in the 2-part total and the 3-part chances alike. ``good_check``: the Good component from the held-out
  Goods a spot check picked (uniform within rank bands, so less selected), where a session has at least
  :data:`CHECK_GOODS_MIN` of them.
* **Negatives' side.** ``bad_own_sd``: the Bad component's spread is the held-out Bads' own. ``bad_kde``: the Bad
  component is a kernel density of the held-out Bads (as #4490 models a random sample of Bads), its weight fitted by
  the 3-part EM, which takes up their selection: the Bads stand for the share of the corpus's negatives they look
  like, not for their count. ``bad_ssl``: the Bad component semi-supervised as ``good_ssl`` does the Good one.
* **Both / the total.** ``own_sds`` (both spreads unpooled), ``both_ssl``, and the counted bound removed
  (``no_bound``) or the total read from the chances alone (``total_post``); then combinations.

Truth swaps from #4937 ride along as references on the same rows: ``post_true`` (the true chances: the ceiling),
``neg_true`` and ``good_true``.

**Find's other corpora (#4452).** Every candidate is also drawn, with the same class model, on three corpora cut
from the withheld half: its negatives alone (``empty``, ~10,900 images, no positive), 200 of its negatives
(``empty200``, #4452's own scenario) and 1,000 of its images at their own prevalence (``small1k``). On the empty
corpora any image kept is wrong: a candidate must keep close to nothing there.

  chances_5040.py measure --results DIR [--results DIR ...] --out DIR [--procs N] [--limit N]
  chances_5040.py report --out DIR [--docs DIR] [--figs DIR]
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
from early_line_4937 import (  # noqa: E402
    FindLine,
    Normal,
    em_fixed,
    fbeta_at,
    isotonic_post,
    monotone,
)
from fold_scale_4936 import PATHS, Snapshot, _ident, _logit, click_of, load_snapshots  # noqa: E402

#: The fewest held-out Goods from a spot check that ``good_check`` reads; with fewer the Good component is today's.
CHECK_GOODS_MIN = 3
#: Each candidate as its pieces (see the module docstring); ``today`` is the shipped line.
CANDIDATES: dict[str, dict] = {
    "today": {},
    "good_own_sd": {"good": "own_sd"},
    "good_ssl": {"good": "ssl"},
    "good_check": {"good": "check"},
    "bad_own_sd": {"bad": "own_sd"},
    "bad_kde": {"bad": "kde"},
    "bad_ssl": {"bad": "ssl"},
    "own_sds": {"good": "own_sd", "bad": "own_sd"},
    "both_ssl": {"good": "ssl", "bad": "ssl"},
    "no_bound": {"bound": False},
    "total_post": {"total": "post"},
    "good_ssl_bad_kde": {"good": "ssl", "bad": "kde"},
    "good_check_bad_kde": {"good": "check", "bad": "kde"},
    "good_ssl_no_bound": {"good": "ssl", "bound": False},
    "bad_kde_no_bound": {"bad": "kde", "bound": False},
    "both_ssl_no_bound": {"good": "ssl", "bad": "ssl", "bound": False},
}
#: The second pass, on the first pass's findings: the Bad component fitted on the corpus split into its two moves, the
#: bound loosened rather than removed, and the three candidates that gained combined.
CANDIDATES_2: dict[str, dict] = {
    "today": {},
    "bad_ssl": {"bad": "ssl"},
    "bad_ssl_mean": {"bad": "ssl_mean"},
    "bad_ssl_sd": {"bad": "ssl_sd"},
    "bound_x4": {"margin": 4.0},
    "bad_ssl_no_bound": {"bad": "ssl", "bound": False},
    "bad_ssl_bound_x4": {"bad": "ssl", "margin": 4.0},
    "bad_ssl_good_own_sd": {"bad": "ssl", "good": "own_sd"},
    "bad_ssl_good_own_sd_no_bound": {"bad": "ssl", "good": "own_sd", "bound": False},
    "good_own_sd_no_bound": {"good": "own_sd", "bound": False},
}
#: The truth swaps read beside them on the withheld half (not on the other corpora).
REFERENCES = ("post_true", "neg_true", "good_true")
#: The corpora every candidate is drawn on: the withheld half, and three cut from it.
CORPORA = ("full", "empty", "empty200", "small1k")
#: The phases of the pick log whose picks a spot check drew (the weak check, the check before Done, the closing one).
CHECK_PHASES = frozenset({"prompt", "check"})


# --- the pieces --------------------------------------------------------------------------------------------------


def _npdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    from vtscore.training.thresholds.labels_line import _norm_pdf

    return _norm_pdf(x, mu, sd)


class Density:
    """A fixed density (a kernel density of the held-out Bads): ``pdf`` only, as the 3-part fit reads it."""

    def __init__(self, pts: np.ndarray):
        from vtscore.training.thresholds.labels_line import _bads_density

        self.pts = np.sort(np.asarray(pts, dtype=np.float64))
        self._pdf = _bads_density(self.pts)
        self.mu = float(self.pts.mean())

    def pdf(self, x):
        return self._pdf(np.asarray(x, dtype=np.float64))


def _own_sd(pts: np.ndarray, fallback: float, floor: float) -> float:
    """A class's own spread (its held-out logits' SD), floored on the corpus as today's pooled spread is."""
    if pts.size < 2:
        return fallback
    return max(float(np.std(pts, ddof=1)), floor)


def em2_ssl(x, good: Normal, pts, floor: float, iterations: int = 500, tol: float = 1e-9):
    """The 2-part fit (``fit_corpus``) with the Good component semi-supervised on *pts* (the held-out Goods).

    ``(EM positives, the fitted Good component)``. Each M-step pools the labelled Goods with the corpus's soft
    positives for the Good component's mean and spread (floored on the corpus); the bulk is fitted as today, its mean
    held below the Good mean.
    """
    from vtscore.training.thresholds.labels_line import PREVALENCE_MAX, PREVALENCE_MIN

    pts = np.asarray(pts, dtype=np.float64)
    m, sp = float(pts.sum()), pts.size
    mu0 = float(np.median(x))
    s0 = max(1.4826 * float(np.median(np.abs(x - mu0))), floor)
    g_mu, g_sd = good.mu, good.sigma
    pi = 0.01
    r = np.zeros_like(x)
    for _ in range(iterations):
        f1 = pi * _npdf(x, g_mu, g_sd)
        f0 = (1.0 - pi) * _npdf(x, mu0, s0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nan_to_num(f1 / (f1 + f0), nan=0.0)
        new_pi = min(max(float(r.mean()), PREVALENCE_MIN), PREVALENCE_MAX)
        w = 1.0 - r
        sw = float(w.sum())
        if sw <= 0:
            break
        sr = float(r.sum())
        n1 = sp + sr
        new_gmu = (m + float((r * x).sum())) / n1
        new_gsd = max(
            math.sqrt((float(((pts - new_gmu) ** 2).sum()) + float((r * (x - new_gmu) ** 2).sum())) / n1), floor
        )
        new_mu0 = min(float((w * x).sum()) / sw, new_gmu)
        new_s0 = max(math.sqrt(float((w * (x - new_mu0) ** 2).sum()) / sw), floor)
        done = (
            abs(new_pi - pi) < tol
            and abs(new_mu0 - mu0) < 1e-7
            and abs(new_s0 - s0) < 1e-7
            and abs(new_gmu - g_mu) < 1e-7
            and abs(new_gsd - g_sd) < 1e-7
        )
        pi, mu0, s0, g_mu, g_sd = new_pi, new_mu0, new_s0, new_gmu, new_gsd
        if done:
            break
    return float(r.sum()), Normal(g_mu, g_sd)


def em3_general(
    x,
    good,
    bad,
    floor: float,
    mu_cap: float,
    *,
    good_pts=None,
    bad_pts=None,
    bad_fit: str = "both",
    iterations: int = 300,
):
    """The 3-part fit (``corpus_posteriors``) with any Good and Bad components, either semi-supervised.

    A component given its labelled points (*good_pts*, *bad_pts*) must be a :class:`Normal`; each M-step pools its
    points with the corpus's soft members for its mean and spread. The bulk's mean is held below the Bad component's.
    ``(chances, w1, wb)``; with neither semi-supervised and both components fixed it is ``corpus_posteriors``.
    *bad_fit* says which of the Bad component's moments the fit moves: ``both``, ``mean`` or ``sd`` (the other kept).
    """
    from vtscore.training.thresholds.labels_line import PREVALENCE_MAX, PREVALENCE_MIN

    mu0 = float(np.median(x))
    s0 = max(1.4826 * float(np.median(np.abs(x - mu0))), floor)
    w1, wb = 0.01, 0.05
    g_mu, g_sd = (good.mu, good.sigma) if good_pts is not None else (math.nan, math.nan)
    b_mu, b_sd = (bad.mu, bad.sigma) if bad_pts is not None else (math.nan, math.nan)
    g = None if good_pts is not None else good.pdf(x)
    b = None if bad_pts is not None else bad.pdf(x)
    cap = mu_cap
    r1 = np.zeros_like(x)
    for _ in range(iterations):
        a1 = w1 * (_npdf(x, g_mu, g_sd) if g is None else g)
        ab = wb * (_npdf(x, b_mu, b_sd) if b is None else b)
        a0 = max(1.0 - w1 - wb, 1e-9) * _npdf(x, mu0, s0)
        tot = a1 + ab + a0
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = np.nan_to_num(a1 / tot, nan=0.0)
            rb = np.nan_to_num(ab / tot, nan=0.0)
        r0 = np.clip(1.0 - r1 - rb, 0.0, 1.0)
        nw1 = min(max(float(r1.mean()), PREVALENCE_MIN), PREVALENCE_MAX)
        nwb = min(max(float(rb.mean()), PREVALENCE_MIN), 0.9)
        moved = 0.0
        if good_pts is not None:
            n1 = good_pts.size + float(r1.sum())
            ng = (float(good_pts.sum()) + float((r1 * x).sum())) / n1
            ns = max(math.sqrt((float(((good_pts - ng) ** 2).sum()) + float((r1 * (x - ng) ** 2).sum())) / n1), floor)
            moved = max(moved, abs(ng - g_mu), abs(ns - g_sd))
            g_mu, g_sd = ng, ns
        if bad_pts is not None:
            nb_ = bad_pts.size + float(rb.sum())
            nm = (float(bad_pts.sum()) + float((rb * x).sum())) / nb_ if bad_fit != "sd" else bad.mu
            ns = max(math.sqrt((float(((bad_pts - nm) ** 2).sum()) + float((rb * (x - nm) ** 2).sum())) / nb_), floor)
            ns = ns if bad_fit != "mean" else bad.sigma
            moved = max(moved, abs(nm - b_mu), abs(ns - b_sd))
            b_mu, b_sd = nm, ns
            cap = b_mu
        s_r0 = float(r0.sum())
        if s_r0 > 1e-12:
            mu0 = min(float((r0 * x).sum()) / s_r0, cap)
            s0 = max(math.sqrt(max(float((r0 * (x - mu0) ** 2).sum()) / s_r0, 0.0)), floor)
        done = abs(nw1 - w1) < 1e-8 and abs(nwb - wb) < 1e-7 and moved < 1e-7
        w1, wb = nw1, nwb
        if done:
            break
    return r1, w1, wb


def draw(fl: FindLine, beta: float, pieces: dict, pts: dict, gate: bool = False) -> dict:
    """Find's line on *fl*'s corpus with a candidate's *pieces*; ``today``'s pieces go through ``FindLine.draw``.

    *pts* holds the labels' held-out logits (``good``, ``bad``, ``check_good``). The counted bound reads the Good
    component the 2-part fit ends with. The class model's cut unless *gate*.
    """
    from vtscore.training.thresholds.labels_line import (
        _BOUND_MARGIN,
        PREVALENCE_MAX,
        PREVALENCE_MIN,
        corpus_cut,
        corpus_expected_fbeta,
        head_count_cut,
        head_count_serves,
    )

    from early_line_4937 import counted_bound, em2

    m = fl.model
    fl_floor = fl.floor
    sigma = max(m.sigma if m.sigma_raw is None else m.sigma_raw, fl_floor)
    gk, bk = pieces.get("good"), pieces.get("bad")
    good = Normal(m.mu_pos, sigma)
    bad = Normal(m.mu_neg, sigma)
    good_ssl = bad_ssl = None
    if gk == "own_sd":
        good = Normal(m.mu_pos, _own_sd(pts["good"], sigma, fl_floor))
    elif gk == "check" and pts["check_good"].size >= CHECK_GOODS_MIN:
        cg = pts["check_good"]
        good = Normal(float(cg.mean()), _own_sd(cg, sigma, fl_floor))
    elif gk == "ssl":
        good_ssl = pts["good"]
    if bk == "own_sd":
        bad = Normal(m.mu_neg, _own_sd(pts["bad"], sigma, fl_floor))
    elif bk == "kde" and pts["bad"].size >= 3:
        bad = Density(pts["bad"])
    elif bk in ("ssl", "ssl_mean", "ssl_sd"):
        bad_ssl = pts["bad"]
    margin = float(pieces.get("margin", _BOUND_MARGIN))
    if good_ssl is None and bad_ssl is None and isinstance(bad, Normal) and margin == _BOUND_MARGIN:
        kw: dict = {"good": good, "bad": bad, "bound": pieces.get("bound", True), "gate": gate}
        if pieces.get("total") == "post":
            kw["total"] = 0.0  # the cut counts the larger of the total and the chances' own sum
        return fl.draw(beta, **kw)
    # The 2-part total.
    if good_ssl is not None:
        em_pos, good2 = em2_ssl(fl.x, good, good_ssl, fl_floor)
    else:
        em_pos, _ = em2(fl.x, good, fl_floor, good.mu)
        good2 = good
    em_share = em_pos / fl.n if fl.n else 0.0
    bnd = counted_bound(good2, fl.x_desc)
    share = min(em_share, margin * bnd) if pieces.get("bound", True) else em_share
    bad_fit = {"ssl_mean": "mean", "ssl_sd": "sd"}.get(str(bk), "both")
    post, w1, wb = em3_general(fl.x, good, bad, fl_floor, m.mu_neg, good_pts=good_ssl, bad_pts=bad_ssl, bad_fit=bad_fit)
    tot = 0.0 if pieces.get("total") == "post" else share * fl.n
    post_desc = np.asarray(post, dtype=np.float64)[fl.order]
    if gate and head_count_serves(m):
        thr = head_count_cut(fl.s_desc, beta)
    else:
        thr = corpus_cut(post_desc, fl.s_desc, tot, beta)
    return {
        "thr": float(thr),
        "prev": min(max(share, PREVALENCE_MIN), PREVALENCE_MAX),
        "em_share": em_share,
        "total": tot,
        "post_sum": float(np.sum(post)),
        "efb": corpus_expected_fbeta(post_desc, tot, beta),
        "post_desc": post_desc,
        "w1": w1,
        "wb": wb,
    }


# --- one snapshot ------------------------------------------------------------------------------------------------


def _model_points(s: Snapshot, labels: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """The logits the class model was fitted on, as ``fit_labels_line`` picks them: ``(Goods, Bads, Good ids, src)``."""
    from vtscore.training.thresholds.labels_line import _pooled_class_model

    orderings = s.orderings()
    folds = _pooled_class_model(orderings) if orderings else None
    if folds is not None and folds.mu_pos > folds.mu_neg:
        sc = np.concatenate([f.held_scores for f in s.folds])
        yy = np.concatenate([f.held_labels for f in s.folds]).astype(np.float64)
        ids = np.concatenate([f.held_ids for f in s.folds])
        src = "folds"
    else:  # the in-sample stand-in: the head's own scores of the labelled items
        voted = np.array([int(i) in labels for i in s.sim_ids])
        sc, ids = s.sim_scores[voted], s.sim_ids[voted]
        yy = np.array([1.0 if labels[int(i)] else 0.0 for i in ids])
        src = "in_sample"
    ok = np.isfinite(sc) & (sc >= 0) & (sc <= 1)
    sc, yy, ids = sc[ok], yy[ok], ids[ok]
    return _logit(sc[yy >= 0.5]), _logit(sc[yy < 0.5]), ids[yy >= 0.5], src


def _read(fl: FindLine, scores, y, beta: float, d: dict) -> dict:
    f, kept, _ = fbeta_at(scores, y, d["thr"], beta)
    return {"fbeta": f, "kept": kept, "prev": d["prev"], "efb": d["efb"], "post_sum": d["post_sum"]}


def _best(scores, y, beta: float) -> tuple[float, int]:
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0, 0
    tp = np.cumsum(y[np.argsort(-scores, kind="stable")])
    k = np.arange(1, y.size + 1)
    b2 = beta * beta
    fb = (1 + b2) * tp / (b2 * n_pos + k)
    return float(fb.max()), int(k[int(np.argmax(fb))])


def measure(
    snap: dict, check_ids: frozenset[int], seed_key: int, patch_space: bool = False, cands: dict | None = None
) -> dict | None:
    """One row: every candidate's line on the withheld half and on the three corpora cut from it."""
    from vtscore.detectors.label_quota import query_holds
    from vtscore.training.thresholds.labels_line import _bads_density, fit_labels_line, head_count_serves

    s = Snapshot(snap)
    beta = s.beta
    y = s.y.astype(np.int64)
    n_pos = int(y.sum())
    if n_pos == 0 or bool(query_holds(int((s.sim_vote >= 0).sum()), 0, patch_space=patch_space)):
        return None
    labels = s.labels_dict()
    line = fit_labels_line(s.orderings() or None, s.sim_scores, s.sim_ids.tolist(), labels)
    if line is None:
        return None
    model = line.model
    good_pts, bad_pts, good_ids, src = _model_points(s, labels)
    check_good = good_pts[np.isin(good_ids, np.fromiter(check_ids, dtype=np.int64))]
    pts = {"good": good_pts, "bad": bad_pts, "check_good": check_good}
    best_f, best_k = _best(s.scores, y, beta)
    row: dict = {
        "t": s.t,
        "beta": beta,
        "line_src": src,
        "n_test": int(y.size),
        "n_pos_test": n_pos,
        "prev_true": n_pos / y.size,
        "model_n_pos": model.n_pos,
        "model_n_neg": model.n_neg,
        "pts_ok": bool(good_pts.size == model.n_pos and bad_pts.size == model.n_neg),
        "n_check_good": int(check_good.size),
        "head_count": bool(head_count_serves(model)),
        "best_fbeta": best_f,
        "best_kept": best_k,
        "mu_pos": model.mu_pos,
        "sd_goods": float(np.std(good_pts, ddof=1)) if good_pts.size > 1 else math.nan,
        "sd_bads": float(np.std(bad_pts, ddof=1)) if bad_pts.size > 1 else math.nan,
        "mu_true": float(s.final[s.pos].mean()),
        "sd_true": float(s.final[s.pos].std(ddof=1)) if n_pos > 1 else math.nan,
    }
    rng = np.random.default_rng(seed_key)
    neg_idx = np.flatnonzero(y == 0)
    sub = {
        "full": np.arange(y.size),
        "empty": neg_idx,
        "empty200": rng.choice(neg_idx, size=min(200, neg_idx.size), replace=False),
        "small1k": np.sort(rng.choice(y.size, size=min(1000, y.size), replace=False)),
    }
    for corpus in CORPORA:
        idx = sub[corpus]
        sc, yy = s.scores[idx], y[idx]
        fl = FindLine(model, sc)
        if corpus != "full":
            bf, bk = _best(sc, yy, beta)
            row[f"{corpus}_best_fbeta"], row[f"{corpus}_best_kept"] = bf, bk
            row[f"{corpus}_n_pos"] = int(yy.sum())
        for name, pieces in (CANDIDATES if cands is None else cands).items():
            d = draw(fl, beta, pieces, pts)
            if d.get("bads_shape"):
                row["bads_shape"] = True
                return row
            r = _read(fl, sc, yy, beta, d)
            tag = name if corpus == "full" else f"{corpus}:{name}"
            for k, v in r.items():
                row[f"{tag}_{k}"] = v
            if corpus == "full":
                kb = best_k
                lo, hi = max(0, kb - 5), min(d["post_desc"].size, kb + 5)
                row[f"{name}_post_edge"] = float(d["post_desc"][lo:hi].mean())
                if name == "today":
                    lib = line.on_corpus(s.scores)
                    from vtscore.training.thresholds.labels_line import corpus_cut

                    lib_cut = corpus_cut(
                        lib.unvoted_posteriors, lib.unvoted_scores, lib.unvoted_share * lib.unvoted_scores.size, beta
                    )
                    row["chk_thr"] = abs(d["thr"] - lib_cut)
                    row["chk_prev"] = abs(d["prev"] - lib.prevalence)
            if name == "today":
                # What Find returns on this corpus today, the head's count line included where it serves.
                g = fl.draw(beta, gate=True)
                f, kept, _ = fbeta_at(sc, yy, g["thr"], beta)
                row[f"{'served' if corpus == 'full' else corpus + ':served'}_fbeta"] = f
                row[f"{'served' if corpus == 'full' else corpus + ':served'}_kept"] = kept
    # The truth swaps (#4937), on the withheld half.
    fl = FindLine(model, s.scores)
    iso = isotonic_post(fl.x, y.astype(np.float64))
    f0 = _bads_density(np.sort(fl.x[y == 0]))(fl.x)
    sig = max(model.sigma_raw if model.sigma_raw is not None else model.sigma, fl.floor)
    refs = {"post_true": {"post": iso}, "good_true": None, "neg_true": None}
    if n_pos > 1:
        refs["neg_true"] = {"post": monotone(fl.x, em_fixed(Normal(model.mu_pos, sig).pdf(fl.x), f0))}
        refs["good_true"] = {"good": Normal(row["mu_true"], row["sd_true"])}
    for name, kw in refs.items():
        if kw is None:
            continue
        d = fl.draw(beta, **kw, gate=False)
        r = _read(fl, s.scores, y, beta, d)
        for k, v in r.items():
            row[f"{name}_{k}"] = v
        kb = best_k
        lo, hi = max(0, kb - 5), min(d["post_desc"].size, kb + 5)
        row[f"{name}_post_edge"] = float(d["post_desc"][lo:hi].mean())
    return row


# --- many cells --------------------------------------------------------------------------------------------------


def _check_ids(cells: str, idx: int) -> dict[str, frozenset[int]]:
    """Per embedder, the ids a spot check picked in this cell's session (its pick log's check phases)."""
    p = os.path.join(cells, f"task_{idx:04d}__picks.csv.gz")
    if not os.path.exists(p):
        p = p[:-3]
    if not os.path.exists(p):
        return {}
    picks = pd.read_csv(p, usecols=["phase", "picked_id", "embedder"])
    picks = picks[picks["phase"].isin(CHECK_PHASES)]
    return {str(e): frozenset(int(i) for i in g["picked_id"]) for e, g in picks.groupby("embedder")}


#: The pick log's embedder for each path.
EMBEDDER = {"binary": "siglip", "region": "siglip+dinov3_patch"}


def _cell_rows(args) -> list[dict]:
    npz, ident, cells, which = args
    cands = {1: CANDIDATES, 2: CANDIDATES_2}[which]
    try:
        snaps = load_snapshots(npz)
        checks = _check_ids(cells, int(ident["idx"]))
    except Exception as exc:  # a truncated file: report, do not stop the pool
        return [{**ident, "error": f"{type(exc).__name__}: {exc}"}]
    rows = []
    for prefix, path in PATHS.items():
        mine = {k: v for k, v in snaps.items() if k.startswith(prefix + "/")}
        listed = sorted(int(k.rsplit("/t", 1)[1]) for k in mine)
        by_step: dict[int, dict] = {}
        for snap in mine.values():
            by_step.setdefault(int(snap["t"]), snap)
        for t, snap in sorted(by_step.items()):
            key = int(ident["idx"]) * 10_000 + t
            try:
                row = measure(snap, checks.get(EMBEDDER[path], frozenset()), key, path == "region", cands)
            except Exception as exc:
                rows.append({**ident, "path": path, "t": t, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if row is not None:
                rows.append({**ident, "path": path, "click": click_of(t, listed), **row})
    return rows


def cmd_measure(a) -> None:
    jobs = []
    for res in a.results:
        cells = os.path.join(res, "cells")
        run = os.path.basename(os.path.dirname(res.rstrip("/")))
        for npz in sorted(glob.glob(os.path.join(cells, "task_*__testscores.npz"))):
            jobs.append((npz, {"run": run, **_ident(cells, npz)}, cells, a.pass_))
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
    if bad is not None and bad.notna().any():
        print(df.loc[bad.notna(), "error"].value_counts().head(10).to_string())
    for col in ("chk_thr", "chk_prev"):
        if col in df:
            print(f"{col}: max {df[col].max():.2e}, n>1e-9 {int((df[col] > 1e-9).sum())}")
    if "pts_ok" in df:
        print(f"pts_ok: {df['pts_ok'].mean():.4f}")


# --- verify: the library's arm against the pricer ------------------------------------------------------------------

#: The arms the library carries (``labels_chances``), each with the pricer's candidate it must reproduce.
LIBRARY_ARMS = {"bad_ssl": ("corpus", None), "bad_ssl_bound_x4": ("corpus", 4.0)}


def _verify_rows(args) -> list[dict]:
    """Per snapshot: the library's line under each arm on the withheld half and the empty corpus, its kept count."""
    from vtscore.detectors.label_quota import query_holds
    from vtscore.eval.live_threshold_rules import labels_chances
    from vtscore.training.thresholds.labels_line import corpus_cut, fit_labels_line

    npz, ident = args
    out = []
    for key, snap in load_snapshots(npz).items():
        path = next((p for pre, p in PATHS.items() if key.startswith(pre + "/")), None)
        if path is None:
            continue
        s = Snapshot(snap)
        y = s.y.astype(np.int64)
        if y.sum() == 0 or query_holds(int((s.sim_vote >= 0).sum()), 0, patch_space=path == "region"):
            continue
        row = {**ident, "path": path, "t": s.t}
        for arm, (bad, margin) in LIBRARY_ARMS.items():
            with labels_chances(bad, margin):
                line = fit_labels_line(s.orderings() or None, s.sim_scores, s.sim_ids.tolist(), s.labels_dict())
                if line is None:
                    break
                for corpus, sc in (("full", s.scores), ("empty", s.scores[y == 0])):
                    lib = line.on_corpus(sc)
                    total = lib.unvoted_share * lib.unvoted_scores.size
                    thr = corpus_cut(lib.unvoted_posteriors, lib.unvoted_scores, total, s.beta)
                    tag = arm if corpus == "full" else f"empty:{arm}"
                    row[f"lib_{tag}_kept"] = int((sc >= thr).sum())
        out.append(row)
    return out


def cmd_verify(a) -> None:
    """Every snapshot's library line under each arm against the pricer's candidate (pass 2's measure)."""
    jobs = []
    for res in a.results:
        cells = os.path.join(res, "cells")
        run = os.path.basename(os.path.dirname(res.rstrip("/")))
        for npz in sorted(glob.glob(os.path.join(cells, "task_*__testscores.npz"))):
            jobs.append((npz, {"run": run, **_ident(cells, npz)}))
    if a.limit:
        jobs = jobs[: a.limit]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        for out in ex.map(_verify_rows, jobs, chunksize=1):
            rows.extend(out)
    # A step filed under several listed clicks is one snapshot (the pricer reads each step once).
    lib = pd.DataFrame(rows).drop_duplicates(["run", "idx", "path", "t"])
    priced = pd.concat([pd.read_csv(f, low_memory=False) for f in a.priced], ignore_index=True)
    priced = priced[priced["error"].isna()] if "error" in priced else priced
    m = lib.merge(priced, on=["run", "idx", "path", "t"], how="inner")
    print(f"{len(lib)} library snapshots, {len(priced)} priced, {len(m)} matched")
    for arm in LIBRARY_ARMS:
        for tag in (arm, f"empty:{arm}"):
            both = m[[f"lib_{tag}_kept", f"{tag}_kept"]].dropna()
            d = (both[f"lib_{tag}_kept"] - both[f"{tag}_kept"]).abs()
            print(f"{tag}: {int((d == 0).sum())} of {len(d)} kept sets the same size; max |diff| {d.max():.0f}")
    if a.save:
        m[[c for c in m if c.startswith("lib_") or c in ("run", "idx", "path", "t")]].to_csv(a.save, index=False)


# --- report ------------------------------------------------------------------------------------------------------

#: What each candidate is called in the report's tables.
LABEL = {
    "today": "today",
    "good_own_sd": "Good: its own spread",
    "good_ssl": "Good: fitted on the corpus (semi-supervised)",
    "good_check": "Good: the spot check's Goods",
    "bad_own_sd": "Bad: its own spread",
    "bad_kde": "Bad: the Bads' density",
    "bad_ssl": "Bad: fitted on the corpus (semi-supervised)",
    "own_sds": "both: their own spreads",
    "both_ssl": "both: fitted on the corpus",
    "no_bound": "no counted bound",
    "total_post": "total: the chances' sum",
    "good_ssl_bad_kde": "Good fitted + the Bads' density",
    "good_check_bad_kde": "check's Goods + the Bads' density",
    "good_ssl_no_bound": "Good fitted, no bound",
    "bad_kde_no_bound": "the Bads' density, no bound",
    "both_ssl_no_bound": "both fitted, no bound",
    "bad_ssl_mean": "Bad fitted: its mean only",
    "bad_ssl_sd": "Bad fitted: its spread only",
    "bound_x4": "the bound at 4x",
    "bad_ssl_no_bound": "Bad fitted, no bound",
    "bad_ssl_bound_x4": "Bad fitted, the bound at 4x",
    "bad_ssl_good_own_sd": "Bad fitted + Good's own spread",
    "bad_ssl_good_own_sd_no_bound": "Bad fitted + Good's own spread, no bound",
    "good_own_sd_no_bound": "Good's own spread, no bound",
    "post_true": "the true chances (ceiling)",
    "neg_true": "the negatives' own shape (truth)",
    "good_true": "the Good component true (truth)",
}
PRESETS = (0.25, 1.0, 4.0)
#: The clicks the report's tables read (Binary from 35: below 30 labels the typed query's set is shown).
CLICKS = {"binary": (35, 40, 50, 75, 100, 150), "region": (25, 35, 50, 75, 100, 150)}


def _by_class(frame: pd.DataFrame, col: str) -> tuple[float, float]:
    """Mean and SE over classes (a class's bands and seeds averaged first: they are not independent)."""
    per = frame.groupby("class")[col].mean().dropna()
    if per.empty:
        return math.nan, math.nan
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per))) if len(per) > 1 else math.nan


def _f(m: float, se: float, digits: int = 3) -> str:
    if not np.isfinite(m):
        return "-"
    return f"{m:+.{digits}f} (± {se:.{digits}f})" if np.isfinite(se) else f"{m:+.{digits}f}"


def _names(df: pd.DataFrame) -> list[str]:
    return [c for c in dict.fromkeys([*CANDIDATES, *CANDIDATES_2, *REFERENCES]) if f"{c}_fbeta" in df]


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["error"].isna()] if "error" in df else df
    df = df[df.get("bads_shape", pd.Series(False, index=df.index)).fillna(False) == False].copy()  # noqa: E712
    df["head_count"] = df["head_count"].astype(bool)
    new: dict[str, pd.Series | np.ndarray] = {}
    for c in _names(df):
        new[f"{c}_dF"] = df[f"{c}_fbeta"] - df["today_fbeta"]
        new[f"{c}_lk"] = np.log2((df[f"{c}_kept"] + 1) / (df["best_kept"] + 1))
        new[f"{c}_lp"] = np.log2(df[f"{c}_prev"] / df["prev_true"])
        new[f"{c}_efb_err"] = df[f"{c}_efb"] - df[f"{c}_fbeta"]
        # What Find returns with the candidate: its cut where the class model serves, the head's count line elsewhere.
        new[f"{c}_served_dF"] = np.where(df["head_count"], 0.0, new[f"{c}_dF"])
        if f"small1k:{c}_fbeta" in df:
            new[f"small1k:{c}_dF"] = df[f"small1k:{c}_fbeta"] - df["small1k:today_fbeta"]
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def tables(df: pd.DataFrame, path: str) -> dict[str, pd.DataFrame]:
    names = _names(df)
    out: dict[str, list[dict]] = {
        k: [] for k in ("gain", "served", "sizes", "prevalence", "efb", "empty", "small", "edge")
    }
    clicks = CLICKS.get(path, sorted(df["click"].unique()))
    for (beta, c), g_all in df[df["click"].isin(clicks)].groupby(["beta", "click"]):
        g = g_all[~g_all["head_count"]]
        base = {"beta": beta, "click": int(c), "class-model snapshots": len(g)}
        gain, served, sizes, prev, efb, edge = (dict(base) for _ in range(6))
        served["all snapshots"] = len(g_all)
        prev["truth: 1.00; today"] = f"{2 ** g['today_lp'].median():.2f}"
        for n in names:
            gain[LABEL.get(n, n)] = _f(*_by_class(g, f"{n}_dF"))
            if n in CANDIDATES:
                served[LABEL.get(n, n)] = _f(*_by_class(g_all, f"{n}_served_dF"))
            sizes[LABEL.get(n, n)] = f"{2 ** g[f'{n}_lk'].median():.2f} / {g[f'{n}_lk'].abs().mean():.2f}"
            prev[LABEL.get(n, n)] = f"{2 ** g[f'{n}_lp'].median():.2f}"
            efb[LABEL.get(n, n)] = _f(*_by_class(g, f"{n}_efb_err"), 2)
            if f"{n}_post_edge" in g:
                edge[LABEL.get(n, n)] = f"{g[f'{n}_post_edge'].median():.2f}"
        for k, v in (
            ("gain", gain),
            ("served", served),
            ("sizes", sizes),
            ("prevalence", prev),
            ("efb", efb),
            ("edge", edge),
        ):
            out[k].append(v)
        for corpus in ("empty", "empty200"):
            row = {**base, "corpus": corpus}
            for n in [n for n in names if f"{corpus}:{n}_kept" in g]:
                k = g[f"{corpus}:{n}_kept"]
                row[LABEL.get(n, n)] = (
                    f"{k.mean():.1f} / {k.median():.0f} / {100 * (k == 0).mean():.0f}% / {k.quantile(0.9):.0f}"
                )
            out["empty"].append(row)
        sm = g[g["small1k_n_pos"] > 0]
        row = {**base, "small-corpus snapshots": len(sm)}
        for n in [n for n in names if f"small1k:{n}_dF" in sm]:
            row[LABEL.get(n, n)] = _f(*_by_class(sm, f"small1k:{n}_dF"))
        out["small"].append(row)
    return {k: pd.DataFrame(v) for k, v in out.items()}


#: The candidates the figures draw, with the ceiling, in the order of their legends.
SHOW = ("good_own_sd", "good_check", "good_ssl", "bad_kde", "bad_ssl", "bad_ssl_bound_x4", "no_bound", "bound_x4")
COLOR = {
    "good_own_sd": "#9ec5f0",
    "good_check": "#5aa0e6",
    "good_ssl": "#1b4f8a",
    "bad_kde": "#c9a227",
    "bad_ssl": "#2a78d6",
    "bad_ssl_bound_x4": "#eb6834",
    "no_bound": "#d62728",
    "bound_x4": "#f4a6a6",
    "post_true": "#9467bd",
}


def figures(df: pd.DataFrame, path: str, figdir) -> None:
    """Per preset over votes: each candidate's gain on the class model's cut, its set against the best, and what it
    keeps on a corpus with no positives."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    betas = [b for b in PRESETS if b in set(df["beta"])]
    clicks = [c for c in CLICKS.get(path, ()) if c in set(df["click"])]
    show = [n for n in SHOW if f"{n}_fbeta" in df] + ["post_true"]
    fig, axes = plt.subplots(3, len(betas), figsize=(5.4 * len(betas), 11), squeeze=False)
    for j, b in enumerate(betas):
        g = df[(df["beta"] == b) & ~df["head_count"]]
        x = np.array(clicks, dtype=float)
        for n in show:
            ls = ":" if n == "post_true" else "-"
            m, se = map(np.array, zip(*[_by_class(g[g["click"] == c], f"{n}_dF") for c in clicks]))
            axes[0][j].plot(x, m, ls, marker="o", ms=3, color=COLOR[n], label=LABEL.get(n, n))
            axes[0][j].fill_between(x, m - se, m + se, color=COLOR[n], alpha=0.12, lw=0)
            axes[1][j].plot(x, [2 ** g[g["click"] == c][f"{n}_lk"].median() for c in clicks], ls, marker="o", ms=3,
                            color=COLOR[n])  # fmt: skip
            if n != "post_true" and f"empty:{n}_kept" in g:
                axes[2][j].plot(x, [g[g["click"] == c][f"empty:{n}_kept"].mean() for c in clicks], ls, marker="o",
                                ms=3, color=COLOR[n])  # fmt: skip
        axes[1][j].plot(x, [2 ** g[g["click"] == c]["today_lk"].median() for c in clicks], "-", marker="o", ms=3,
                        color="#333333", label="today")  # fmt: skip
        axes[2][j].plot(x, [g[g["click"] == c]["empty:today_kept"].mean() for c in clicks], "-", marker="o", ms=3,
                        color="#333333", label="today")  # fmt: skip
        axes[0][j].axhline(0, color="k", lw=0.8)
        axes[1][j].axhline(1, color="k", lw=0.8)
        axes[1][j].set_yscale("log", base=2)
        axes[2][j].set_yscale("log")
        titles = (
            f"beta {b:g}: F-beta gain over today's line, same heads",
            f"beta {b:g}: set size / the best set's (median)",
            f"beta {b:g}: images kept on the corpus with no positives (mean)",
        )
        for i, t in enumerate(titles):
            axes[i][j].set_title(t, fontsize=9)
            axes[i][j].set_xlabel("vote")
            axes[i][j].grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    axes[1][0].legend(fontsize=7)
    fig.suptitle(f"{path}: the class model's cut, each candidate swapped in (dotted: the true chances, the ceiling)",
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(figdir / "candidates.png", dpi=110)
    plt.close(fig)


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
    """Literal examples at one preset and click: the arm's median, best and worst sessions against today's line.

    Each row: the withheld half's top (positives green, negatives grey) with today's cut, the arm's and the best;
    beside it each image's chance of a positive under today's 3-part fit, the arm's and the truth's (isotonic).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    from vtscore.eval.live_threshold_rules import labels_chances
    from vtscore.training.thresholds.labels_line import fit_labels_line

    arm = a.arm
    bad, margin = LIBRARY_ARMS[arm]
    df = prep(load_passes(a.out))
    g = df[(df["path"] == a.path) & (df["beta"] == a.beta) & (df["click"] == a.click) & ~df["head_count"]]
    g = g[g["best_kept"] >= 10].sort_values(f"{arm}_dF")
    moved = g[g[f"{arm}_dF"].abs() > 1e-9]
    picks = [("hurt most", moved.iloc[0]), ("median of those it moved", moved.iloc[len(moved) // 2]),
             ("helped most", moved.iloc[-1])]  # fmt: skip
    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 3.4 * len(picks)), squeeze=False)
    rows = []
    for i, (why, r) in enumerate(picks):
        s = Snapshot(_find_snapshot(a.results, r["run"], int(r["idx"]), int(r["t"]), a.path))
        y = s.y.astype(np.int64)
        lines = {}
        for name, (b_, m_) in (("today", (None, None)), (arm, (bad, margin))):
            with labels_chances(b_, m_):
                line = fit_labels_line(s.orderings() or None, s.sim_scores, s.sim_ids.tolist(), s.labels_dict())
                assert line is not None
                lines[name] = line.on_corpus(s.scores)
        x = _logit(s.scores)
        best_x = float(np.sort(x)[::-1][int(r["best_kept"]) - 1])
        cuts = {}
        from vtscore.training.thresholds.labels_line import corpus_cut

        for name, ln in lines.items():
            thr = corpus_cut(
                ln.unvoted_posteriors, ln.unvoted_scores, ln.unvoted_share * ln.unvoted_scores.size, s.beta
            )
            cuts[name] = float(_logit([thr])[0])
        lo = min(*cuts.values(), best_x, float(np.quantile(x[y == 1], 0.1))) - 0.5
        hi = float(x.max()) + 0.2
        ax = axes[i][0]
        bins = np.linspace(lo, hi, 50)
        ax.hist(x[(y == 0) & (x >= lo)], bins=bins, color="#999999", alpha=0.6, label="withheld negatives")
        ax.hist(x[(y == 1) & (x >= lo)], bins=bins, color="#2ca02c", alpha=0.8, label="withheld positives")
        ax.set_yscale("log")
        ax.set_ylim(0.5, None)
        ax.axvline(cuts["today"], color="k", lw=1.5, label=f"today's cut ({int(r['today_kept'])} kept)")
        ax.axvline(cuts[arm], color=COLOR[arm], lw=1.5, label=f"{LABEL[arm]} ({int(r[f'{arm}_kept'])} kept)")
        ax.axvline(best_x, color="#9467bd", lw=1.5, ls="--", label=f"best cut ({int(r['best_kept'])} kept)")
        ax.set_title(f"{r['cell']} seed {int(r['seed'])}, vote {int(r['t'])}, beta {a.beta:g}: {why}", fontsize=9)
        ax.set_xlabel("final model's logit")
        ax.set_ylabel("images")
        ax.legend(fontsize=6)
        ax = axes[i][1]
        o = np.argsort(x)
        sel = x[o] >= lo
        for name, ln, color in (
            ("today's 3-part fit", lines["today"], "#333333"),
            (LABEL[arm], lines[arm], COLOR[arm]),
        ):
            post = np.interp(x, _logit(ln.unvoted_scores[::-1]), ln.unvoted_posteriors[::-1])
            ax.plot(x[o][sel], post[o][sel], color=color, label=name)
        ax.plot(
            x[o][sel], isotonic_post(x, y.astype(np.float64))[o][sel], color="#9467bd", label="the truth (isotonic)"
        )
        for name, c in (("today", "k"), (arm, COLOR[arm])):
            ax.axvline(cuts[name], color=c, lw=1)
        ax.axvline(best_x, color="#9467bd", lw=1, ls="--")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("final model's logit")
        ax.set_ylabel("chance of a positive")
        ax.legend(fontsize=6)
        rows.append({
            "picked as": why, "session": r["cell"], "seed": int(r["seed"]), "vote": int(r["t"]),
            "kept today / arm / best": f"{int(r['today_kept'])} / {int(r[f'{arm}_kept'])} / {int(r['best_kept'])}",
            "F today / arm / best": f"{r['today_fbeta']:.2f} / {r[f'{arm}_fbeta']:.2f} / {r['best_fbeta']:.2f}",
            "chances' sum today / arm / positives": f"{r['today_post_sum']:.0f} / {r[f'{arm}_post_sum']:.0f} / {int(r['n_pos_test'])}",
            "chance at the best set's edge today / arm": f"{r['today_post_edge']:.2f} / {r[f'{arm}_post_edge']:.2f}",
        })  # fmt: skip
    fig.tight_layout()
    tag = {0.25: "025", 1.0: "1", 4.0: "4"}[float(a.beta)]
    figdir = Path(a.figs) / a.path
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"examples_{arm}_b{tag}_vote{a.click}.png", dpi=110)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    if a.docs:
        (Path(a.docs) / a.path).mkdir(parents=True, exist_ok=True)
        t.to_csv(Path(a.docs) / a.path / f"examples_{arm}_b{tag}_vote{a.click}.csv", index=False)


def load_passes(files: list[str]) -> pd.DataFrame:
    """Every measure's rows, one per snapshot: the second pass's candidates joined onto the first pass's rows."""
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    first = [f for f in frames if "good_ssl_fbeta" in f]
    second = [f for f in frames if "bad_ssl_bound_x4_fbeta" in f]
    rest = [f for f in frames if "good_ssl_fbeta" not in f and "bad_ssl_bound_x4_fbeta" not in f]
    keys = ["run", "idx", "path", "t"]
    one = pd.concat(first + rest, ignore_index=True) if first or rest else None
    if not second:
        return one if one is not None else pd.DataFrame()
    two = pd.concat(second, ignore_index=True)
    if one is None:
        return two
    extra = [c for c in two if c not in one or c in keys]
    return one.merge(two[extra], on=keys, how="left")


def cmd_report(a) -> None:
    from pathlib import Path

    df = prep(load_passes(a.out))
    for path, g in df.groupby("path"):
        tabs = Path(a.docs or os.path.join(os.path.dirname(a.out[0]), "tables")) / path
        tabs.mkdir(parents=True, exist_ok=True)
        for name, t in tables(g, path).items():
            t.to_csv(tabs / f"{name}.csv", index=False)
            print(f"\n== {path}: {name}")
            print(t.to_string(index=False))
        figures(g, path, Path(a.figs or os.path.join(os.path.dirname(a.out[0]), "figures")) / path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--results", action="append", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--procs", type=int, default=8)
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--pass", dest="pass_", type=int, default=1, choices=(1, 2))
    v = sub.add_parser("verify")
    v.add_argument("--results", action="append", required=True)
    v.add_argument("--priced", action="append", required=True, help="pass 2's snapshots.csv.gz (repeatable)")
    v.add_argument("--procs", type=int, default=8)
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--save")
    r = sub.add_parser("report")
    r.add_argument("--out", action="append", required=True, help="a measure's snapshots.csv.gz (repeatable)")
    r.add_argument("--docs")
    r.add_argument("--figs")
    e = sub.add_parser("examples")
    e.add_argument("--results", action="append", required=True)
    e.add_argument("--out", action="append", required=True, help="the measures' snapshots.csv.gz (repeatable)")
    e.add_argument("--arm", default="bad_ssl_bound_x4", choices=tuple(LIBRARY_ARMS))
    e.add_argument("--path", default="binary")
    e.add_argument("--beta", type=float, default=1.0)
    e.add_argument("--click", type=int, default=50)
    e.add_argument("--docs")
    e.add_argument("--figs", required=True)
    a = ap.parse_args(argv)
    {"measure": cmd_measure, "verify": cmd_verify, "report": cmd_report, "examples": cmd_examples}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
