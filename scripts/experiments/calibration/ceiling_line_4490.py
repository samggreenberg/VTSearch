#!/usr/bin/env python3
"""#4490: why the labels line goes too deep with every label known, and a rule that fixes it.

Reads the withheld-half snapshots ``run_cells.py`` saves under ``CALIB_SAVE_TEST_SCORES=1``
(``task_NNNN__testscores.npz``): the full-label ceiling's (``ceiling``) and each session's at chosen clicks
(``t5`` ... ``last``, ``final``), each with the class model and the calibration folds it was drawn from.  Every
rule is replayed on the same scores, so two rules differ in the rule and nothing else.

    python ceiling_line_4490.py price --cells DIR [--cells DIR ...] [--key ceiling] --out rows.csv
    python ceiling_line_4490.py summary rows.csv [rows.csv ...]
    python ceiling_line_4490.py mechanism --cells CEILING_CELLS --out mech.csv
    python ceiling_line_4490.py pair --ceiling ceil_rows.csv --session b025=rows.csv --session b1=... --out pair.csv
    python ceiling_line_4490.py subsample --cells CEILING_CELLS --out sub.csv
    python ceiling_line_4490.py figures --mech mech.csv --ceiling ceil_rows.csv --session b025=rows.csv ... --example NPZ --out DIR

The rules (``RULES``):

* ``shipped`` - the labels line as it ships (#4452, #4492).
* ``rep_kde`` - the candidate: when the Bads look like a random sample of the corpus (at least ``REP_MIN_BADS``
  of them, and at least ``REP_SHARE`` of the corpus scoring above their median held-out score), the negatives are
  modelled by the Bads' own held-out scores (a Gaussian KDE on the logit scale) instead of a normal, the
  positives' share is fitted on the corpus by EM, and the cut is the counted F-beta cut over those chances.
  Otherwise the shipped line.
* ``kde`` - the same fit with no gate (an ablation: it is wrong for a session's selected Bads).
* ``near_q0.02`` - the issue's first suggestion: the class model from every Good and the Bads whose held-out
  score would rank in the corpus's top 2%.
* ``mix`` - the negatives as a normal mixture fitted to the Bads (components by BIC; one component = shipped).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from _cells_paths import testscore_files  # noqa: E402
from vtscore.training.thresholds import labels_line as LL  # noqa: E402

BETAS = (0.25, 1.0, 4.0)
TAG = {0.25: "1/4", 1.0: "1", 4.0: "4"}
#: The gate's two constants: the fewest Bads whose KDE sees the negatives' tail (a random 30 of a ceiling's Bads
#: lost 0.09 F at beta 1/4; 100 gained), and the share of the corpus above the Bads' median below which the Bads
#: are a selected sample (a ceiling's sit at 0.27-0.52, a 150-click session's at a median of 0.03).
REP_MIN_BADS = 100
REP_SHARE = 0.2


# ----------------------------------------------------------------------------- snapshots
def load(npz_path: str | Path) -> dict[str, dict]:
    """``{key: snapshot}`` for one cell; keys like ``siglip/whole_image/ceiling``."""
    z = np.load(npz_path, allow_pickle=False)
    out: dict[str, dict] = {}
    for k in z.files:
        prefix, name = k.rsplit("/", 1)
        out.setdefault(prefix, {})[name] = z[k]
    for snap in out.values():
        snap.update(json.loads(str(snap.pop("meta"))))
    return out


def folds_of(snap: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = snap.get("fold_index")
    if idx is None or idx.size == 0:
        return []
    return [
        (snap["fold_scores"][idx == i].astype(np.float64), snap["fold_labels"][idx == i].astype(np.float64))
        for i in np.unique(idx)
    ]


def model_of(snap: dict) -> LL.ClassScoreModel | None:
    m = snap.get("model")
    if not m:
        return None
    names = {f.name for f in fields(LL.ClassScoreModel)}
    return LL.ClassScoreModel(**{k: v for k, v in m.items() if k in names})


def folds_xy(snap: dict) -> tuple[np.ndarray, np.ndarray]:
    """The held-out Goods' and Bads' logit scores."""
    fs = np.asarray(snap["fold_scores"], dtype=np.float64)
    fy = np.asarray(snap["fold_labels"])
    ok = np.isfinite(fs) & (fs >= 0) & (fs <= 1)
    return LL._logit(fs[ok & (fy == 1)]), LL._logit(fs[ok & (fy == 0)])


def fold_model_matches(snap: dict) -> bool:
    """Whether the saved model is the one the folds imply (not the in-sample fallback of a one-Good session)."""
    m, fm = model_of(snap), LL.class_score_model(folds_of(snap))
    return m is not None and fm is not None and abs(fm.mu_pos - m.mu_pos) < 1e-6 and abs(fm.mu_neg - m.mu_neg) < 1e-6


# ----------------------------------------------------------------------------- scoring
def fbeta(k: int, tp: int, n_pos: int, beta: float) -> float:
    if k <= 0 or tp <= 0 or n_pos <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * tp / (b2 * n_pos + k)


def kept(scores: np.ndarray, labels: np.ndarray, thr: float, beta: float) -> dict:
    keep = scores >= thr
    k, tp, n_pos = int(keep.sum()), int(labels[keep].sum()), int(labels.sum())
    return {"k": k, "tp": tp, "n_pos": n_pos, "f": fbeta(k, tp, n_pos, beta)}


def best_cut(scores: np.ndarray, labels: np.ndarray, beta: float) -> dict:
    """The best F-beta over every top-k of the ranking (the ranking's ceiling, no line)."""
    y = labels[np.argsort(-scores, kind="stable")].astype(np.float64)
    tp = np.cumsum(y)
    b2 = beta * beta
    f = (1 + b2) * tp / (b2 * y.sum() + np.arange(1, y.size + 1)) if y.sum() else np.zeros_like(y)
    i = int(np.argmax(f))
    return {"k_best": i + 1, "f_best": float(f[i])}


# ----------------------------------------------------------------------------- rules
def shipped(snap: dict, scores: np.ndarray):
    model = model_of(snap)
    line = None if model is None else LL._line_on(model, scores, None, None)
    return None if line is None else line.threshold


def bads_kde(xn: np.ndarray):
    """A Gaussian KDE of the Bads' logit scores (Silverman's bandwidth), binned on a fine grid."""
    n = xn.size
    sd = float(np.std(xn, ddof=1)) if n > 1 else 0.25
    iqr = float(np.subtract(*np.percentile(xn, [75, 25]))) / 1.349 if n > 3 else sd
    # Silverman's rule, floored: identical Bads (a session's first two can tie) would give a zero-width spike.
    h = max(0.9 * (min(sd, iqr) if iqr > 0 else sd) * n ** (-0.2), 0.01)
    lo, hi = float(xn.min()) - 12 * h, float(xn.max()) + 12 * h
    grid = np.linspace(lo, hi, 8001)
    step = grid[1] - grid[0]
    counts = np.bincount(np.clip(np.rint((xn - lo) / step).astype(int), 0, grid.size - 1), minlength=grid.size)
    # Never wider than the grid: np.convolve's "same" output takes the longer input's length.
    half = min(int(np.ceil(12 * h / step)), (grid.size - 1) // 2)
    dens = np.convolve(counts.astype(np.float64), norm.pdf(np.arange(-half, half + 1) * step / h), mode="same")
    dens /= n * h

    def pdf(x: np.ndarray) -> np.ndarray:
        out = np.interp(x, grid, dens, left=0.0, right=0.0)
        far = (x < lo) | (x > hi)
        if far.any():
            edge = np.where(x[far] < lo, xn.min(), xn.max())
            out[far] = norm.pdf((x[far] - edge) / h) / (n * h)
        return out

    return pdf


def bads_mixture(xn: np.ndarray, max_comp: int = 4):
    """A normal mixture for the Bads' logit scores, components by BIC; ``(pdf, k)``."""
    from sklearn.mixture import GaussianMixture  # noqa: PLC0415

    best = None
    for k in range(1, max_comp + 1):
        if xn.size < 10 * k:
            break
        g = GaussianMixture(k, random_state=0).fit(xn.reshape(-1, 1))
        bic = g.bic(xn.reshape(-1, 1))
        if best is None or bic < best[0]:
            best = (bic, k, g)
    if best is None:
        return None, 0
    _, k, g = best
    w, mu, sd = g.weights_, g.means_.ravel(), np.sqrt(g.covariances_.ravel())
    return (lambda x: (w * norm.pdf((x[:, None] - mu) / sd) / sd).sum(axis=1)), k


def shape_line(snap: dict, scores: np.ndarray, pdf0):
    """Goods ~ the class model's Good normal (spread floored on this corpus), negatives ~ *pdf0*; the positives'
    share by EM on the corpus; each item's chance from it; the counted F-beta cut (``corpus_cut``)."""
    model = model_of(snap)
    u = LL._finite_unit(scores)
    x = LL._logit(u)
    fm = model.floored(LL.corpus_sigma_floor(u))
    f1, f0 = norm.pdf(x, fm.mu_pos, fm.sigma), pdf0(x)
    pi, r = 0.01, np.zeros_like(x)
    for _ in range(500):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nan_to_num(pi * f1 / (pi * f1 + (1 - pi) * f0), nan=0.0)
        new = min(max(float(r.mean()), LL.PREVALENCE_MIN), LL.PREVALENCE_MAX)
        done = abs(new - pi) < 1e-10
        pi = new
        if done:
            break
    order = np.argsort(-u, kind="stable")
    post, desc = r[order], u[order]
    total = float(post.sum())
    return lambda beta: LL.corpus_cut(post, desc, total, beta)


def bads_share_above_median(snap: dict, scores: np.ndarray) -> float:
    """The share of the corpus scoring above the Bads' median held-out score: ~0.5 for a random sample of Bads."""
    _, xn = folds_xy(snap)
    if xn.size == 0:
        return float("nan")
    x = np.sort(LL._logit(LL._finite_unit(scores)))
    return 1.0 - np.searchsorted(x, np.median(xn)) / x.size


def representative(snap: dict, scores: np.ndarray, *, share: float = REP_SHARE, min_bads: int = REP_MIN_BADS) -> bool:
    _, xn = folds_xy(snap)
    return xn.size >= min_bads and bads_share_above_median(snap, scores) >= share


def rep_kde(snap: dict, scores: np.ndarray):
    if fold_model_matches(snap) and representative(snap, scores):
        return shape_line(snap, scores, bads_kde(folds_xy(snap)[1]))
    return shipped(snap, scores)


def kde(snap: dict, scores: np.ndarray):
    if not fold_model_matches(snap):
        return shipped(snap, scores)
    return shape_line(snap, scores, bads_kde(folds_xy(snap)[1]))


def mix(snap: dict, scores: np.ndarray):
    if not fold_model_matches(snap):
        return shipped(snap, scores)
    pdf0, k = bads_mixture(folds_xy(snap)[1])
    return shipped(snap, scores) if pdf0 is None or k <= 1 else shape_line(snap, scores, pdf0)


def near_bads(q: float):
    def rule(snap: dict, scores: np.ndarray):
        if not fold_model_matches(snap):
            return shipped(snap, scores)
        cut = float(np.quantile(LL._logit(LL._finite_unit(scores)), 1.0 - q))
        kept_folds = [
            (s[(y >= 0.5) | (LL._logit(s) >= cut)], y[(y >= 0.5) | (LL._logit(s) >= cut)]) for s, y in folds_of(snap)
        ]
        model = LL.class_score_model(kept_folds)
        line = None if model is None else LL._line_on(model, scores, None, None)
        return shipped(snap, scores) if line is None else line.threshold

    return rule


RULES = {"shipped": shipped, "rep_kde": rep_kde, "kde": kde, "near_q0.02": near_bads(0.02), "mix": mix}


# ----------------------------------------------------------------------------- price
def _price_cell(args) -> list[dict]:
    path, keys, rule_names = args
    rows = []
    for key, snap in load(path).items():
        short = key.rsplit("/", 1)[1]
        if keys and short not in keys:
            continue
        s = np.asarray(snap["scores"], dtype=np.float64)
        y = np.asarray(snap["labels"], dtype=np.float64)
        # A session is scored at its own preset; the ceiling (no preset) at each.
        betas = BETAS if short == "ceiling" else (float(snap["beta"]),)
        fns = {}
        for rn in rule_names:
            try:
                fns[rn] = RULES[rn](snap, s)
            except Exception as e:  # noqa: BLE001 - a rule that fails on a cell is reported, not fatal
                print(f"{path} {key} {rn}: {e!r}", file=sys.stderr)
                fns[rn] = None
        for b in betas:
            best = best_cut(s, y, b)
            for rn, fn in fns.items():
                thr = float("nan") if fn is None else float(fn(b))
                m = (
                    kept(s, y, thr, b)
                    if math.isfinite(thr)
                    else {"k": -1, "tp": -1, "n_pos": int(y.sum()), "f": np.nan}
                )
                rows.append({"dir": str(Path(path).parent), "cell": Path(path).name.split("__")[0], "key": short,
                             "t": snap.get("t"), "rule": rn, "beta": b, "thr": thr, **m, **best})  # fmt: skip
    return rows


def price(cells: list[str], keys: list[str], rules: list[str], jobs: int) -> pd.DataFrame:
    work = [(str(p), set(keys), rules) for d in cells for p in testscore_files(d)]
    with ProcessPoolExecutor(jobs) as ex:
        rows = [r for rr in ex.map(_price_cell, work, chunksize=2) for r in rr]
    return pd.DataFrame(rows)


def summary(d: pd.DataFrame, base: str = "shipped") -> pd.DataFrame:
    """Per key x beta x rule: mean F, the paired difference from *base* (SE), returned set, P, R, the best cut."""
    piv = d.pivot_table(index=["dir", "cell", "key", "beta"], columns="rule", values="f")
    out = []
    for (key, beta, rule), g in d.groupby(["key", "beta", "rule"]):
        sub = piv.xs((key, beta), level=("key", "beta"))
        delta = (sub[rule] - sub[base]).dropna()
        g = g[g.k >= 0]
        out.append({"key": key, "beta": beta, "rule": rule, "n": len(g), "F": g.f.mean(), "dF": delta.mean(),
                    "se": delta.std(ddof=1) / math.sqrt(max(len(delta), 2)),
                    "changed%": 100 * (delta.abs() > 1e-9).mean(),
                    "k_med": g.k.median(), "k_p90": g.k.quantile(0.9),
                    "P": (g.tp / g.k.where(g.k > 0)).mean(), "R": (g.tp / g.n_pos.where(g.n_pos > 0)).mean(),
                    "F_best": g.f_best.mean()})  # fmt: skip
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- mechanism
def _mech_cell(path: str) -> dict:
    snap = load(path)["siglip/whole_image/ceiling"]
    s, y = snap["scores"].astype(np.float64), snap["labels"].astype(np.float64)
    m = LL.class_score_model(folds_of(snap))
    _, xn = folds_xy(snap)
    line = LL._line_on(m, s, None, None)
    order = np.argsort(-s, kind="stable")
    ys, post = np.cumsum(y[order]), line.unvoted_posteriors
    n_neg = float((1 - y).sum())
    sd_neg = float(xn.std(ddof=1))
    r = {"cell": Path(path).name[:9], "n_good": int(m.n_pos), "n_bad": int(m.n_neg), "n_pos": int(y.sum()),
         "est_total": line.unvoted_share * s.size, "share_above_bad_median": bads_share_above_median(snap, s)}  # fmt: skip
    for z in (2, 3, 4):
        r[f"tail{z}"] = float((xn > m.mu_neg + z * sd_neg).mean()) / float(norm.sf(z))
    for b in BETAS:
        thr = line.threshold(b)
        k = int((s >= thr).sum())
        tag = TAG[b]
        r[f"k {tag}"], r[f"tp {tag}"], r[f"expected tp {tag}"] = (
            k,
            float(ys[k - 1]) if k else 0.0,
            float(post[:k].sum()),
        )
        r[f"fp {tag}"] = k - r[f"tp {tag}"]
        r[f"fp normal {tag}"] = m.survival(thr)[1] * n_neg
        r[f"fp bads {tag}"] = float((xn >= LL._logit([thr])[0]).mean()) * n_neg
    return r


def mechanism(cells: str, jobs: int) -> pd.DataFrame:
    with ProcessPoolExecutor(jobs) as ex:
        return pd.DataFrame(list(ex.map(_mech_cell, [str(p) for p in testscore_files(cells)], chunksize=4)))


# ----------------------------------------------------------------------------- the issue's pairing
def pair(ceiling: pd.DataFrame, sessions: dict[str, pd.DataFrame], rules: list[str]) -> pd.DataFrame:
    """The ceiling under each rule against the session's shipped line at click 150, paired on cell and seed."""
    out = []
    for tag, s in sessions.items():
        s = s[(s.key == "last") & (s.rule == "shipped")][["cell", "f", "beta"]].rename(columns={"f": "f_session"})
        b = float(s.beta.iloc[0])
        for rule in rules:
            c = ceiling[(ceiling.beta == b) & (ceiling.rule == rule)].merge(s.drop(columns="beta"), on="cell")
            d = c.f - c.f_session
            out.append({"preset": TAG[b], "ceiling rule": rule, "n": len(c), "ceiling F": c.f.mean(),
                        "best cut": c.f_best.mean(), "returned median": c.k.median(), "returned p90": c.k.quantile(0.9),
                        "precision": (c.tp / c.k.where(c.k > 0)).mean(), "recall": (c.tp / c.n_pos).mean(),
                        "session F at 150": c.f_session.mean(), "ceiling - session": d.mean(),
                        "se": d.std(ddof=1) / math.sqrt(len(d)), "worse %": 100 * (d < 0).mean()})  # fmt: skip
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- fewer random Bads
def _subsample_cell(path: str) -> list[dict]:
    from dataclasses import asdict  # noqa: PLC0415

    base = load(path)["siglip/whole_image/ceiling"]
    s, y = base["scores"].astype(np.float64), base["labels"].astype(np.float64)
    out = []
    for n_bad in (30, 100, 300, 1000, 0):
        snap = dict(base)
        if n_bad:
            rng = np.random.default_rng([4490, int(Path(path).name[5:9]), n_bad])
            neg = np.flatnonzero(base["fold_labels"] == 0)
            keep = np.sort(
                np.concatenate([np.flatnonzero(base["fold_labels"] == 1), rng.choice(neg, n_bad, replace=False)])
            )
            for f in ("fold_scores", "fold_labels", "fold_index"):
                snap[f] = base[f][keep]
            snap["model"] = asdict(LL.class_score_model(folds_of(snap)))
        for rn in ("shipped", "rep_kde", "kde"):
            fn = RULES[rn](snap, s)
            for b in BETAS:
                out.append({"cell": Path(path).name[:9], "n_bad": n_bad or "all", "rule": rn, "beta": b,
                            **kept(s, y, fn(b), b)})  # fmt: skip
    return out


def subsample(cells: str, jobs: int) -> pd.DataFrame:
    with ProcessPoolExecutor(jobs) as ex:
        return pd.DataFrame(
            [r for rr in ex.map(_subsample_cell, [str(p) for p in testscore_files(cells)], chunksize=4) for r in rr]
        )


# ----------------------------------------------------------------------------- figures
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(mech: pd.DataFrame, ceiling: pd.DataFrame, sessions: dict[str, pd.DataFrame], example: str, out: Path):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK})

    # 1. One cell: the Bads' held-out scores against the class model's normal, and the two cuts.
    snap = load(example)["siglip/whole_image/ceiling"]
    s = snap["scores"].astype(np.float64)
    m = LL.class_score_model(folds_of(snap))
    xp, xn = folds_xy(snap)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    _style(ax)
    bins = np.linspace(min(xn.min(), xp.min()) - 0.2, max(xn.max(), xp.max()) + 0.2, 70)
    width = bins[1] - bins[0]
    ax.hist(xn, bins=bins, color=BLUE, alpha=0.55, label=f"Bads' held-out scores ({xn.size:,})")
    ax.hist(xp, bins=bins, color=ORANGE, alpha=0.75, label=f"Goods' held-out scores ({xp.size})")
    xx = np.linspace(bins[0], bins[-1], 600)
    ax.plot(
        xx, xn.size * width * norm.pdf(xx, m.mu_neg, m.sigma), color=INK, lw=1.5, label="the class model's Bad normal"
    )
    ax.plot(xx, xn.size * width * bads_kde(xn)(xx), color=AQUA, lw=2, label="the Bads' own shape (KDE)")
    for rule, ls in (("shipped", "--"), ("rep_kde", "-")):
        thr = RULES[rule](snap, s)(1.0)
        ax.axvline(LL._logit([thr])[0], color=INK2, ls=ls, lw=1.2)
        ax.text(LL._logit([thr])[0], 0.9, f" {rule.replace('rep_kde', 'fix')} cut, beta 1", transform=ax.get_xaxis_transform(),
                color=INK2, fontsize=8, rotation=90, va="top", ha="right")  # fmt: skip
    ax.set_yscale("log")
    ax.set_ylim(0.5, None)
    ax.set_xlabel("held-out score (logit)")
    ax.set_ylabel("images per bin (log)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "tail_example.png", dpi=150)
    plt.close(fig)

    # 2. Wrong images above the shipped cut: the truth, the normal's prediction, the Bads' own prediction.
    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=SURFACE)
    _style(ax)
    xs = np.arange(len(BETAS))
    for j, (col, lab, color) in enumerate(
        (
            ("fp", "true", INK2),
            ("fp normal", "the class model's normal predicts", BLUE),
            ("fp bads", "the Bads' own scores predict", AQUA),
        )
    ):
        vals = [mech[f"{col} {TAG[b]}"].median() for b in BETAS]
        bars = ax.bar(xs + (j - 1) * 0.26, vals, width=0.24, color=color, label=lab)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(xs, [f"beta {TAG[b]}" for b in BETAS])
    ax.set_ylabel("wrong images above the cut\n(median over cells)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "wrong_above_cut.png", dpi=150)
    plt.close(fig)

    # 3. The session's line over clicks against the ceiling's, shipped and fixed, per preset.
    keys = ["t5", "t10", "t25", "t50", "t100", "last"]
    clicks = [5, 10, 25, 50, 100, 150]
    fig, axes = plt.subplots(1, len(sessions), figsize=(3.3 * len(sessions), 3.2), sharey=True, facecolor=SURFACE)
    for ax, (tag, d) in zip(np.atleast_1d(axes), sessions.items()):
        _style(ax)
        b = float(d.beta.iloc[0])
        sh = d[d.rule == "shipped"]
        ax.plot(
            clicks,
            [sh[sh.key == k].f.mean() for k in keys],
            color=BLUE,
            lw=2,
            marker="o",
            ms=4,
            label="session (shipped line)",
        )
        c = ceiling[ceiling.beta == b]
        cells = set(sh.cell)
        c = c[c.cell.isin(cells)]
        for rule, ls, color, lab in (
            ("shipped", "--", ORANGE, "every label, shipped line"),
            ("rep_kde", "-", ORANGE, "every label, fixed line"),
        ):
            ax.axhline(c[c.rule == rule].f.mean(), color=color, ls=ls, lw=1.6, label=lab)
        ax.axhline(c[c.rule == "shipped"].f_best.mean(), color=GRAY, ls=":", lw=1.4, label="every label, best cut")
        ax.set_xscale("log")
        ax.set_xticks(clicks, [str(x) for x in clicks])
        ax.set_title(f"beta {TAG[b]}", fontsize=10)
        ax.set_xlabel("click")
    np.atleast_1d(axes)[0].set_ylabel("F-beta of the withheld half\nabove Find's line")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ceiling_vs_session.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI
def _sessions(specs: list[str]) -> dict[str, pd.DataFrame]:
    return {tag: pd.read_csv(path) for tag, path in (s.split("=", 1) for s in specs)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("price")
    p.add_argument("--cells", action="append", required=True)
    p.add_argument("--key", action="append", default=[])
    p.add_argument("--rule", action="append", default=[])
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary")
    p.add_argument("rows", nargs="+")
    for name in ("mechanism", "subsample"):
        p = sub.add_parser(name)
        p.add_argument("--cells", required=True)
        p.add_argument("--jobs", type=int, default=8)
        p.add_argument("--out", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--ceiling", required=True)
    p.add_argument("--session", action="append", required=True, help="tag=rows.csv")
    p.add_argument("--out", required=True)
    p = sub.add_parser("figures")
    p.add_argument("--mech", required=True)
    p.add_argument("--ceiling", required=True)
    p.add_argument("--session", action="append", required=True, help="tag=rows.csv")
    p.add_argument("--example", required=True, help="a ceiling task_NNNN__testscores.npz")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    if a.cmd == "price":
        d = price(a.cells, a.key, a.rule or list(RULES), a.jobs)
        d.to_csv(a.out, index=False)
        print(f"{d.cell.nunique()} cells, {len(d)} rows -> {a.out}")
    elif a.cmd == "summary":
        print(summary(pd.concat(pd.read_csv(r) for r in a.rows)).round(3).to_string(index=False))
    elif a.cmd == "mechanism":
        d = mechanism(a.cells, a.jobs)
        d.to_csv(a.out, index=False)
        print(d.describe(percentiles=[0.1, 0.5, 0.9]).round(2).T.to_string())
    elif a.cmd == "subsample":
        d = subsample(a.cells, a.jobs)
        d.to_csv(a.out, index=False)
        print(d.pivot_table(index=["n_bad", "beta"], columns="rule", values="f").round(3).to_string())
    elif a.cmd == "pair":
        d = pair(pd.read_csv(a.ceiling), _sessions(a.session), ["shipped", "rep_kde"])
        d.to_csv(a.out, index=False)
        print(d.round(3).to_string(index=False))
    elif a.cmd == "figures":
        figures(pd.read_csv(a.mech), pd.read_csv(a.ceiling), _sessions(a.session), a.example, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
