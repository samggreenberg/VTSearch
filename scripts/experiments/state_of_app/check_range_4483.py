"""#4483: price the spot check's precision range on coverage against width.

The check shows a **likely range** for the share of the set it audited that is
right (:meth:`~vtscore.training.thresholds.SpotCheck.range`): each band's
Clopper-Pearson interval with each tail at ``alpha / bands`` (the union bound
over the bands, #4272), weighted by band size.  At 5 picks a band it held the
truth every time in #4474's review and averaged ~0.7 wide.  This prices other
constructions on the same per-band counts:

* ``cp_union`` - the shipped range (reproduced exactly; the realized pass checks it
  against the harness's ``range_lo`` / ``range_hi``).
* ``hg_union`` - the same tail, but each band's bound is the exact hypergeometric one
  for a band of known size (a band of 8 with 5 of 5 right holds at least 5).
* ``cp_plain`` / ``hg_plain`` - the same without the union split: each tail at ``alpha``.
* ``bayes90`` / ``bayes80`` - each band's count of right items under a uniform prior,
  convolved over the bands: the central 90% / 80% of the set's share right;
  ``bayes90j`` the same at 90% under a Jeffreys (beta-binomial 1/2, 1/2) prior.
* ``wald90`` - the stratified normal interval, for reference.

Two sources, each written as one row per check with its bands as ``N:n:r:K``
strata (size, picks, right, truly right):

* ``realized RUN...`` - every end-of-run check in State-of-the-App runs: the pick
  log's check rounds are its bands in order (one round of picks a band), the
  ``last`` rank frame is the ranking it walked, so its truth is exact.
* ``simulate RUN...`` - the shipped walk (:class:`SpotCheck`, its own draws) on the
  ``step`` rank frames at clicks 25 / 51 / 100, the regime a check during a session
  sees; recall is read against the frame's true count of unvoted positives (the
  mixture's count is not logged).  Also writes the strata of the line's own kept
  set (``floor_count`` at that step), from the same picks.

``report OUT`` scores every construction and writes ``summary.csv``, the figures
and the examples.  Work dir and report: ``docs/experiments/2026-10-15-check-range-4483/``.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from vtscore.training.thresholds.spot_check import CHECK_ALPHA, SpotCheck, band_edges, bands_for

#: The frames a check during a session is simulated on (#4664: the weak check
#: first starts at a median click of 17-18 and ends at 42-48; the check before
#: Done runs at a median click of 57-63, #4909).
SIM_FRAMES = (25, 51, 100)

METHODS = ("cp_union", "hg_union", "cp_plain", "hg_plain", "bayes90", "bayes80", "bayes90j", "wald90")


# ------------------------------------------------------------------ strata


def strata_str(strata: list[tuple[int, int, int, int]]) -> str:
    return ";".join(f"{N}:{n}:{r}:{K}" for N, n, r, K in strata)


def parse_strata(s: str) -> list[tuple[int, int, int, int]]:
    if not isinstance(s, str) or not s:
        return []
    return [tuple(int(x) for x in part.split(":")) for part in s.split(";")]  # type: ignore[misc]


def _ranks(s: object) -> np.ndarray:
    if not isinstance(s, str) or not s.strip():
        return np.zeros(0, dtype=np.int64)
    return np.asarray(s.split(), dtype=np.int64)


def _count_in(pos: np.ndarray, lo: int, hi: int) -> int:
    return int(np.count_nonzero((pos >= lo) & (pos < hi)))


# ------------------------------------------------------------------ the bounds


@functools.lru_cache(maxsize=None)
def cp_bounds(N: int, n: int, r: int, tail: float) -> tuple[float, float]:
    """The shipped band bound (:func:`~vtscore.training.thresholds.spot_check.likely_range`)."""
    from vtscore.training.thresholds.spot_check import likely_range  # noqa: PLC0415

    if n <= 0:
        return 0.0, 1.0
    rng = likely_range(r, n, N, tail)
    return rng.lo, rng.hi


@functools.lru_cache(maxsize=None)
def hg_bounds(N: int, n: int, r: int, tail: float) -> tuple[float, float]:
    """Exact one-sided hypergeometric bounds on a band's share right, each at *tail*.

    The lower bound is the least K under which seeing *r* or more right of *n*
    picks has probability above *tail*; the upper the greatest K under which
    seeing *r* or fewer does.  A census is exact.
    """
    from scipy.stats import hypergeom  # noqa: PLC0415

    if n <= 0:
        return 0.0, 1.0
    if n >= N:
        return r / N, r / N
    K = np.arange(N + 1)
    sf = hypergeom.sf(r - 1, N, K, n)  # P(R >= r | K)
    cdf = hypergeom.cdf(r, N, K, n)  # P(R <= r | K)
    lo = int(K[np.argmax(sf > tail)]) if r > 0 else 0
    ok = np.flatnonzero(cdf > tail)
    hi = int(ok[-1]) if r < n else N
    return lo / N, hi / N


@functools.lru_cache(maxsize=None)
def band_posterior(N: int, n: int, r: int, a: float = 1.0) -> np.ndarray:
    """P(K | r of n right) for a band of *N* under a beta-binomial(a, a) prior on K = 0..N (a = 1: uniform)."""
    from scipy.stats import betabinom, hypergeom  # noqa: PLC0415

    K = np.arange(N + 1)
    prior = betabinom.pmf(K, N, a, a)
    w = prior if n <= 0 else prior * hypergeom.pmf(r, N, K, n)
    return w / w.sum()


def bayes_bounds(strata: list[tuple[int, int, int, int]], level: float, a: float = 1.0) -> tuple[float, float]:
    """The central *level* interval of the set's share right: the bands' posteriors convolved."""
    total = np.ones(1)
    for N, n, r, _K in strata:
        total = np.convolve(total, band_posterior(N, n, r, a))
    k = sum(s[0] for s in strata)
    cdf = np.cumsum(total)
    tail = (1.0 - level) / 2.0
    lo = int(np.searchsorted(cdf, tail - 1e-12))
    hi = int(np.searchsorted(cdf, 1.0 - tail - 1e-12))
    return lo / k, min(hi, k) / k


def wald_bounds(strata: list[tuple[int, int, int, int]], z: float = 1.6449) -> tuple[float, float]:
    k = sum(s[0] for s in strata)
    est = var = 0.0
    for N, n, r, _K in strata:
        if n <= 0:
            return 0.0, 1.0
        p = r / n
        est += N * p
        if n < N:
            s2 = p * (1 - p) * n / (n - 1) if n > 1 else 0.25
            var += N * N * (1 - n / N) * s2 / n
    est /= k
    half = z * math.sqrt(var) / k
    return max(0.0, est - half), min(1.0, est + half)


def summed(strata: list[tuple[int, int, int, int]], fn, tail: float) -> tuple[float, float]:
    k = sum(s[0] for s in strata)
    lo = hi = 0.0
    for N, n, r, _K in strata:
        a, b = fn(N, n, r, tail)
        lo += N * a
        hi += N * b
    return lo / k, hi / k


def method_bounds(
    strata: list[tuple[int, int, int, int]], alpha: float = CHECK_ALPHA
) -> dict[str, tuple[float, float]]:
    B = max(1, len(strata))
    return {
        "cp_union": summed(strata, cp_bounds, alpha / B),
        "hg_union": summed(strata, hg_bounds, alpha / B),
        "cp_plain": summed(strata, cp_bounds, alpha),
        "hg_plain": summed(strata, hg_bounds, alpha),
        "bayes90": bayes_bounds(strata, 0.90),
        "bayes80": bayes_bounds(strata, 0.80),
        "bayes90j": bayes_bounds(strata, 0.90, 0.5),
        "wald90": wald_bounds(strata),
    }


def truth_and_point(strata: list[tuple[int, int, int, int]]) -> tuple[float, float]:
    k = sum(s[0] for s in strata)
    truth = sum(s[3] for s in strata) / k
    if any(n <= 0 for _N, n, _r, _K in strata):
        return truth, float("nan")
    return truth, sum(N * r / n for N, n, r, _K in strata) / k


# ------------------------------------------------------------------ realized


def _cells(run: Path) -> list[Path]:
    return sorted((run / "results" / "cells").glob("task_*__picks.csv.gz"))


def realized_cell(picks_path: str, run_label: str) -> dict | None:
    """One end-of-run check: its bands' counts from the pick log, its truth from the ``last`` frame."""
    stem = picks_path[: -len("__picks.csv.gz")]
    picks = pd.read_csv(picks_path)
    if "check_kind" in picks.columns:
        end = picks[(picks["phase"] == "check") & (picks["check_kind"].fillna("") == "end")]
    else:
        end = picks[picks["phase"] == "check"]
    if end.empty:
        return None
    frames = pd.read_csv(stem + "__rankframes.csv.gz")
    last = frames[frames["kind"] == "last"]
    if last.empty:
        return None
    last = last.iloc[-1]
    main = pd.read_csv(
        stem + ".csv.gz",
        usecols=[
            "t",
            "floor_status",
            "floor_count",
            "range_lo",
            "range_hi",
            "check_labelled",
            "check_right",
            "check_audited",
            "beta",
        ],
    )
    row = main[main["t"] == main["t"].max()].iloc[0]
    before = main[main["t"] == int(last["t"])]
    kept = int(before["floor_count"].iloc[0]) if not before.empty else -1
    if str(row["floor_status"]) != "checked" or int(row["check_audited"]) < 1:
        return None
    n_pool = int(last["n_pool"])
    rounds = [g["picked_label"].astype(int).to_numpy() for _, g in end.groupby("t", sort=True)]
    first_pool = int(end.sort_values("t")["n_pool"].iloc[0])
    edges = band_edges(n_pool)
    audited = int(row["check_audited"])
    B = bands_for(audited, edges)
    pos = _ranks(last["pool_pos_ranks"])
    strata = []
    for b in range(B):
        N = edges[b + 1] - edges[b]
        labels = rounds[b] if b < len(rounds) else np.zeros(0, dtype=int)
        strata.append((N, int(labels.size), int(labels.sum()), _count_in(pos, edges[b], edges[b + 1])))
    return {
        "source": "realized",
        "run": run_label,
        "cell": Path(stem).name,
        "seed": int(picks["seed"].iloc[0]),
        "category": str(picks["category"].iloc[0]),
        "embedder": str(picks["embedder"].iloc[0]) if "embedder" in picks.columns else "",
        "beta": float(row["beta"]),
        "t": int(last["t"]),
        "draw": 0,
        "n_pool": n_pool,
        "n_pool_pos": int(last["n_pool_pos"]),
        "audited": audited,
        "kept": kept,
        "rounds": len(rounds),
        "start_ok": int(first_pool == n_pool - rounds[0].size),
        "edge_ok": int(edges[B] == audited),
        "log_lo": float(row["range_lo"]),
        "log_hi": float(row["range_hi"]),
        "log_labelled": int(row["check_labelled"]),
        "log_right": int(row["check_right"]),
        "strata": strata_str(strata),
        "kept_strata": "",
    }


def run_realized(runs: list[str], out: Path, jobs: int) -> None:
    tasks = []
    for spec in runs:
        label, _, path = spec.partition("=")
        for p in _cells(Path(path)):
            tasks.append((str(p), label))
    with ProcessPoolExecutor(jobs) as ex:
        rows = [r for r in ex.map(realized_cell, *zip(*tasks), chunksize=8) if r is not None]
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "realized.csv.gz", index=False)
    # The reproduction check: the shipped construction on these strata is the harness's range.
    bad = 0
    for r in df.itertuples():
        lo, hi = method_bounds(parse_strata(r.strata))["cp_union"]
        st = parse_strata(r.strata)
        if (
            abs(lo - r.log_lo) > 1e-5
            or abs(hi - r.log_hi) > 1e-5
            or sum(s[1] for s in st) != r.log_labelled
            or sum(s[2] for s in st) != r.log_right
        ):
            bad += 1
    check = {
        "checks": len(df),
        "range_mismatch": bad,
        "start_mismatch": int((df["start_ok"] == 0).sum()),
        "edge_mismatch": int((df["edge_ok"] == 0).sum()),
    }
    (out / "realized_check.json").write_text(json.dumps(check, indent=1) + "\n")
    print(json.dumps(check))


# ------------------------------------------------------------------ simulated


def _walk(n_pool: int, pos: np.ndarray, beta: float, n_pos: float, seed: int) -> SpotCheck:
    posset = set(int(p) for p in pos)
    check = SpotCheck.start_balance(range(n_pool), beta, n_pos, seed=seed)
    while check.running:
        check.record({i: (i in posset) for i in check.pending})
    return check


def _strata_over(check: SpotCheck, pos: np.ndarray, top: int) -> list[tuple[int, int, int, int]]:
    """The strata of the top *top* ranks: each band's part above *top*, with the picks that fell in it."""
    edges = check.edges
    labelled = np.asarray(sorted(check.labels), dtype=np.int64)
    right = np.asarray([check.labels[int(i)] for i in labelled], dtype=bool)
    out = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], min(edges[b + 1], top)
        if lo >= top:
            break
        mask = (labelled >= lo) & (labelled < hi)
        out.append((hi - lo, int(mask.sum()), int(right[mask].sum()), _count_in(pos, lo, hi)))
    return out


def simulate_cell(picks_path: str, run_label: str, draws: int, npos_scale: float) -> list[dict]:
    stem = picks_path[: -len("__picks.csv.gz")]
    frames = pd.read_csv(stem + "__rankframes.csv.gz")
    main = pd.read_csv(stem + ".csv.gz", usecols=["t", "floor_count", "beta", "seed", "category"])
    beta = float(main["beta"].dropna().iloc[0])
    rows = []
    steps = frames[frames["kind"] == "step"].sort_values("t")
    for target in SIM_FRAMES:
        # A frame is recorded at the step that reaches its click, which a round of several votes can step past.
        fr = steps[(steps["t"] >= target) & (steps["t"] < target + 10)]
        if fr.empty:
            continue
        fr = fr.iloc[0]
        t = int(fr["t"])
        n_pool = int(fr["n_pool"])
        pos = _ranks(fr["pool_pos_ranks"])
        at = main[main["t"] == t]
        kept = int(at["floor_count"].iloc[0]) if not at.empty else -1
        n_pos = max(1.0, float(pos.size) * npos_scale)
        for d in range(draws):
            check = _walk(n_pool, pos, beta, n_pos, _seed(Path(stem).name, run_label, t, d))
            strata = []
            for b in range(check.bands):
                lo, hi = check.edges[b], check.edges[b + 1]
                N = hi - lo
                n = r = 0
                for cid, ok in check.labels.items():
                    if lo <= cid < hi:
                        n += 1
                        r += int(ok)
                strata.append((N, n, r, _count_in(pos, lo, hi)))
            rows.append(
                {
                    "source": "simulated",
                    "run": run_label,
                    "cell": Path(stem).name,
                    "seed": int(main["seed"].iloc[0]),
                    "category": str(main["category"].iloc[0]),
                    "beta": beta,
                    "t": target,
                    "frame_t": t,
                    "draw": d,
                    "n_pool": n_pool,
                    "n_pool_pos": int(pos.size),
                    "audited": int(check.k),
                    "kept": kept,
                    "rounds": int(check.round),
                    "strata": strata_str(strata),
                    "kept_strata": strata_str(_strata_over(check, pos, kept)) if kept > 0 else "",
                }
            )
    return rows


def _seed(*parts: object) -> int:
    import hashlib  # noqa: PLC0415

    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def run_simulated(runs: list[str], out: Path, jobs: int, draws: int, npos_scale: float, tag: str) -> None:
    tasks = []
    for spec in runs:
        label, _, path = spec.partition("=")
        for p in _cells(Path(path)):
            tasks.append((str(p), label, draws, npos_scale))
    rows: list[dict] = []
    with ProcessPoolExecutor(jobs) as ex:
        for part in ex.map(simulate_cell, *zip(*tasks), chunksize=4):
            rows.extend(part)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"simulated{tag}.csv.gz", index=False)
    print(f"simulated{tag}: {len(rows)} checks")


# ------------------------------------------------------------------ report


def score(df: pd.DataFrame, column: str = "strata") -> pd.DataFrame:
    """One row per (check, method): the range, its width, and whether it held the truth."""
    recs = []
    for r in df.itertuples(index=False):
        strata = parse_strata(getattr(r, column))
        if not strata:
            continue
        truth, point = truth_and_point(strata)
        for m, (lo, hi) in method_bounds(strata).items():
            recs.append(
                {
                    "source": r.source,
                    "run": r.run,
                    "cell": r.cell,
                    "category": r.category,
                    "seed": r.seed,
                    "beta": r.beta,
                    "t": r.t,
                    "draw": r.draw,
                    "k": sum(s[0] for s in strata),
                    "labelled": sum(s[1] for s in strata),
                    "method": m,
                    "lo": lo,
                    "hi": hi,
                    "width": hi - lo,
                    "truth": truth,
                    "point": point,
                    "covered": float(lo - 1e-9 <= truth <= hi + 1e-9),
                    "above": float(lo > truth + 1e-9),
                    "below": float(hi < truth - 1e-9),
                }
            )
    return pd.DataFrame(recs)


def summarize(sc: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = sc.groupby(keys + ["method"], sort=False)
    out = g.agg(
        checks=("covered", "size"),
        coverage=("covered", "mean"),
        above=("above", "mean"),
        below=("below", "mean"),
        width=("width", "mean"),
        lo=("lo", "mean"),
        hi=("hi", "mean"),
        truth=("truth", "mean"),
        k=("k", "median"),
        labelled=("labelled", "mean"),
    ).reset_index()
    return out


def _score_file(path: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return score(df, column)


def run_report(out: Path, jobs: int) -> None:
    parts = []
    for name, column, label in (
        ("realized.csv.gz", "strata", "audited"),
        ("simulated.csv.gz", "strata", "audited"),
        ("simulated.csv.gz", "kept_strata", "kept"),
        ("simulated_npos2.csv.gz", "strata", "audited"),
    ):
        p = out / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        chunks = np.array_split(df, max(1, jobs * 4))
        with ProcessPoolExecutor(jobs) as ex:
            scored = pd.concat(list(ex.map(score, chunks, [column] * len(chunks))), ignore_index=True)
        scored["set"] = label
        scored["file"] = name.split(".")[0]
        parts.append(scored)
    sc = pd.concat(parts, ignore_index=True)
    sc.to_csv(out / "scored.csv.gz", index=False)
    summ = summarize(sc, ["file", "set", "run", "beta", "t"])
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string())


#: How the report names each construction.
METHOD_NAMES = {
    "cp_union": "shipped (CP, tail split over bands)",
    "hg_union": "exact band bound, tail split",
    "cp_plain": "CP, no split",
    "hg_plain": "exact band bound, no split",
    "bayes90": "90% over the set",
    "bayes80": "80% over the set",
    "bayes90j": "90% over the set, Jeffreys prior",
    "wald90": "normal 90%",
}
METHOD_COLORS = {
    "cp_union": "#7f7f7f",
    "hg_union": "#1f77b4",
    "cp_plain": "#bcbd22",
    "hg_plain": "#17becf",
    "bayes90": "#d62728",
    "bayes80": "#ff7f0e",
    "bayes90j": "#8c564b",
    "wald90": "#9467bd",
}
PRESETS = ((0.25, "beta 1/4"), (1.0, "beta 1"), (4.0, "beta 4"))


def run_figures(out: Path, docs: Path, featured: str) -> None:
    """The report's figures and tables, from ``scored.csv.gz``."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sc = pd.read_csv(out / "scored.csv.gz")
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    (docs / "tables").mkdir(parents=True, exist_ok=True)
    # The closing checks of the 400-click runs (every preset) and the simulated checks at click 51.
    real = sc[(sc["file"] == "realized") & sc["run"].str.startswith("s")]
    sim = sc[(sc["file"] == "simulated") & (sc["set"] == "audited") & (sc["t"] == 51)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (beta, name) in zip(axes, PRESETS):
        for df, marker, label in ((real, "o", "closing check (realized)"), (sim, "s", "check at click 51 (simulated)")):
            g = df[np.isclose(df["beta"], beta)].groupby("method").agg(cov=("covered", "mean"), w=("width", "mean"))
            for m in METHODS:
                if m not in g.index:
                    continue
                ax.scatter(
                    g.loc[m, "w"],
                    g.loc[m, "cov"],
                    marker=marker,
                    s=60,
                    facecolors=METHOD_COLORS[m] if marker == "o" else "none",
                    edgecolors=METHOD_COLORS[m],
                    label=f"{METHOD_NAMES[m]}" if (marker == "o" and beta == 0.25) else None,
                )
        ax.axhline(0.9, color="#444", lw=0.8, ls="--")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("mean width of the range", fontsize=9)
        ax.set_xlim(0, 0.85)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("share of checks whose range held the truth", fontsize=9)
    axes[0].set_ylim(0.5, 1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], marker="o", ls="", color="#444", label="closing check, realized"),
        plt.Line2D([], [], marker="s", ls="", mfc="none", color="#444", label="check at click 51, simulated"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Coverage against width, per construction (dashed: 90%)", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(docs / "figures" / "coverage_width.png", dpi=130)
    plt.close(fig)

    # Per check: the shipped range and the featured ones against the truth, on one simulated
    # check a session at click 51 (its first draw): the regime a user's check sees.
    feat = [m for m in featured.split(",") if m]
    one = sim[sim["draw"] == 0]
    fig, axes = plt.subplots(len(PRESETS), 1, figsize=(13, 9), sharex=True)
    rng = np.random.default_rng(4483)
    for ax, (beta, name) in zip(axes, PRESETS):
        base = one[np.isclose(one["beta"], beta) & (one["method"] == "cp_union")]
        cells = base.sample(n=min(80, len(base)), random_state=int(rng.integers(1 << 30)))
        cells = cells.sort_values(["truth", "lo"]).reset_index(drop=True)
        x = np.arange(len(cells))
        sub = one[np.isclose(one["beta"], beta)].set_index(["run", "cell", "method"])
        offsets = np.linspace(-0.25, 0.25, 1 + len(feat))
        for off, m in zip(offsets, ["cp_union", *feat]):
            lo = [sub.loc[(r.run, r.cell, m), "lo"] for r in cells.itertuples()]
            hi = [sub.loc[(r.run, r.cell, m), "hi"] for r in cells.itertuples()]
            ax.vlines(x + off, lo, hi, color=METHOD_COLORS[m], lw=2.2, label=METHOD_NAMES[m] if beta == 0.25 else None)
        ax.scatter(x, cells["truth"], color="black", s=10, zorder=3, label="truth" if beta == 0.25 else None)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"{name}\nshare right", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[-1].set_xlabel(
        "80 sessions per preset, drawn at random: one simulated check at click 51 each, sorted by the truth", fontsize=9
    )
    fig.legend(loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Each check's range against its truth", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(docs / "figures" / "ranges_per_check.png", dpi=130)
    plt.close(fig)

    summ = pd.read_csv(out / "summary.csv")
    num = summ.select_dtypes("number").columns
    summ[num] = summ[num].round(3)
    summ.to_csv(docs / "tables" / "summary.csv", index=False)

    # What the state line would read for the outcomes a check most often has.
    shown = ["cp_union", "hg_union", "hg_plain", "bayes90", "bayes90j", "wald90"]
    rows = []
    for label, strata in (
        *((f"top 8, {r} of 5 right", [(8, 5, r, 0)]) for r in range(6)),
        ("top 16, 5 + 5 of 10 right", [(8, 5, 5, 0), (8, 5, 5, 0)]),
        ("top 16, 3 + 1 of 10 right", [(8, 5, 3, 0), (8, 5, 1, 0)]),
        ("top 32, 5 + 4 + 3 of 15 right", [(8, 5, 5, 0), (8, 5, 4, 0), (16, 5, 3, 0)]),
        ("top 32, 2 + 1 + 0 of 15 right", [(8, 5, 2, 0), (8, 5, 1, 0), (16, 5, 0, 0)]),
        (
            "top 128, 5 + 5 + 4 + 2 + 1 of 25 right",
            [(8, 5, 5, 0), (8, 5, 5, 0), (16, 5, 4, 0), (32, 5, 2, 0), (64, 5, 1, 0)],
        ),
    ):
        b = method_bounds(strata)
        rows.append({"outcome": label, **{m: f"{round(b[m][0] * 100)}–{round(b[m][1] * 100)}%" for m in shown}})
    pd.DataFrame(rows).to_csv(docs / "tables" / "what_users_read.csv", index=False)

    # Literal checks: random ones per preset, and every featured construction's misses at beta 1.
    keep = ["run", "cell", "category", "seed", "beta", "t", "k", "labelled", "truth"]
    wide = real.pivot_table(index=keep, columns="method", values=["lo", "hi"]).reset_index()
    wide.columns = [c if isinstance(c, str) else (c[0] if not c[1] else f"{c[1]}_{c[0]}") for c in wide.columns]
    strata_of = pd.read_csv(out / "realized.csv.gz").set_index(["run", "cell"])["strata"]
    wide["bands N:picks:right:truly_right"] = [strata_of.loc[(r.run, r.cell)] for r in wide.itertuples()]
    picks = [
        wide[np.isclose(wide["beta"], beta)].sample(n=4, random_state=4483 + i) for i, (beta, _) in enumerate(PRESETS)
    ]
    for m in feat:
        miss = wide[(wide[f"{m}_lo"] > wide["truth"] + 1e-9) | (wide[f"{m}_hi"] < wide["truth"] - 1e-9)]
        picks.append(miss[np.isclose(miss["beta"], 1.0)].head(4))
    ex = pd.concat(picks).drop_duplicates(["run", "cell"])
    cols = [*keep, "bands N:picks:right:truly_right"] + [f"{m}_{e}" for m in shown for e in ("lo", "hi")]
    ex[cols].round(3).to_csv(docs / "tables" / "examples.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("realized")
    a.add_argument("runs", nargs="+", help="label=RUN_DIR")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    s = sub.add_parser("simulate")
    s.add_argument("runs", nargs="+", help="label=RUN_DIR")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    s.add_argument("--draws", type=int, default=20)
    s.add_argument("--npos-scale", type=float, default=1.0)
    s.add_argument("--tag", default="")
    r = sub.add_parser("report")
    r.add_argument("out", type=Path)
    r.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    f = sub.add_parser("figures")
    f.add_argument("out", type=Path)
    f.add_argument("--docs", type=Path, required=True)
    f.add_argument("--featured", default="hg_union,bayes90")
    args = ap.parse_args(argv)
    if args.cmd == "realized":
        run_realized(args.runs, args.out, args.jobs)
    elif args.cmd == "simulate":
        run_simulated(args.runs, args.out, args.jobs, args.draws, args.npos_scale, args.tag)
    elif args.cmd == "figures":
        run_figures(args.out, args.docs, args.featured)
    else:
        run_report(args.out, args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
