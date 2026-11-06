"""#4876: re-price #4736's typed-query hold at every count from 0 to 60, on one build, photos and app screens.

A typed-query session shows the typed query's own set on Find until its labels number the hold (30, on a
single-vector space), and the trained head's set from then on. The hold changes what Find returns, never what
Autopilot asks: the harness trains the head and picks with it at every vote either way, and a held click
writes the typed query's set as its untagged row while the head's own rows are kept, tagged ``hold:head``
(``voting_iterations.QUERY_HOLD_HEAD_TAG``). So one session set run with the shipped hold prices every count up
to it exactly: hold *h* shows the head's row at each vote whose labels number *h* or more, and the query's set
below that. Hold 0 is the head from the label quota (the arm before #4736, ``CALIB_QUERY_HOLD=0``); a count under
the quota is the same as 0, since the quota's own query tier (#4758) is not the hold. A count past the shipped one
(#5556) needs no head row: the query's set depends on the text sort, the withheld half and the line's beta only,
never the votes, so it is one value a session, and hold *h* > 30 shows it at every vote whose labels number fewer
than *h*. Each session's stop (the first ``done`` click, as ``stopping.stopping_points`` reads it) is read too: the
hold cannot move it, since the stop rule reads only the head's line, and ``check`` confirms that.

Each vote scores F-beta at the session's preset of the set Find returns on the withheld half; a vote with no
row carries the last one (a check round writes one row for its votes) and a vote before the first row scores 0,
as ``apptitude/analyze.py`` reads them. The session mean is over votes 1-150 (the owner's number, #4668) and the
short one over 1-50.

    python hold_4876.py extract --cells <run>/results/cells --out X.npz [--embedder siglip]
    python hold_4876.py price --arm photos:b1=X.npz --arm screens:b1=Y.npz ... --out <dir>
    python hold_4876.py figures --arm photos:b1=X.npz --arm screens:b1=Y.npz ... --out <dir>
    python hold_4876.py stop --arm photos:b1=X.npz ... --out <dir>     # #5556: at the stop, and the path to it
    python hold_4876.py check --run <hold-h run>/results/cells --control <shipped run>/results/cells --hold H

``check`` is the proof that the offline price is the run's: for every cell the hold-*h* run shares with the
control (same grid index), the picks and the stop click must be identical and the run's curve must equal the
control's priced at *h*.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))
import _cells_io  # noqa: E402
from stopping import STOPPING_PHASE, km_median  # noqa: E402

HORIZON = 150
SHIPPED = 30
MAX_HOLD = 60
HOLDS = np.arange(MAX_HOLD + 1)
HOLD_TAG = "hold:head"
SPANS = {"1-150": (1, 150), "1-50": (1, 50), "1-25": (1, 25), "51-150": (51, 150)}
COLS = [
    "dataset", "category", "embedder", "seed", "t", "phase", "beta", "fbeta", "precision", "recall", "n_good",
    "n_bad", "app_trained", "detector_tier", "n_test_pos", "n_test_neg", "pool_variant", "gmm_variant", "schedule",
]  # fmt: skip
#: Picks columns that must match between a hold-h run and the control (everything but bookkeeping).
PICK_KEYS = ["t", "phase", "picked_id", "picked_label", "n_good", "n_bad"]


def fbeta_pr(p: np.ndarray, r: np.ndarray, beta: float) -> np.ndarray:
    """F-beta from precision and recall; an empty set (precision NaN) scores 0."""
    p = np.nan_to_num(np.asarray(p, dtype=float))
    r = np.nan_to_num(np.asarray(r, dtype=float))
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b2 * p + r > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)


def carry(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Values at votes 1..HORIZON, each the last row at or before it; 0 before the first row."""
    out = np.zeros(HORIZON)
    order = np.argsort(t, kind="stable")
    t, v = np.asarray(t, dtype=float)[order], np.asarray(v, dtype=float)[order]
    idx = np.searchsorted(t, np.arange(1, HORIZON + 1), side="right") - 1
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    return out


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().isin(("", "nan", "None"))


def read_cell(args: tuple[str, str | None]) -> dict | None:
    """One cell file -> its curve at every hold count, or ``None`` for an empty cell / another embedder."""
    path, embedder = args
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    if df.empty or (embedder and str(df["embedder"].iloc[0]) != embedder):
        return None
    df = df[df["t"] <= HORIZON]
    check = _cells_io.check_rows(df)
    base = _cells_io._base_rows(df)
    ordinary = base[~_cells_io.check_rows(base)].groupby("t", as_index=False).last()
    beta = float(base["beta"].dropna().iloc[0]) if base["beta"].notna().any() else 1.0
    pv = df["pool_variant"].fillna("").astype(str).str.strip()
    held_all = df[
        (df["gmm_variant"].fillna("").astype(str) == HOLD_TAG)
        & _blank(df["schedule"])
        & pv.isin(_cells_io.BASE_POOL_VARIANTS)
        & ~check
    ]
    held = held_all.groupby("t", as_index=False).last()
    t = ordinary["t"].to_numpy(dtype=float)
    shown = ordinary["fbeta"].fillna(0.0).to_numpy(dtype=float)
    labels = (ordinary["n_good"] + ordinary["n_bad"]).to_numpy(dtype=float)
    is_query = (ordinary["detector_tier"].astype(str) == "query").to_numpy()
    query_vals = shown[is_query]
    query_f = float(query_vals[0]) if len(query_vals) else np.nan
    trained = ordinary["app_trained"].fillna(0).astype(bool).to_numpy()
    lab_at = dict(zip(ordinary["t"].astype(int), labels))
    head_t = held["t"].astype(int).to_numpy()
    head_f = fbeta_pr(held["precision"].to_numpy(), held["recall"].to_numpy(), beta)
    head_lab = np.array([lab_at.get(int(x), np.nan) for x in head_t])
    pos = {int(x): i for i, x in enumerate(t.astype(int))}
    curves = np.empty((len(HOLDS), HORIZON), dtype=np.float32)
    for h in HOLDS:
        f = shown.copy()
        for x, ff, lab in zip(head_t, head_f, head_lab):
            if lab >= h and int(x) in pos:
                f[pos[int(x)]] = ff
        if h > SHIPPED and len(query_vals):
            # Past the shipped count the query's set is shown where the head's was: one value a session.
            f[(labels >= SHIPPED) & (labels < h) & ~is_query] = query_f
        curves[h] = carry(t, f)
    done = ordinary["phase"].fillna("").astype(str).str.strip().to_numpy() == STOPPING_PHASE
    t_stop = float(t[np.argmax(done)]) if done.any() else np.nan
    first = base.iloc[0]
    n_pos, n_neg = float(first["n_test_pos"]), float(first["n_test_neg"])
    return {
        "dataset": str(first["dataset"]),
        "category": str(first["category"]),
        "seed": int(first["seed"]),
        "beta": beta,
        "prevalence": n_pos / (n_pos + n_neg) if n_pos + n_neg > 0 else np.nan,
        "query_f": query_f,
        "n_held": len(held),
        "t_stop": t_stop,
        "labels_at_stop": float(labels[np.argmax(done)]) if done.any() else np.nan,
        # Invariants the price rests on: one head row a held vote, and only below the shipped count.
        "held_dupes": int(len(held_all) - len(held)),
        "held_past_hold": int(np.nansum(head_lab >= SHIPPED)),
        "held_unmatched": int(sum(int(x) not in pos for x in head_t)),
        # #5556: the query's set is one value a session (the price past the shipped count rests on it).
        "query_spread": float(np.ptp(query_vals)) if len(query_vals) else 0.0,
        "query_past_hold": int(np.sum(is_query & (labels >= SHIPPED) & trained)),
        "tiers_past_hold": ",".join(sorted(set(ordinary.loc[labels >= SHIPPED, "detector_tier"].astype(str)))),
        "first_held": float(head_t.min()) if len(head_t) else np.nan,
        "last_held": float(head_t.max()) if len(head_t) else np.nan,
        "labels": carry(t, labels).astype(np.float32),
        "shown": carry(t, shown).astype(np.float32),
        "head": carry(head_t, head_f).astype(np.float32) if len(head_t) else np.full(HORIZON, np.nan, np.float32),
        "curves": curves,
    }


def extract(a: argparse.Namespace) -> None:
    files = [str(f) for f in _cells_io.main_frame_files(Path(a.cells))]
    with ProcessPoolExecutor(a.workers) as ex:
        got = list(ex.map(read_cell, [(f, a.embedder) for f in files], chunksize=8))
    runs = [g for g in got if g is not None]
    print(f"{a.cells}: {len(files)} cells, {len(runs)} read ({a.embedder or 'any embedder'})")
    if not runs:
        raise SystemExit("no runs")
    bad = {k: sum(r[k] for r in runs) for k in ("held_dupes", "held_past_hold", "held_unmatched", "query_past_hold")}
    print(f"invariants (all should be 0): {bad}; runs with no held vote: {sum(r['n_held'] == 0 for r in runs)}")
    spread = max(r["query_spread"] for r in runs)
    print(
        f"query's set one value a session: largest spread {spread:.2e}; no query row: "
        f"{sum(np.isnan(r['query_f']) for r in runs)}"
    )
    tiers = pd.Series([r["tiers_past_hold"] for r in runs]).value_counts()
    print("tiers shown at 30+ labels (runs):", dict(tiers))
    print(f"stopped: {sum(np.isfinite(r['t_stop']) for r in runs)} of {len(runs)}")
    if a.stops:
        st = pd.read_csv(a.stops, usecols=["category", "seed", "stopped", "t_stop"])
        got = pd.DataFrame(
            {
                "category": [r["category"] for r in runs],
                "seed": [r["seed"] for r in runs],
                "mine": [r["t_stop"] for r in runs],
            }
        )
        m = got.merge(st, on=["category", "seed"], how="outer", indicator=True)
        same = (m["mine"] == m["t_stop"]) | (m["mine"].isna() & m["t_stop"].isna())
        print(
            f"stop vs {a.stops}: {int(same.sum())} of {len(m)} sessions agree "
            f"(unmatched {int((m['_merge'] != 'both').sum())})"
        )
        if not same.all():
            raise SystemExit("the cells' stop disagrees with stops.csv")
    scal = ("dataset", "category", "seed", "beta", "prevalence", "query_f", "n_held", "first_held", "last_held",
            "t_stop", "labels_at_stop")  # fmt: skip
    np.savez_compressed(
        a.out,
        **{k: np.array([r[k] for r in runs]) for k in scal},
        **{k: np.stack([r[k] for r in runs]) for k in ("labels", "shown", "head", "curves")},
        source=str(a.cells),
    )
    print(f"wrote {a.out}")


def _span_means(curves: np.ndarray) -> dict[str, np.ndarray]:
    """Per run and hold, the session mean over each span: ``{span: (runs, holds)}``."""
    return {k: curves[:, :, lo - 1 : hi].mean(axis=2) for k, (lo, hi) in SPANS.items()}


def _bounds(curves: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """What a judge the app could run might reach, per run and span, read with hindsight.

    ``vote``: at every vote the better of the two sets (the query's and the head's), the most any rule that
    switches back and forth could reach. ``run``: per run the better of no hold and the shipped hold, the
    most a rule that picks one of the two per session could reach.
    """
    out: dict[str, dict[str, np.ndarray]] = {"vote": {}, "run": {}}
    per_vote = np.maximum(curves[:, 0, :], curves[:, SHIPPED, :])
    for k, (lo, hi) in SPANS.items():
        out["vote"][k] = per_vote[:, lo - 1 : hi].mean(axis=1)
        out["run"][k] = np.maximum(curves[:, 0, lo - 1 : hi].mean(axis=1), curves[:, SHIPPED, lo - 1 : hi].mean(axis=1))
    return out


def clustered_se(d: np.ndarray, cluster: np.ndarray) -> float:
    """SE of the mean of *d*, clustered (runs of one class share a cluster), with the G/(G-1) correction."""
    resid = d - d.mean()
    sums = pd.Series(resid).groupby(np.asarray(cluster)).sum().to_numpy()
    g = len(sums)
    return float(np.sqrt(g / (g - 1) * (sums**2).sum()) / len(d)) if g > 1 else float("nan")


def _cls(category: np.ndarray) -> np.ndarray:
    return np.array([str(c).split("@")[0] for c in category])


def _load(spec: str) -> tuple[str, str, dict]:
    key, _, path = spec.partition("=")
    dataset, _, preset = key.partition(":")
    z = dict(np.load(path, allow_pickle=False))
    return dataset, preset, z


def price(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, per_run, curve_rows = [], [], []
    arms = [_load(s) for s in a.arm]
    for dataset, preset, z in arms:
        m = _span_means(z["curves"])
        bounds = _bounds(z["curves"])
        n = len(z["seed"])
        cls = _cls(z["category"])
        seeds = np.unique(z["seed"])
        for span, v in m.items():
            d = v - v[:, [SHIPPED]]
            by_seed = np.stack([d[z["seed"] == s].mean(axis=0) for s in seeds])
            cols = [(int(h), v[:, h], d[:, h], by_seed[:, h]) for h in HOLDS]
            for name, b in bounds.items():
                db = b[span] - v[:, SHIPPED]
                cols.append((f"best_per_{name}", b[span], db, np.array([db[z["seed"] == s].mean() for s in seeds])))
            for h, val, dh, bs in cols:
                rows.append(
                    {
                        "dataset": dataset,
                        "preset": preset,
                        "span": span,
                        "hold": h,
                        "runs": n,
                        "mean": float(val.mean()),
                        "minus_shipped": float(dh.mean()),
                        "se": clustered_se(dh, cls),
                        "seed_min": float(bs.min()),
                        "seed_max": float(bs.max()),
                        "better": float((dh > 1e-9).mean()),
                        "worse": float((dh < -1e-9).mean()),
                    }
                )
        for i in range(n):
            per_run.append(
                {
                    "dataset": dataset,
                    "preset": preset,
                    "category": z["category"][i],
                    "seed": int(z["seed"][i]),
                    "prevalence": float(z["prevalence"][i]),
                    "query_f": float(z["query_f"][i]),
                    "first_held": float(z["first_held"][i]),
                    **{f"m150_h{h}": float(m["1-150"][i, h]) for h in HOLDS},
                    **{f"m50_h{h}": float(m["1-50"][i, h]) for h in HOLDS},
                }
            )
        for h in (0, 10, 15, 20, SHIPPED):
            mean_curve = z["curves"][:, h, :].mean(axis=0)
            curve_rows += [
                {"dataset": dataset, "preset": preset, "hold": h, "t": t + 1, "fbeta": float(mean_curve[t])}
                for t in range(HORIZON)
            ]
        head = np.nanmean(z["head"], axis=0)
        curve_rows += [
            {"dataset": dataset, "preset": preset, "hold": "head", "t": t + 1, "fbeta": float(head[t])}
            for t in range(HORIZON)
        ]
    holds = pd.DataFrame(rows)
    runs = pd.DataFrame(per_run)
    holds.to_csv(out / "holds.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "curves.csv", index=False)
    (out / "summary.md").write_text(summary(holds, runs))
    print((out / "summary.md").read_text())


def _pm(row: pd.Series) -> str:
    return f"{row['minus_shipped']:+.3f} ± {row['se']:.3f}"


def summary(holds: pd.DataFrame, runs: pd.DataFrame) -> str:
    show = ["0", "8", "10", "15", "20", "25"]
    holds = holds.assign(hold=holds["hold"].astype(str))
    lines = [
        "# #4736's typed-query hold, re-priced (#4876)",
        "",
        "Each cell: session mean at hold *h* minus the shipped hold (30), mean ± SE clustered on the class; "
        "positive means *h* beats the shipped hold. Hold 0 is the head from the label quota. The two last "
        "columns are hindsight bounds: the better set at every vote, and the better of h=0 and h=30 per run.",
        "",
    ]
    for span in ("1-150", "1-50", "1-25"):
        lines += [
            f"## Votes {span}",
            "",
            "| dataset | preset | runs | at 30 | "
            + " | ".join(f"h={h}" for h in show)
            + " | best h | seeds (h=0) | better set a vote | better of 0/30 a run |",
            "|---|---|---|---|" + "---|" * len(show) + "---|---|---|---|",
        ]
        for (ds, pr), g in holds[holds["span"] == span].groupby(["dataset", "preset"], sort=False):
            g = g.set_index("hold")
            ints = g[g.index.str.isdigit()]
            best = ints["mean"].astype(float).idxmax()
            lines.append(
                f"| {ds} | {pr} | {int(g['runs'].iloc[0])} | {g.loc[str(SHIPPED), 'mean']:.3f} | "
                + " | ".join(_pm(g.loc[h]) for h in show)
                + f" | {best} ({g.loc[best, 'minus_shipped']:+.3f}) | "
                f"{g.loc['0', 'seed_min']:+.3f} to {g.loc['0', 'seed_max']:+.3f} | "
                f"{_pm(g.loc['best_per_vote'])} | {_pm(g.loc['best_per_run'])} |"
            )
        lines.append("")
    lines += [
        "## Runs better / worse with no hold (h=0 vs 30, votes 1-50)",
        "",
        "| dataset | preset | better | tie | worse |",
        "|---|---|---|---|---|",
    ]
    for (ds, pr), g in holds[(holds["span"] == "1-50") & (holds["hold"] == "0")].groupby(
        ["dataset", "preset"], sort=False
    ):
        r = g.iloc[0]
        lines.append(f"| {ds} | {pr} | {r['better']:.0%} | {1 - r['better'] - r['worse']:.0%} | {r['worse']:.0%} |")
    lines += ["", "## By class (h=0 minus 30, votes 1-50, mean over seeds and bands)", ""]
    runs = runs.assign(cls=runs["category"].str.split("@").str[0], d50=runs["m50_h0"] - runs[f"m50_h{SHIPPED}"])
    for ds, g in runs.groupby("dataset", sort=False):
        t = g.pivot_table(index="cls", columns="preset", values="d50", aggfunc="mean")
        t = t.reindex(t.mean(axis=1).sort_values(ascending=False).index)
        lines += [f"### {ds}", "", "| class | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
        lines += [f"| {c} | " + " | ".join(f"{v:+.3f}" for v in t.loc[c]) + " |" for c in t.index]
        lines.append("")
    return "\n".join(lines)


#: Categorical slots (validated: light surface, adjacent CVD ΔE >= 9): blue, orange, aqua.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
PRESET_TITLE = {"b025": "beta ¼", "b1": "beta 1", "b4": "beta 4"}


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def figures(a: argparse.Namespace) -> None:
    """``fig_hold.png`` (the gain against the hold count) and ``fig_by_vote.png`` (the set on screen per vote)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = {(d, p): z for d, p, z in (_load(s) for s in a.arm)}
    presets = [p for p in PRESET_TITLE if any(k[1] == p for k in arms)]
    names = {"photos": ("photos (COCO Better)", BLUE), "screens": ("app screens (Apptitude, 1%)", ORANGE)}

    fig, axes = plt.subplots(2, len(presets), figsize=(10, 5.6), sharex=True, squeeze=False)
    for col, pr in enumerate(presets):
        for row, span in enumerate(("1-150", "1-50")):
            ax = axes[row, col]
            _style(ax)
            ax.axhline(0, color=MUTED, linewidth=1)
            for ds, (label, color) in names.items():
                z = arms.get((ds, pr))
                if z is None:
                    continue
                v = _span_means(z["curves"])[span]
                d = v - v[:, [SHIPPED]]
                cls = _cls(z["category"])
                m = d.mean(axis=0)
                se = np.array([clustered_se(d[:, h], cls) for h in HOLDS])
                ax.fill_between(HOLDS, m - 2 * se, m + 2 * se, color=color, alpha=0.15, linewidth=0)
                ax.plot(HOLDS, m, color=color, linewidth=2, label=label)
                ax.annotate(f"{m[0]:+.3f}", (0, m[0]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK)
            if row == 0:
                ax.set_title(PRESET_TITLE[pr], fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(f"gain over the shipped hold\nsession mean, votes {span}", fontsize=8, color=INK)
            if row == 1:
                ax.set_xlabel("hold (labels before the head's set shows)", fontsize=8, color=INK)
            ax.set_xticks([0, 7, 10, 15, 20, 25, 30])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_hold.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(presets), figsize=(10, 5.6), sharex=True, squeeze=False)
    t = np.arange(1, HORIZON + 1)
    span = slice(0, 50)
    for col, pr in enumerate(presets):
        for row, ds in enumerate(("photos", "screens")):
            ax = axes[row, col]
            _style(ax)
            z = arms.get((ds, pr))
            if z is None:
                continue
            c = z["curves"]
            lines = (
                ("shipped (hold 30)", c[:, SHIPPED, :].mean(axis=0), BLUE, "-"),
                ("no hold", c[:, 0, :].mean(axis=0), ORANGE, "-"),
                ("better set, hindsight", np.maximum(c[:, 0, :], c[:, SHIPPED, :]).mean(axis=0), AQUA, "--"),
            )
            for label, y, color, ls in lines:
                ax.plot(t[span], y[span], color=color, linewidth=2, linestyle=ls, label=label)
            if row == 0:
                ax.set_title(PRESET_TITLE[pr], fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(f"{names[ds][0]}\nmean F-beta on screen", fontsize=8, color=INK)
            if row == 1:
                ax.set_xlabel("vote", fontsize=8, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_by_vote.png", dpi=150)
    plt.close(fig)

    # Per seed: no hold minus the shipped hold, one dot a seed, so the spread is read off runs, not a mean.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), squeeze=False)
    for col, span in enumerate(("1-150", "1-50")):
        ax = axes[0, col]
        _style(ax)
        ax.axhline(0, color=MUTED, linewidth=1)
        for k, pr in enumerate(presets):
            for j, (ds, (label, color)) in enumerate(names.items()):
                z = arms.get((ds, pr))
                if z is None:
                    continue
                v = _span_means(z["curves"])[span]
                d = v[:, 0] - v[:, SHIPPED]
                per_seed = [d[z["seed"] == sd].mean() for sd in np.unique(z["seed"])]
                x = k + (j - 0.5) * 0.3
                ax.scatter(
                    np.full(len(per_seed), x),
                    per_seed,
                    s=36,
                    color=color,
                    edgecolor="white",
                    linewidth=1,
                    zorder=3,
                    label=label if k == 0 else None,
                )
                ax.hlines(d.mean(), x - 0.1, x + 0.1, color=INK, linewidth=2, zorder=4)
        ax.set_xticks(range(len(presets)), [PRESET_TITLE[p] for p in presets])
        ax.set_title(f"votes {span}", fontsize=10, color=INK)
        if col == 0:
            ax.set_ylabel("no hold minus shipped hold\nsession mean, one dot a seed", fontsize=8, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "fig_seeds.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out}/fig_hold.png, {out}/fig_by_vote.png, {out}/fig_seeds.png")


#: #5556: the counts the report names side by side, and the shades its by-vote figure draws them in (light to dark).
NAMED = (20, 25, 30, 35, 40)
NAMED_SHADES = ("#9ec5ef", "#5d9de3", BLUE, "#1d5aa3", "#0f3566")
BAND_COLORS = {"small": BLUE, "medium": ORANGE, "large": AQUA}
READ_TITLE = {"at_stop": "the objective where the app says stop", "path": "the mean objective on the way to the stop"}


def _band(category: np.ndarray) -> np.ndarray:
    return np.array([str(c).split("@")[1] if "@" in str(c) else "" for c in category])


def _stop_reads(z: dict) -> dict[str, np.ndarray]:
    """Per session and count, ``{read: (runs, holds)}``: the value at the stop, and the mean over votes 1 to it.

    A session never told to stop is read at its last click (``HORIZON``), so its path runs over every vote.
    """
    curves = z["curves"]
    stopped = np.isfinite(z["t_stop"])
    t_end = np.where(stopped, z["t_stop"], HORIZON).astype(int)
    n = len(t_end)
    at_stop = curves[np.arange(n), :, t_end - 1].astype(float)
    mask = (np.arange(HORIZON)[None, :] < t_end[:, None]).astype(np.float32)
    path = np.einsum("nht,nt->nh", curves, mask, dtype=np.float64) / t_end[:, None]
    return {"at_stop": at_stop, "path": path}


def _labels_end(z: dict) -> np.ndarray:
    """Labels where each session is read: at its stop, or at its last click if it never stopped."""
    return np.where(np.isfinite(z["t_stop"]), z["labels_at_stop"], z["labels"][:, -1])


def _decide(d: dict[str, np.ndarray], se: dict[str, np.ndarray]) -> tuple[int, str]:
    """The issue's rule (owner, 2026-11-04), on per-count gains over 30 and their clustered SEs, read literally.

    The stop click is the same at every count (read, and confirmed by ``check``), so no count stops earlier: the
    stop's part is the objective there. The count best at the stop wins if it beats 30 there by 2 SE; a count 2 SE
    worse there is out. Otherwise, among the counts that tie 30 at the stop, the one with the best path wins if it
    beats 30's path by 2 SE. Otherwise 30. A count the rule does not pick is never chosen for clearing 2 SE alone.
    """
    best = int(np.argmax(d["at_stop"]))
    if d["at_stop"][best] > 2 * se["at_stop"][best]:
        return best, "the stop: the best count there beats 30's objective by 2 SE"
    tie = [h for h in HOLDS if d["at_stop"][h] >= -2 * se["at_stop"][h]]
    best = int(max(tie, key=lambda k: d["path"][k]))
    if d["path"][best] > 2 * se["path"][best]:
        return (
            best,
            f"the path: ties 30 at the stop, and its path beats 30's by 2 SE ({d['path'][best]:+.4f} ± {se['path'][best]:.4f})",
        )
    return SHIPPED, (
        f"keep 30: the best path ({best}, {d['path'][best]:+.4f} ± {se['path'][best]:.4f}) does not beat 30's by 2 SE"
    )


def _scope_rows(reads, cls, lab_end, sel) -> tuple[list[dict], dict, dict]:
    rows, d_all, se_all = [], {}, {}
    for read, v in reads.items():
        v = v[sel]
        c = cls[sel]
        dd = v - v[:, [SHIPPED]]
        d_all[read] = dd.mean(axis=0)
        se_all[read] = np.array([clustered_se(dd[:, h], c) for h in HOLDS])
        best = int(np.argmax(d_all[read]))
        # Counts within 2 SE of the best, the SE of each count's difference to the best, paired within session.
        near = [
            int(h) for h in HOLDS if (v[:, best] - v[:, h]).mean() <= 2 * clustered_se(v[:, best] - v[:, h], c) + 1e-12
        ]
        for h in HOLDS:
            lo, hi = min(h, SHIPPED), max(h, SHIPPED)
            moved = (lab_end[sel] >= lo) & (lab_end[sel] < hi)
            rows.append(
                {
                    "read": read,
                    "hold": int(h),
                    "runs": int(sel.sum()),
                    "mean": float(v[:, h].mean()),
                    "minus_30": float(d_all[read][h]),
                    "se": float(se_all[read][h]),
                    "share_below": float((lab_end[sel] < h).mean()),
                    "share_moved": float(moved.mean()),
                    "minus_30_moved": float(dd[moved, h].mean()) if moved.any() else np.nan,
                    "better": float((dd[:, h] > 1e-9).mean()),
                    "worse": float((dd[:, h] < -1e-9).mean()),
                    "best": best,
                    "near_lo": min(near),
                    "near_hi": max(near),
                }
            )
    return rows, d_all, se_all


def _gap(z: dict, sel: np.ndarray, cls: np.ndarray) -> pd.DataFrame:
    """The head's set minus the query's at each label count, paired within a session.

    The head's set is hold 0's (``hold:head`` below 30 labels, the untagged row from 30), read at the last vote whose
    labels number at most *L*. ``before_stop`` keeps only the sessions not yet told to stop at *L*.
    """
    labels, head, q = z["labels"][sel], z["curves"][sel, 0, :].astype(float), z["query_f"][sel].astype(float)
    lab_end = _labels_end(z)[sel]
    c = cls[sel]
    out = []
    for L in range(1, HORIZON + 1):
        idx = (labels <= L).sum(axis=1) - 1  # labels never fall, so this is the last vote at or below L
        ok = (idx >= 0) & np.isfinite(q) & (labels[:, -1] >= L)
        g = np.where(ok, head[np.arange(len(idx)), np.clip(idx, 0, None)] - q, np.nan)
        pre = ok & (lab_end > L)
        out.append(
            {
                "labels": L,
                "runs": int(ok.sum()),
                "gap": float(np.nanmean(g)) if ok.any() else np.nan,
                "se": clustered_se(g[ok], c[ok]) if ok.sum() > 1 else np.nan,
                "share_head_ahead": float((g[ok] > 1e-9).mean()) if ok.any() else np.nan,
                "runs_before_stop": int(pre.sum()),
                "gap_before_stop": float(np.nanmean(g[pre])) if pre.any() else np.nan,
            }
        )
    return pd.DataFrame(out)


def _crossing(g: pd.DataFrame) -> dict:
    """Where the mean gap turns positive for good, and the stretch of label counts whose 2 SE band covers zero."""
    gap, se, lab = g["gap"].to_numpy(), g["se"].to_numpy(), g["labels"].to_numpy()
    neg = np.where(~(gap > 0))[0]
    cross = int(lab[neg[-1] + 1]) if len(neg) and neg[-1] + 1 < len(lab) else (int(lab[0]) if not len(neg) else None)
    flat = lab[np.abs(gap) <= 2 * se]
    return {
        "crosses_at": cross,
        "near_zero_from": int(flat.min()) if len(flat) else "",
        "near_zero_to": int(flat.max()) if len(flat) else "",
        "near_zero_counts": int(len(flat)),
    }


def stop_price(a: argparse.Namespace) -> None:
    """#5556: every count from 0 to 60, read where the app says stop and on the path there."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [_load(s) for s in a.arm]
    rows, km_rows, gaps, decisions, cross_rows, pooled = [], [], [], [], [], {}
    for dataset, preset, z in arms:
        reads = _stop_reads(z)
        cls, band, lab_end = _cls(z["category"]), _band(z["category"]), _labels_end(z)
        stopped = np.isfinite(z["t_stop"])
        pooled[(dataset, preset)] = (reads, cls, z)
        for scope in ["all", *sorted(set(band) - {""})]:
            sel = np.ones(len(cls), bool) if scope == "all" else band == scope
            key = {"dataset": dataset, "preset": preset, "scope": scope}
            r, d, se = _scope_rows(reads, cls, lab_end, sel)
            rows += [{**key, **x} for x in r]
            h, why = _decide(d, se)
            decisions.append({**key, "count": h, "why": why})
            ts = z["t_stop"][sel]
            fired = ts[np.isfinite(ts)]
            km_rows.append(
                {
                    **key,
                    "runs": int(sel.sum()),
                    "fire_rate": float(stopped[sel].mean()),
                    "km_t_stop": km_median(fired, np.full(int((~stopped[sel]).sum()), float(HORIZON))),
                    "q25": float(np.percentile(fired, 25)),
                    "q50": float(np.percentile(fired, 50)),
                    "q75": float(np.percentile(fired, 75)),
                    "labels_at_stop_median": float(np.nanmedian(z["labels_at_stop"][sel])),
                    "labels_at_stop_min": float(np.nanmin(z["labels_at_stop"][sel])),
                    **{f"share_stop_below_{n}": float((lab_end[sel] < n).mean()) for n in NAMED},
                }
            )
            g = _gap(z, sel, cls)
            gaps.append(g.assign(**key))
            cross_rows.append({**key, **_crossing(g)})
    # One count for every preset against one a beta: pooled over presets, a class one cluster across them.
    common = []
    for scope in ["all"]:
        for read in ("at_stop", "path"):
            dd = np.concatenate([v[0][read] - v[0][read][:, [SHIPPED]] for v in pooled.values()])
            c = np.concatenate([v[1] for v in pooled.values()])
            m = dd.mean(axis=0)
            se = np.array([clustered_se(dd[:, h], c) for h in HOLDS])
            best_one = int(np.argmax(m))
            per_beta = np.concatenate(
                [v[0][read][:, int(np.argmax((v[0][read] - v[0][read][:, [SHIPPED]]).mean(axis=0)))] - v[0][read][:, best_one]
                 for v in pooled.values()]
            )  # fmt: skip
            common.append(
                {
                    "scope": scope,
                    "read": read,
                    "one_count": best_one,
                    "one_count_minus_30": float(m[best_one]),
                    "one_count_se": float(se[best_one]),
                    "per_beta_minus_one_count": float(per_beta.mean()),
                    "per_beta_minus_one_count_se": clustered_se(per_beta, c),
                    **{f"minus_30_h{n}": float(m[n]) for n in NAMED},
                }
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "stop_holds.csv", index=False, float_format="%.6f")
    km = pd.DataFrame(km_rows)
    km.to_csv(out / "stop_clicks.csv", index=False, float_format="%.4f")
    gap = pd.concat(gaps, ignore_index=True)
    gap.to_csv(out / "gap.csv", index=False, float_format="%.6f")
    dec = pd.DataFrame(decisions)
    dec.to_csv(out / "decision.csv", index=False)
    com = pd.DataFrame(common)
    com.to_csv(out / "one_count.csv", index=False, float_format="%.6f")
    cr = pd.DataFrame(cross_rows)
    cr.to_csv(out / "crossing.csv", index=False)
    (out / "summary_stop.md").write_text(stop_summary(tab, km, dec, com, cr))
    print((out / "summary_stop.md").read_text())
    if not a.no_figures:
        stop_figures(arms, tab, gap, out)


def stop_summary(tab, km, dec, com, cr) -> str:
    def cell(g, h):
        r = g[g["hold"] == h].iloc[0]
        return f"{r['minus_30']:+.4f} ± {r['se']:.4f}"

    lines = [
        "# The typed-query hold's count, where the app says stop and on the way there (#5556)",
        "",
        "Every cell: the count minus 30 (the shipped hold), mean ± SE clustered on the class. Positive means the "
        "count beats 30. `at_stop` is the objective where the app says stop (the last click if it never does); "
        "`path` is each session's mean over votes 1 to there.",
        "",
        "## The stop click (the same at every count: the stop rule reads only the head)",
        "",
        "| dataset | preset | scope | runs | fired | KM median | quartiles (fired) | labels at stop (median, min) | "
        + " | ".join(f"stop below {n}" for n in NAMED) + " |",
        "|---|---|---|---|---|---|---|---|" + "---|" * len(NAMED),
    ]  # fmt: skip
    for _, r in km.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['preset']} | {r['scope']} | {r['runs']} | {r['fire_rate']:.0%} | {r['km_t_stop']:.0f} | "
            f"{r['q25']:.0f} / {r['q50']:.0f} / {r['q75']:.0f} | {r['labels_at_stop_median']:.0f}, {r['labels_at_stop_min']:.0f} | "
            + " | ".join(f"{r[f'share_stop_below_{n}']:.1%}" for n in NAMED) + " |"
        )  # fmt: skip
    for read in ("at_stop", "path"):
        lines += [
            "",
            f"## {READ_TITLE[read].capitalize()} (`{read}`)",
            "",
            "| dataset | preset | scope | at 30 | " + " | ".join(f"h={h}" for h in (0, *NAMED, 50, 60) if h != SHIPPED)
            + " | best h (gain) | within 2 SE of best |",
            "|---|---|---|---|" + "---|" * (len(NAMED) + 2) + "---|---|",
        ]  # fmt: skip
        for (ds, pr, sc), g in tab[tab["read"] == read].groupby(["dataset", "preset", "scope"], sort=False):
            r30 = g[g["hold"] == SHIPPED].iloc[0]
            best = int(r30["best"])
            lines.append(
                f"| {ds} | {pr} | {sc} | {r30['mean']:.4f} | "
                + " | ".join(cell(g, h) for h in (0, *NAMED, 50, 60) if h != SHIPPED)
                + f" | {best} ({g[g['hold'] == best].iloc[0]['minus_30']:+.4f}) | {int(r30['near_lo'])}-{int(r30['near_hi'])} |"
            )  # fmt: skip
    lines += [
        "",
        "## The sessions a count moves at the stop (stop between the count and 30 labels)",
        "",
        "| dataset | preset | scope | " + " | ".join(f"h={h}: share, gain there" for h in NAMED if h != SHIPPED) + " |",
        "|---|---|---|" + "---|" * (len(NAMED) - 1),
    ]
    for (ds, pr, sc), g in tab[tab["read"] == "at_stop"].groupby(["dataset", "preset", "scope"], sort=False):
        cells = []
        for h in NAMED:
            if h == SHIPPED:
                continue
            r = g[g["hold"] == h].iloc[0]
            cells.append(f"{r['share_moved']:.1%}, {r['minus_30_moved']:+.3f}")
        lines.append(f"| {ds} | {pr} | {sc} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Head's set minus the query's, by label count",
        "",
        "| dataset | preset | scope | positive for good from | 2 SE covers zero (labels) | counts near zero |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in cr.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['preset']} | {r['scope']} | {r['crosses_at']} | "
            f"{r['near_zero_from']}-{r['near_zero_to']} | {r['near_zero_counts']} |"
        )
    lines += ["", "## One count for every preset, against one a beta", "",
              "| read | best single count | its gain over 30 | per-beta bests minus it | " + " | ".join(f"h={n}" for n in NAMED) + " |",
              "|---|---|---|---|" + "---|" * len(NAMED)]  # fmt: skip
    for _, r in com.iterrows():
        lines.append(
            f"| {r['read']} | {r['one_count']} | {r['one_count_minus_30']:+.4f} ± {r['one_count_se']:.4f} | "
            f"{r['per_beta_minus_one_count']:+.4f} ± {r['per_beta_minus_one_count_se']:.4f} | "
            + " | ".join(f"{r[f'minus_30_h{n}']:+.4f}" for n in NAMED) + " |"
        )  # fmt: skip
    lines += [
        "",
        "## The decision rule, applied",
        "",
        "| dataset | preset | scope | count | why |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['dataset']} | {r['preset']} | {r['scope']} | {r['count']} | {r['why']} |" for _, r in dec.iterrows()
    ]
    return "\n".join(lines) + "\n"


def stop_figures(arms, tab: pd.DataFrame, gap: pd.DataFrame, out: Path) -> None:
    """``fig_stop.png`` (both reads against the count), ``fig_counts_by_vote.png``, ``fig_gap.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    presets = [p for p in PRESET_TITLE if any(x[1] == p for x in arms)]
    ds0 = arms[0][0]

    fig, axes = plt.subplots(2, len(presets), figsize=(10, 5.6), sharex=True, squeeze=False)
    for col, pr in enumerate(presets):
        for row, read in enumerate(("at_stop", "path")):
            ax = axes[row, col]
            _style(ax)
            ax.axhline(0, color=MUTED, linewidth=1)
            ax.axvline(SHIPPED, color=GRID, linewidth=1.5, zorder=0)
            for scope, color, lw in (
                ("small", BLUE, 1.2),
                ("medium", ORANGE, 1.2),
                ("large", AQUA, 1.2),
                ("all", INK, 2.2),
            ):
                g = tab[
                    (tab["dataset"] == ds0) & (tab["preset"] == pr) & (tab["read"] == read) & (tab["scope"] == scope)
                ]
                if g.empty:
                    continue
                if scope == "all":
                    ax.fill_between(g["hold"], g["minus_30"] - 2 * g["se"], g["minus_30"] + 2 * g["se"], color=INK,
                                    alpha=0.12, linewidth=0)  # fmt: skip
                ax.plot(
                    g["hold"],
                    g["minus_30"],
                    color=color,
                    linewidth=lw,
                    label=scope if scope == "all" else f"{scope} objects",
                )
            if row == 0:
                ax.set_title(PRESET_TITLE[pr], fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(f"gain over hold 30\n{'at the stop' if read == 'at_stop' else 'mean, votes 1 to the stop'}",
                              fontsize=8, color=INK)  # fmt: skip
            if row == 1:
                ax.set_xlabel("hold (labels before the head's set shows)", fontsize=8, color=INK)
            ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_stop.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(presets), figsize=(10, 3.4), sharey=True, squeeze=False)
    t = np.arange(1, 61)
    for col, pr in enumerate(presets):
        ax = axes[0, col]
        _style(ax)
        z = next(zz for d, p, zz in arms if p == pr)
        for h, color in zip(NAMED, NAMED_SHADES):
            ax.plot(t, z["curves"][:, h, :60].mean(axis=0), color=color, linewidth=2.2 if h == SHIPPED else 1.4,
                    label=f"hold {h}")  # fmt: skip
        ax.set_title(PRESET_TITLE[pr], fontsize=10, color=INK)
        ax.set_xlabel("vote", fontsize=8, color=INK)
        if col == 0:
            ax.set_ylabel("mean F-beta on screen", fontsize=8, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "fig_counts_by_vote.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(presets), figsize=(10, 3.4), sharey=True, squeeze=False)
    for col, pr in enumerate(presets):
        ax = axes[0, col]
        _style(ax)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.axvline(SHIPPED, color=GRID, linewidth=1.5, zorder=0)
        g = gap[(gap["dataset"] == ds0) & (gap["preset"] == pr) & (gap["scope"] == "all") & (gap["labels"] <= 80)]
        ax.fill_between(
            g["labels"], g["gap"] - 2 * g["se"], g["gap"] + 2 * g["se"], color=BLUE, alpha=0.18, linewidth=0
        )
        ax.plot(g["labels"], g["gap"], color=BLUE, linewidth=2, label="every session")
        ax.plot(g["labels"], g["gap_before_stop"], color=ORANGE, linewidth=1.4, label="sessions not yet told to stop")
        ax.set_title(PRESET_TITLE[pr], fontsize=10, color=INK)
        ax.set_xlabel("labels", fontsize=8, color=INK)
        if col == 0:
            ax.set_ylabel("head's set minus the query's\nF-beta, paired in session", fontsize=8, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "fig_gap.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out}/fig_stop.png, {out}/fig_counts_by_vote.png, {out}/fig_gap.png")


def _picks_file(cell: Path) -> Path:
    """A cell's picks sidecar: ``task_N__picks.csv``, gzipped beside a gzipped cell (#5092)."""
    name = cell.name
    stem, gz = (name[: -len(".csv.gz")], ".gz") if name.endswith(".csv.gz") else (name[: -len(".csv")], "")
    return cell.with_name(f"{stem}__picks.csv{gz}")


def _picks(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path, usecols=lambda c: c in PICK_KEYS)
    return p[[c for c in PICK_KEYS if c in p.columns]].reset_index(drop=True)


def check(a: argparse.Namespace) -> None:
    run_dir, ctl_dir = Path(a.run), Path(a.control)
    files = sorted(_cells_io.main_frame_files(run_dir))
    rows = []
    for f in files:
        c = ctl_dir / f.name
        got_run, got_ctl = read_cell((str(f), None)), read_cell((str(c), None)) if c.exists() else None
        if got_run is None or got_ctl is None:
            rows.append({"cell": f.name, "status": "missing"})
            continue
        same_picks = _picks(_picks_file(f)).equals(_picks(_picks_file(c)))
        run_curve = got_run["shown"]  # the run as it showed
        priced = got_ctl["curves"][a.hold]
        rows.append(
            {
                "cell": f.name,
                "category": got_run["category"],
                "seed": got_run["seed"],
                "status": "ok",
                "same_picks": bool(same_picks),
                "same_stop": bool(
                    got_run["t_stop"] == got_ctl["t_stop"]
                    or (np.isnan(got_run["t_stop"]) and np.isnan(got_ctl["t_stop"]))
                ),
                "run_t_stop": got_run["t_stop"],
                "control_t_stop": got_ctl["t_stop"],
                "max_abs_diff": float(np.abs(run_curve - priced).max()),
                "run_mean": float(run_curve.mean()),
                "priced_mean": float(priced.mean()),
                "run_last_held": got_run["last_held"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    ok = df[df["status"] == "ok"]
    print(
        f"{run_dir} vs {ctl_dir} at hold {a.hold}: {len(ok)}/{len(df)} cells compared, "
        f"picks identical {int(ok['same_picks'].sum())}/{len(ok)}, "
        f"stop click identical {int(ok['same_stop'].sum())}/{len(ok)}, "
        f"max |run - priced| {ok['max_abs_diff'].max():.2e}, "
        f"session mean run {ok['run_mean'].mean():.6f} vs priced {ok['priced_mean'].mean():.6f}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--cells", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--embedder", default=None, help="keep this embedder's cells only (a two-arm grid)")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--stops", default=None, help="analyze.py's stops.csv, to cross-check each session's stop")
    p = sub.add_parser("price")
    p.add_argument("--arm", action="append", required=True, help="dataset:preset=X.npz")
    p.add_argument("--out", required=True)
    f = sub.add_parser("figures")
    f.add_argument("--arm", action="append", required=True, help="dataset:preset=X.npz (photos, screens)")
    f.add_argument("--out", required=True)
    st = sub.add_parser("stop")
    st.add_argument("--arm", action="append", required=True, help="dataset:preset=X.npz")
    st.add_argument("--out", required=True)
    st.add_argument("--no-figures", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--run", required=True)
    c.add_argument("--control", required=True)
    c.add_argument("--hold", type=int, required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"extract": extract, "price": price, "figures": figures, "stop": stop_price, "check": check}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
