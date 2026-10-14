"""#4876: re-price #4736's typed-query hold at every count from 0 to 30, on one build, photos and app screens.

A typed-query session shows the typed query's own set on Find until its labels number the hold (30, on a
single-vector space), and the trained head's set from then on. The hold changes what Find returns, never what
Autopilot asks: the harness trains the head and picks with it at every vote either way, and a held click
writes the typed query's set as its untagged row while the head's own rows are kept, tagged ``hold:head``
(``voting_iterations.QUERY_HOLD_HEAD_TAG``). So one session set run with the shipped hold prices every count up
to it exactly: hold *h* shows the head's row at each vote whose labels number *h* or more, and the query's set
below that. Hold 0 is the head from the label quota (the arm before #4736, ``CALIB_QUERY_HOLD=0``); a count under
the quota is the same as 0, since the quota's own query tier (#4758) is not the hold.

Each vote scores F-beta at the session's preset of the set Find returns on the withheld half; a vote with no
row carries the last one (a check round writes one row for its votes) and a vote before the first row scores 0,
as ``rico_suave/analyze.py`` reads them. The session mean is over votes 1-150 (the owner's number, #4668) and the
short one over 1-50.

    python hold_4876.py extract --cells <run>/results/cells --out X.npz [--embedder siglip]
    python hold_4876.py price --arm photos:b1=X.npz --arm screens:b1=Y.npz ... --out <dir>
    python hold_4876.py figures --arm photos:b1=X.npz --arm screens:b1=Y.npz ... --out <dir>
    python hold_4876.py check --run <hold-h run>/results/cells --control <shipped run>/results/cells --hold H

``check`` is the proof that the offline price is the run's: for every cell the hold-*h* run shares with the
control (same grid index), the picks must be identical and the run's curve must equal the control's priced at *h*.
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

HORIZON = 150
SHIPPED = 30
HOLDS = np.arange(SHIPPED + 1)
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
        curves[h] = carry(t, f)
    q = ordinary.loc[ordinary["detector_tier"].astype(str) == "query", "fbeta"]
    first = base.iloc[0]
    n_pos, n_neg = float(first["n_test_pos"]), float(first["n_test_neg"])
    return {
        "dataset": str(first["dataset"]),
        "category": str(first["category"]),
        "seed": int(first["seed"]),
        "beta": beta,
        "prevalence": n_pos / (n_pos + n_neg) if n_pos + n_neg > 0 else np.nan,
        "query_f": float(q.iloc[0]) if len(q) else np.nan,
        "n_held": len(held),
        # Invariants the price rests on: one head row a held vote, and only below the shipped count.
        "held_dupes": int(len(held_all) - len(held)),
        "held_past_hold": int(np.nansum(head_lab >= SHIPPED)),
        "held_unmatched": int(sum(int(x) not in pos for x in head_t)),
        "first_held": float(head_t.min()) if len(head_t) else np.nan,
        "last_held": float(head_t.max()) if len(head_t) else np.nan,
        "labels": carry(t, labels).astype(np.float32),
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
    bad = {k: sum(r[k] for r in runs) for k in ("held_dupes", "held_past_hold", "held_unmatched")}
    print(f"invariants (all should be 0): {bad}; runs with no held vote: {sum(r['n_held'] == 0 for r in runs)}")
    scal = ("dataset", "category", "seed", "beta", "prevalence", "query_f", "n_held", "first_held", "last_held")
    np.savez_compressed(
        a.out,
        **{k: np.array([r[k] for r in runs]) for k in scal},
        **{k: np.stack([r[k] for r in runs]) for k in ("labels", "head", "curves")},
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
    names = {"photos": ("photos (COCO Better)", BLUE), "screens": ("app screens (Rico Suave, 1%)", ORANGE)}

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
        same_picks = _picks(f.with_name(f.stem + "__picks.csv")).equals(_picks(c.with_name(c.stem + "__picks.csv")))
        run_curve = got_run["curves"][SHIPPED]  # the run as it showed (no held row beyond its own hold)
        priced = got_ctl["curves"][a.hold]
        rows.append(
            {
                "cell": f.name,
                "category": got_run["category"],
                "seed": got_run["seed"],
                "status": "ok",
                "same_picks": bool(same_picks),
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
    p = sub.add_parser("price")
    p.add_argument("--arm", action="append", required=True, help="dataset:preset=X.npz")
    p.add_argument("--out", required=True)
    f = sub.add_parser("figures")
    f.add_argument("--arm", action="append", required=True, help="dataset:preset=X.npz (photos, screens)")
    f.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--run", required=True)
    c.add_argument("--control", required=True)
    c.add_argument("--hold", type=int, required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"extract": extract, "price": price, "figures": figures, "check": check}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
