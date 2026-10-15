"""Are Autopilot's stopping rules any good (#3560): where the clicks before Done go, and what comes after.

The State of the App analyzer reports where the app said stop (``stops.csv``, the summary's stopping
block).  This script asks the questions that block leaves open, over one review set per preset:

* **where the clicks before the stop go**, stretch by stretch (opening, Hard, New, the mid-session check,
  the check before Done), and which light each stretch waited on (``stopping.stage_table``,
  ``stopping.binding_note``);
* **what comes after the stop**: the objective at every click from 60 before the stop to 80 after it,
  per session that fired, so the report can draw the stop against the curve it sits on;
* **whether Smart's lapses mean anything**: Smart is the light that breaks Done after the stop; if the
  sessions it breaks gain more afterwards than the sessions it leaves alone, its lapses carry news about
  the objective, and if not they are noise as far as the user is concerned.

Two steps, because the first reads every cell and the second commits a few kilobytes.

``read`` (on a GRID node; ~1 min a set at 8 CPUs) writes, per preset, the per-session
``stops-<beta>.csv``, ``margins-<beta>.csv`` and ``around-<beta>.csv`` into a work dir::

    python stop_rules_3560.py read --cells 0.25=<b025>/results/cells --cells 1=<b1>/... --cells 4=<b4>/... \\
        --embedder siglip --work /expscratch/$USER/stop-3560/work

``report`` reduces the work dir to what a report commits: ``stages.csv``, ``margins.csv``,
``around_stop.csv``, ``smart_lapses.csv``, ``tables.md`` and ``figures/{stages,around_stop}.png``::

    python stop_rules_3560.py report --work <work> --out docs/experiments/<date>-stopping-rules-3560
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
import _cells_io  # noqa: E402
import objective  # noqa: E402
import stopping  # noqa: E402

KEYS = ("embedder", "category", "seed")
METRICS = (objective.OBJECTIVE, "average_precision")
#: Clicks before and after the stop the objective is read at, per session that fired.
BEFORE, AFTER = 60, 80
#: The window the figure draws, over the sessions that cover all of it: a fixed population, so the curve's
#: shape is not the late stoppers running out of budget on the right (they are the hard sessions).
WINDOW = (-40, 60)
#: How a table and a figure name each preset.
PRESET = {0.25: "β 1/4", 1.0: "β 1", 4.0: "β 4"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
#: One hue per stretch, in the stretches' order (the reference categorical palette, first five slots,
#: validated light on SURFACE; three sit under 3:1, so every segment is labelled and the table carries it).
STAGE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def _around(g: pd.DataFrame, t_stop: float) -> np.ndarray:
    """The objective at each click from ``t_stop - BEFORE`` to ``t_stop + AFTER``; NaN outside the run.

    A click's value is the last row at or before it: a check round's five clicks share the row that closed
    it, as they share its detector.
    """
    t = g["t"].to_numpy(dtype=float)
    v = pd.to_numeric(g[objective.OBJECTIVE], errors="coerce").to_numpy(dtype=float)
    clicks = t_stop + np.arange(-BEFORE, AFTER + 1)
    idx = np.searchsorted(t, clicks, side="right") - 1
    out = np.where((idx >= 0) & (clicks <= t[-1]), v[np.clip(idx, 0, None)], np.nan)
    return out


def _read_cell(path: str, embedder: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _cells_io._base_rows(pd.read_csv(path, low_memory=False))
    df = df[~_cells_io.check_rows(df)]
    if embedder:
        df = df[df["embedder"] == embedder]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    stops = stopping.stopping_points(df, keys=KEYS, metrics=METRICS)
    mg = stopping.margins(df, keys=KEYS)
    around = []
    for _, r in stops[stops["stopped"]].iterrows():
        g = df[(df["category"] == r["category"]) & (df["seed"] == r["seed"])].sort_values("t")
        around.append(
            pd.DataFrame(
                {
                    "category": r["category"],
                    "seed": r["seed"],
                    "k": np.arange(-BEFORE, AFTER + 1),
                    objective.OBJECTIVE: _around(g, float(r["t_stop"])),
                }
            )
        )
    return stops, mg, pd.concat(around, ignore_index=True) if around else pd.DataFrame()


def read(cells: list[str], embedder: str, work: Path, nproc: int) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for spec in cells:
        beta, path = spec.split("=", 1)
        files = [str(p) for p in _cells_io.main_frame_files(path)]
        with Pool(nproc) as pool:
            parts = pool.map(partial(_read_cell, embedder=embedder), files, chunksize=8)
        for i, name in enumerate(("stops", "margins", "around")):
            frame = pd.concat([p[i] for p in parts if not p[i].empty], ignore_index=True)
            frame.to_csv(work / f"{name}-{beta}.csv", index=False, float_format="%.6g")
        print(f"beta {beta}: {len(files)} cells", flush=True)


def _band(category: pd.Series) -> pd.Series:
    return category.astype(str).str.split("@").str[1].fillna("")


def _class(category: pd.Series) -> pd.Series:
    return category.astype(str).str.split("@").str[0]


def _mean_se(values: pd.Series, category: pd.Series) -> tuple[float, float]:
    """Mean over sessions and its SE over the per-class means: a class's sessions share their hard cases."""
    v = values.dropna()
    by_class = v.groupby(_class(category.loc[v.index])).mean()
    return float(v.mean()), float(by_class.std() / np.sqrt(len(by_class))) if len(by_class) > 1 else float("nan")


def load(work: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stops, mg, around = [], [], []
    for beta in sorted(PRESET):
        tag = f"{beta:g}"
        if not (work / f"stops-{tag}.csv").exists():
            continue
        s = pd.read_csv(work / f"stops-{tag}.csv")
        s["stopped"] = s["stopped"].astype(bool)
        m = pd.read_csv(work / f"margins-{tag}.csv")
        a = pd.read_csv(work / f"around-{tag}.csv")
        for df in (s, m, a):
            df["arm"] = PRESET[beta]
            df["beta"] = beta
        for df in (s, m):
            df["band"] = _band(df["category"])
        stops.append(s)
        mg.append(m)
        around.append(a)
    frames = [pd.concat(x, ignore_index=True) for x in (stops, mg, around)]
    # Presets in beta order, not label order ("β 1" sorts before "β 1/4").
    order = pd.CategoricalDtype([PRESET[b] for b in sorted(PRESET)], ordered=True)
    for df in frames:
        df["arm"] = df["arm"].astype(order)
    return frames[0], frames[1], frames[2]


def stage_rows(stops: pd.DataFrame) -> pd.DataFrame:
    """One row per preset and scope (``all``, each band): the stopping summary plus two paired gains with SEs."""
    metric = objective.OBJECTIVE
    rows = []
    for (arm, beta), g in stops.groupby(["arm", "beta"], sort=False, observed=True):
        for scope, h in [("all", g), *((str(b), x) for b, x in g.groupby("band", sort=True))]:
            row = stopping.summarise(h.assign(scope=scope), by=("scope",), metrics=METRICS).iloc[0].to_dict()
            fired = h[h["stopped"]]
            row["arm"], row["beta"] = arm, beta
            row["mean_gain_after_stop"], row["se_gain_after_stop"] = _mean_se(
                fired[f"{metric}_delta"], fired["category"]
            )
            row["mean_hard_exit_to_stop"], row["se_hard_exit_to_stop"] = _mean_se(
                fired[f"{metric}_at_stop"] - fired[f"{metric}_at_hard_exit"], fired["category"]
            )
            row["mean_clicks_hard_exit_to_stop"] = float((fired["t_stop"] - fired["t_hard_exit"]).mean())
            rows.append(row)
    table = pd.DataFrame(rows)
    lead = ["arm", "beta", "scope", "n_runs", "n_fired", "fire_rate", "km_t_stop"]
    return table[lead + [c for c in table.columns if c not in lead]]


def smart_lapse_rows(stops: pd.DataFrame) -> pd.DataFrame:
    """Do the sessions whose Done Smart breaks gain more after the stop than the ones it leaves alone?

    Within class, so the comparison is not which classes Smart flaps on: the per-class difference of means,
    over the classes that have sessions of both kinds, with its SE over those classes.
    """
    metric = objective.OBJECTIVE
    rows = []
    for (arm, beta), g in stops.groupby(["arm", "beta"], sort=False, observed=True):
        fired = g[g["stopped"]].copy()
        fired["broke"] = fired["done_lapses_smart"] > 0
        fired["cls"] = _class(fired["category"])
        per = fired.groupby(["cls", "broke"])[f"{metric}_delta"].mean().unstack().dropna()
        diff = per[True] - per[False] if {True, False} <= set(per.columns) else pd.Series(dtype=float)
        rows.append(
            {
                "arm": arm,
                "beta": beta,
                "n_fired": len(fired),
                "share_broken": float(fired["broke"].mean()),
                "gain_broken": float(fired.loc[fired["broke"], f"{metric}_delta"].mean()),
                "gain_kept": float(fired.loc[~fired["broke"], f"{metric}_delta"].mean()),
                "n_classes": len(diff),
                "within_class_diff": float(diff.mean()) if len(diff) else float("nan"),
                "se_within_class_diff": float(diff.std() / np.sqrt(len(diff))) if len(diff) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def around_rows(around: pd.DataFrame) -> pd.DataFrame:
    """The objective's change since the stop, per click of :data:`WINDOW`, over the sessions that cover it all.

    ``mean`` / ``q25`` / ``q75`` are of each session's objective at click *k* minus at its stop; ``level``
    is the mean objective itself, and ``n`` the sessions (the same at every *k*).
    """
    metric = objective.OBJECTIVE
    lo, hi = WINDOW
    a = around[(around["k"] >= lo) & (around["k"] <= hi)].copy()
    run = ["arm", "category", "seed"]
    covered = a.groupby(run, observed=True)[metric].transform(lambda x: x.notna().all())
    a = a[covered.astype(bool)]
    at_stop = a[a["k"] == 0].set_index(run)[metric].rename("at_stop")
    a = a.join(at_stop, on=run)
    a["gain"] = a[metric] - a["at_stop"]
    g = a.groupby(["arm", "beta", "k"], sort=False, observed=True)
    out = g["gain"].agg(mean="mean", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), n="count")
    out["level"] = g[metric].mean()
    return out.reset_index()


def _axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_axisbelow(True)


def stages_figure(table: pd.DataFrame, out: Path) -> None:
    """Mean clicks per stretch before the stop, one bar per preset and band, over the sessions that fired."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    order = []
    for arm in table["arm"].unique():
        for scope in ("all", "large", "medium", "small"):
            hit = table[(table["arm"] == arm) & (table["scope"] == scope)]
            if len(hit):
                order.append(hit.iloc[0])
    fig, ax = plt.subplots(figsize=(9.0, 0.42 * len(order) + 1.6), facecolor=SURFACE)
    _axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    names = [n for n, _ in stopping.STAGES]
    for y, r in enumerate(order):
        left = 0.0
        for name, color in zip(names, STAGE_COLORS, strict=True):
            w = float(r[f"mean_clicks_{name}"])
            if w <= 0:
                continue
            # A 2 px surface gap between segments.
            ax.barh(y, w, left=left, color=color, height=0.72, edgecolor=SURFACE, linewidth=1.5)
            if w >= 6:
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=7.5, color=INK)
            left += w
        ax.text(left + 1.5, y, f"{left:.0f}  ({r['fire_rate']:.0%} fired)", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{r['arm']}  {r['scope']}" if r["scope"] != "all" else f"{r['arm']}  all" for r in order])
    ax.invert_yaxis()
    ax.set_xlim(0, max(float(sum(r[f"mean_clicks_{n}"] for n in names)) for r in order) * 1.22)
    ax.set_xlabel("clicks before Autopilot says Done (mean of the sessions that got there)", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE_COLORS]
    ax.legend(
        handles,
        [stopping.STAGE_LABELS[n] for n in names],
        ncol=5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8,
        labelcolor=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def around_figure(rows: pd.DataFrame, out: Path) -> None:
    """The objective gained since Done, one panel per preset: mean and inter-quartile band, clicks from Done."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    arms = list(rows["arm"].unique())
    fig, axes = plt.subplots(1, len(arms), figsize=(3.6 * len(arms), 3.4), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, arm in zip(np.atleast_1d(axes), arms, strict=True):
        _axes(ax)
        ax.grid(True, color=GRID, linewidth=0.6)
        g = rows[rows["arm"] == arm].sort_values("k")
        ax.fill_between(g["k"], g["q25"], g["q75"], color="#2a78d6", alpha=0.18, linewidth=0)
        ax.plot(g["k"], g["mean"], color="#2a78d6", linewidth=2)
        ax.axvline(0, color=INK, linewidth=1, linestyle="--")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        at = float(g.loc[g["k"] == 0, "level"].iloc[0])
        ax.annotate(f"Done, at {at:.2f}", (0, 0), xytext=(5, 6), textcoords="offset points", fontsize=8, color=INK)
        last = g.iloc[-1]
        ax.annotate(
            f"{float(last['mean']):+.3f}",
            (last["k"], last["mean"]),
            xytext=(-2, 6),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            ha="right",
        )
        ax.set_title(f"{arm}  ({int(g['n'].iloc[0])} sessions)", color=INK, fontsize=10)
        ax.set_xlabel("clicks from Done", color=INK)
    np.atleast_1d(axes)[0].set_ylabel(f"{objective.OBJECTIVE} gained since Done", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _f(v: float, digits: int = 3) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:+.{digits}f}"


def report(work: Path, out: Path) -> None:
    stops, mg, around = load(work)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    table = stage_rows(stops)
    table.to_csv(out / "stages.csv", index=False, float_format="%.4f")
    msum = stopping.summarise_margins(mg, by=("arm",))
    msum_band = stopping.summarise_margins(mg, by=("arm", "band"))
    pd.concat([msum.assign(band="all"), msum_band], ignore_index=True).to_csv(
        out / "margins.csv", index=False, float_format="%.4f"
    )
    lapses = smart_lapse_rows(stops)
    lapses.to_csv(out / "smart_lapses.csv", index=False, float_format="%.4f")
    arows = around_rows(around)
    arows.to_csv(out / "around_stop.csv", index=False, float_format="%.4f")
    stages_figure(table, out / "figures" / "stages.png")
    around_figure(arows, out / "figures" / "around_stop.png")

    metric = objective.OBJECTIVE
    by_arm = stopping.summarise(stops, by=("arm",), metrics=METRICS)
    by_band = stopping.summarise(stops, by=("arm", "band"), metrics=METRICS)
    md = [
        "## Where the app said stop",
        "",
        stopping.stopping_table(by_arm, metric=metric),
        "",
        stopping.stopping_table(by_band, metric=metric),
        "",
        "## Where the clicks before the stop went",
        "",
        stopping.stage_table(by_arm, metric=metric),
        "",
        stopping.stage_table(by_band, metric=metric),
        "",
        stopping.binding_note(by_arm),
        "",
        "## Paired gains, mean ± SE over class means",
        "",
        "| preset | after the stop | clicks after the stop | Hard's end to the stop | clicks |",
        "|---|---|---|---|---|",
    ]
    for _, r in table[table["scope"] == "all"].iterrows():
        md.append(
            f"| {r['arm']} | {_f(r['mean_gain_after_stop'])} ± {r['se_gain_after_stop']:.3f} | "
            f"{r['median_clicks_after_stop']:.0f} | "
            f"{_f(r['mean_hard_exit_to_stop'])} ± {r['se_hard_exit_to_stop']:.3f} | "
            f"{r['mean_clicks_hard_exit_to_stop']:.0f} |"
        )
    md += [
        "",
        "## Smart's lapses after the stop",
        "",
        "| preset | fired | Done broken by Smart | gain after stop, broken | kept | within class, broken − kept |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in lapses.iterrows():
        md.append(
            f"| {r['arm']} | {int(r['n_fired'])} | {r['share_broken']:.0%} | {_f(r['gain_broken'])} | "
            f"{_f(r['gain_kept'])} | {_f(r['within_class_diff'])} ± {r['se_within_class_diff']:.3f} "
            f"({int(r['n_classes'])} classes) |"
        )
    md += [
        "",
        "## How close each gate came",
        "",
        stopping.margin_table(msum),
        "",
        stopping.margin_table(msum_band),
        "",
    ]
    (out / "tables.md").write_text("\n".join(md))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--cells", action="append", required=True, help="BETA=<run dir>/results/cells")
    r.add_argument("--embedder", default="", help="keep only this embedder's sessions (an A/B's control arm)")
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--nproc", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "read":
        read(args.cells, args.embedder, args.work, args.nproc)
    else:
        report(args.work, args.out)


if __name__ == "__main__":
    main()
