#!/usr/bin/env python3
"""The review across floors (#4408): each P's returned set, read off sessions that aimed at that P.

A review runs one set of sessions per floor P (``CALIB_MIN_PRECISION``), each
analyzed on its own (``analyze.sh``). This puts them side by side, reading each
P only from its own sessions:

* ``returned_at_own_p.png`` -- per P, over clicks: the returned set's precision
  against P (top) and its recall against the oracle's recall at P (bottom). When
  the 50% run is given, the same P read off the 50% sessions is drawn light, so
  the figure shows what running each P on its own changed.
* With ``--kind balance`` the runs are keyed by beta (#4413): ``returned_at_own_beta.png`` (the
  returned set's F-beta as a share of the best cut, per beta over clicks), ``objective_at_own_beta.png``
  (the objective over clicks, with the end after the check) and ``perbeta_summary.md``: the objective
  first (#4427: F-beta of the withheld set above the app's threshold, before and after the check, with
  the returned set's size), then the returned set from the rank frames, the early dip against the
  typed query's set, and the spot check (#4474).
* ``perp_summary.md`` -- per P and point (text, 25, 50, final, ceiling):
  precision, its gap to P, the share of runs meeting P, recall, the oracle's
  recall at P and the share of it returned; then each P's sessions' AP and
  Goods at the end.

    python perp.py --run 0.1=DIR --run 0.5=DIR --run 0.9=DIR --out OUT

Each DIR is an ``analysis-binary`` directory (``analyze.sh``'s output).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import HEADLINE_POINTS, _md, returned_at_p  # noqa: E402
from figures import INK, SURFACE, _axes  # noqa: E402

REF = 0.5  # the floor whose sessions the other floors were read off before #4408


def _tag(floor: float) -> str:
    return f"p{round(floor * 100)}"


def _curve(analysis: Path, floor: float, at: list[int] | None = None) -> pd.DataFrame:
    c = pd.read_csv(analysis / "curves.csv")
    steps = pd.read_csv(analysis / "line_steps.csv")
    clicks = at or sorted({0, *steps["t"].astype(int).unique().tolist()})
    t = _tag(floor)
    return c[c["t"].isin(clicks)].groupby("t")[[f"precision_{t}", f"recall_{t}", f"oracle_recall_{t}"]].mean()


def figure(runs: dict[float, Path], out: Path) -> None:
    floors = sorted(runs)
    fig, axes = plt.subplots(2, len(floors), figsize=(4.4 * len(floors), 6.4), facecolor=SURFACE, sharex=True)
    for j, floor in enumerate(floors):
        t = _tag(floor)
        own = _curve(runs[floor], floor)
        series = [(own, "its own sessions", 2.0, 1.0)]
        if REF in runs and floor != REF:
            series.append((_curve(runs[REF], floor), f"read off P = {REF:.0%} sessions", 1.4, 0.4))
        for m, label, lw, alpha in series:
            axes[0, j].plot(m.index, m[f"precision_{t}"], color="#2a6fdb", lw=lw, alpha=alpha, marker="o", ms=2.5)
            axes[1, j].plot(m.index, m[f"recall_{t}"], color="#2a6fdb", lw=lw, alpha=alpha, marker="o", ms=2.5,
                            label=label)  # fmt: skip
        axes[1, j].plot(own.index, own[f"oracle_recall_{t}"], color=INK, ls="--", lw=1.1, label="oracle at P")
        axes[0, j].axhline(floor, color=INK, ls=":", lw=1.2)
        for row, col in ((0, f"precision_{t}"), (1, f"recall_{t}")):
            v = own[col].iloc[-1]
            axes[row, j].annotate(f"{v:.2f}", (own.index[-1], v), xytext=(4, 0), textcoords="offset points",
                                  va="center", color=INK, fontsize=8)  # fmt: skip
        axes[0, j].set_title(f"P = {floor:.0%}: precision against P (dotted)", color=INK, fontsize=10, loc="left")
        axes[1, j].set_title("recall against the oracle's at P (dashed)", color=INK, fontsize=10, loc="left")
        for row in (0, 1):
            _axes(axes[row, j])
            axes[row, j].set_ylim(0, 1.02)
        axes[1, j].set_xlabel("clicks", color=INK)
        axes[1, j].legend(fontsize=7, frameon=False, loc="lower right")
    axes[0, 0].set_ylabel("precision of the returned set", color=INK)
    axes[1, 0].set_ylabel("recall of the returned set", color=INK)
    fig.tight_layout()
    fig.savefig(out / "returned_at_own_p.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def summary(runs: dict[float, Path], out: Path) -> None:
    md = [
        "# The returned set at each P, each read off its own sessions (#4408)",
        "",
        "`precision` against the target P (`gap` = precision - P), `meets` the share of runs at or above P, "
        "`recall` against `oracle_recall` (the most any cut of the same ranking returns at or above P; "
        "`share` = recall / oracle recall). Points: text sort, 25 and 50 clicks, the end, full labels.",
        "",
    ]
    rows, ends = [], []
    for floor, d in sorted(runs.items()):
        lines = pd.read_csv(d / "lines.csv", dtype={"point": str})
        t = returned_at_p(lines[lines["floor"].round(4) == round(floor, 4)], ["arm"]).reset_index()
        t.insert(0, "sessions_at", f"{floor:.0%}")
        rows.append(t)
        cells = pd.read_csv(d / "cells.csv")
        ends.append(
            {
                "sessions_at": f"{floor:.0%}",
                "runs": len(cells),
                "final_ap": cells["final_ap"].mean(),
                "ceiling_ap": cells["ceiling_ap"].mean(),
                "goods_found": cells["positives_found"].mean(),
                "check_confirmed": (cells["check_status"] == "confirmed").mean() if "check_status" in cells else None,
            }
        )
        if REF in runs and floor != REF:
            ref_lines = pd.read_csv(runs[REF] / "lines.csv", dtype={"point": str})
            r = returned_at_p(ref_lines[ref_lines["floor"].round(4) == round(floor, 4)], ["arm"]).reset_index()
            r.insert(0, "sessions_at", f"{REF:.0%} (read at {floor:.0%})")
            rows.append(r[r["point"].astype(str) == "final"])
    table = pd.concat(rows, ignore_index=True)
    table["point"] = pd.Categorical(table["point"].astype(str), HEADLINE_POINTS, ordered=True)
    md += [_md(table, index=False), "", "## Each P's sessions at the end", "", _md(pd.DataFrame(ends).round(3), False)]
    (out / "perp_summary.md").write_text("\n".join(md) + "\n")


def _beta_tag(beta: float) -> str:
    return "b" + (f"{beta:g}".replace(".", "") if beta < 1 else f"{beta:g}")


def figure_balance(runs: dict[float, Path], out: Path) -> None:
    """Per beta, over clicks: the returned set's F-beta as a share of the best cut, each beta off its own sessions (#4413).

    Click 0 is the text sort. Solid: each sort's own line in the app (the text sort's blind GMM cut, the
    detector's labels line); dashed: set-constant top-K on both. One rule per comparison (owner, 2026-10-04).
    Every run at every recorded click, as its session shows it (:func:`_by_rule`, #4631).
    """
    betas = sorted(runs)
    fig, axes = plt.subplots(1, len(betas), figsize=(4.4 * len(betas), 3.9), facecolor=SURFACE, sharey=True)
    axes = [axes] if len(betas) == 1 else list(axes)
    for ax, beta in zip(axes, betas, strict=True):
        for rule, series in _share_by_rule(runs[beta], beta).items():
            ls = "-" if rule == "app line" else "--"
            ax.plot(series.index, series.to_numpy(), color="#2a6fdb", ls=ls, lw=2, marker="o", ms=2.5, label=rule)
            ax.annotate(f"{series.iloc[-1]:.2f}", (series.index[-1], series.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=8)  # fmt: skip
        ax.axhline(1.0, color=INK, ls=":", lw=1.2)
        ax.set_title(f"beta = {beta:g}: F-beta / best cut", color=INK, fontsize=10, loc="left")
        _axes(ax)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("clicks", color=INK)
    axes[0].set_ylabel("F-beta of the returned set / best cut", color=INK)
    axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "returned_at_own_beta.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _by_rule(d: Path, beta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(balances at the text point, every run at every recorded click)`` for *beta*, each with a ``rule`` column.

    The clicks are read as each session shows them (``analyze.session_balance_steps``, #4631): a run's frame
    once the app shows its detector, the typed query's set before that and throughout a run that never trained,
    so a mean at a click is over every run, not the ones with a frame there.
    """
    from analyze import session_balance_steps  # noqa: PLC0415

    steps = pd.read_csv(d / "balance_steps.csv")
    bal = pd.read_csv(d / "balances.csv", dtype={"point": str})
    if "rule" not in steps:  # an analysis from before the rule column: the app's line only
        steps = steps.assign(rule="app line")
        bal = bal.assign(rule="app line")
    steps = steps[steps["beta"].round(4) == round(beta, 4)]
    bal = bal[(bal["beta"].round(4) == round(beta, 4)) & (bal["point"] == "text")]
    return bal, session_balance_steps(steps, bal)


def _share_by_rule(d: Path, beta: float, read: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> dict[str, pd.Series]:
    """Per rule: the returned set's share of the best cut over clicks, the text sort at click 0, every run at every
    click (*read* is :func:`_by_rule`'s output when the caller has it)."""
    bal, steps = read if read is not None else _by_rule(d, beta)
    out = {}
    for rule in ("app line", "top-K"):
        s = steps[steps["rule"] == rule]
        if s.empty:
            continue
        t0 = float(bal[bal["rule"] == rule]["fb_share"].mean())
        out[rule] = pd.concat([pd.Series({0: t0}), s.groupby("t")["fb_share"].mean()]).sort_index()
    return out


def figure_objective(runs: dict[float, Path], out: Path) -> None:
    """Per beta, over clicks: the objective (#4427), unchecked at each click, and the end after the check."""
    betas = sorted(runs)
    fig, axes = plt.subplots(1, len(betas), figsize=(4.4 * len(betas), 3.9), facecolor=SURFACE, sharey=True)
    axes = [axes] if len(betas) == 1 else list(axes)
    for ax, beta in zip(axes, betas, strict=True):
        c = pd.read_csv(runs[beta] / "curves.csv")
        cells = pd.read_csv(runs[beta] / "cells.csv")  # every run, the never-trained at the typed query (#4631)
        if "thr_fbeta" in c:
            m = c.groupby("t")["thr_fbeta"].mean().dropna()
            ax.plot(m.index, m.to_numpy(), color="#2a6fdb", lw=2, label="unchecked, at each click")
            if "thr_fbeta_final" in cells and len(m):
                ax.scatter([m.index.max() + 6], [cells["thr_fbeta_final"].mean()], color="#d9480f", zorder=3,
                           label="after the check")  # fmt: skip
        ax.set_title(f"beta = {beta:g}: the objective", color=INK, fontsize=10, loc="left")
        _axes(ax)
        ax.set_xlabel("clicks", color=INK)
    axes[0].set_ylabel("F-beta of the withheld set above the threshold", color=INK)
    axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "objective_at_own_beta.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _objective_rows(runs: dict[float, Path]) -> pd.DataFrame:
    """Per beta, off its own sessions: the objective before and after the check, and what the line returned.

    Over every run (#4631): one that never trained scores its typed query's set at every point, and its check's
    effect is 0, so the effect is the difference of the two columns beside it.
    """
    rows = []
    for beta, d in sorted(runs.items()):
        c = pd.read_csv(d / "cells.csv")
        if "thr_fbeta_final" not in c:
            continue
        effect = c["thr_fbeta_final"] - c["thr_fbeta_unchecked"]
        se = effect.std() / max(len(effect), 1) ** 0.5
        rows.append({
            "beta": f"{beta:g}",
            "runs": len(c),
            "F at 25": c.get("thr_fbeta_25", pd.Series(dtype=float)).mean(),
            "F at 50": c.get("thr_fbeta_50", pd.Series(dtype=float)).mean(),
            "F unchecked": c["thr_fbeta_unchecked"].mean(),
            "F after the check": c["thr_fbeta_final"].mean(),
            "check's effect": f"{effect.mean():+.3f} ± {se:.3f}",
            "precision after": c["thr_precision_final"].mean(),
            "recall after": c["thr_recall_final"].mean(),
            "returned median, unchecked / after": f"{c['thr_returned_unchecked'].median():.0f} / {c['thr_returned_final'].median():.0f}",
            "over 200, unchecked / after": f"{(c['thr_returned_unchecked'] > 200).mean():.0%} / {(c['thr_returned_final'] > 200).mean():.0%}",
            "check votes": c["check_votes"].mean() if "check_votes" in c else float("nan"),
        })  # fmt: skip
    return pd.DataFrame(rows)


#: The returned set's path through a session (#4519): these clicks, then after the end-of-run check.
PATH_CLICKS = (25, 50, 100, 150)
#: The path's first point: the text sort under its own blind GMM cut, before any vote.
TYPED_QUERY = "typed query"


def _path_rows(runs: dict[float, Path]) -> pd.DataFrame:
    """Per beta, off its own sessions: the returned set's precision, recall and F-beta at each click of
    :data:`PATH_CLICKS` and after the check (#4519, the owner's precision-recall curves).

    Means over every run (#4631): a session still in the opening, or one that never trained, at the typed
    query's set; an empty set at precision 0.  ``returned`` is the median size.  A run analyzed before #4519
    has no per-click precision or recall, and its rows read NaN.

    The path starts at the typed query (owner, 2026-10-06: Photo Finish's notch at click 0): the text sort
    cut at its own line in the app, the set a user gets before voting, read off ``balances.csv`` (the
    ``app line`` rule at the ``text`` point) over the same runs.
    """
    rows = []
    for beta, d in sorted(runs.items()):
        c = pd.read_csv(d / "cells.csv")
        rows.append(_typed_query_row(d, beta))
        points = [(str(t), f"_{t}") for t in PATH_CLICKS] + [("after the check", "_final")]
        for point, suffix in points:
            col = lambda m, s=suffix: c.get(f"thr_{m}{s}", pd.Series(np.nan, index=c.index))  # noqa: E731
            have = col("fbeta").notna()
            rows.append({
                "beta": f"{beta:g}",
                "point": point,
                "precision": col("precision")[have].mean(),
                "recall": col("recall")[have].mean(),
                "fbeta": col("fbeta")[have].mean(),
                "returned, median": col("returned")[have].median(),
                "runs": int(have.sum()),
            })  # fmt: skip
    return pd.DataFrame(rows)


def _typed_query_row(d: Path, beta: float) -> dict:
    """The typed query's own returned set at *beta*: the text sort at its own line in the app, every run."""
    row = {"beta": f"{beta:g}", "point": TYPED_QUERY}
    bal = pd.read_csv(d / "balances.csv", dtype={"point": str})
    if "rule" not in bal:  # an analysis from before the rule column: the app's line only
        bal = bal.assign(rule="app line")
    bal = bal[(bal["beta"].round(4) == round(beta, 4)) & (bal["point"] == "text") & (bal["rule"] == "app line")]
    if bal.empty:
        return {**row, "precision": np.nan, "recall": np.nan, "fbeta": np.nan, "returned, median": np.nan, "runs": 0}
    return {
        **row,
        "precision": bal["precision"].mean(),
        "recall": bal["recall"].mean(),
        "fbeta": bal["fbeta"].mean(),
        "returned, median": bal["k"].median(),
        "runs": len(bal),
    }


def figure_path(runs: dict[float, Path], out: Path) -> None:
    """``precision_recall_path.png`` (#4519): per beta, the returned set's (recall, precision) at each click of
    :data:`PATH_CLICKS`, joined in order, ending on the point after the check (✓)."""
    # The typed query sits at precision ~0.01, recall ~0.9: off any useful scale, so the picture starts at
    # the first click; the table and CSV carry it.
    rows = _path_rows(runs)
    rows = rows[rows["point"] != TYPED_QUERY]
    fig, ax = plt.subplots(figsize=(5.6, 4.6), facecolor=SURFACE)
    for (beta, g), color in zip(rows.groupby("beta", sort=False), ("#2a6fdb", "#d9480f", "#2b8a3e", "#7048e8")):
        g = g.dropna(subset=["precision", "recall"])
        if g.empty:
            continue
        ax.plot(g["recall"], g["precision"], color=color, lw=1.8, marker="o", ms=4, label=f"beta {beta}")
        for _, r in g.iterrows():
            tag = "✓" if r["point"] == "after the check" else str(r["point"])
            ax.annotate(tag, (r["recall"], r["precision"]), textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=color)  # fmt: skip
    ax.set_xlabel("recall of the returned set", color=INK)
    ax.set_ylabel("precision of the returned set", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("the returned set through the session", color=INK, fontsize=10, loc="left")
    _axes(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "precision_recall_path.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


#: The clicks the early-dip table reads the session at: past the earliest hand-over (click 23, #4605).
DIP_CLICKS = (25, 50, 100)
#: The early-dip table's word for a series that never falls below the text sort's level.
NEVER_BELOW = "never below"


def _back_by(series: pd.Series, level: float) -> int | str:
    """The click by which *series* is back at *level* after its lowest point past click 0 (:data:`NEVER_BELOW`
    when it never falls below it)."""
    past = series.loc[1:]
    if past.empty or past.min() >= level - 1e-9:
        return NEVER_BELOW
    low = past.idxmin()
    back = series.loc[low:][series.loc[low:] >= level - 1e-9]
    return int(back.index[0]) if len(back) else "not by the end"


def _dip_rows(runs: dict[float, Path]) -> pd.DataFrame:
    """Per beta and rule: the text sort's set at click 0 against the session's over clicks, the same rule on both.

    Every run at every click as its session shows it (#4605, #4631): the typed query's set until the app shows the
    run's detector, and throughout a run that never trained. So the dip is the one a user meets at the hand-over,
    not the opening's own detectors, which no screen shows.
    """
    rows = []
    for beta, d in sorted(runs.items()):
        bal, steps = read = _by_rule(d, beta)
        for rule, m in _share_by_rule(d, beta, read).items():
            s = steps[steps["rule"] == rule]
            t_rows = bal[bal["rule"] == rule]
            f = s.groupby("t")["fbeta"].mean()
            k = s.groupby("t")["k"].median()
            text_f = float(t_rows["fbeta"].mean()) if len(t_rows) else float("nan")
            text_k = float(t_rows["k"].median()) if len(t_rows) else float("nan")
            low_t = int(m.loc[1:].idxmin()) if len(m) > 1 else 0
            at = " / ".join(str(t) for t in DIP_CLICKS)
            rows.append({
                "beta": f"{beta:g}", "rule": rule,
                "text sort: F": text_f, "text sort: returned": text_k, "text sort: share of best": m.loc[0],
                f"session: F at {at}": " / ".join(f"{f.get(t, float('nan')):.2f}" for t in DIP_CLICKS),
                f"session: returned at {at}": " / ".join(f"{k.get(t, float('nan')):.0f}" for t in DIP_CLICKS),
                "session F >= text sort's again by click": _back_by(f, text_f),
                "lowest share": m.loc[low_t], "at click": low_t,
                "share >= text sort's again by click": _back_by(m, float(m.loc[0])),
            })  # fmt: skip
    return pd.DataFrame(rows)


def _check_rows(runs: dict[float, Path]) -> pd.DataFrame:
    """Per beta: the spot check's votes, its range and how often the range held the audited set's truth."""
    rows = []
    for beta, d in sorted(runs.items()):
        c = pd.read_csv(d / "cells.csv")
        if "check_votes" not in c:
            continue
        c = c[c["check_votes"].fillna(0) > 0]
        rows.append({
            "beta": f"{beta:g}", "checks": len(c), "votes": c["check_votes"].mean(),
            "range lo": c["range_lo"].mean(), "range hi": c["range_hi"].mean(),
            "width": (c["range_hi"] - c["range_lo"]).mean(), "truth": c["check_truth"].mean(),
            "covered": c["check_covered"].mean(),
        })  # fmt: skip
    return pd.DataFrame(rows)


def summary_balance(runs: dict[float, Path], out: Path) -> None:
    from analyze import returned_at_beta  # noqa: PLC0415

    path = _path_rows(runs)
    path.to_csv(out / "precision_recall_path.csv", index=False)
    md = [
        "# Each balance read off its own sessions (#4413)",
        "",
        "## The objective",
        "",
        "F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and "
        "50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; "
        "the check's paired effect; the returned set's size.",
        "",
        _md(_objective_rows(runs).round(3), index=False),
        "",
        "## The returned set through the session",
        "",
        "Per beta, the precision and recall of the same withheld set from the typed query (the text sort at "
        "its own blind GMM cut, before any vote), through 25, 50, 100 and 150 clicks, to after the check (#4519): "
        "the path a preset's returned set takes as the user clicks "
        "(`precision_recall_path.png`, `precision_recall_path.csv`). Means over every run, a session still in "
        "the opening or one that never trained at the typed query's set (#4605, #4631); `returned` is the "
        "median size.",
        "",
        _md(path.round(3), index=False),
        "",
        "## The returned set at each balance (rank frames)",
        "",
        "One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - "
        "the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full "
        "labels (#4486). "
        "`top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort "
        "and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the "
        "same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text "
        "sort, 25 and 50 clicks, the end, full labels.",
        "",
    ]
    rows = []
    for beta, d in sorted(runs.items()):
        b = pd.read_csv(d / "balances.csv", dtype={"point": str})
        t = returned_at_beta(b[b["beta"].round(4) == round(beta, 4)], ["arm"]).reset_index()
        t.insert(0, "sessions_at", f"beta {beta:g}")
        rows.append(t)
    table = pd.concat(rows, ignore_index=True)
    table["point"] = pd.Categorical(table["point"].astype(str), HEADLINE_POINTS, ordered=True)
    md += [_md(table, index=False), ""]
    md += [
        "## The early dip",
        "",
        "The session's returned set against the text sort's, the same rule on both sides (#4384): under "
        "`app line` the text sort's own line in the app against the detector's labels line; under `top-K` the "
        "old cap on both. Every run at every click as its session shows it: the typed query's set until the app "
        "shows the run's detector (#4605), and throughout a run that never trained (#4631). F-beta, what each "
        "returned, and the click by which the session's F-beta is back at the text sort's after its lowest "
        "point; then the share of the best cut, which rises with the clicks.",
        "",
        _md(_dip_rows(runs).round(3), index=False),
        "",
        "## The spot check",
        "",
        "`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.",
        "",
        _md(_check_rows(runs).round(3), index=False),
    ]
    (out / "perbeta_summary.md").write_text("\n".join(md) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="P=analysis dir (or beta=dir with --kind balance)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--kind", choices=("floor", "balance"), default="floor", help="the preference the runs are keyed by"
    )
    args = ap.parse_args(argv)
    runs = {float(k): Path(v) for k, _, v in (r.partition("=") for r in args.run)}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.kind == "balance":
        figure_balance(runs, args.out)
        figure_objective(runs, args.out)
        figure_path(runs, args.out)
        summary_balance(runs, args.out)
    else:
        figure(runs, args.out)
        summary(runs, args.out)
    print(f"perp -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
