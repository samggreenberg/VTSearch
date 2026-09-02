"""Aggregate the Autopilot budget sweep into the report's tables.

Reads the two frames :mod:`run_autopilot_sweep` writes — ``autopilot_prod_steps.csv``
(one row per step at the shipped cut) and ``autopilot_prod_budget.csv`` (one row
per step and inclusion ``k``) — and writes ``autopilot_prod_tables.md`` next to
them.

The headline metric is the **budget violation rate**: among steps at inclusion
>= 0, how often the realized held-out FNR exceeds the advertised ``alpha(k)``
cap.  Supporting tables give the operating point the shipped cut lands on,
threshold placement against the oracle reference, how much of the budget the
safe blend rescues, and the vote composition Autopilot arrives at.

Both frames come from the harness (issue #3408), so every number here describes
the shipped detector: the ``linear_svm`` head, the app's phase machine, the
acquisition offset, and the production Train/Calibrate split.  Nothing is
recomputed — ``excess_fnr`` in particular is
:func:`vtscore.eval.arms_inclusion._inclusion_sweep_rows`', not a restatement.
The one derived column is ``excess_pos``, the excess floored at zero, which is
how the committed 2026-07-30 tables reported it; the signed mean is printed
beside it because a *negative* excess (the cut placed conservatively low) is the
mechanism the study's verdict rests on and a floor hides it.

These tables do **not** reproduce the committed ``autopilot_tables.md``: that is
the 2026-07-30 run's record, and the earlier harness measured a different
configuration.

Usage::

    python summarize_autopilot.py [--steps CSV] [--budget CSV] [--out PATH]
"""

from __future__ import annotations

import argparse

import common

common.setup_env()

import pandas as pd  # noqa: E402

#: Vote counts the tables are read at.  Every step is in the frames; these are
#: the four the study reported, kept so the shape stays comparable.
CHECKPOINTS = (12, 24, 50, 100)


def _arm_group(arm: str) -> str:
    return "agnews" if str(arm).startswith("agnews:") else str(arm)


def _fmt(x, digits: int = 3) -> str:
    return "-" if x is None or pd.isna(x) else f"{float(x):.{digits}f}"


def _at_checkpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Rows at the reported vote counts, with the arm group attached."""
    d = df[df["t"].isin(CHECKPOINTS)].copy()
    d["arm_group"] = d["dataset"].map(_arm_group)
    return d


def budget_table(budget: pd.DataFrame) -> str:
    """Budget compliance at inclusion >= 0, by arm group x vote count."""
    d = _at_checkpoints(budget[budget["inclusion_k"] >= 0])
    d["violated"] = d["sweep_fnr"] > d["alpha"]
    d["excess_pos"] = d["excess_fnr"].clip(lower=0.0)
    lines = [
        "| arm group | votes | rows | FNR (mean) | alpha cap (mean) | violation rate "
        "| excess (mean, floored) | excess (p90) | excess (mean, signed) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (grp, t), g in d.groupby(["arm_group", "t"], sort=True):
        lines.append(
            f"| {grp} | {t} | {len(g)} | {_fmt(g['sweep_fnr'].mean())} | {_fmt(g['alpha'].mean())} "
            f"| {_fmt(g['violated'].mean())} | {_fmt(g['excess_pos'].mean())} "
            f"| {_fmt(g['excess_pos'].quantile(0.9))} | {_fmt(g['excess_fnr'].mean())} |"
        )
    return "\n".join(lines)


def excess_by_inclusion_table(budget: pd.DataFrame) -> str:
    """Floored excess per inclusion value, one column per vote count.

    This is the study's headline trajectory: whether the budget converges as the
    calibration set grows, and at which ``k`` the cap becomes finer than a few
    dozen calibration positives can certify.
    """
    d = _at_checkpoints(budget[budget["inclusion_k"] >= 0])
    d["excess_pos"] = d["excess_fnr"].clip(lower=0.0)
    piv = d.pivot_table(index="inclusion_k", columns="t", values="excess_pos", aggfunc="mean")
    votes = [c for c in CHECKPOINTS if c in piv.columns]
    header = " | ".join(f"excess @ {v}" for v in votes)
    lines = [f"| inclusion | alpha cap | {header} |", "|---:|---:|" + "---:|" * len(votes)]
    for k in sorted(d["inclusion_k"].unique()):
        alpha = float(d.loc[d["inclusion_k"] == k, "alpha"].iloc[0])
        cells = " | ".join(_fmt(piv.at[k, v]) for v in votes)
        lines.append(f"| {int(k):+d} | {alpha:.4f} | {cells} |")
    return "\n".join(lines)


def operating_point_table(steps: pd.DataFrame) -> str:
    """Where the shipped cut actually lands, by arm group x vote count."""
    d = _at_checkpoints(steps)
    lines = [
        "| arm group | votes | recall | precision | FPR | flagged / test pool |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for (grp, t), g in d.groupby(["arm_group", "t"], sort=True):
        frac = (g["n_flagged"] / (g["n_test_pos"] + g["n_test_neg"])).mean()
        lines.append(
            f"| {grp} | {t} | {_fmt(g['recall'].mean())} | {_fmt(g['precision'].mean())} "
            f"| {_fmt(g['fpr'].mean())} | {_fmt(frac)} |"
        )
    return "\n".join(lines)


def threshold_placement_table(steps: pd.DataFrame) -> str:
    """Signed shipped-minus-oracle threshold error, and the cost it buys.

    The sign is the mechanism: a cut placed *below* oracle over-admits (recall
    bought with precision), one placed above under-admits (misses, which is what
    the ``alpha(k)`` budget prices).  ``regret`` is the harness's own
    cost difference against the same-step oracle.
    """
    d = _at_checkpoints(steps)
    d["gap"] = d["threshold"] - d["oracle_threshold"]
    lines = [
        "| arm group | votes | threshold - oracle (mean) | (p90) | regret (mean) |",
        "|---|---:|---:|---:|---:|",
    ]
    for (grp, t), g in d.groupby(["arm_group", "t"], sort=True):
        lines.append(
            f"| {grp} | {t} | {_fmt(g['gap'].mean())} | {_fmt(g['gap'].quantile(0.9))} | {_fmt(g['regret'].mean())} |"
        )
    return "\n".join(lines)


def blend_table(steps: pd.DataFrame, budget: pd.DataFrame) -> str:
    """Does the safe blend rescue the budget at the default inclusion?

    The budget frame's ``sweep_fnr`` at ``k=0`` is the raw cross-calibration cut;
    the steps frame's ``fnr`` is the shipped one, which blends that cut with the
    population fit.  Paired on the same step, so the difference is the blend.
    """
    key = ["dataset", "seed", "t"]
    raw = _at_checkpoints(budget[budget["inclusion_k"] == 0]).set_index(key)
    shipped = _at_checkpoints(steps).set_index(key)
    j = raw[["alpha", "sweep_fnr"]].join(shipped[["fnr"]], how="inner").reset_index()
    j["violated_raw"] = j["sweep_fnr"] > j["alpha"]
    j["violated_shipped"] = j["fnr"] > j["alpha"]
    lines = [
        "| votes | violation (raw conformal) | violation (shipped blend) | FNR (raw) | FNR (shipped) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for t, g in j.groupby("t", sort=True):
        lines.append(
            f"| {t} | {_fmt(g['violated_raw'].mean())} | {_fmt(g['violated_shipped'].mean())} "
            f"| {_fmt(g['sweep_fnr'].mean())} | {_fmt(g['fnr'].mean())} |"
        )
    return "\n".join(lines)


def composition_table(steps: pd.DataFrame) -> str:
    """What Autopilot's vote set looks like, and which phase it reached.

    The Good:Bad ratio is emergent under the phase machine rather than a knob,
    and the phase says *why*: a run still in ``hard`` is refining the boundary,
    one in ``new`` has moved on to atlas coverage.
    """
    d = _at_checkpoints(steps)
    d["good_frac"] = d["n_good"] / (d["n_good"] + d["n_bad"])
    lines = ["| arm group | votes | good vote frac (mean) | phases reached |", "|---|---:|---:|---|"]
    for (grp, t), g in d.groupby(["arm_group", "t"], sort=True):
        phases = ", ".join(f"{p} x{n}" for p, n in g["phase"].value_counts().items())
        lines.append(f"| {grp} | {t} | {_fmt(g['good_frac'].mean(), 2)} | {phases} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the Autopilot budget sweep.")
    parser.add_argument("--steps", default=str(common.RESULTS / "autopilot_prod_steps.csv"))
    parser.add_argument("--budget", default=str(common.RESULTS / "autopilot_prod_budget.csv"))
    parser.add_argument("--out", default=str(common.RESULTS / "autopilot_prod_tables.md"))
    args = parser.parse_args(argv)

    steps = pd.read_csv(args.steps)
    budget = pd.read_csv(args.budget)

    cells = len(steps.drop_duplicates(subset=["dataset", "seed"]))
    head = steps["head"].dropna().unique()
    headline = _at_checkpoints(budget[budget["inclusion_k"] >= 0])
    rate = (headline["sweep_fnr"] > headline["alpha"]).mean()

    sections = [
        "# Autopilot budget sweep: summary tables",
        "",
        f"{cells} cells (arm x seed), head `{'/'.join(head)}`, "
        f"steps read at {', '.join(str(c) for c in CHECKPOINTS)} votes. "
        f"Budget violation rate at inclusion >= 0: {rate:.3f}.",
        "",
        "## Conformal budget compliance (inclusion >= 0)",
        "",
        budget_table(budget),
        "",
        "## Excess by inclusion value",
        "",
        excess_by_inclusion_table(budget),
        "",
        "## Operating point at the shipped cut",
        "",
        operating_point_table(steps),
        "",
        "## Threshold placement vs oracle",
        "",
        threshold_placement_table(steps),
        "",
        "## Safe-blend mitigation reach (inclusion 0)",
        "",
        blend_table(steps, budget),
        "",
        "## Vote composition and phase",
        "",
        composition_table(steps),
        "",
    ]
    with open(args.out, "w") as f:
        f.write("\n".join(sections))
    common.log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
