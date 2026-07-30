"""Aggregate the **Autopilot** selection-bias sweep into the report's tables.

Reads ``docs/experiments/inclusion-knob/autopilot_sweep.csv`` (written by
:mod:`run_autopilot_sweep`, whose biased arm is the repo's own Autopilot
selector) and writes ``autopilot_tables.md`` next to it.

The headline metric is the **budget violation rate**: among conformal cells at
inclusion >= 0, how often the realized held-out FNR exceeds the advertised
``alpha(k)`` cap, split by vote policy.  Supporting tables quantify threshold
placement against the oracle reference, the safe-blend's mitigation reach, the
vote-set composition drift, and - because margin sampling is expected to
mis-place the cut in the *opposite* direction from top-greedy labeling - the
operating point (recall vs precision vs FPR) each policy actually lands on.

Usage::

    python summarize_autopilot.py [--csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _arm_group(arm: str) -> str:
    return "agnews" if arm.startswith("agnews:") else arm


def _fmt(x: float, digits: int = 3) -> str:
    return "-" if pd.isna(x) else f"{x:.{digits}f}"


def budget_table(df: pd.DataFrame) -> str:
    """Conformal budget compliance at inclusion >= 0, by policy x arm group."""
    d = df[(df["design"] == "conformal") & (df["inclusion"] >= 0)].copy()
    d["violated"] = d["fnr"] > d["alpha_cap"]
    lines = [
        "| arm group | policy | cells | FNR (mean) | alpha cap (mean) | violation rate | FNR excess (mean) | FNR excess (p90) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (grp, policy), g in d.groupby([d["arm"].map(_arm_group), "policy"], sort=True):
        lines.append(
            f"| {grp} | {policy} | {len(g)} | {_fmt(g['fnr'].mean())} | {_fmt(g['alpha_cap'].mean())} "
            f"| {_fmt(g['violated'].mean())} | {_fmt(g['fnr_excess'].mean())} "
            f"| {_fmt(g['fnr_excess'].quantile(0.9))} |"
        )
    return "\n".join(lines)


def violation_by_inclusion_table(df: pd.DataFrame) -> str:
    """Violation rate per inclusion value (conformal only), uniform vs autopilot."""
    d = df[(df["design"] == "conformal") & (df["inclusion"] >= 0)].copy()
    d["violated"] = d["fnr"] > d["alpha_cap"]
    piv_v = d.pivot_table(index="inclusion", columns="policy", values="violated", aggfunc="mean")
    piv_e = d.pivot_table(index="inclusion", columns="policy", values="fnr_excess", aggfunc="mean")
    lines = [
        "| inclusion | alpha cap | violation (uniform) | violation (autopilot) | FNR excess (uniform) | FNR excess (autopilot) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in sorted(d["inclusion"].unique()):
        alpha = float(d.loc[d["inclusion"] == k, "alpha_cap"].iloc[0])
        lines.append(
            f"| {k:+d} | {alpha:.4f} | {_fmt(piv_v.at[k, 'uniform'])} | {_fmt(piv_v.at[k, 'autopilot'])} "
            f"| {_fmt(piv_e.at[k, 'uniform'])} | {_fmt(piv_e.at[k, 'autopilot'])} |"
        )
    return "\n".join(lines)


def inflation_table(df: pd.DataFrame) -> str:
    """Conformal-vs-oracle threshold gap by policy (inclusion 0, the default)."""
    d = df[df["inclusion"] == 0]
    key = ["arm", "seed", "n_votes", "policy"]
    conf = d[d["design"] == "conformal"].set_index(key)["threshold"]
    orac = d[d["design"] == "oracle"].set_index(key)["threshold"]
    gap = (conf - orac).rename("gap").reset_index()
    gap["arm_group"] = gap["arm"].map(_arm_group)
    lines = [
        "| arm group | policy | threshold - oracle (mean) | (p90) |",
        "|---|---|---:|---:|",
    ]
    for (grp, policy), g in gap.groupby(["arm_group", "policy"], sort=True):
        lines.append(f"| {grp} | {policy} | {_fmt(g['gap'].mean())} | {_fmt(g['gap'].quantile(0.9))} |")
    return "\n".join(lines)


def blend_table(df: pd.DataFrame) -> str:
    """Does the production safe-blend rescue the budget?  By vote count."""
    d = df[(df["inclusion"] >= 0) & (df["design"].isin(["conformal", "blend"])) & (df["policy"] == "autopilot")].copy()
    d["violated"] = d["fnr"] > d["alpha_cap"]
    piv_v = d.pivot_table(index="n_votes", columns="design", values="violated", aggfunc="mean")
    piv_e = d.pivot_table(index="n_votes", columns="design", values="fnr_excess", aggfunc="mean")
    lines = [
        "| votes | violation (conformal) | violation (blend) | FNR excess (conformal) | FNR excess (blend) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n in sorted(d["n_votes"].unique()):
        lines.append(
            f"| {n} | {_fmt(piv_v.at[n, 'conformal'])} | {_fmt(piv_v.at[n, 'blend'])} "
            f"| {_fmt(piv_e.at[n, 'conformal'])} | {_fmt(piv_e.at[n, 'blend'])} |"
        )
    return "\n".join(lines)


def composition_table(df: pd.DataFrame) -> str:
    """Vote-set composition drift per policy: class ratio + quantile shift."""
    cell_cols = ["arm", "seed", "n_votes", "policy", "vote_pos_frac", "cal_pos_q25", "test_pos_q25"]
    cells = df[cell_cols].drop_duplicates(subset=["arm", "seed", "n_votes", "policy"]).copy()
    cells["q25_shift"] = cells["cal_pos_q25"] - cells["test_pos_q25"]
    cells["arm_group"] = cells["arm"].map(_arm_group)
    lines = [
        "| arm group | policy | vote pos frac (mean) | cal q25 - pool q25 (mean) | (p90) |",
        "|---|---|---:|---:|---:|",
    ]
    for (grp, policy), g in cells.groupby(["arm_group", "policy"], sort=True):
        lines.append(
            f"| {grp} | {policy} | {_fmt(g['vote_pos_frac'].mean(), 2)} "
            f"| {_fmt(g['q25_shift'].mean())} | {_fmt(g['q25_shift'].quantile(0.9))} |"
        )
    return "\n".join(lines)


def operating_point_table(df: pd.DataFrame) -> str:
    """Where each policy's default (k=0) cut actually lands.

    Top-greedy labeling mis-places the cut *high* (misses); margin sampling is
    expected to mis-place it *low* (false alarms).  Recall vs precision vs FPR
    distinguishes the two failure directions, which ``fnr_excess`` alone cannot.
    """
    d = df[(df["design"] == "conformal") & (df["inclusion"] == 0)]
    lines = [
        "| arm group | policy | recall | precision | FPR | n included / pool |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (grp, policy), g in d.groupby([d["arm"].map(_arm_group), "policy"], sort=True):
        frac = (g["n_included"] / g["pool_size"]).mean()
        lines.append(
            f"| {grp} | {policy} | {_fmt(g['recall'].mean())} | {_fmt(g['precision'].mean())} "
            f"| {_fmt(g['fpr'].mean())} | {_fmt(frac)} |"
        )
    return "\n".join(lines)


def threshold_direction_table(df: pd.DataFrame) -> str:
    """Signed conformal-minus-oracle threshold error by policy x vote count."""
    d = df[df["inclusion"] == 0]
    key = ["arm", "seed", "n_votes", "policy"]
    conf = d[d["design"] == "conformal"].set_index(key)["threshold"]
    orac = d[d["design"] == "oracle"].set_index(key)["threshold"]
    gap = (conf - orac).rename("gap").reset_index()
    piv = gap.pivot_table(index="n_votes", columns="policy", values="gap", aggfunc="mean")
    lines = ["| votes | threshold - oracle (uniform) | (autopilot) |", "|---:|---:|---:|"]
    for n in sorted(gap["n_votes"].unique()):
        lines.append(f"| {n} | {_fmt(piv.at[n, 'uniform'])} | {_fmt(piv.at[n, 'autopilot'])} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the selection-bias sweep.")
    parser.add_argument("--csv", default=str(common.RESULTS / "autopilot_sweep.csv"))
    parser.add_argument("--out", default=str(common.RESULTS / "autopilot_tables.md"))
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    n_cells = len(df.drop_duplicates(subset=["arm", "seed", "n_votes", "policy"]))
    headline = df[(df["design"] == "conformal") & (df["inclusion"] >= 0)].copy()
    headline["violated"] = headline["fnr"] > headline["alpha_cap"]
    by_policy = headline.groupby("policy")["violated"].mean()

    sections = [
        "# Selection-bias sweep: summary tables",
        "",
        f"{n_cells} cells (arm x seed x votes x policy); "
        f"budget violation rate at inclusion >= 0: "
        f"uniform {by_policy.get('uniform', np.nan):.3f}, autopilot {by_policy.get('autopilot', np.nan):.3f}.",
        "",
        "## Conformal budget compliance (inclusion >= 0)",
        "",
        budget_table(df),
        "",
        "## Violation rate by inclusion value",
        "",
        violation_by_inclusion_table(df),
        "",
        "## Operating point at the default (inclusion 0)",
        "",
        operating_point_table(df),
        "",
        "## Threshold placement vs oracle (inclusion 0)",
        "",
        inflation_table(df),
        "",
        "### Signed threshold error by vote count",
        "",
        threshold_direction_table(df),
        "",
        "## Safe-blend mitigation reach (autopilot policy)",
        "",
        blend_table(df),
        "",
        "## Vote-set composition drift",
        "",
        composition_table(df),
        "",
    ]
    out = args.out
    with open(out, "w") as f:
        f.write("\n".join(sections))
    common.log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
