#!/usr/bin/env python
"""#4776's report tables, rendered as markdown from the analysis CSVs (no number copied by hand).

Reads ``handover_4776.py``'s ``rules.csv`` (one per bench) and ``prior_4776.py``'s ``prior_ab.csv``
and writes ``tables.md``:

* the hand-over rules on faces and on COCO: Find's F-beta, session mean over votes 1-50 and 1-150,
  each rule minus today, mean +- SE over runs;
* AP of the head started from the centroid (in the shadow, ranking only) against the centroid's;
* the arm whose head starts from the centroid against today, run for run.

A run that never finds a Good (COCO: 18 of 720 a preset) gives nothing to Test under any rule or
head, so it differs by zero; ``--runs-of coco-b1=720`` counts it in (#4631: every run in every
average), scaling each difference and SE by runs-with-rows / all runs.

Usage::

    python handover_4776_tables.py --rules <rules.csv> ... --ab <prior_ab.csv> ... [--runs-of SET=N ...] --out <tables.md>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RULES = [
    ("never", "centroid kept (no hand-over)"),
    ("goods>=5", "head from 5 Goods"),
    ("bads>=8", "head from 8 Bads"),
    ("bads>=16", "head from 16 Bads"),
    ("bads>=32", "head from 32 Bads"),
    ("judge:mrr", "judge: held-out Goods' MRR"),
    ("judge:mrr sticky", "judge: MRR, head kept once it wins"),
    ("judge:logrank", "judge: mean log rank"),
    ("oracle", "per-click oracle (a bound, not a rule)"),
]
FACE_SETS = [f"k{k}-b{b}-{s}" for s in ("1024", "640") for k in (1, 4) for b in ("025", "1", "4")]
COCO_SETS = ["coco-b025", "coco-b1", "coco-b4"]
BETA = {"025": "β ¼", "1": "β 1", "4": "β 4"}


def _label(s: str) -> str:
    if s.startswith("coco"):
        return BETA[s.split("-b")[1]]
    k, b, size = s.split("-")
    return f"{k[1]} photo{'s' if k != 'k1' else ''}, {BETA[b[1:]]}" + ("" if size == "1024" else f", {size}")


def _pm(m: float, se: float) -> str:
    return f"{m:+.3f} ± {se:.3f}"


def _scale(df: pd.DataFrame, runs_of: dict[str, int]) -> pd.DataFrame:
    """Count runs with no rows as zero differences: scale every difference and its SE."""
    df = df.copy()
    for s, n in runs_of.items():
        m = df["set"] == s
        if not m.any():
            continue
        f = df.loc[m, "runs"] / n
        for c in [c for c in df.columns if c.startswith("d") and ("_1_" in c or "_51_" in c)]:
            df.loc[m, c] = df.loc[m, c] * f
    return df


def rules_table(r: pd.DataFrame, sets: list[str], span: str) -> str:
    sets = [s for s in sets if (r["set"] == s).any()]
    head = "| rule | " + " | ".join(_label(s) for s in sets) + " |"
    lines = [head, "|---|" + "---|" * len(sets)]
    for rule, name in RULES:
        cells = []
        for s in sets:
            x = r[(r["set"] == s) & (r["rule"] == rule)]
            cells.append("" if x.empty else _pm(float(x[f"dF_{span}"].iloc[0]), float(x[f"dF_{span}_se"].iloc[0])))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    today = []
    for s in sets:
        x = r[(r["set"] == s) & (r["rule"] == "today")]
        today.append("" if x.empty else f"{float(x[f'F_{span}'].iloc[0]):.3f}")
    lines.append("| *today, for scale (F-beta)* | " + " | ".join(today) + " |")
    return "\n".join(lines)


def prior_ap_table(r: pd.DataFrame, sets: list[str]) -> str:
    sets = [s for s in sets if (r["set"] == s).any()]
    lines = [
        "| set | centroid kept, 1-50 | head from centroid, 1-50 | centroid kept, 1-150 | head from centroid, 1-150 |",
        "|---|---|---|---|---|",
    ]
    for s in sets:
        n = r[(r["set"] == s) & (r["rule"] == "never")].iloc[0]
        p = r[(r["set"] == s) & (r["rule"] == "prior_ap_s10")].iloc[0]
        lines.append(
            f"| {_label(s)} | {_pm(n['dAP_1_50'], n['dAP_1_50_se'])} | {_pm(p['dAP_1_50'], p['dAP_1_50_se'])} "
            f"| {_pm(n['dAP_1_150'], n['dAP_1_150_se'])} | {_pm(p['dAP_1_150'], p['dAP_1_150_se'])} |"
        )
    return "\n".join(lines)


def ab_table(a: pd.DataFrame, sets: list[str]) -> str:
    sets = [s for s in sets if (a["set"] == s).any()]
    lines = [
        "| set | runs | today, 1-50 | Δ 1-50 | today, 1-150 | Δ 1-150 | Δ 51-150 | wins / losses (1-150) | ΔAP 1-50 | ΔAP 1-150 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sets:
        x = a[a["set"] == s].iloc[0]
        lines.append(
            f"| {_label(s)} | {int(x['runs'])} | {x['F_ctl_1_50']:.3f} | {_pm(x['dF_1_50'], x['dF_1_50_se'])} "
            f"| {x['F_ctl_1_150']:.3f} | {_pm(x['dF_1_150'], x['dF_1_150_se'])} | {_pm(x['dF_51_150'], x['dF_51_150_se'])} "
            f"| {x['win_1_150']:.0%} / {x['loss_1_150']:.0%} | {_pm(x['dAP_1_50'], x['dAP_1_50_se'])} "
            f"| {_pm(x['dAP_1_150'], x['dAP_1_150_se'])} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rules", type=Path, action="append", default=[])
    ap.add_argument("--ab", type=Path, action="append", default=[])
    ap.add_argument("--runs-of", action="append", default=[], help="SET=N: count runs with no rows as zero differences")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs_of = {k: int(v) for k, _, v in (x.partition("=") for x in args.runs_of)}
    r = _scale(pd.concat([pd.read_csv(p) for p in args.rules], ignore_index=True), runs_of) if args.rules else None
    a = _scale(pd.concat([pd.read_csv(p) for p in args.ab], ignore_index=True), runs_of) if args.ab else None
    out = ["# #4776 tables (generated by handover_4776_tables.py)\n"]
    if r is not None:
        for span, words in (("1_50", "votes 1-50"), ("1_150", "votes 1-150")):
            out.append(f"## Hand-over rules, faces 1024 px: Find's F-beta over {words}, rule minus today\n")
            out.append(rules_table(r, FACE_SETS[:6], span) + "\n")
            out.append(f"## Hand-over rules, faces 640 px: {words}\n")
            out.append(rules_table(r, FACE_SETS[6:], span) + "\n")
            out.append(f"## Hand-over rules, COCO Better Binary: {words}\n")
            out.append(rules_table(r, COCO_SETS, span) + "\n")
        out.append("## AP of the withheld half, minus today's head (shadow; ranking only)\n")
        out.append(prior_ap_table(r, [*FACE_SETS, *COCO_SETS]) + "\n")
    if a is not None:
        out.append("## The arm: head started from the centroid, minus today, run for run\n")
        out.append(ab_table(a, [*FACE_SETS, *COCO_SETS]) + "\n")
    args.out.write_text("\n".join(out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
