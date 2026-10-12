#!/usr/bin/env python
"""#4786's report tables, rendered as markdown from the analysis CSVs (no number copied by hand).

Reads ``line_4786.py``'s ``rules.csv`` and ``first_clicks.csv`` (one pair per bench, or one for
both) and ``prior_4776.py``'s ``prior_ab.csv`` for the full-session arm, and writes ``tables.md``:

* the count rules at beta 4, each at the shipped gate: Find's F-beta, session mean over votes
  1-50 and 1-150, each rule minus today, mean +- SE over runs;
* the head's count line under every gate;
* the head's count line at beta 1/4 and 1, where it does not ship;
* the arm (the rule in every fit's line, full sessions) against today, run for run;
* the first trained click: what each line keeps and what it scores there.

A run that never finds a Good (COCO) gives nothing to Test under any rule, so it differs by zero;
``--runs-of coco-b4=720`` counts it in (#4631: every run in every average), scaling each
difference and SE by runs-with-rows / all runs.

Usage::

    python line_4786_tables.py --rules <rules.csv> ... --first <first_clicks.csv> ... \
        [--ab <prior_ab.csv> ...] [--runs-of SET=N ...] [--gate neg20] --out <tables.md>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RULES = [
    ("headcountb1", "the head's count, never fewer than the beta-1 line's (shipped)"),
    ("headcount", "the head's count (the centroid's rule on the head's scores)"),
    ("headcountmin", "the fewer of today's and the head's count"),
    ("carry", "the centroid's count carried"),
    ("carrymin", "the fewer of today's and the centroid's count"),
    ("cap2g", "today's, at most 2 x the Goods in hand"),
    ("cap3g", "today's, at most 3 x the Goods in hand"),
    ("scaled", "today's, at most beta^0.708 x the beta-1 line's"),
    ("b1line", "the beta-1 labels line"),
    ("centroid", "the centroid's own Test (#4776's hand-over)"),
]
GATES = [
    ("16", "fewer than 16 Bads in hand"),
    ("32", "fewer than 32 Bads in hand"),
    ("48", "fewer than 48 Bads in hand"),
    ("inf", "every trained click"),
    ("neg12", "fewer than 12 held-out Bads"),
    ("neg20", "fewer than 20 held-out Bads"),
    ("neg32", "fewer than 32 held-out Bads"),
    ("neg20pos6", "thin: fewer than 20 held-out Bads and 6 held-out Goods"),
    ("neg32pos4", "thin: fewer than 32 held-out Bads and 4 held-out Goods"),
    ("neg32pos6", "thin: fewer than 32 held-out Bads and 6 held-out Goods (shipped)"),
    ("neg32pos8", "thin: fewer than 32 held-out Bads and 8 held-out Goods"),
    ("neg1000pos6", "fewer than 6 held-out Goods, any Bads"),
    ("sep1.5", "d' below 1.5"),
    ("sep2", "d' below 2"),
]
FACE_B4 = [f"k{k}-b4-{s}" for s in ("1024", "640") for k in (1, 4)]


def _fmt(d: float, se: float) -> str:
    return f"{d:+.3f} ± {se:.3f}"


def _scale(rules: pd.DataFrame, runs_of: dict[str, int]) -> pd.DataFrame:
    out = rules.copy()
    for name, n in runs_of.items():
        m = out["set"] == name
        if not m.any():
            continue
        f = out.loc[m, "runs"].astype(float) / float(n)
        for c in [c for c in out.columns if c.startswith("dF_")]:
            out.loc[m, c] = out.loc[m, c] * f
        out.loc[m, "runs"] = n
    return out


def _table(rules: pd.DataFrame, sets: list[str], names: list[tuple[str, str]], span: str) -> list[str]:
    have = [s for s in sets if (rules["set"] == s).any()]
    lines = ["| rule | " + " | ".join(have) + " |", "|---|" + "---|" * len(have)]
    for rule, label in names:
        row = [label]
        for s in have:
            r = rules[(rules["set"] == s) & (rules["rule"] == rule)]
            row.append(_fmt(float(r[f"dF_{span}"].iloc[0]), float(r[f"dF_{span}_se"].iloc[0])) if len(r) else "")
        lines.append("| " + " | ".join(row) + " |")
    today = ["*today, for scale (F-beta)*"]
    for s in have:
        r = rules[(rules["set"] == s) & (rules["rule"] == "today")]
        today.append(f"{float(r[f'F_{span}'].iloc[0]):.3f}" if len(r) else "")
    lines.append("| " + " | ".join(today) + " |")
    runs = ["*runs*"] + [
        str(int(rules[(rules["set"] == s) & (rules["rule"] == "today")]["runs"].iloc[0])) for s in have
    ]
    lines.append("| " + " | ".join(runs) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rules", action="append", type=Path, required=True)
    ap.add_argument("--first", action="append", type=Path, default=[])
    ap.add_argument("--ab", action="append", type=Path, default=[])
    ap.add_argument("--runs-of", action="append", default=[], help="SET=N: count N runs, the missing ones as zero")
    ap.add_argument("--gate", default="neg20")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs_of = {k: int(v) for k, v in (x.split("=") for x in args.runs_of)}
    rules = _scale(pd.concat([pd.read_csv(p) for p in args.rules], ignore_index=True), runs_of)
    sets = list(dict.fromkeys(rules["set"]))
    b4 = [s for s in sets if s.endswith("-b4") or "-b4-" in s]
    low = [s for s in sets if s not in b4]
    g = args.gate
    out: list[str] = []
    for span, title in (("1_50", "votes 1-50"), ("1_150", "votes 1-150")):
        out += [f"### The count rules at beta 4, until the gate ({g}), {title}", ""]
        names = [(f"{r}@{g}", label) for r, label in RULES] + [("oracle", "per-click oracle (a bound, not a rule)")]
        out += _table(rules, b4, names, span) + [""]
    for span, title in (("1_50", "votes 1-50"), ("1_150", "votes 1-150")):
        out += [f"### The head's count under each gate, beta 4, {title}", ""]
        out += _table(rules, b4, [(f"headcountb1@{x}", label) for x, label in GATES], span) + [""]
    if low:
        for span, title in (("1_50", "votes 1-50"), ("1_150", "votes 1-150")):
            out += [f"### Were it drawn at beta 1/4 and 1 ({g}), {title}", ""]
            out += _table(
                rules,
                low,
                [(f"headcountb1@{g}", "the head's count, floored at beta 1"), ("oracle", "per-click oracle")],
                span,
            )
            out += [""]
    if args.ab:
        ab = pd.concat([pd.read_csv(p) for p in args.ab], ignore_index=True)
        out += [
            "### The arm against today, run for run (full sessions)",
            "",
            "| set | runs | today, 1-50 | Δ 1-50 | today, 1-150 | Δ 1-150 | Δ 51-150 | wins / losses (1-150) | ΔAP 1-50 | kept at vote 8, today → arm (median) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in ab.iterrows():
            n = runs_of.get(str(r["set"]), int(r["runs"]))
            f = float(r["runs"]) / n  # runs with no row differ by zero (#4631)
            out.append(
                f"| {r['set']} | {n} | {r['F_ctl_1_50']:.3f} | {_fmt(r['dF_1_50'] * f, r['dF_1_50_se'] * f)} | "
                f"{r['F_ctl_1_150']:.3f} | {_fmt(r['dF_1_150'] * f, r['dF_1_150_se'] * f)} | "
                f"{_fmt(r['dF_51_150'] * f, r['dF_51_150_se'] * f)} | "
                f"{r['win_1_150'] * f:.0%} / {r['loss_1_150'] * f:.0%} | {_fmt(r['dAP_1_50'] * f, r['dAP_1_50_se'] * f)} | "
                f"{r['kept_ctl_median_8']:g} → {r['kept_arm_median_8']:g} |"
            )
        out.append("")
    if args.first:
        fc = pd.concat([pd.read_csv(p) for p in args.first], ignore_index=True)
        out += [
            "### The first trained click",
            "",
            "| set | runs | withheld photos (median) | today keeps (median / p90) | the head's count keeps (median / p90) | F-beta: centroid before → today | → the head's count |",
            "|---|---|---|---|---|---|---|",
        ]
        for s, d in fc.groupby("set", sort=False):
            out.append(
                f"| {s} | {len(d)} | {d['n_test_pos'].median():g} | {d['k_test'].median():g} / {d['k_test'].quantile(0.9):.0f} | "
                f"{d['k_headcountb1'].median():g} / {d['k_headcountb1'].quantile(0.9):.0f} | "
                f"{d['last_centroid_f'].mean():.2f} → {d['f_today'].mean():.2f} | {d['f_headcountb1'].mean():.2f} |"
            )
        out.append("")
    args.out.write_text("\n".join(out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
