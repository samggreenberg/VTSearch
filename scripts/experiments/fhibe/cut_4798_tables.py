#!/usr/bin/env python
"""#4798's tables, from ``cut_4798.py``'s ``rules.csv`` (one per preset) and ``prior_4776.py``'s ``prior_ab.csv``.

Writes ``tables.md``: every fix to the total and every gate rule, re-cut on today's sessions, as the
mean paired difference from today ± SE over runs (votes 1-150, then 1-50), faces from 1 and from 4
photos and COCO at each preset; then the arm (the Goods-only gate in every fit's line, its own
sessions) against today.

Usage::

    python cut_4798_tables.py --rules b025=<rules.csv> --rules b1=... --rules b4=... \
        [--ab b4=<prior_ab.csv> ...] --out tables.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BETAS = (("b025", "β ¼"), ("b1", "β 1"), ("b4", "β 4"))
FIXES = ("ptotal", "pchances", "pmin", "capsum", "captotal", "ncap")
GATES = ("headgoods", "floorhead", "caphead", "caphead-dry")


def _signed(x: float) -> str:
    """``+0.012`` / ``−0.012`` (a true minus), and ``0.000`` when it rounds to nothing."""
    v = round(float(x), 3)
    return "0.000" if v == 0 else f"{v:+.3f}".replace("-", "−")


def _cell(row: pd.Series | None, span: str) -> str:
    if row is None:
        return "–"
    return f"{_signed(row[f'dF_{span}'])} ± {row[f'dF_{span}_se']:.3f}"


def _bench(sets: list[str]) -> list[tuple[str, str]]:
    """The columns: (header suffix, set) for faces from 1 photo, faces from 4, COCO."""
    out = []
    for prefix, name in (("k1-", "faces 1"), ("k4-", "faces 4"), ("coco", "COCO")):
        s = [x for x in sets if x.startswith(prefix)]
        if s:
            out.append((name, s[0]))
    return out


def rules_table(rules: dict[str, pd.DataFrame], names: tuple[str, ...], prefix: str, span: str) -> list[str]:
    head = ["rule"]
    cols = []
    for b, label in BETAS:
        if b not in rules:
            continue
        for name, s in _bench(sorted(rules[b]["set"].unique())):
            head.append(f"{label}, {name}")
            cols.append((b, s))
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in names:
        cells = [f"`{n}`"]
        for b, s in cols:
            r = rules[b]
            hit = r[(r["rule"] == f"{prefix}{n}") & (r["set"] == s)]
            cells.append(_cell(hit.iloc[0] if len(hit) else None, span))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def arm_table(abs_: dict[str, pd.DataFrame]) -> list[str]:
    lines = [
        "| preset | set | runs | today, 1–50 | Δ 1–50 | Δ 1–150 | runs gaining / losing (1–150) | ΔAP 1–150 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b, label in BETAS:
        if b not in abs_:
            continue
        for _name, s in _bench(sorted(abs_[b]["set"].unique())):
            r = abs_[b][abs_[b]["set"] == s].iloc[0]
            lines.append(
                f"| {label} | {s} | {int(r['runs'])} | {r['F_ctl_1_50']:.3f} | {_cell(r, '1_50')} | {_cell(r, '1_150')} | "
                f"{100 * r['win_1_150']:.0f}% / {100 * r['loss_1_150']:.0f}% | {_signed(r['dAP_1_150'])} |"
            )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rules", action="append", required=True, help="BETA=rules.csv")
    ap.add_argument("--ab", action="append", default=[], help="BETA=prior_ab.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rules = {k: pd.read_csv(v) for k, _, v in (s.partition("=") for s in args.rules)}
    abs_ = {k: pd.read_csv(v) for k, _, v in (s.partition("=") for s in args.ab)}
    md = ["# #4798's tables", ""]
    for span, words in (("1_150", "votes 1–150"), ("1_50", "votes 1–50")):
        md += [f"## The fixes to the total, {words}", "", *rules_table(rules, FIXES, "cut:", span), ""]
    for span, words in (("1_150", "votes 1–150"), ("1_50", "votes 1–50")):
        md += [f"## Moving the gate, {words}", "", *rules_table(rules, GATES, "", span), ""]
    md += ["## The per-click oracle, votes 1–150 (a bound)", "", *rules_table(rules, ("oracle",), "", "1_150"), ""]
    if abs_:
        md += ["## The arm: the Goods-only gate in every fit's line, against today", "", *arm_table(abs_), ""]
    args.out.write_text("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
