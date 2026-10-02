#!/usr/bin/env python
"""Compare arbitrary arms (analysis-binary dirs) against a control, at one beta (#4428 sweep, #4427 walk arms).

The objective first (owner 2026-10-01): the returned set's F-beta at the session's beta on the withheld half,
then AP, Goods, the kept size, precision, recall, the share of the best cut, the check's votes, and the
own-corpus diagnostic (kept-set precision there, positives left unvoted, positives in hand).  Deltas are
paired by cell (category x seed), ± SE.

    python compare_arms.py --beta 1 --control <dir> --arm name=<dir> [name=<dir> ...] [--out file.md]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

KEY = ["category", "seed"]


def _own_balance(d: Path, beta: float) -> pd.DataFrame:
    b = pd.read_csv(d / "balances.csv")
    b = b[(b["point"] == "final") & (b["beta"].sub(beta).abs() < 1e-9)]
    return b[KEY + ["band", "k", "precision", "recall", "fbeta", "oracle_fbeta", "fb_share"]]


def _cells(d: Path) -> pd.DataFrame:
    c = pd.read_csv(d / "cells.csv")
    c = c[~c["never_trained"].astype(bool)]
    keep = ["band", "final_ap", "goods_150", "check_votes", "check_status"]
    for extra in (
        "thr_fbeta_unchecked",
        "thr_fbeta_final",
        "thr_walk_effect",
        "thr_precision_final",
        "thr_recall_final",
        "thr_returned_unchecked",
        "thr_returned_final",
        "final_in_hand",
        "final_pool_precision",
        "final_pool_positives",
        "final_pool_k",
    ):
        if extra in c:
            keep.append(extra)
    return c[KEY + keep]


def _paired(ctl: pd.DataFrame, arm: pd.DataFrame, col: str) -> tuple[float, float, int]:
    m = ctl.merge(arm, on=KEY, suffixes=("_c", "_a"))
    diff = (m[f"{col}_a"] - m[f"{col}_c"]).dropna()
    if diff.empty:
        return float("nan"), float("nan"), 0
    se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    return float(diff.mean()), float(se), int(len(diff))


def _fmt(x: float, digits: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _pm(m: float, se: float, digits: int = 3) -> str:
    return f"{'+' if m >= 0 else ''}{_fmt(m, digits)} ± {_fmt(se, digits)}"


def compare(beta: float, control: tuple[str, Path], arms: list[tuple[str, Path]]) -> str:
    rows = {name: (_cells(d), _own_balance(d, beta)) for name, d in [control, *arms]}
    ctl_c, ctl_b = rows[control[0]]
    out = [f"## beta {beta:g}: click 150 (control: {control[0]})\n"]
    if all("thr_fbeta_final" in c for c, _b in rows.values()):
        out.append(
            "**The objective: the withheld set above the app's threshold** (F-beta at the session's beta; paired d vs the control):\n"
        )
        out.append(
            "| arm | unchecked | after the check | walk's effect | precision after | recall after | returned (unchecked → checked) | d after the check | d unchecked |"
        )
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, (c, _b) in rows.items():
            d_after = _paired(ctl_c, c, "thr_fbeta_final") if name != control[0] else (0.0, float("nan"), 0)
            d_before = _paired(ctl_c, c, "thr_fbeta_unchecked") if name != control[0] else (0.0, float("nan"), 0)
            eff = c["thr_walk_effect"]
            out.append(
                f"| {name} | {_fmt(c['thr_fbeta_unchecked'].mean())} | **{_fmt(c['thr_fbeta_final'].mean())}** | "
                f"{_pm(eff.mean(), eff.std(ddof=1) / math.sqrt(len(eff)))} | {_fmt(c['thr_precision_final'].mean())} | "
                f"{_fmt(c['thr_recall_final'].mean())} | {_fmt(c['thr_returned_unchecked'].mean(), 1)} → {_fmt(c['thr_returned_final'].mean(), 1)} | "
                f"{'–' if name == control[0] else _pm(*d_after[:2])} | {'–' if name == control[0] else _pm(*d_before[:2])} |"
            )
        out.append("")
        out.append("The rank-count reading (the balance's rule re-drawn on the fresh ranking; secondary):\n")
    out.append(
        "| arm | cells | **F-beta at the line** | share of best | kept | precision | recall | final AP | Goods @150 | check votes | own-corpus kept precision | left unvoted | in hand |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, (c, b) in rows.items():
        own = (
            f"{_fmt(c['final_pool_precision'].mean())} | {_fmt(c['final_pool_positives'].mean(), 1)} | {_fmt(c['final_in_hand'].mean(), 1)}"
            if "final_in_hand" in c
            else "– | – | –"
        )
        out.append(
            f"| {name} | {len(c)} | **{_fmt(b['fbeta'].mean())}** | {_fmt(b['fb_share'].mean())} | {_fmt(b['k'].mean(), 1)} | "
            f"{_fmt(b['precision'].mean())} | {_fmt(b['recall'].mean())} | {_fmt(c['final_ap'].mean())} | "
            f"{_fmt(c['goods_150'].mean(), 1)} | {_fmt(c['check_votes'].mean(), 1)} | {own} |"
        )
    out.append("")
    out.append("Paired against the control (cell = category x seed), mean difference ± SE:\n")
    out.append(
        "| arm | **d F-beta** | d share | d kept | d precision | d recall | d AP | d Goods | d check votes | n |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, (c, b) in rows.items():
        if name == control[0]:
            continue
        bb = [_paired(ctl_b, b, col) for col in ("fbeta", "fb_share", "k", "precision", "recall")]
        cc = [_paired(ctl_c, c, col) for col in ("final_ap", "goods_150", "check_votes")]
        out.append(
            f"| {name} | **{_pm(*bb[0][:2])}** | {_pm(*bb[1][:2])} | {_pm(*bb[2][:2], 1)} | {_pm(*bb[3][:2])} | "
            f"{_pm(*bb[4][:2])} | {_pm(*cc[0][:2])} | {_pm(*cc[1][:2], 1)} | {_pm(*cc[2][:2], 1)} | {bb[0][2]} |"
        )
    out.append("")
    out.append("By size band, F-beta at the line (paired d vs the control):\n")
    out.append("| band | " + " | ".join(n if n == control[0] else f"{n} (d)" for n in rows) + " |")
    out.append("|---|" + "---:|" * len(rows))
    for band in ("small", "medium", "large"):
        cb = ctl_b[ctl_b["band"] == band]
        cells_out = []
        for name, (_c, b) in rows.items():
            bb = b[b["band"] == band]
            if name == control[0]:
                cells_out.append(_fmt(bb["fbeta"].mean()))
            else:
                m, _se, _n = _paired(cb, bb, "fbeta")
                cells_out.append(f"{_fmt(bb['fbeta'].mean())} ({'+' if m >= 0 else ''}{_fmt(m)})")
        out.append(f"| {band} | " + " | ".join(cells_out) + " |")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--control", required=True, help="name=dir or dir (the arm's analysis-binary)")
    ap.add_argument("--arm", action="append", default=[], help="name=dir")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    def parse(spec: str) -> tuple[str, Path]:
        name, _, d = spec.partition("=") if "=" in spec else (Path(spec).parent.name, "", spec)
        return name, Path(d)

    text = compare(args.beta, parse(args.control), [parse(a) for a in args.arm])
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
