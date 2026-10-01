#!/usr/bin/env python
"""Compare the acquisition arms per beta (#4409 re-keyed under #4413 step 5).

For each beta, the three arms - ``ctl`` (the shipped line - 4 acquisition),
``x1.0`` (the acquisition cut at the F-beta argmax) and ``x0.5`` (at half its
depth) - are read off their ``analysis-binary`` tables and set side by side:
the ranking (final AP, Goods found), the returned set at the session's own
beta at click 150 (kept, precision, recall, F-beta, share of the best cut),
and the check.  Deltas against the control are paired by cell (category x
seed) with a standard error over the cells, and broken out by size band.

    python compare_acq.py --root /expscratch/sgreenberg/p-aware-acq-4409 --betas 0.5 1 2 --out acq_summary.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ARMS = ("ctl", "1.0", "0.5")
KEY = ["category", "seed"]


def _arm_dir(root: Path, beta: str, arm: str) -> Path:
    return root / f"b{beta}-x{arm}" / "analysis-binary"


def _own_balance(d: Path, beta: float) -> pd.DataFrame:
    b = pd.read_csv(d / "balances.csv")
    b = b[(b["point"] == "final") & (b["beta"].sub(beta).abs() < 1e-9)]
    return b[KEY + ["band", "k", "precision", "recall", "fbeta", "oracle_fbeta", "fb_share"]]


def _cells(d: Path) -> pd.DataFrame:
    c = pd.read_csv(d / "cells.csv")
    c = c[~c["never_trained"].astype(bool)]
    keep = [
        "band",
        "final_ap",
        "ceiling_ap",
        "goods_25",
        "goods_50",
        "goods_100",
        "goods_150",
        "positives_found",
        "check_status",
        "check_votes",
    ]
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


def compare(root: Path, beta: str) -> str:  # noqa: C901
    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for arm in ARMS:
        d = _arm_dir(root, beta, arm)
        if not (d / "cells.csv").exists():
            continue
        tables[arm] = (_cells(d), _own_balance(d, float(beta)))
    if "ctl" not in tables:
        return f"## beta {beta}\n\n(no control analysis yet)\n"
    out = [f"## beta {beta}: the returned set at its own balance, click 150 (Binary, 5 seeds)\n"]
    out.append(
        "| arm | cells | final AP | Goods @25 / 50 / 100 / 150 | check votes | kept | precision | recall | F-beta | best F-beta | share | checked |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, (c, b) in tables.items():
        checked = (c["check_status"] == "checked").mean() if "check_status" in c else float("nan")
        goods = " / ".join(_fmt(c[f"goods_{t}"].mean(), 1) for t in (25, 50, 100, 150))
        out.append(
            f"| {'line-4 (shipped)' if arm == 'ctl' else 'argmax x' + arm} | {len(c)} | {_fmt(c['final_ap'].mean())} | "
            f"{goods} | {_fmt(c['check_votes'].mean(), 1)} | "
            f"{_fmt(b['k'].mean(), 1)} | {_fmt(b['precision'].mean())} | {_fmt(b['recall'].mean())} | {_fmt(b['fbeta'].mean())} | "
            f"{_fmt(b['oracle_fbeta'].mean())} | **{_fmt(b['fb_share'].mean())}** | {_fmt(checked, 2)} |"
        )
    out.append("")
    ctl_c, ctl_b = tables["ctl"]
    out.append("Paired against the control (cell = category x seed), mean difference ± SE:\n")
    out.append("| arm | d share | d F-beta | d precision | d recall | d kept | d Goods @150 | d final AP | n |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, (c, b) in tables.items():
        if arm == "ctl":
            continue
        cells = [_paired(ctl_b, b, col) for col in ("fb_share", "fbeta", "precision", "recall", "k")]
        more = [_paired(ctl_c, c, col) for col in ("goods_150", "final_ap")]
        row = " | ".join(f"{_fmt(m)} ± {_fmt(se)}" for m, se, _n in cells + more)
        out.append(f"| argmax x{arm} | {row} | {cells[0][2]} |")
    out.append("")
    out.append(
        "By size band at click 150 (share of the best F-beta, then F-beta / Goods / final AP; paired d vs the control):\n"
    )
    out.append("| band | metric | " + " | ".join("line-4" if a == "ctl" else f"x{a} (d)" for a in tables) + " |")
    out.append("|---|---|" + "---:|" * len(tables))
    for band in ("small", "medium", "large"):
        for metric, src, digits in (
            ("share", "b", 3),
            ("F-beta", "b", 3),
            ("Goods @150", "c", 1),
            ("final AP", "c", 3),
        ):
            col = {"share": "fb_share", "F-beta": "fbeta", "Goods @150": "goods_150", "final AP": "final_ap"}[metric]
            cells_out = []
            ctl_t = ctl_b if src == "b" else ctl_c
            cb = ctl_t[ctl_t["band"] == band]
            for arm, (c, b) in tables.items():
                t = b if src == "b" else c
                bb = t[t["band"] == band]
                if arm == "ctl":
                    cells_out.append(_fmt(bb[col].mean(), digits))
                else:
                    m, _se, _n = _paired(cb, bb, col)
                    cells_out.append(f"{_fmt(bb[col].mean(), digits)} ({'+' if m >= 0 else ''}{_fmt(m, digits)})")
            out.append(f"| {band} | {metric} | " + " | ".join(cells_out) + " |")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/expscratch/sgreenberg/p-aware-acq-4409"))
    ap.add_argument("--betas", nargs="+", default=["0.5", "1", "2"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    text = "# Acquisition at the F-beta argmax vs the shipped line - 4, per beta (#4409 / #4413 step 5)\n\n"
    text += "\n".join(compare(args.root, b) for b in args.betas)
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
