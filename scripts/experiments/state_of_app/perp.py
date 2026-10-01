#!/usr/bin/env python3
"""The review across floors (#4408): each P's returned set, read off sessions that aimed at that P.

A review runs one set of sessions per floor P (``CALIB_MIN_PRECISION``), each
analyzed on its own (``analyze.sh``). This puts them side by side, reading each
P only from its own sessions:

* ``returned_at_own_p.png`` -- per P, over clicks: the returned set's precision
  against P (top) and its recall against the oracle's recall at P (bottom). When
  the 50% run is given, the same P read off the 50% sessions is drawn light, so
  the figure shows what running each P on its own changed.
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="P=analysis dir, e.g. 0.9=/path/analysis-binary")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    runs = {float(k): Path(v) for k, _, v in (r.partition("=") for r in args.run)}
    args.out.mkdir(parents=True, exist_ok=True)
    figure(runs, args.out)
    summary(runs, args.out)
    print(f"perp -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
