"""Paired readout of ``sota_documents_arms.py``'s keypoint-cut arms against the full-keypoint control (#4934).

Reads ``<root>/<arm>/rep<r>/<class>/steps.csv`` (and ``clicks.csv``). For each cut arm and each
class, the paired difference (arm - control) is averaged over the replicates, then bootstrapped
over classes (``vote_curve.paired_ci``), so a resample keeps a class's replicates together. A
session that ended before click *v* (no unlabelled page left in its click half) holds its last
state. The pre-registered rule (#4934, owner 2026-10-15): an arm passes when the 95% lower bound
of both the AP and the F1-at-the-line difference at 25 clicks is at least -0.01; the passing arm
that verifies the fewest keypoints ships for the CPU path.

    python keypoint_cut_analyze.py <root> [--control all] [--out <dir>]
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vote_curve import paired_ci  # noqa: E402

CLICKS = (0, 10, 25, 50)
DECIDE_AT = 25
MARGIN = -0.01
#: The readouts: ranking AP on the test half, and the returned set's F1 (beta 1) and F-beta at 1/4 and 4.
READOUTS = ("ap", "gate_f1", "gate_fb025", "gate_fb4")


def load(root: Path) -> dict[str, dict[tuple[str, int], list[dict[str, str]]]]:
    """``{arm: {(class, replicate): steps}}`` for every finished call under *root*."""
    runs: dict[str, dict[tuple[str, int], list[dict[str, str]]]] = {}
    for f in sorted(root.glob("*/rep*/*/steps.csv")):
        if not (f.parent / "done").exists():
            continue
        arm, rep, slug = f.parts[-4], int(f.parts[-3][3:]), f.parts[-2]
        runs.setdefault(arm, {})[(slug.replace("__", "/", 1), rep)] = list(csv.DictReader(f.open(encoding="utf-8")))
    return runs


def at(steps: list[dict[str, str]], v: int, key: str) -> float:
    """*key* at click *v*, or at the session's last click if it ended earlier."""
    row = next((r for r in steps if int(r["v"]) == v), steps[-1])
    return float(row[key]) if row[key] not in ("", "nan") else float("nan")


def retrain(root: Path, arm: str) -> dict[str, float]:
    """Median retrain seconds after a Good and after a Bad, and the p90 of every step, for *arm*."""
    after: dict[str, list[float]] = {"good": [], "bad": [], "all": []}
    for f in root.glob(f"{arm}/rep*/*/steps.csv"):
        steps = list(csv.DictReader(f.open(encoding="utf-8")))
        clicks_f = f.parent / "clicks.csv"
        label = {}
        if clicks_f.exists():
            label = {int(c["click"]): c["label"] for c in csv.DictReader(clicks_f.open(encoding="utf-8"))}
        for r in steps:
            s = float(r["retrain_s"])
            after["all"].append(s)
            lab = label.get(int(r["v"]))
            if lab in after:
                after[lab].append(s)
    return {
        "good_median": statistics.median(after["good"]) if after["good"] else float("nan"),
        "bad_median": statistics.median(after["bad"]) if after["bad"] else float("nan"),
        "p90": float(np.percentile(after["all"], 90)) if after["all"] else float("nan"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--control", default="all")
    ap.add_argument("--out", type=Path, help="write summary.md and paired.csv here")
    args = ap.parse_args(argv)
    runs = load(args.root)
    if args.control not in runs:
        raise SystemExit(f"no finished {args.control!r} runs under {args.root}")
    ctrl = runs[args.control]
    lines = [f"# Keypoint cuts vs `{args.control}`: {args.root}", ""]
    rows: list[dict[str, Any]] = []
    verdicts: list[tuple[str, bool]] = []
    for arm in sorted(a for a in runs if a != args.control):
        sessions = sorted(set(runs[arm]) & set(ctrl))
        classes = sorted({c for c, _ in sessions})
        lines.append(f"## `{arm}`: {len(sessions)} paired sessions, {len(classes)} classes")
        lines.append("")
        lines.append("| readout | " + " | ".join(f"{v} clicks" for v in CLICKS) + " |")
        lines.append("|---|" + "---|" * len(CLICKS))
        lower: dict[str, float] = {}
        for key in READOUTS:
            cells = []
            for v in CLICKS:
                per_class = []
                for c in classes:
                    d = [at(runs[arm][s], v, key) - at(ctrl[s], v, key) for s in sessions if s[0] == c]
                    d = [x for x in d if np.isfinite(x)]
                    if d:
                        per_class.append(float(np.mean(d)))
                diffs = np.array(per_class)
                if not len(diffs):
                    cells.append("n/a")
                    continue
                mean, lo, hi = paired_ci(diffs)
                cells.append(f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                rows.append(
                    {"arm": arm, "readout": key, "v": v, "mean": mean, "lo": lo, "hi": hi, "n_classes": len(diffs)}
                )
                if v == DECIDE_AT:
                    lower[key] = lo
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        passed = all(lower.get(k, -np.inf) >= MARGIN for k in ("ap", "gate_f1"))
        verdicts.append((arm, passed))
        t, tc = retrain(args.root, arm), retrain(args.root, args.control)
        lines.append("")
        lines.append(
            f"Retrain, median after a Good {t['good_median']:.2f} s (control {tc['good_median']:.2f}), "
            f"after a Bad {t['bad_median']:.2f} s ({tc['bad_median']:.2f}), p90 {t['p90']:.2f} s ({tc['p90']:.2f})."
        )
        lines.append(
            f"**Pre-registered rule at {DECIDE_AT} clicks (lower bounds >= {MARGIN}):** AP {lower.get('ap', float('nan')):+.3f}, "
            f"F1 {lower.get('gate_f1', float('nan')):+.3f} -> **{'PASS' if passed else 'FAIL'}**"
        )
        # The classes that lose most AP at the decision click, both replicates averaged.
        worst = sorted(
            (
                np.mean(
                    [at(runs[arm][s], DECIDE_AT, "ap") - at(ctrl[s], DECIDE_AT, "ap") for s in sessions if s[0] == c]
                ),
                c,
            )
            for c in classes
        )[:5]
        lines.append("Largest AP losses at 25 clicks: " + ", ".join(f"{c} {d:+.2f}" for d, c in worst))
        lines.append("")
    lines.append("## Verdicts")
    lines.append("")
    lines.extend(f"- `{arm}`: {'PASS' if ok else 'FAIL'}" for arm, ok in verdicts)
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.md").write_text(text, encoding="utf-8")
        with (args.out / "paired.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["arm"])
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
