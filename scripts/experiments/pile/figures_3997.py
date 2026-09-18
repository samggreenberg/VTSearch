#!/usr/bin/env python3
"""Figures for the VG-diversity study (#3997).

Two, and each is there because a number in the report is easy to misread without
it:

* ``fig_matched.png`` -- pooled vs composition-matched provenance AUC. The point
  is that about half the pooled signal is class and band, which a table of two
  columns states but does not make felt.
* ``fig_hardness.png`` -- the per-cell paired differences, as a distribution
  rather than a mean. 58 cells straddling zero with a slight lean is a very
  different claim from "off-COCO is easier", and the histogram is what stops the
  mean being read as the latter.

    python figures_3997.py --probe provenance_probe.json \
        --hardness provenance_hardness.json --outdir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CHANCE = 0.5
MIN_ARM = 10


def fig_matched(probe: dict, out: Path) -> None:
    names = [k for k, v in probe.items() if "positives_matched" in v]
    pooled = [probe[k]["positives_pooled"]["auc"] for k in names]
    matched = [probe[k]["positives_matched"]["auc"] for k in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - 0.19, pooled, 0.36, label="pooled", color="#9ecae1")
    ax.bar(x + 0.19, matched, 0.36, label="matched within class@band", color="#3182bd")
    ax.axhline(CHANCE, color="#d62728", lw=1.2, ls="--")
    ax.annotate("chance", (0.012, CHANCE), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", color="#d62728", fontsize=8)
    for xi, (p, m) in enumerate(zip(pooled, matched)):
        ax.annotate(f"{p:.3f}", (xi - 0.19, p), ha="center", va="bottom", fontsize=7)
        ax.annotate(f"{m:.3f}", (xi + 0.19, m), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0.48, 0.60)
    ax.set_ylabel("provenance AUC (positives only)")
    ax.set_title("Half the readable provenance signal is class and band composition")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _deltas(rows: list[dict], arm: str) -> np.ndarray:
    out = []
    for r in rows:
        if len(r["above"].get(arm, [])) < MIN_ARM:
            continue
        c = np.log1p(np.asarray(r["above"]["coco"], dtype=float)).mean()
        o = np.log1p(np.asarray(r["above"][arm], dtype=float)).mean()
        out.append(-(o - c))  # a cost, so negate: + means off-COCO ranks better
    return np.asarray(out)


def fig_hardness(hard: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(hard), figsize=(3.6 * len(hard), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (embedder, rows) in zip(axes, hard.items()):
        d = _deltas(rows, "off")
        ax.hist(d, bins=18, color="#3182bd", alpha=0.85)
        ax.axvline(0, color="#444", lw=1)
        ax.axvline(d.mean(), color="#d62728", lw=1.6)
        ax.annotate(f"mean {d.mean():+.3f}\n{(d > 0).sum()} of {len(d)} cells > 0",
                    (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8, color="#d62728")
        ax.set_title(embedder, fontsize=10)
        ax.set_xlabel("per-cell Δ log1p(rank)\n(+ = off-COCO ranks better)", fontsize=8)
    axes[0].set_ylabel("cells")
    fig.suptitle("Off-COCO positives are not harder: per-cell differences straddle zero, leaning easier",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--hardness", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig_matched(json.loads(args.probe.read_text()), args.outdir / "fig_matched.png")
    fig_hardness(json.loads(args.hardness.read_text()), args.outdir / "fig_hardness.png")
    print(f"wrote {args.outdir}/fig_matched.png and fig_hardness.png")


if __name__ == "__main__":
    main()
