"""Does down-weighting a hard-view starting face in the example centroid recover #5376's beta-4 loss? (#5389)

Reads the analyses ``sota_face_analyze.sh`` wrote over the same K = 4 sessions, one per arm
(``--arm TAG=<analysis>``; the first is the control, weight 1, the plain mean), and the sessions'
list (``--people``: ``kind`` flagged when the worst start has Face Quality under 14, else clear).
``--swap`` (optional) is #5376's arm with the hard face left out of the session entirely, read as a
reference level only: it ran at another commit, so it is not paired.

Writes to ``--out``, and prints the summary:

* ``paired.csv``: one row per person, size, beta and arm (the person as the first 8 characters of the id);
* ``summary.csv``: per size, beta, kind and arm, the mean of each metric at 50 votes and after the
  session, and the arm's paired difference from the control with a bootstrap 95% interval over people;
  ``identical`` says whether every session matched the control exactly (the no-op check on clear ones);
* ``f50.png``: F at 50 votes on the flagged sessions per beta, per arm, with #5376's leave-it-out.

    python hard_face_weight_5389_compare.py --people people.csv --arm ctl=<a> --arm w0=<a> ... --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BETAS = {"025": 0.25, "1": 1.0, "4": 4.0}
SIZES = ("1024", "640")
METRICS = (
    "thr_fbeta_50",
    "thr_precision_50",
    "thr_recall_50",
    "thr_returned_50",
    "ap_50",
    "thr_fbeta_final",
    "final_ap",
)


def cells(analysis: str, btag: str, size: str) -> pd.DataFrame:
    return pd.read_csv(Path(analysis) / f"k4-b{btag}-{size}" / "cells.csv").set_index("category")[list(METRICS)]


def boot(x: np.ndarray, rng: np.random.Generator, n: int = 4000) -> tuple[float, float]:
    if len(x) == 0:
        return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def figure(summary: pd.DataFrame, arms: list[str], path: Path, title: str) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    s = summary[(summary.kind == "flagged") & (summary.metric == "thr_fbeta_50")]
    cols = [*arms, "swap"] if "swap" in set(s.arm) else arms
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    width = 0.8 / len(cols)
    palette = ["#c0504d", "#f2a65a", "#9bbb59", "#4bacc6", "#4f81bd"]
    for ax, size in zip(axes, SIZES):
        for i, arm in enumerate(cols):
            g = s[(s["size"] == size) & (s.arm == arm)].sort_values("beta")
            x = np.arange(len(g)) + (i - (len(cols) - 1) / 2) * width
            label = "hard face out (#5376, not paired)" if arm == "swap" else arm
            ax.bar(x, g.mean_arm, width, label=label, color=palette[i % len(palette)])
        ax.set_xticks(np.arange(len(BETAS)), [f"β {b:g}" for b in BETAS.values()])
        ax.set_title(f"{size} px")
        ax.set_ylim(0.6, 1.0)
    axes[0].set_ylabel("F at 50 votes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(cols), fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=120)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--people", required=True)
    ap.add_argument("--arm", action="append", required=True, help="TAG=<analysis dir>; the first is the control")
    ap.add_argument("--swap", help="#5376's leave-it-out analysis, a reference level only")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--title", default="Four-photo sessions with one hard-view start: its weight in the example centroid"
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arms = dict(a.split("=", 1) for a in args.arm)
    names = list(arms)
    ctl = names[0]
    people = pd.read_csv(args.people, dtype={"res": str})
    rng = np.random.default_rng(5389)
    rows, summary = [], []
    for size in SIZES:
        kind = people[people.res == size].set_index("category").kind
        for btag, beta in BETAS.items():
            c = cells(arms[ctl], btag, size)
            ref = cells(args.swap, btag, size) if args.swap else None
            for name in names[1:]:
                a = cells(arms[name], btag, size)
                j = c.join(a, lsuffix="_ctl", rsuffix="_arm", how="inner").join(kind, how="inner")
                j["size"], j["beta"], j["arm"] = size, beta, name
                rows.append(j.reset_index())
                for k, g in j.groupby("kind"):
                    for m in METRICS:
                        d = (g[f"{m}_arm"] - g[f"{m}_ctl"]).dropna().to_numpy()
                        lo, hi = boot(d, rng)
                        summary.append(
                            dict(
                                size=size,
                                beta=beta,
                                kind=k,
                                arm=name,
                                metric=m,
                                n=len(d),
                                mean_ctl=g[f"{m}_ctl"].mean(),
                                mean_arm=g[f"{m}_arm"].mean(),
                                diff=d.mean() if len(d) else np.nan,
                                lo=lo,
                                hi=hi,
                                identical=bool(np.array_equal(g[f"{m}_arm"], g[f"{m}_ctl"], equal_nan=True)),
                            )
                        )
            # The control and #5376's leave-it-out as levels, on the flagged sessions.
            flagged = kind[kind == "flagged"].index
            for name, frame in ((ctl, c), *((("swap", ref),) if ref is not None else ())):
                g = frame.reindex(flagged.intersection(frame.index))
                for m in METRICS:
                    summary.append(
                        dict(size=size, beta=beta, kind="flagged", arm=name, metric=m, n=len(g), mean_arm=g[m].mean())
                    )
    paired = pd.concat(rows)
    paired["category"] = paired.category.str[:8]
    paired.to_csv(args.out / "paired.csv", index=False)
    out = pd.DataFrame(summary)
    out.to_csv(args.out / "summary.csv", index=False)
    figure(out, names, args.out / "f50.png", args.title)
    key = out[out.metric.isin(("thr_fbeta_50", "thr_precision_50", "thr_returned_50", "thr_recall_50"))]
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(key.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
