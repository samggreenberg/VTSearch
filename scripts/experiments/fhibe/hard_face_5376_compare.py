"""Is the beta-4 gap of a four-photo session with a hard starting face the face or the person? (#5376)

Reads two analyses ``sota_face_analyze.sh`` wrote over the same flagged K = 4 sessions: ``--ctl``,
the sessions as they ran, and ``--swap``, each hard face (Face Quality under 14) swapped for the next
clear photo of the person or dropped (``hard_face_5376.py``). It also reads ``--ref``, the arm's own
analysis of every session (#5325's shipped arm), for the unflagged people and their K = 1 sessions.
The swap log comes from the swap arm's cell logs.

Writes to ``--out``, and prints the summary:

* ``paired.csv``: one row per person, size and beta (the person as the first 8 characters of the id);
* ``summary.csv``: mean F-beta at 50 votes and after the session, control vs swap, with a bootstrap
  95% interval over people, split by whether the hard face was swapped or dropped;
* ``k1.csv``: the check that needs no run, the same flagged people's one-photo sessions from a clear
  photo (the K = 1 example is the first of the four): K4 - K1 at 50 votes, flagged vs the rest, and the
  difference with a bootstrap 95% interval;
* ``f50.png``: F at 50 votes per beta, as run, with the hard face out, and the other people.

    python hard_face_5376_compare.py --ctl <a> --swap <a> --ref <a> --swap-logs <runs>/<swap>-k4-b{beta}/logs --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BETAS = {"025": 0.25, "1": 1.0, "4": 4.0}
SIZES = ("1024", "640")
COLS = ["category", "thr_fbeta_50", "thr_fbeta_final"]


def cells(analysis: str, k: int, btag: str, size: str) -> pd.DataFrame:
    return pd.read_csv(Path(analysis) / f"k{k}-b{btag}-{size}" / "cells.csv")[COLS].set_index("category")


def swaps(pattern: str) -> pd.DataFrame:
    rows = []
    for btag, beta in BETAS.items():
        for f in glob.glob(pattern.replace("{beta}", btag) + "/cell-*.out"):
            for line in open(f, errors="replace"):
                if line.startswith("hard_face_5376 swap: "):
                    d = json.loads(line.split(": ", 1)[1])
                    n_hard = sum(q < 14 for q in d["original_q"])
                    rows.append(
                        dict(
                            index=d["index"],
                            beta=beta,
                            category=d["category"],
                            n_hard=n_hard,
                            k_after=len(d["examples"]),
                            first_q=d["original_q"][0],
                            kind="dropped" if len(d["examples"]) < len(d["original"]) else "swapped",
                        )
                    )
    return pd.DataFrame(rows)


def boot(x: np.ndarray, rng: np.random.Generator, n: int = 4000) -> tuple[float, float]:
    if len(x) == 0:
        return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def k1_check(ref: str, paired: pd.DataFrame, path: Path, rng: np.random.Generator) -> None:
    rows = []
    for size in SIZES:
        for btag, beta in BETAS.items():
            k4, k1 = cells(ref, 4, btag, size), cells(ref, 1, btag, size)
            j = k4.join(k1, lsuffix="_k4", rsuffix="_k1", how="inner")
            here = paired[(paired["size"] == size) & (paired.beta == beta)]
            short = j.index.str[:8]
            d = (j.thr_fbeta_50_k4 - j.thr_fbeta_50_k1).to_numpy()
            # A flagged person counts only when the one-photo session started from a clear face (the first
            # of the four), so K1 is the same person without the hard face. The rest are every unflagged one.
            clear = short.isin(set(here[here.first_q >= 14].category))
            f, r = d[clear], d[~short.isin(set(here.category))]
            bs = [rng.choice(f, len(f)).mean() - rng.choice(r, len(r)).mean() for _ in range(4000)]
            rows.append(
                dict(
                    size=size,
                    beta=beta,
                    n_flagged=len(f),
                    k1_flagged=j.thr_fbeta_50_k1[clear].mean(),
                    k1_rest=j.thr_fbeta_50_k1[~short.isin(set(here.category))].mean(),
                    k4_minus_k1_flagged=f.mean(),
                    k4_minus_k1_rest=r.mean(),
                    did=f.mean() - r.mean(),
                    lo=np.percentile(bs, 2.5),
                    hi=np.percentile(bs, 97.5),
                )
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    s = summary[(summary.kind == "all") & (summary.metric == "thr_fbeta_50")]
    for ax, size in zip(axes, SIZES):
        g = s[s["size"] == size].sort_values("beta")
        x = np.arange(len(g))
        ax.bar(x - 0.27, g.ctl, 0.27, label="as run (hard face in)", color="#c0504d")
        ax.bar(x, g.swap, 0.27, label="hard face swapped or dropped", color="#4f81bd")
        ax.bar(x + 0.27, g.rest_k4, 0.27, label="other people", color="#a5a5a5")
        ax.set_xticks(x, [f"β {b:g}" for b in g.beta])
        ax.set_title(f"{size} px, {int(g.n.iloc[0])} people")
        ax.set_ylim(0.7, 1.0)
    axes[0].set_ylabel("F at 50 votes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Four-photo sessions whose worst start has Face Quality < 14")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=120)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--swap", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--swap-logs", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sw = swaps(args.swap_logs)
    rng = np.random.default_rng(5376)
    rows, summary = [], []
    for size in SIZES:
        for btag, beta in BETAS.items():
            c, s = cells(args.ctl, 4, btag, size), cells(args.swap, 4, btag, size)
            ref4, ref1 = cells(args.ref, 4, btag, size), cells(args.ref, 1, btag, size)
            j = c.join(s, lsuffix="_ctl", rsuffix="_swap", how="inner")
            j = j.join(ref4.add_suffix("_arm"), how="left").join(ref1.add_suffix("_k1"), how="left")
            # The swap log carries the size only through the cell index (size-major, 562 people a size).
            log = sw[(sw.beta == beta) & ((sw["index"] >= 562) == (size == "640"))].set_index("category")
            j = j.join(log[["kind", "n_hard", "k_after", "first_q"]], how="left")
            j["size"], j["beta"] = size, beta
            rows.append(j.reset_index())
            rest = ref4.drop(index=j.index, errors="ignore")
            for kind, g in [("all", j), *j.groupby("kind")]:
                for metric in ("thr_fbeta_50", "thr_fbeta_final"):
                    d = (g[f"{metric}_swap"] - g[f"{metric}_ctl"]).dropna().to_numpy()
                    lo, hi = boot(d, rng)
                    summary.append(
                        dict(
                            size=size,
                            beta=beta,
                            kind=kind,
                            metric=metric,
                            n=len(d),
                            ctl=g[f"{metric}_ctl"].mean(),
                            swap=g[f"{metric}_swap"].mean(),
                            diff=d.mean() if len(d) else np.nan,
                            lo=lo,
                            hi=hi,
                            ctl_equals_arm=bool(np.allclose(g[f"{metric}_ctl"], g[f"{metric}_arm"], equal_nan=True)),
                            rest_k4=rest[metric].mean(),
                        )
                    )
    paired = pd.concat(rows)
    paired["category"] = paired.category.str[:8]
    paired.to_csv(args.out / "paired.csv", index=False)
    out = pd.DataFrame(summary)
    k1_check(args.ref, paired, args.out / "k1.csv", rng)
    figure(out, args.out / "f50.png")
    out.to_csv(args.out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
