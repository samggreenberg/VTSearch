"""#4110's figures, from ``analyze_4110.py``'s output and the arms' cells.

    python figures_4110.py --base /expscratch/$USER/det-4110 --analysis /expscratch/$USER/det-4110/analysis \
        --out docs/experiments/2026-10-15-node-determinism-4110/figures

* ``verdicts.png``    - per pair, the share of cells in each verdict of ``analyze_4110.py``;
* ``fbeta_gap.png``   - the paired gap at every vote, every cell and the mean: #3840's same-arm repeat
  (2026-09-22, cost, Intel minus AMD), today's dev and today's dev pinned (F-beta, Intel minus AMD), and
  what the pins move on one node type (AMD, pinned minus today);
* ``cell_seconds.png`` - one cell's wall time per arm, the pins' price.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_4110 as A  # noqa: E402

VERDICT_COLORS = {
    "identical": "#3b7d3b",
    "diagnostics": "#9cc28a",
    "threshold": "#e3c35a",
    "objective": "#d9822b",
    "trajectory": "#b5402f",
}
VERDICT_LABELS = ["bit-identical", "diagnostics only", "threshold", "objective", "other picks"]
ORDER = [
    ("a_base", "a_rep", "same arm twice, AMD"),
    ("i_base", "a_base", "today: Intel vs AMD"),
    ("i_npy", "a_base", "numpy pin: Intel vs AMD"),
    ("i_pin", "a_pin", "numpy + MKL: Intel vs AMD"),
    ("i_full", "a_full", "all three pins: Intel vs AMD"),
]


def verdicts(pairs: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    labels = []
    for k, (left, right, label) in enumerate(ORDER):
        sub = pairs[(pairs["left"] == left) & (pairs["right"] == right)]
        n = len(sub)
        labels.append(f"{label}  (n={n})")
        x0 = 0.0
        for v, c in VERDICT_COLORS.items():
            share = (sub["verdict"] == v).mean() if n else 0.0
            ax.barh(k, share, left=x0, color=c, label=v if k == 0 else None)
            x0 += share
    ax.set_yticks(range(len(ORDER)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of cells")
    ax.legend(VERDICT_LABELS, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=5, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "verdicts.png", dpi=150)
    plt.close(fig)


def _gaps(base: Path, left: str, right: str) -> pd.DataFrame:
    lc, rc = A._cells(base / left), A._cells(base / right)
    frames = []
    for c in sorted(lc.keys() & rc.keys()):
        a, b = A._clicks(lc[c]), A._clicks(rc[c])
        m = a[["t", "fbeta"]].merge(b[["t", "fbeta"]], on="t", suffixes=("_l", "_r"))
        m["gap"] = m["fbeta_l"] - m["fbeta_r"]
        m["cell"] = c
        frames.append(m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["t", "gap", "cell"])


def _gaps_3840(frames: Path, vendors: Path) -> pd.DataFrame:
    """#3840's same-arm repeat (``v_rep`` vs ``v_baseline``), cross-vendor cells only, as Intel minus AMD."""
    key = ["dataset", "embedder", "style", "category", "seed"]
    df = pd.read_csv(frames, usecols=["grid", *key, "t", "cost"])
    rep, first = df[df.grid == "v_rep"], df[(df.grid == "v_baseline") & df.seed.isin([0, 1])]
    m = rep.merge(first, on=[*key, "t"], suffixes=("_rep", "_first"))
    m["cell"] = m[key].astype(str).agg("/".join, axis=1)
    v = pd.read_csv(vendors).set_index("cell")["pair"]
    m["pair"] = m["cell"].map(v)
    m = m[m["pair"].isin(["AMD->Intel", "Intel->AMD"])].copy()
    # first run -> repeat: Intel minus AMD is (repeat - first) when the repeat ran on Intel.
    sign = np.where(m["pair"] == "AMD->Intel", 1.0, -1.0)
    m["gap"] = sign * (m["cost_rep"] - m["cost_first"])
    return m[["cell", "t", "gap"]]


def fbeta_gap(base: Path, out: Path, frames_3840: Path | None, vendors_3840: Path | None) -> None:
    panels = [
        ("i_base", "a_base", "today's dev, Intel minus AMD"),
        ("i_full", "a_full", "pinned, Intel minus AMD"),
        ("a_full", "a_base", "AMD, pinned minus today"),
    ]
    with_3840 = frames_3840 is not None and vendors_3840 is not None
    fig, axes = plt.subplots(1, len(panels) + with_3840, figsize=(4.4 * (len(panels) + with_3840), 3.4), sharey=True)
    if with_3840:
        g = _gaps_3840(frames_3840, vendors_3840)
        ax = axes[0]
        for _, one in g.groupby("cell"):
            ax.plot(one["t"], one["gap"], color="#888888", lw=0.6, alpha=0.5)
        ax.plot(*zip(*g.groupby("t")["gap"].mean().items(), strict=True), color="#1f4e8c", lw=2)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"#3840, 2026-09-22, Intel minus AMD: {g['cell'].nunique()} cells (cost)", fontsize=9)
        ax.set_xlabel("vote")
        axes = axes[1:]
        fig.axes[0].set_ylabel("paired gap")
    for ax, (left, right, title) in zip(axes, panels, strict=True):
        g = _gaps(base, left, right)
        for _, one in g.groupby("cell"):
            # A vote one session wrote no base row for is a gap in the line, not a straight segment across it.
            full = one.set_index("t")["gap"].reindex(range(int(one["t"].min()), int(one["t"].max()) + 1))
            ax.plot(full.index, full.to_numpy(), color="#888888", lw=0.6, alpha=0.5)
        if len(g):
            mean = g.groupby("t")["gap"].mean()
            ax.plot(mean.index, mean.to_numpy(), color="#1f4e8c", lw=2, label="mean over cells")
        ax.axhline(0, color="black", lw=0.5)
        n = g["cell"].nunique() if len(g) else 0
        ax.set_title(f"{title}: {n} cells (F-beta)", fontsize=9)
        ax.set_xlabel("vote")
    if not with_3840:
        axes[0].set_ylabel("paired gap")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fbeta_gap.png", dpi=150)
    plt.close(fig)


def cell_seconds(base: Path, out: Path) -> None:
    arms = ["i_base", "i_full", "a_base", "a_full"]
    data = [A.cell_seconds(base / a).to_numpy() / 60 for a in arms]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.boxplot(data, showfliers=False)
    for k, d in enumerate(data, start=1):
        ax.scatter(np.full(d.size, k) + np.random.default_rng(k).uniform(-0.15, 0.15, d.size), d, s=6, color="#555555")
    ax.set_xticks(range(1, len(arms) + 1), ["Intel\ntoday", "Intel\npinned", "AMD\ntoday", "AMD\npinned"])
    ax.set_ylabel("minutes a cell")
    fig.tight_layout()
    fig.savefig(out / "cell_seconds.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames-3840", type=Path, help="#3840's steps.csv.gz, for the September panel")
    ap.add_argument("--vendors-3840", type=Path, help="#3840's cells by node vendor (cells_3840_by_vendor.csv)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.analysis / "pairs.csv")
    verdicts(pairs, args.out)
    fbeta_gap(args.base, args.out, args.frames_3840, args.vendors_3840)
    cell_seconds(args.base, args.out)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
