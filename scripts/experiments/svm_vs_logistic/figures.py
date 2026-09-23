"""#3197 figures, from the analysis CSVs only (``analyze_stage_a.py`` / ``analyze_stage_b.py``).

python figures.py --analysis DIR --out docs/experiments/2026-09-22-svm-vs-logistic-3197/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREEN, GREY, RED = "#2a6fdb", "#e07b00", "#2f9e44", "#8a8a8a", "#c92a2a"


def _err(ax, x, df, color, label, marker="o"):
    ax.errorbar(x, df["mean"], yerr=2 * df["se"], color=color, marker=marker, capsize=3, label=label, lw=1.6)


def fig_stage_b(an: Path, out: Path) -> None:
    p = pd.read_csv(an / "stageB_paired.csv")
    p = p[(p.env == "ALL")]
    arms = ["linear", "linconv", "mlp", "svmc01", "svmc10"]
    labels = {
        "linear": "logistic (shipped fit)",
        "linconv": "logistic, 2000 epochs",
        "mlp": "MLP",
        "svmc01": "SVM C=0.1",
        "svmc10": "SVM C=10",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, m, title in zip(
        axes,
        ["cost", "oracle_cost", "regret"],
        ["cost at the shipped cut", "oracle cost (ranking only)", "regret (the cut's own loss)"],
    ):
        sub = p[(p.window == "aulc") & (p.metric == m)].set_index("arm").reindex(arms)
        y = np.arange(len(arms))
        ax.errorbar(sub["mean"], y, xerr=2 * sub["se"], fmt="o", color=BLUE, capsize=3)
        ax.axvline(0, color=GREY, lw=1)
        ax.set_yticks(y, [labels[a] for a in arms])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("arm − shipped SVM (area under the curve, ±2 SE)\n> 0: the shipped SVM is better")
    fig.suptitle("Stage B, 480 paired cells per arm: the SVM's in-loop win is in the ranking, not the cut", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "stageB_paired_aulc.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for arm, c in (("linear", ORANGE), ("linconv", GREEN), ("mlp", GREY)):
        rows = []
        for w in ("t10", "t20", "t40", "t80", "t150"):
            r = p[(p.window == w) & (p.metric == "oracle_cost") & (p.arm == arm)]
            if len(r):
                rows.append((int(w[1:]), r["mean"].iloc[0], r["se"].iloc[0]))
        if rows:
            t, m_, s_ = zip(*rows)
            ax.errorbar(t, m_, yerr=2 * np.array(s_), color=c, marker="o", capsize=3, label=labels[arm])
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 40, 80, 150], ["10", "20", "40", "80", "150"])
    ax.set_xlabel("clicks")
    ax.set_ylabel("oracle cost: arm − shipped SVM (±2 SE)")
    ax.set_title("The ranking gap by click (positive = SVM ranks better)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "stageB_ranking_gap_by_click.png", dpi=130)
    plt.close(fig)


def fig_stage_a(an: Path, out: Path) -> None:
    lv = pd.read_csv(an / "A_base_levels.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    cs = [0.01, 0.1, 1, 10, 100]
    for size, ls in ((4, ":"), (8, "--"), (32, "-")):
        s = lv[lv["size"] == size].set_index("arm")["test_ap"]
        svm = [s.get("svm" if c == 1 else f"svm_C{c:g}") for c in cs]
        lrc = [s.get(f"lrconv_C{c:g}") for c in cs]
        ax.plot(cs, svm, ls, color=BLUE, marker="o", label=f"SVM, {size} Goods")
        ax.plot(cs, lrc, ls, color=GREEN, marker="s", label=f"logistic (converged), {size} Goods")
        ax.axhline(s.get("lr"), color=ORANGE, ls=ls, lw=1)
    ax.axvline(1, color=GREY, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("C  (orange lines: the shipped logistic head, which has no C)")
    ax.set_ylabel("held-out average precision")
    ax.set_title("Both losses on the same regularisation path (fixed votes)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    g = pd.read_csv(an / "A_geometry.csv")
    g = g[g["size"] == 8].set_index("arm")
    order = [
        "svm_C0.01",
        "svm_C0.1",
        "svm",
        "svm_C10",
        "svm_C100",
        "svm_hinge",
        "lrconv_C1",
        "lr",
        "lr_zeroinit",
        "lr_ep2000",
    ]
    order = [o for o in order if o in g.index]
    ax.barh(
        range(len(order)), g.loc[order, "active_frac"], color=[BLUE if o.startswith("svm") else ORANGE for o in order]
    )
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("fraction of votes inside the SVM margin (y·f(x) < 1), 8 Goods + 32 Bads")
    ax.set_title("At the shipped C, ~95% of votes are inside the margin", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "stageA_regpath_and_margin.png", dpi=130)
    plt.close(fig)

    m = pd.read_csv(an / "A_mechanism.csv")
    m = m[m.metric == "test_ap"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    specs = [
        ("dilute", "easy votes added (× the hard core)", "M2: easy-vote dilution"),
        ("noise", "fraction of votes flipped", "M3: label noise"),
        ("ratio", "Bad:Good ratio", "M5: class ratio"),
    ]
    for ax, (cond, xl, title) in zip(axes, specs):
        for b, c in (("lr", ORANGE), ("lrconv_C1", GREEN)):
            s = m[(m.condition == cond) & (m.a == "svm") & (m.b == b)]
            if "size" in s and s["size"].notna().any() and cond != "ratio":
                for size, mk in zip(sorted(s["size"].dropna().unique()), ("o", "s")):
                    ss = s[s["size"] == size].sort_values("level")
                    x = ss["level"].to_numpy()
                    _err(
                        ax,
                        x + (0.0 if b == "lr" else 0.01 * max(1, x.max())),
                        ss,
                        c,
                        f"SVM − {b}, {int(size)} Goods",
                        mk,
                    )
            else:
                ss = s.sort_values("level")
                _err(ax, ss["level"].to_numpy(), ss, c, f"SVM − {b}")
        ax.axhline(0, color=GREY, lw=1)
        ax.set_xlabel(xl)
        ax.set_ylabel("Δ held-out AP (±2 SE); > 0 = SVM better")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "stageA_mechanisms.png", dpi=130)
    plt.close(fig)

    rp = an / "A_replay.csv"
    if rp.exists():
        r = pd.read_csv(rp)
        r = r[r.metric == "test_oracle_cost"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, cond in zip(axes, ["replay_svm", "replay_linear"]):
            for (a, b), c in ((("svm", "lr"), BLUE), (("lrconv_C1", "lr"), GREEN), (("lrconv_C1", "svm"), GREY)):
                s = r[(r.condition == cond) & (r.a == a) & (r.b == b)].sort_values("level")
                if len(s):
                    _err(ax, s["level"].to_numpy(), s, c, f"{a} − {b}")
            ax.axhline(0, color=GREY, lw=1)
            ax.set_xscale("log")
            ax.set_xticks([10, 20, 40, 80, 150], ["10", "20", "40", "80", "150"])
            ax.set_xlabel("clicks of the replayed trajectory")
            ax.set_title(f"votes Autopilot collected under the {cond.split('_')[1]} head", fontsize=10)
        axes[0].set_ylabel("Δ held-out oracle cost (±2 SE); < 0 = first arm ranks better")
        axes[0].legend(fontsize=8)
        fig.suptitle("Same votes, different heads: Stage B's vote sets refitted", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "stageA_replay.png", dpi=130)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_stage_b(args.analysis, args.out)
    fig_stage_a(args.analysis, args.out)
    print(sorted(p.name for p in args.out.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
