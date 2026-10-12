#!/usr/bin/env python
"""The figures of #4780's report: where the pronoun gap sits, what hairstyle does to it, and the line.

Reads ``face_gap_4780.py``'s output (``stages.csv``, ``people.csv``) and draws, for the sets named in
``--sets`` (``LABEL=SET[:STATIC_CELL]``, the set as ``--set`` named it, the static cell its margin
comes from):

* ``fig_stages.png`` - the she/her minus he/him gap at each stage of a session, +-1 SE, one marker
  per set, from 1 and from 4 photos;
* ``fig_curves.png`` - the objective at every click from 1 photo, per pronoun, one panel per set;
* ``fig_hairstyles.png`` - the final ranking's F at the true count against how many hairstyles a
  person wears across their photos, per pronoun, one panel per set (each person faint behind the means);
* ``fig_line.png`` - what the line costs (F at the line minus F at the true count) against the
  embedder's margin (own photos' cosine minus the closest impostor's), per pronoun, one panel per set.

Marks are group means; the hairstyle figure also draws each person as an unlabelled faint dot.

    python face_gap_4780_figures.py --analysis <dir> --sets "FaceNet=fn:facenet_roll,ArcFace=arc:arcface_roll" --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
#: Sets (embedders) take the reference palette's blue and aqua; pronouns its orange and violet.
SET_COLORS = ("#2a78d6", "#1baf7a")
PRONOUN = {"She/her/hers": ("#eb6834", "she/her"), "He/him/his": ("#4a3aa7", "he/him")}
STAGE_LABELS = {
    "click0_ap": "click 0: example sort, AP",
    "session_mean": "mean over clicks 1-150 (F at the line)",
    "f_line": "after the check: F at the line",
    "final_f_true_count": "final ranking: F at the true count",
    "ceiling_f_true_count": "full-label ceiling: F at the true count",
    "ceiling_ap": "full-label ceiling: AP",
    "line_loss": "the line's cost: F at the line - at the true count",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )


def fig_stages(stages: pd.DataFrame, sets: list[tuple[str, str, str]], out: Path) -> None:
    s = stages[stages["contrast"] == "pronoun"]
    rows = list(STAGE_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, sharex=True)
    for ax, k in zip(axes, (1, 4)):
        for j, (label, name, _) in enumerate(sets):
            d = s[(s["set"] == name) & (s["K"] == k)].set_index("stage")
            y = np.arange(len(rows)) + (j - (len(sets) - 1) / 2) * 0.22
            x = [d.loc[r, "gap"] if r in d.index else np.nan for r in rows]
            e = [d.loc[r, "se"] if r in d.index else np.nan for r in rows]
            ax.errorbar(
                x, y, xerr=e, fmt="o", ms=5, color=SET_COLORS[j], ecolor=SET_COLORS[j], elinewidth=1.4, capsize=0
            )
            if k == 1:
                ax.text(x[0] + e[0] + 0.004, y[0], label, color=INK2, ha="left", va="center", fontsize=8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_title("from 1 photo" if k == 1 else "from 4 photos", loc="left", fontsize=10)
    axes[0].set_yticks(range(len(rows)), [STAGE_LABELS[r] for r in rows])
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=SET_COLORS[j], label=lab) for j, (lab, _, _) in enumerate(sets)
    ]
    axes[1].legend(handles=handles, frameon=False, loc="lower left", fontsize=8)
    fig.supxlabel("she/her minus he/him (F-beta or AP, beta 1, 1024 px), +-1 SE", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig_stages.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_curves(curves: pd.DataFrame, sets: list[tuple[str, str, str]], out: Path) -> None:
    """The objective at every click from 1 photo, per pronoun, one panel per set."""
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, name, _) in zip(axes, sets):
        d = curves[(curves["set"] == name) & (curves["K"] == 1)]
        for p, (color, short) in PRONOUN.items():
            q = d[d["pronoun"] == p].sort_values("t")
            ax.fill_between(q["t"], q["mean"] - q["se"], q["mean"] + q["se"], color=color, alpha=0.15, lw=0)
            ax.plot(q["t"], q["mean"], color=color, lw=1.8)
            if len(q):
                ax.text(q["t"].iloc[-1] + 3, q["mean"].iloc[-1], short, color=INK2, va="center", fontsize=8)
        ax.set_xlim(0, 168)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("click (0 = the example sort)")
    axes[0].set_ylabel("F-beta of the withheld photos at the app's line")
    fig.suptitle(
        "From 1 photo, beta 1, 1024 px: what the session shows at every click, +-1 SE", x=0.01, ha="left", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out / "fig_curves.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _binned(d: pd.DataFrame, x: str, y: str, edges: list[float], labels: list[str]) -> pd.DataFrame:
    b = pd.cut(d[x], edges, labels=labels, include_lowest=True)
    g = d.groupby([b, "pronoun"], observed=True)[y]
    return pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.count()), "n": g.count()}).reset_index()


def _pronoun_panels(ppl, sets, x, y, edges, labels, xlabel, ylabel, title, path, min_n=8, people=False) -> None:
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, name, _) in zip(axes, sets):
        d = ppl[(ppl["set"] == name) & (ppl["K"] == 1) & ppl["pronoun"].isin(PRONOUN)]
        xcol = x(name) if callable(x) else x
        if xcol not in d:
            continue
        e = edges(d[xcol]) if callable(edges) else edges
        t = _binned(d, xcol, y, e, labels)
        if people:  # each person, faint, behind the means: the spread the means summarise
            rng = np.random.default_rng(4780)
            for i, (p, (color, _)) in enumerate(PRONOUN.items()):
                q = d[d["pronoun"] == p]
                b = pd.cut(q[xcol], e, labels=False, include_lowest=True).to_numpy(float)
                off = (i - 0.5) * 0.16
                ax.scatter(b + off + rng.uniform(-0.05, 0.05, len(b)), q[y], s=5, color=color, alpha=0.18, lw=0)
        for p, (color, short) in PRONOUN.items():
            q = t[(t["pronoun"] == p) & (t["n"] >= min_n)]
            pos = [labels.index(v) for v in q[xcol]]
            ax.errorbar(pos, q["mean"], yerr=q["se"], fmt="-o", ms=5, lw=1.6, color=color, elinewidth=1.2, capsize=0)
            if len(pos):
                ax.text(pos[-1] + 0.12, q["mean"].iloc[-1], short, color=INK2, va="center", fontsize=8)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlim(-0.3, len(labels) - 0.3)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--sets", required=True, help="LABEL=SET:STATIC_CELL,...")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sets = []
    for spec in args.sets.split(","):
        label, rest = spec.split("=", 1)
        name, _, cell = rest.partition(":")
        sets.append((label, name, cell))
    cells = {name: cell for _, name, cell in sets}
    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = pd.read_csv(args.analysis / "stages.csv")
    ppl = pd.read_csv(args.analysis / "people.csv", dtype={"skin_tone": str})
    fig_stages(stages, sets, args.out)
    if (args.analysis / "curves_by_group.csv").exists():
        fig_curves(pd.read_csv(args.analysis / "curves_by_group.csv"), sets, args.out)
    _pronoun_panels(
        ppl,
        sets,
        "hairstyles",
        "final_f_true_count",
        [0, 1, 2, 3, 99],
        ["1", "2", "3", "4+"],
        "distinct hairstyles across the person's photos",
        "final ranking: F at the true count",
        "From 1 photo: the ranking against how much the person's hair changes between photos",
        args.out / "fig_hairstyles.png",
        people=True,
    )
    _pronoun_panels(
        ppl,
        sets,
        lambda n: f"static_{cells[n]}_margin",
        "line_loss",
        lambda v: list(np.nanquantile(v, [0, 0.2, 0.4, 0.6, 0.8, 1])),
        ["1st", "2nd", "3rd", "4th", "5th"],
        "margin quintile (own photos vs closest impostor)",
        "F at the line minus at the true count",
        "From 1 photo: what the line costs, against the embedder's margin for the person",
        args.out / "fig_line.png",
    )
    print(f"wrote fig_stages.png, fig_hairstyles.png, fig_line.png -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
