#!/usr/bin/env python3
"""State of the App (#4159): the two production paths side by side (#4655).

Each path has its own review and its own report (owner, 2026-09-24), and a
reader of the Region Photo report wants the comparison that matters: the
DINOv3 region path against the SigLIP binary path, each as it ships. Both
reports commit the same two tables, under the same rules (#4603, #4605,
#4631), so this reads them straight from the two report directories and needs
nothing from the GRID:

* ``objective_by_click.csv`` -- per preset, the objective at every click
  (``by_click.py``): the typed query's set until the app shows a run's
  detector, then its line;
* ``precision_recall_path.csv`` -- per preset, the returned set at the typed
  query, 25, 50, 100 and 150 clicks and after the check (``perp.py``).

It writes, into ``--out``:

* ``vs_binary_objective.png`` -- per preset, both paths' objective over
  clicks, with each path's end after the check;
* ``vs_binary_path.png`` -- per preset, both paths' returned set as precision
  against recall, from the typed query (which the two share) through 25, 50,
  100 (unlabelled) and 150 clicks to after the check (✓);

and prints the table the report carries: per preset, each path's objective at
each point, the region path's lead, and the returned set after the check.

Not paired by seed: each path is averaged over every run its review has, and
the two reviews ran different seed counts. The report says how many.

    python compare_paths.py --binary docs/experiments/<date>-state-of-the-app-binary-photo \\
        --region docs/experiments/<date>-state-of-the-app-region-photo \\
        --out docs/experiments/<date>-state-of-the-app-region-photo/figures
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import COLORS, INK, MUTED, SURFACE, _axes  # noqa: E402

#: The two paths, in the order every figure and table lists them.
PATHS = ("SigLIP binary", "DINOv3 region")
#: The returned set's points, as ``perp.py`` names them, left to right.
POINTS = ("typed query", "25", "50", "100", "150", "after the check")
#: How far right of the last click the point after the check sits, as in ``perp.py``'s figure.
CHECK_DX = 6


def preset(beta: float) -> str:
    """``1/4``, ``1``, ``4``: a preset as the app and the reports name it."""
    return str(Fraction(beta).limit_denominator(64))


def load(binary: Path, region: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both reports' per-click objective and returned-set path, each row tagged with its ``path``."""
    by_click, along = [], []
    for name, d in zip(PATHS, (binary, region), strict=True):
        by_click.append(pd.read_csv(d / "objective_by_click.csv").assign(path=name))
        along.append(pd.read_csv(d / "precision_recall_path.csv", dtype={"point": str}).assign(path=name))
    clicks, path = pd.concat(by_click, ignore_index=True), pd.concat(along, ignore_index=True)
    for frame, what in ((clicks, "objective_by_click.csv"), (path, "precision_recall_path.csv")):
        betas = {name: set(frame.loc[frame["path"] == name, "beta"]) for name in PATHS}
        if betas[PATHS[0]] != betas[PATHS[1]]:
            raise SystemExit(f"{what}: the two reports ran different presets: {betas}")
    missing = set(POINTS) - set(path["point"])
    if missing:
        raise SystemExit(f"precision_recall_path.csv lacks {sorted(missing)}")
    return clicks, path


def _runs(frame: pd.DataFrame, name: str) -> int:
    return int(frame.loc[frame["path"] == name, "runs"].max())


def figure_objective(clicks: pd.DataFrame, path: pd.DataFrame, out: Path) -> Path:
    """``vs_binary_objective.png``: per preset, both paths' objective over clicks and after the check."""
    betas = sorted(clicks["beta"].unique())
    fig, axes = plt.subplots(1, len(betas), figsize=(4.4 * len(betas), 3.9), facecolor=SURFACE, sharey=True)
    axes = [axes] if len(betas) == 1 else list(axes)
    for ax, beta in zip(axes, betas, strict=True):
        for name in PATHS:
            c = clicks[(clicks["beta"] == beta) & (clicks["path"] == name)].sort_values("t")
            end = path[(path["beta"] == beta) & (path["path"] == name) & (path["point"] == "after the check")]
            color = COLORS[name]
            ax.plot(c["t"], c["fbeta"], color=color, lw=2, label=f"{name} ({_runs(clicks, name):,} runs)")
            x_end = int(c["t"].max()) + CHECK_DX
            y_end = float(end["fbeta"].iloc[0])
            ax.scatter([x_end], [y_end], color=color, zorder=3)
            ax.annotate(f"{y_end:.2f}", (x_end, y_end), xytext=(5, 0), textcoords="offset points", va="center",
                        color=INK, fontsize=8)  # fmt: skip
        ax.set_title(f"preset {preset(beta)}: the objective", color=INK, fontsize=10, loc="left")
        _axes(ax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("clicks (the dot: after the check)", color=INK)
    axes[0].set_ylabel("F-beta of the withheld set above the threshold", color=INK)
    axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    target = out / "vs_binary_objective.png"
    fig.savefig(target, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return target


def figure_path(path: pd.DataFrame, out: Path) -> Path:
    """``vs_binary_path.png``: per preset, both paths' returned set as precision against recall."""
    betas = sorted(path["beta"].unique())
    fig, axes = plt.subplots(1, len(betas), figsize=(4.4 * len(betas), 4.2), facecolor=SURFACE, sharey=True)
    axes = [axes] if len(betas) == 1 else list(axes)
    for ax, beta in zip(axes, betas, strict=True):
        for name in PATHS:
            g = path[(path["beta"] == beta) & (path["path"] == name)].set_index("point").loc[list(POINTS)]
            color = COLORS[name]
            ax.plot(g["recall"], g["precision"], color=color, lw=1.8, marker="o", ms=4, label=name)
            for point, r in g.iterrows():
                # The typed query is the two paths' shared start, labelled once below; 100 sits on top of 150
                # by then, and the two labels would print over each other.
                if point in ("typed query", "100"):
                    continue
                tag = "✓" if point == "after the check" else str(point)
                ax.annotate(tag, (r["recall"], r["precision"]), textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=color)  # fmt: skip
        start = path[(path["beta"] == beta) & (path["point"] == "typed query")]
        ax.annotate("typed query", (start["recall"].mean(), start["precision"].mean()), textcoords="offset points",
                    xytext=(-6, -12), ha="right", fontsize=7, color=MUTED)  # fmt: skip
        ax.set_title(f"preset {preset(beta)}: the returned set", color=INK, fontsize=10, loc="left")
        _axes(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("recall of the returned set", color=INK)
    axes[0].set_ylabel("precision of the returned set", color=INK)
    axes[-1].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    target = out / "vs_binary_path.png"
    fig.savefig(target, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return target


def _signed(v: float, digits: int = 2) -> str:
    """``+0.07`` / ``−0.02``, with the reports' minus sign rather than a hyphen."""
    return f"{v:+.{digits}f}".replace("-", "−")


def table(path: pd.DataFrame) -> str:
    """The report's table: per preset, each path's objective at each point, the lead, and the end's set."""
    head = ["preset", "path", *POINTS, "precision ✓", "recall ✓", "returned ✓, median"]
    lines = ["| " + " | ".join(head) + " |", "|---:|---|" + "---:|" * (len(head) - 2)]
    numbers = ["fbeta", "precision", "recall", "returned, median"]
    for beta in sorted(path["beta"].unique()):
        at = {name: path[(path["beta"] == beta) & (path["path"] == name)].set_index("point")[numbers] for name in PATHS}
        for name in PATHS:
            g, end = at[name], at[name].loc["after the check"]
            cells = [f"{g.loc[p, 'fbeta']:.2f}" for p in POINTS]
            cells += [f"{end['precision']:.2f}", f"{end['recall']:.2f}", f"{end['returned, median']:.0f}"]
            lines.append(f"| {preset(beta)} | {name} | " + " | ".join(cells) + " |")
        lead = at[PATHS[1]] - at[PATHS[0]].reindex(at[PATHS[1]].index)
        cells = [_signed(lead.loc[p, "fbeta"]) for p in POINTS]
        end = lead.loc["after the check"]
        cells += [_signed(end["precision"]), _signed(end["recall"]), _signed(end["returned, median"], 0)]
        lines.append(f"| {preset(beta)} | *region − binary* | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=Path, required=True, help="the Binary Photo report's directory")
    ap.add_argument("--region", type=Path, required=True, help="the Region Photo report's directory")
    ap.add_argument("--out", type=Path, required=True, help="where the two figures go")
    args = ap.parse_args(argv)
    clicks, path = load(args.binary, args.region)
    args.out.mkdir(parents=True, exist_ok=True)
    for target in (figure_objective(clicks, path, args.out), figure_path(path, args.out)):
        print(f"wrote {target}", file=sys.stderr)
    print(table(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
