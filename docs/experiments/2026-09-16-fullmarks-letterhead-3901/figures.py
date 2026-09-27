"""Figures for the #3901 letterhead count, from ``calls.txt`` alone.

python figures.py            # writes fig_calls_by_author.png beside this file
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = {"M": "printed mark", "T": "typeset letterhead only", "U": "unreadable", "N": "no letterhead"}
COLOURS = {"M": "#2f6f9f", "T": "#9cc3de", "U": "#d9d9d9", "N": "#e8e8e8"}


def read_calls(path: Path) -> dict[str, dict[str, list[int]]]:
    calls: dict[str, dict[str, list[int]]] = {}
    author = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sheet = re.match(r"sheet\d+ (.+)", line)
        if sheet:
            author = sheet.group(1)
            calls[author] = {}
            continue
        key, _, rest = line.partition(" ")
        if key in LABELS and author is not None:
            calls[author][key] = [int(x) for x in rest.split()]
    return calls


def main() -> None:
    calls = read_calls(HERE / "calls.txt")
    authors = sorted(calls, key=lambda a: -len(calls[a].get("M", [])))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    left = [0] * len(authors)
    for key in ("M", "T", "U", "N"):
        widths = [len(calls[a].get(key, [])) for a in authors]
        ax.barh(authors, widths, left=left, color=COLOURS[key], label=LABELS[key], edgecolor="white")
        left = [start + w for start, w in zip(left, widths)]
    ax.invert_yaxis()
    ax.set_xlabel("bands, of 40 sampled per author")
    ax.set_xlim(0, 40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_calls_by_author.png", dpi=150)


if __name__ == "__main__":
    main()
