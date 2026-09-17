"""Figure for the #3914 check, from ``measurements/calls.json`` and ``measurements/hits.json``.

python figures.py      # writes fig_top_hits.png beside this file
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = {
    "blank": "blank or bordered photocopy",
    "article": "news clipping / article",
    "logo_image": "standalone logo image",
    "typed": "typed letter, memo, email",
    "black": "toner-black scan",
    "illustration": "illustration",
    "roster_mark": "carries the class mark",
}
ORDER = ("roster_mark", "blank", "article", "logo_image", "typed", "black", "illustration")


def main() -> None:
    calls = json.loads((HERE / "measurements" / "calls.json").read_text(encoding="utf-8"))
    hits = json.loads((HERE / "measurements" / "hits.json").read_text(encoding="utf-8"))
    kinds: Counter[str] = Counter()
    industries: Counter[str] = Counter()
    for cid, entry in calls.items():
        if cid.startswith("_"):
            continue
        kinds.update(entry["calls"])
        kinds["roster_mark"] += len(entry["roster_mark"])
        industries.update(h["industry"] for h in hits[cid]["hits"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    names = [LABELS[k] for k in ORDER]
    values = [kinds.get(k, 0) for k in ORDER]
    colours = ["#b03a2e"] + ["#9cc3de"] * (len(ORDER) - 1)
    ax1.barh(names, values, color=colours)
    ax1.invert_yaxis()
    ax1.set_xlabel("pages, of the 80 top-ranked (8 per class)")
    ax1.set_title("what SigLIP's top non-Tobacco UCSF pages show", fontsize=10)
    for y, v in enumerate(values):
        ax1.text(v + 0.5, y, str(v), va="center", fontsize=8)
    inds = sorted(industries, key=lambda k: -industries[k])
    ax2.barh(inds, [industries[i] for i in inds], color="#2f6f9f")
    ax2.invert_yaxis()
    ax2.set_xlabel("pages")
    ax2.set_title("their UCSF industry", fontsize=10)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_top_hits.png", dpi=150)


if __name__ == "__main__":
    main()
