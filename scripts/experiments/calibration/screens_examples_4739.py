#!/usr/bin/env python3
"""#4739: the screens a photo detector returns, as literal examples for the report.

Reads the ``__find`` prediction dumps (``VTS_DUMP_TEST_SCORES``) of a few ``screens_only`` cells re-run on the
study's frozen tree, and draws, per cell, the screens its last detector returns (score at or above the Find line),
highest first, each captioned with its score, the line, its Apptitude class and its Rico id. Writes the figure and
``screens_examples.csv`` (every returned screen).

    python screens_examples_4739.py --dumps DIR --out FIGDIR [--per-cell 8]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

RICO = Path("/exp/scale26/datasets/external/Rico/combined")


def returned(dump: Path) -> pd.DataFrame:
    """The screens *dump*'s detector returns, highest score first."""
    d = pd.read_csv(dump)
    d = d[d["score"] >= d["threshold"]].sort_values("score", ascending=False)
    return d.assign(cell=dump.name.split("__")[0])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="VTS_DUMP_TEST_SCORES dir holding <tag>__find.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--rico", default=str(RICO), help="Rico's combined/ dir (<id>.jpg)")
    args = ap.parse_args(argv)
    dumps = sorted(Path(args.dumps).glob("*__find.csv"))
    if not dumps:
        print(f"no *__find.csv under {args.dumps}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = [returned(p) for p in dumps]
    allr = pd.concat(frames, ignore_index=True)
    allr.to_csv(out / "screens_examples.csv", index=False, float_format="%.4f")

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    n = args.per_cell
    fig, axes = plt.subplots(len(frames), n, figsize=(1.55 * n, 3.3 * len(frames)), squeeze=False)
    for i, frame in enumerate(frames):
        target = frame["target_category"].iloc[0] if len(frame) else dumps[i].name
        for j in range(n):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j >= len(frame):
                ax.axis("off")
                continue
            r = frame.iloc[j]
            rico_id = Path(str(r["filename"])).stem
            img = Path(args.rico) / f"{rico_id}.jpg"
            if img.is_file():
                with Image.open(img) as im:
                    im.thumbnail((180, 320))
                    ax.imshow(im)
            ax.set_xlabel(
                f"{r['all_categories']}\nRico {rico_id}\n{r['score']:.3f} (line {r['threshold']:.3f})", fontsize=6.5
            )
            if j == 0:
                ax.set_ylabel(f"{target}\n{len(frame)} returned", fontsize=8)
    fig.suptitle(
        "Screens a photo-trained detector returns (beta 4, seed 0, its last detector), highest score first", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out / "screens_examples.png", dpi=130)
    plt.close(fig)
    print(allr.groupby(["cell", "target_category"]).size().to_string())
    print(allr.groupby(["target_category", "all_categories"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
