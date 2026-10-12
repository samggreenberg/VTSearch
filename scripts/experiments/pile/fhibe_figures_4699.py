#!/usr/bin/env python3
"""Figure for the #4699 supply census: MTCNN recall falls off a pixel cliff, not a band.

Left: recall against the face box's short side in the pixels the app sees,
pooled over every resolution ``fhibe_supply.py detect`` measured. Right: where
the small band's faces sit on that axis at each resolution.

    python fhibe_figures_4699.py --out <census dir> --outdir docs/experiments/2026-10-09-fhibe-supply-4699
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fhibe_supply import IOU_HIT, app_keeps, iou, read_annotations  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"full": "#2a78d6", "1024": "#eb6834", "640": "#1baf7a"}
LABEL = {"full": "full resolution", "1024": "long side 1024", "640": "long side 640"}
CLIFF = 24


def main() -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.ticker  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()

    rows = [r for r in read_annotations(a.out) if r["humans"] == 1]
    det = {}
    for f in sorted((a.out / "detect").glob("shard-*.jsonl")):
        for line in f.open():
            rec = json.loads(line)
            det[rec["img"]] = rec
    rows = [r for r in rows if r["img"] in det]

    edges = [*range(8, 48, 4), 48, 64, 96, 128, 192, 256, 512, 1 << 20]
    hits: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    small_px: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        rec = det[r["img"]]
        gt = [r["x1"], r["y1"], r["x2"], r["y2"]]
        for res, s in rec["scale"].items():
            kept = [d for d in rec["det"][res] if app_keeps(d, s)]
            hit = max((iou(gt, d) for d in kept), default=0.0) >= IOU_HIT
            px = min(r["x2"] - r["x1"], r["y2"] - r["y1"]) * s
            j = next((k for k in range(len(edges) - 1) if px < edges[k + 1]), None)
            if j is not None:
                hits[j][0] += hit
                hits[j][1] += 1
            if r["band"] == "small":
                small_px[res].append(px)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(CLIFF, color=INK2, lw=1, ls="--")
        ticks = [8, 16, 24, 32, 64, 128, 256, 512, 1024]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    xs = [((edges[j] * min(edges[j + 1], 1024)) ** 0.5) for j in sorted(hits) if hits[j][1] >= 20]
    ys = [100 * hits[j][0] / hits[j][1] for j in sorted(hits) if hits[j][1] >= 20]
    ax1.plot(xs, ys, color=SERIES["full"], lw=2, marker="o", ms=5)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("face box short side, pixels the app sees (log)")
    ax1.set_ylabel("MTCNN hit rate, % (IoU >= 0.5)")
    ax1.set_title("MTCNN finds a face by its pixels: ~95% from 24 px up", loc="left", fontsize=11)
    ax1.text(CLIFF * 1.05, 8, f"{CLIFF} px", color=INK2)

    # Each label sits right of its own curve at a staggered height, so none overlap.
    for res, at in (("640", 0.88), ("1024", 0.62), ("full", 0.36)):
        v = sorted(small_px[res])
        if not v:
            continue
        ys = [100 * (k + 1) / len(v) for k in range(len(v))]
        ax2.plot(v, ys, color=SERIES[res], lw=2)
        below = 100 * sum(1 for p in v if p < CLIFF) / len(v)
        ax2.text(
            v[int(at * len(v))] * 1.12,
            100 * at,
            f"{LABEL[res]}\n{below:.0f}% under {CLIFF} px",
            color=INK,
            fontsize=9,
            va="center",
        )
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("small-band face short side, pixels the app sees (log)")
    ax2.set_ylabel("% of small-band photos at or below")
    ax2.set_title("FHIBE's small band only crosses the cliff at 640 px", loc="left", fontsize=11)

    fig.tight_layout()
    a.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.outdir / "fig_recall_by_px.png", facecolor="#fcfcfb")
    print(a.outdir / "fig_recall_by_px.png")


if __name__ == "__main__":
    main()
