"""Render a browse-map mockup: 2-D UMAP scatter + Toponymy signposts.

One PNG per layer, plus a combined "coarse + fine" view approximating what
VTSBrowse would show mid-zoom. Points are colored by ground-truth category
(legend capped), signpost text placed at each topic's 2-D anchor.

Usage::

    python visualize.py esc50 topo_clap_audioset_hf [--layers 0 1 2]
"""

from __future__ import annotations

import argparse

import common

common.setup_env()

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("run", help="e.g. topo_clap_audioset_hf")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = common.ds_dir(args.dataset)
    run = common.load_json(out / f"{args.run}.json")
    meta = common.load_json(out / "meta.json")[: run["n_clips"]]
    xy = np.array(run["umap2d"])
    gt = [m["category"] for m in meta]
    cats = sorted(set(gt))
    cat_i = {c: i for i, c in enumerate(cats)}
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    layers = args.layers if args.layers is not None else [lyr["layer"] for lyr in run["layers"]]
    for li in layers:
        layer = run["layers"][li]
        fig, ax = plt.subplots(figsize=(14, 11))
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=[colors[cat_i[g] % 20] for g in gt],
            s=14,
            alpha=0.6,
            linewidths=0,
        )
        pad_x = (xy[:, 0].max() - xy[:, 0].min()) * 0.06
        pad_y = (xy[:, 1].max() - xy[:, 1].min()) * 0.06
        ax.set_xlim(xy[:, 0].min() - pad_x, xy[:, 0].max() + pad_x)
        ax.set_ylim(xy[:, 1].min() - pad_y, xy[:, 1].max() + pad_y)
        for name, anchor in zip(layer["topic_names"], layer["anchors"]):
            if anchor is None:
                continue
            ax.annotate(
                str(name),
                anchor,
                ha="center",
                va="center",
                fontsize=9 if layer["n_topics"] > 25 else 12,
                weight="bold",
                color="#111",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", alpha=0.85),
            )
        ax.set_title(
            f"{args.dataset} — {args.run} — layer {li} "
            f"({layer['n_topics']} topics, coverage {1 - layer['noise_frac']:.0%})"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        png = out / f"map_{args.run}_layer{li}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=110)
        plt.close(fig)
        print(f"[saved] {png}")


if __name__ == "__main__":
    main()
