"""Render 2-D layout scatter grids colored by taxonomy — the eyeball companion
to the quantitative sweep.

For a cached ``(dataset, embedder)`` matrix, re-fit UMAP at a few points of the
grid (seed 0) and scatter the layout colored by the coarsest taxonomy level, so
the report can show *what* an ``n_neighbors`` change or the compaction toggle
does to cluster geometry — the thing the AUROC number can't picture.

    python visualize.py <dataset> <embedder>

Writes PNGs to the figures dir. Matplotlib only (no browser).
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C

PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]


def coarse_labels(tag: str):
    """Return (label_per_point, level_name) using the coarsest scorable level."""
    d = np.load(C.MATRIX_DIR / f"{tag}.npz", allow_pickle=True)
    leaf = d["leaf"].astype(str)
    dataset = tag.split("__", 1)[0]
    spec = C.ROSTER_BY_NAME[dataset]
    if spec.taxonomy == "esc50":
        from common import _ESC50_SUPER_OF

        return np.array([_ESC50_SUPER_OF.get(c, "?") for c in leaf]), "supercategory"
    if spec.taxonomy == "inat":
        lin = C.load_inat_lineage()
        return np.array([lin.get(c, ["?"])[2] for c in leaf]), "class"  # biological class
    if spec.taxonomy == "fsd50k":
        # coarse = the single most-specific root each clip carries (for coloring)
        names = d["ml_names"].astype(str)
        isroot = d["ml_isroot"]
        ml = d["ml_labels"]
        root_cols = np.where(isroot)[0]
        lab = []
        for row in ml:
            hits = [c for c in root_cols if row[c]]
            lab.append(names[hits[0]] if hits else "other")
        return np.array(lab), "AudioSet top"
    if spec.taxonomy == "places365":
        # coarse = indoor/outdoor where encoded, else 'other'
        io = []
        for c in leaf:
            io.append(
                "indoor" if c.endswith(("_indoor", "_interior")) else "outdoor" if c.endswith("_outdoor") else "other"
            )
        return np.array(io), "indoor/outdoor"
    return leaf, "class"


def _scatter(ax, coords, labels, title, top_k=8):
    order = [lab for lab, _ in sorted(_freq(labels).items(), key=lambda x: -x[1])]
    keep = set(order[:top_k])
    for i, lab in enumerate(order[:top_k]):
        m = labels == lab
        ax.scatter(coords[m, 0], coords[m, 1], s=4, c=PALETTE[i % len(PALETTE)], label=str(lab)[:18], linewidths=0)
    rest = ~np.isin(labels, list(keep))
    if rest.any():
        ax.scatter(coords[rest, 0], coords[rest, 1], s=3, c="#DDDDDD", linewidths=0, zorder=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _freq(labels):
    u, c = np.unique(labels, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


def visualize(dataset: str, embedder: str):
    tag = f"{dataset}__{embedder}"
    d = np.load(C.MATRIX_DIR / f"{tag}.npz", allow_pickle=True)
    X = d["X"].astype(np.float32)
    ids = list(map(int, d["ids"]))
    labels, level = coarse_labels(tag)

    from vtscore.projection.compaction import compact_layout
    from vtscore.projection.umap_projection import fit_projection

    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # (1) n_neighbors sweep, raw layouts
    nns = [5, 15, 50, 200]
    fig, axes = plt.subplots(1, len(nns), figsize=(3.1 * len(nns), 3.3))
    for ax, nn in zip(axes, nns):
        proj = fit_projection(X, ids, n_neighbors=min(nn, len(ids) - 1), min_dist=0.1, random_state=0, compact=False)
        _scatter(ax, proj.coords, labels, f"n_neighbors = {nn}")
    axes[0].legend(markerscale=2.5, fontsize=6, loc="upper right", framealpha=0.85)
    fig.suptitle(
        f"{dataset} · {embedder} — 2-D layout vs n_neighbors (colored by {level}); min_dist=0.1, raw", fontsize=10
    )
    fig.tight_layout()
    p1 = C.FIG_DIR / f"nn_grid_{tag}.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    print("wrote", p1)

    # (2) compaction eyeball: raw vs compacted at nn=15
    proj = fit_projection(X, ids, n_neighbors=min(15, len(ids) - 1), min_dist=0.1, random_state=0, compact=False)
    comp = compact_layout(proj.coords)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))
    _scatter(axes[0], proj.coords, labels, "raw UMAP")
    _scatter(axes[1], comp, labels, "compacted (compact=True)")
    fig.suptitle(f"{dataset} · {embedder} — compaction closes empty space (n_neighbors=15)", fontsize=10)
    fig.tight_layout()
    p2 = C.FIG_DIR / f"compaction_{tag}.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    visualize(sys.argv[1], sys.argv[2])
