"""Diagnostic: measure duplicate-rate in HAC region trees.

Throwaway script supporting the HAC-duplicates investigation.  Builds
region trees over synthetic patch grids + saliency maps (the structural
properties we care about — box duplicates and cell-set duplicates —
depend on grid geometry and the Voronoi-by-spatial-distance leaf
assignment, not on the patch contents).  Reports:

* % of internal HAC nodes whose merged box exactly equals at least one
  ancestor / descendant / sibling / any-other region's box.
* % of internal HAC nodes whose merged box equals one of their two
  children's boxes (the specific case the user flagged: "merging C into
  AB might make no change at all").
* % of internal HAC nodes whose underlying cell set equals an earlier
  node's cell set (this is the *true* duplicate test — same patches
  pooled, regardless of merge order).
* For pairs of nodes with the same bounding box, the distribution of
  cosine similarity between their stored vectors (to quantify how much
  the merge-order-dependent averaging shifts vectors that "should" be
  identical under saliency-weighted re-pooling).

Run::

    python scripts/inspect_hac_duplicates.py --k 12 --grid 16 --n 200 --seed 0
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

import numpy as np

from vtsearch.models.patch_regions import (
    PatchEmbedOutput,
    build_region_tree,
    propose_leaves,
    build_hac_tree,
)


@dataclass
class TreeStats:
    n_internals: int
    n_box_eq_child: int
    n_box_eq_any_earlier: int
    n_cells_eq_earlier: int
    same_box_cos_sims: list[float]
    cells_eq_box_eq: int
    cells_neq_box_eq: int


def _assign_cells_to_leaves(saliency: np.ndarray, k: int) -> list[set[tuple[int, int]]]:
    """Recompute the per-leaf cell sets the same way ``propose_leaves`` does."""
    h, w = saliency.shape
    flat = saliency.reshape(-1).astype(np.float32, copy=False)
    seed_idx = np.argsort(-flat, kind="stable")[:k]
    seeds = [(int(i // w), int(i % w)) for i in seed_idx]
    rows = np.arange(h, dtype=np.float32)[:, None]
    cols = np.arange(w, dtype=np.float32)[None, :]
    dist = np.empty((k, h, w), dtype=np.float32)
    for i, (sr, sc) in enumerate(seeds):
        dist[i] = (rows - sr) ** 2 + (cols - sc) ** 2
    assignments = np.argmin(dist, axis=0)
    cell_sets: list[set[tuple[int, int]]] = []
    for i, (sr, sc) in enumerate(seeds):
        mask = assignments == i
        if not mask.any():
            cell_sets.append({(sr, sc)})
            continue
        ys, xs = np.where(mask)
        cell_sets.append({(int(y), int(x)) for y, x in zip(ys, xs)})
    return cell_sets


def analyse_one(rng: np.random.Generator, grid: int, k: int, dim: int) -> TreeStats:
    patch_grid = rng.standard_normal((grid, grid, dim)).astype(np.float32)
    patch_grid = patch_grid / np.maximum(
        np.linalg.norm(patch_grid, axis=-1, keepdims=True), 1e-12
    )
    raw_sal = rng.random((grid, grid)).astype(np.float32)
    saliency = raw_sal / raw_sal.sum()
    cls_vec = rng.standard_normal(dim).astype(np.float32)
    cls_vec = cls_vec / max(float(np.linalg.norm(cls_vec)), 1e-12)

    output = PatchEmbedOutput(cls_vec=cls_vec, patch_grid=patch_grid, patch_saliency=saliency)
    regions = build_region_tree(output, k=k, alpha=0.5)
    # regions: [CLS, leaves x K, internals x (K-1)]
    leaf_cells = _assign_cells_to_leaves(saliency, k)
    # cell sets per region index in `regions`: CLS = all cells; leaf i = leaf_cells[i-1]
    cell_sets: list[frozenset[tuple[int, int]]] = [None] * len(regions)  # type: ignore[list-item]
    all_cells = frozenset((r, c) for r in range(grid) for c in range(grid))
    cell_sets[0] = all_cells
    for i in range(k):
        cell_sets[i + 1] = frozenset(leaf_cells[i])
    for idx in range(k + 1, len(regions)):
        ci, cj = regions[idx].children  # type: ignore[misc]
        cell_sets[idx] = cell_sets[ci] | cell_sets[cj]

    n_internals = len(regions) - 1 - k
    n_box_eq_child = 0
    n_box_eq_any_earlier = 0
    n_cells_eq_earlier = 0
    same_box_sims: list[float] = []
    cells_eq_box_eq = 0
    cells_neq_box_eq = 0

    # Index nodes by box so we can detect duplicates against any earlier node
    # (not just children).  Cells likewise.
    box_to_idxs: dict[tuple[float, float, float, float], list[int]] = {}
    cells_to_idxs: dict[frozenset[tuple[int, int]], list[int]] = {}
    # Seed with leaves + CLS so internal-vs-leaf duplicates count too.
    for i in range(k + 1):
        box_to_idxs.setdefault(regions[i].box, []).append(i)
        cells_to_idxs.setdefault(cell_sets[i], []).append(i)

    for idx in range(k + 1, len(regions)):
        node = regions[idx]
        ci, cj = node.children  # type: ignore[misc]
        if node.box == regions[ci].box or node.box == regions[cj].box:
            n_box_eq_child += 1
        if node.box in box_to_idxs:
            n_box_eq_any_earlier += 1
            for peer in box_to_idxs[node.box]:
                v1 = np.asarray(node.vec, dtype=np.float32)
                v2 = np.asarray(regions[peer].vec, dtype=np.float32)
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom > 0:
                    same_box_sims.append(float(v1 @ v2) / denom)
                if cell_sets[idx] == cell_sets[peer]:
                    cells_eq_box_eq += 1
                else:
                    cells_neq_box_eq += 1
        if cell_sets[idx] in cells_to_idxs:
            n_cells_eq_earlier += 1
        box_to_idxs.setdefault(node.box, []).append(idx)
        cells_to_idxs.setdefault(cell_sets[idx], []).append(idx)

    return TreeStats(
        n_internals=n_internals,
        n_box_eq_child=n_box_eq_child,
        n_box_eq_any_earlier=n_box_eq_any_earlier,
        n_cells_eq_earlier=n_cells_eq_earlier,
        same_box_cos_sims=same_box_sims,
        cells_eq_box_eq=cells_eq_box_eq,
        cells_neq_box_eq=cells_neq_box_eq,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="# of synthetic images")
    p.add_argument("--k", type=int, default=12, help="leaves per image")
    p.add_argument("--grid", type=int, default=16, help="patch grid side (16=DINOv2 224)")
    p.add_argument("--dim", type=int, default=64, help="vector dim (low for speed)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    totals = {
        "internals": 0,
        "box_eq_child": 0,
        "box_eq_any_earlier": 0,
        "cells_eq_earlier": 0,
        "cells_eq_box_eq": 0,
        "cells_neq_box_eq": 0,
    }
    all_sims: list[float] = []
    for _ in range(args.n):
        stats = analyse_one(rng, args.grid, args.k, args.dim)
        totals["internals"] += stats.n_internals
        totals["box_eq_child"] += stats.n_box_eq_child
        totals["box_eq_any_earlier"] += stats.n_box_eq_any_earlier
        totals["cells_eq_earlier"] += stats.n_cells_eq_earlier
        totals["cells_eq_box_eq"] += stats.cells_eq_box_eq
        totals["cells_neq_box_eq"] += stats.cells_neq_box_eq
        all_sims.extend(stats.same_box_cos_sims)

    T = totals["internals"]
    print(f"# images:                                  {args.n}")
    print(f"# internal HAC nodes (per image):          {T // args.n} (K-1={args.k - 1})")
    print(f"# internals total:                         {T}")
    print()
    print(f"box == one of own children's box:          {totals['box_eq_child']:>6} "
          f"({100 * totals['box_eq_child'] / T:.1f}%)")
    print(f"box == some earlier region's box:          {totals['box_eq_any_earlier']:>6} "
          f"({100 * totals['box_eq_any_earlier'] / T:.1f}%)")
    print(f"underlying cell set == earlier region's:   {totals['cells_eq_earlier']:>6} "
          f"({100 * totals['cells_eq_earlier'] / T:.1f}%)")
    print()
    print(f"of the box-equal pairs:")
    print(f"  cells also equal (true duplicates):      {totals['cells_eq_box_eq']}")
    print(f"  cells differ      (box-only duplicates): {totals['cells_neq_box_eq']}")
    print()
    if all_sims:
        sims = np.array(all_sims)
        print(f"cosine(stored vecs) between same-box pairs:")
        print(f"  N pairs:   {len(sims)}")
        print(f"  min/mean/max: {sims.min():.4f} / {sims.mean():.4f} / {sims.max():.4f}")
        print(f"  % > 0.99:  {100 * (sims > 0.99).mean():.1f}%")
        print(f"  % > 0.999: {100 * (sims > 0.999).mean():.1f}%")
    else:
        print("(no same-box pairs found)")


if __name__ == "__main__":
    main()
