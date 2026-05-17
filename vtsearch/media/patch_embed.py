"""Patch-based region trees for image embedders.

Patch-based image encoders (DINOv2, DINOv3, EUPE) return a vector per spatial
patch plus a CLS token.  This module turns that raw output into a small,
hierarchical region set per image:

  * ``PatchEmbedOutput`` — the wire format the embedder hands us:
    CLS vector + per-patch grid + per-patch saliency.

  * ``propose_leaves`` — clusters the patch grid into K spatially-coherent
    region proposals, each carrying a bounding box (normalised image
    coordinates) and a saliency-weighted-mean vector.

  * ``build_hac_tree`` — runs hierarchical agglomerative clustering over the
    K leaves to add ``K - 1`` internal merge nodes.  The result is a strict
    binary tree with exactly ``2K - 1`` region nodes.

  * ``build_region_tree`` — the top-level entry point.  Combines the CLS
    full-image vector, the HAC leaves, and the HAC internals into the flat
    list that gets pickled as ``media["patch_regions"]``.

The HAC builder is intentionally vector-only (numpy / no torch dependency);
the embedders run the model forward pass and hand us numpy arrays.

This module does *not* score regions, store them on disk, or care about
``MediaEmbedder`` registration — those concerns live in the loader pipeline
and in :mod:`vtsearch.training.region_similarity`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------


class PatchEmbedOutput(NamedTuple):
    """One forward pass of a patch-based image embedder, in numpy.

    All vectors are L2-normalised; all dtypes are float32 on the CPU side.
    """

    cls_vec: np.ndarray
    """Pooled global representation, shape ``(D,)``."""

    patch_grid: np.ndarray
    """Per-patch vectors, shape ``(H, W, D)``.

    ``H`` and ``W`` are embedder-specific: DINOv2 ViT-B/14 at 224² gives
    ``16 × 16``; DINOv3 ViT-B/16 and EUPE ViT-B/16 at 224² give ``14 × 14``.
    """

    patch_saliency: np.ndarray
    """Per-patch importance, shape ``(H, W)``, non-negative, sums to ~1.0.

    For DINOv3 this is the final-block CLS→patch attention averaged across
    heads.  For EUPE (whose SDPA does not return weights) it is a softmaxed
    cosine similarity between each patch and the CLS vector.  Either way:
    "how much each patch contributes to the global representation."
    """


@dataclass
class RegionVector:
    """One node in the per-image region tree.

    The flat list ``media["patch_regions"]`` follows the convention:

    * index 0 is the CLS-pooled full-image node (``children = None``);
    * indices ``1 .. K`` are HAC leaves (``children = None``);
    * indices ``K + 1 ..`` are HAC internal nodes (``children = (i, j)``,
      both pointing earlier in the same list).
    """

    box: tuple[float, float, float, float]
    """Normalised image coordinates ``(x0, y0, x1, y1)``, each in ``[0, 1]``.

    Bounding box of the underlying cells.  Lossy — an L-shaped cell set
    has a bounding box strictly larger than the cell union.  Kept around
    as a cheap rectangular hint for UI rendering; the *true* region
    footprint is :attr:`cell_mask` (leaves) or the union of children's
    cell masks (internals).
    """

    vec: np.ndarray
    """L2-normalised vector for this region, shape ``(D,)``.

    For HAC nodes (leaves and internals) this is the L2-normalised
    saliency-weighted mean of the underlying patch vectors — computed
    additively from :attr:`weight` and the children's weighted sums, so
    every internal's vector equals what we would have gotten by
    re-pooling the patches inside its cell union from scratch.  Merge
    order does not affect the result.

    For the CLS full-image node this is the embedder's CLS-token output
    (not a patch pool).

    Stored as **float16** in the dataset pickle and cast to float32 when
    read into RAM.
    """

    children: Optional[tuple[int, int]] = None
    """Indices of the two children if this is an internal HAC node.

    ``None`` for leaves and for the CLS full-image node.  Encodes the merge
    structure of the agglomerative tree without us having to walk the list
    twice.
    """

    cell_mask: Optional[np.ndarray] = None
    """Per-patch occupancy mask, shape ``(H, W)`` bool, **leaves only**.

    Set by :func:`propose_leaves` to the Voronoi-by-spatial-distance
    assignment for this leaf.  ``None`` on internals and on the CLS
    full-image node — derive the internal cell mask on demand as the
    union of the children's masks (walk :attr:`children` until you hit
    leaves; their masks are disjoint by construction, so OR them).
    """

    weight: float = 0.0
    """Sum of patch saliencies inside :attr:`cell_mask`.

    On leaves: ``saliency[cell_mask].sum()``.  On internals: the sum of
    the two children's weights (associative because leaf cell sets are
    disjoint, so the same equals ``saliency[union_mask].sum()``).  Used
    by :func:`build_hac_tree` to combine vectors additively without
    referring back to the patch grid.  ``0.0`` on the CLS node (which is
    not a patch pool).
    """


# ---------------------------------------------------------------------------
# Leaf proposal
# ---------------------------------------------------------------------------


def _l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Return *v* L2-normalised along *axis*."""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, eps)


# ---------------------------------------------------------------------------
# Embedder-output adapters
# ---------------------------------------------------------------------------


def hf_vit_to_patch_output(
    outputs,
    *,
    num_register_tokens: int = 0,
    batch_index: int = 0,
) -> Optional[PatchEmbedOutput]:
    """Turn a HuggingFace ViT ``ModelOutput`` into a :class:`PatchEmbedOutput`.

    Shared adapter for DINOv2 (``num_register_tokens=0``) and DINOv3
    (``num_register_tokens=4``).  Expects the token layout
    ``[CLS, R1..R_k, P1..P_N]`` and a square spatial patch grid.

    ``outputs`` must have ``last_hidden_state`` and ``attentions`` set —
    pass ``output_attentions=True`` to the model forward call.

    *batch_index* selects which image in a batched forward to extract
    (default 0 preserves the single-image call sites).

    Returns ``None`` if the patch grid isn't square (the loader treats
    that as "no regions for this image", same as a forward-pass failure).
    """
    import torch  # noqa: PLC0415

    hidden = outputs.last_hidden_state[batch_index]  # (T, D)
    attn = outputs.attentions[-1][batch_index]  # (heads, T, T) — last block
    skip = 1 + num_register_tokens
    cls_vec = hidden[0]
    patch_tokens = hidden[skip:]
    num_patches = patch_tokens.shape[0]
    side = int(round(num_patches**0.5))
    if side * side != num_patches:
        return None
    patch_grid = patch_tokens.reshape(side, side, -1)
    cls_to_patches = attn[:, 0, skip:].mean(dim=0)
    saliency = cls_to_patches.reshape(side, side)
    saliency = saliency / torch.clamp(saliency.sum(), min=1e-8)
    return PatchEmbedOutput(
        cls_vec=_norm_torch(cls_vec).astype(np.float32),
        patch_grid=_norm_torch(patch_grid).astype(np.float32),
        patch_saliency=saliency.detach().cpu().float().numpy().astype(np.float32),
    )


def _norm_torch(t: "torch.Tensor") -> np.ndarray:
    """L2-normalise a torch tensor along its last axis and return numpy."""
    v = t.detach().cpu().float().numpy()
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def eupe_features_to_patch_output(
    features: dict,
    *,
    batch_index: int = 0,
) -> Optional[PatchEmbedOutput]:
    """Turn a :func:`facebookresearch/EUPE`-style ``forward_features`` dict
    into a :class:`PatchEmbedOutput`.

    EUPE's forward already separates CLS, storage tokens, and patch tokens
    into named keys (``x_norm_clstoken``, ``x_storage_tokens``,
    ``x_norm_patchtokens``), so we don't need a register-token slice — the
    storage tokens are already absent from ``x_norm_patchtokens``.

    *batch_index* selects which image in a batched forward to extract
    (default 0 preserves the single-image call sites).

    Saliency is the **CLS-cosine-similarity proxy**: each patch's softmaxed
    cosine similarity to the CLS vector.  EUPE's attention path uses
    ``torch.nn.functional.scaled_dot_product_attention`` which does not
    return weights, so the standard CLS→patch attention map isn't available
    without monkey-patching SDPA.  This proxy gives us a meaningful "how
    much does each patch contribute to the global representation" signal at
    no model-internals cost.

    Returns ``None`` if the patch grid isn't square.
    """
    import torch  # noqa: PLC0415

    cls = features["x_norm_clstoken"][batch_index]  # (D,)
    patches = features["x_norm_patchtokens"][batch_index]  # (N, D)
    num_patches = patches.shape[0]
    side = int(round(num_patches**0.5))
    if side * side != num_patches:
        return None
    patch_grid = patches.reshape(side, side, -1)

    cls_n = cls / torch.clamp(cls.norm(), min=1e-8)
    patches_n = patches / torch.clamp(patches.norm(dim=-1, keepdim=True), min=1e-8)
    sims = patches_n @ cls_n  # (N,) cosine similarity per patch to CLS
    # Softmax with a moderate temperature so a few cells dominate without
    # collapsing to a one-hot. Empirically tunable in the caltech101_s
    # sweep alongside K and α.
    saliency = torch.softmax(sims * 4.0, dim=-1).reshape(side, side)

    return PatchEmbedOutput(
        cls_vec=_norm_torch(cls).astype(np.float32),
        patch_grid=_norm_torch(patch_grid).astype(np.float32),
        patch_saliency=saliency.detach().cpu().float().numpy().astype(np.float32),
    )


def propose_leaves(
    patch_grid: np.ndarray,
    patch_saliency: np.ndarray,
    k: int,
) -> list[RegionVector]:
    """Cluster a patch grid into K spatially-coherent leaf regions.

    Algorithm: pick the top-K saliency peaks as seeds, then assign every
    remaining patch cell to its **nearest seed by Euclidean spatial
    distance** (ties broken by saliency).  Each leaf's box is the tight
    bounding box around its cells, in normalised image coordinates.  Each
    leaf's vector is the saliency-weighted mean of its constituent patch
    vectors, L2-normalised.

    This is a deliberately simple baseline.  It produces spatially-
    coherent, non-overlapping leaves with complete grid coverage.  The
    quality of the HAC merges (which the MLP and similarity actually score)
    depends more on the relative arrangement of leaves than on the exact
    boundaries; the simple baseline is good enough for v1.  Future work
    can swap in SLIC superpixels or watershed if the empirical sweep on
    caltech101_s shows clear leaf-quality wins.

    Parameters
    ----------
    patch_grid : ndarray, shape (H, W, D), float32, L2-normalised
        Per-patch vectors out of the embedder.
    patch_saliency : ndarray, shape (H, W), float32, non-negative
        Per-patch importance; not required to sum to 1.
    k : int
        Number of leaves to produce.  Must be ``>= 1`` and ``<= H * W``.

    Returns
    -------
    list[RegionVector] of length ``k``, leaves in saliency-peak order.
    """
    if patch_grid.ndim != 3:
        raise ValueError(f"patch_grid must be (H, W, D); got shape {patch_grid.shape}")
    height, width, _ = patch_grid.shape
    if patch_saliency.shape != (height, width):
        raise ValueError(f"patch_saliency must be ({height}, {width}); got {patch_saliency.shape}")
    num_cells = height * width
    if not 1 <= k <= num_cells:
        raise ValueError(f"k must be in [1, {num_cells}]; got {k}")

    flat_sal = patch_saliency.reshape(-1).astype(np.float32, copy=False)
    seed_indices = np.argsort(-flat_sal, kind="stable")[:k]
    seeds = [(int(idx // width), int(idx % width)) for idx in seed_indices]

    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]

    # For each cell, distance² to each seed: shape (k, H, W). Pick argmin.
    dist_stack = np.empty((k, height, width), dtype=np.float32)
    for i, (sr, sc) in enumerate(seeds):
        dist_stack[i] = (rows - sr) ** 2 + (cols - sc) ** 2
    assignments = np.argmin(dist_stack, axis=0)

    leaves: list[RegionVector] = []
    for i, (sr, sc) in enumerate(seeds):
        mask = assignments == i
        if not mask.any():
            # Degenerate: this seed is shadowed by another (only happens at
            # k close to num_cells with tied distances). Fall back to a
            # 1-cell leaf at the seed itself so we still emit K leaves.
            mask = np.zeros_like(mask)
            mask[sr, sc] = True

        # Bounding box in patch-grid coordinates → normalised image coords.
        ys, xs = np.where(mask)
        row_lo, row_hi = int(ys.min()), int(ys.max())
        col_lo, col_hi = int(xs.min()), int(xs.max())
        box = (
            col_lo / width,
            row_lo / height,
            (col_hi + 1) / width,
            (row_hi + 1) / height,
        )

        # Saliency-weighted-mean vector.  We use floored saliencies so a
        # cell with literally zero saliency still contributes its vector
        # (otherwise a flat-saliency image would produce a zero leaf vec).
        # The *floored* weight is what feeds into the HAC weighted-pool
        # merger downstream — we also stash it on the leaf so internals
        # can combine vectors additively without referring back to the
        # patch grid.
        weights = np.maximum(patch_saliency[mask], 1e-8).astype(np.float32)
        weight_total = float(weights.sum())
        norm_weights = weights / weight_total
        vecs = patch_grid[mask].astype(np.float32, copy=False)
        leaf_vec = (vecs * norm_weights[:, None]).sum(axis=0)
        leaf_vec = _l2_normalize(leaf_vec)

        leaves.append(
            RegionVector(
                box=box,
                vec=leaf_vec,
                children=None,
                cell_mask=mask.astype(bool, copy=True),
                weight=weight_total,
            )
        )

    return leaves


# ---------------------------------------------------------------------------
# Hierarchical agglomerative clustering
# ---------------------------------------------------------------------------


def _box_centroid(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _merge_boxes(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _affinity(
    a: RegionVector,
    b: RegionVector,
    alpha: float,
) -> float:
    """Higher is closer.  Tunable blend of cosine + spatial proximity.

    ``alpha`` weights the cosine half; ``1 - alpha`` weights spatial
    proximity.  ``alpha = 1`` is pure-cosine (clusters visually-similar but
    spatially-distant regions, e.g. two faces in a crowd).  ``alpha = 0`` is
    pure-adjacency (merges anything that touches).  ``alpha = 0.5`` is the
    default starting point; final value pinned by the caltech101_s sweep.
    """
    cosine = float(a.vec.astype(np.float32) @ b.vec.astype(np.float32))
    ax, ay = _box_centroid(a.box)
    bx, by = _box_centroid(b.box)
    # Centroid distance is bounded by sqrt(2) for boxes in [0,1]²; turn it
    # into a similarity in [0, 1].
    spatial = 1.0 - min(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / (2.0**0.5), 1.0)
    return alpha * cosine + (1.0 - alpha) * spatial


def build_hac_tree(
    leaves: list[RegionVector],
    alpha: float,
    *,
    patch_grid: np.ndarray,
    patch_saliency: np.ndarray,
) -> list[RegionVector]:
    """Build a strict binary HAC tree on top of *leaves*.

    Returns a flat list of size ``2K - 1`` where the first ``K`` entries
    are the input leaves (verbatim) and the remaining ``K - 1`` entries
    are internal merge nodes, each with ``children`` pointing earlier in
    the list.

    Merge rule: at each step, find the pair of currently-live nodes with
    the highest :func:`_affinity` and merge them.  *patch_grid* and
    *patch_saliency* are required so leaf weighted sums can be derived
    from each leaf's ``cell_mask`` once at the start; internals then
    combine vectors **additively** (``new_weighted_sum =
    a.weighted_sum + b.weighted_sum``).  Because leaf cell sets are
    disjoint by construction, this is exactly the saliency-weighted mean
    of the patches in the merged cell union — order-independent, and
    equal to what we would get by re-pooling from the patch grid each
    merge.

    The merged node's box is the union of the two child boxes (loose
    bounding rectangle — still kept for cheap UI hints, even though the
    true footprint is the union of cell masks).

    Complexity: ``O(K³)`` from the brute-force argmax inner loop.  For
    ``K ≤ 16`` this is negligible (<5 ms / image).
    """
    if not leaves:
        raise ValueError("leaves must be non-empty")
    height, width, _ = patch_grid.shape
    if patch_saliency.shape != (height, width):
        raise ValueError(f"patch_saliency must be ({height}, {width}); got {patch_saliency.shape}")

    nodes: list[RegionVector] = list(leaves)
    # Parallel array of unnormalised saliency-weighted sums, one per node.
    # Carried only inside build; not stored on RegionVector.  Floored
    # saliencies (matching propose_leaves) keep zero-saliency cells from
    # silently dropping out of the mean.
    floored_sal = np.maximum(patch_saliency.astype(np.float32, copy=False), 1e-8)
    weighted_sums: list[np.ndarray] = []
    for leaf in leaves:
        mask = leaf.cell_mask
        if mask is None:
            raise ValueError("build_hac_tree requires leaves with cell_mask set (propose_leaves now populates this)")
        sal_slice = floored_sal[mask]
        vec_slice = patch_grid[mask].astype(np.float32, copy=False)
        weighted_sums.append((vec_slice * sal_slice[:, None]).sum(axis=0))

    live: list[int] = list(range(len(leaves)))

    while len(live) > 1:
        best_pair: Optional[tuple[int, int]] = None
        best_score = -np.inf
        for i_idx, i in enumerate(live):
            for j in live[i_idx + 1 :]:
                score = _affinity(nodes[i], nodes[j], alpha)
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        assert best_pair is not None
        a_idx, b_idx = best_pair
        a, b = nodes[a_idx], nodes[b_idx]

        merged_sum = weighted_sums[a_idx] + weighted_sums[b_idx]
        merged_weight = float(a.weight + b.weight)
        merged_vec = _l2_normalize(merged_sum)
        merged_box = _merge_boxes(a.box, b.box)
        nodes.append(
            RegionVector(
                box=merged_box,
                vec=merged_vec,
                children=(a_idx, b_idx),
                cell_mask=None,
                weight=merged_weight,
            )
        )
        weighted_sums.append(merged_sum)
        new_idx = len(nodes) - 1
        live = [x for x in live if x not in (a_idx, b_idx)]
        live.append(new_idx)

    return nodes


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Vote attribution (v2): on-the-fly box → vote vector
# ---------------------------------------------------------------------------


def box_to_vote_vector(
    patch_grid: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    """Pool a user-drawn box into one vote vector via uniform mean.

    Selects the patch cells whose centers fall inside the normalised box,
    averages their vectors with equal weight, L2-normalises.  No saliency
    weighting — every selected cell contributes equally.

    Same patch set → same vector, regardless of how the user assembled the
    box.  Two boxes that select the same cells produce the same result.
    The pre-normalisation sum is additive across disjoint cell sets, so a
    hypothetical "multi-box vote" that unioned cells from several boxes
    would still pool consistently.

    Deliberately *not* an attempt to recover the vector of whatever HAC
    region happens to span the same patches — v1's HAC builder re-L2-
    normalises at every internal merge, so no pooling rule can match a
    specific HAC node's vector without replicating its exact merge chain.
    See ``docs/plans/patch-embedder.md`` ("v2 → Backend semantics §1").

    Parameters
    ----------
    patch_grid : ndarray, shape (H, W, D)
        Per-patch vectors out of the embedder (already L2-normalised), as
        stored in ``media["patch_grid"]``.  Float16 (pickle dtype) or
        float32 input is accepted; the result is always float32.
    box : (x0, y0, x1, y1)
        Normalised image coordinates in ``[0, 1]``; ``x`` runs along the
        grid columns, ``y`` along the rows.  Same convention as
        :func:`propose_leaves` and :class:`vtsearch.datasets.labelset.LabeledElement.region_box`.
        Swapped corners are tolerated; out-of-range coordinates are clamped
        to the unit square.

    Returns
    -------
    ndarray, shape (D,), float32, L2-normalised.

    Notes
    -----
    * Inclusion rule: a patch at grid position ``(row, col)`` is included
      iff its center ``((col + 0.5) / W, (row + 0.5) / H)`` falls inside
      the closed rectangle ``[x_lo, x_hi] × [y_lo, y_hi]`` (clamped corners).
    * Empty-selection fallback: if the (possibly very thin) box contains
      no cell centers, snap to the single cell whose center is closest to
      the box center.  Callers always get a well-defined unit vector.
    """
    if patch_grid.ndim != 3:
        raise ValueError(f"patch_grid must be (H, W, D); got shape {patch_grid.shape}")
    if len(box) != 4:
        raise ValueError(f"box must be a 4-tuple; got {box!r}")

    height, width, _ = patch_grid.shape
    x0, y0, x1, y1 = (float(v) for v in box)
    x_lo, x_hi = max(0.0, min(x0, x1)), min(1.0, max(x0, x1))
    y_lo, y_hi = max(0.0, min(y0, y1)), min(1.0, max(y0, y1))

    col_centers = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    row_centers = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    inside_x = (col_centers >= x_lo) & (col_centers <= x_hi)
    inside_y = (row_centers >= y_lo) & (row_centers <= y_hi)
    mask = inside_y[:, None] & inside_x[None, :]

    if not mask.any():
        cx = 0.5 * (x_lo + x_hi)
        cy = 0.5 * (y_lo + y_hi)
        col_idx = int(np.argmin(np.abs(col_centers - cx)))
        row_idx = int(np.argmin(np.abs(row_centers - cy)))
        mask = np.zeros((height, width), dtype=bool)
        mask[row_idx, col_idx] = True

    vecs = patch_grid[mask].astype(np.float32, copy=False)
    pooled = vecs.mean(axis=0)
    return _l2_normalize(pooled)


def to_fp16(regions: list[RegionVector]) -> list[RegionVector]:
    """Cast every region vector to float16 for pickling.

    Used by the loader pipeline to compress ``media["patch_regions"]``
    before it lands in the dataset pickle.  Vectors are rehydrated to
    float32 by callers that score them (similarity, MLP).
    """
    return [
        RegionVector(
            box=r.box,
            vec=r.vec.astype(np.float16, copy=False),
            children=r.children,
            cell_mask=r.cell_mask,
            weight=r.weight,
        )
        for r in regions
    ]


def build_region_tree(
    output: PatchEmbedOutput,
    *,
    k: int = 12,
    alpha: float = 0.5,
) -> list[RegionVector]:
    """Build the full ``media["patch_regions"]`` list from one embedder output.

    Layout of the returned list:

    * index 0       — CLS-pooled full-image node, box ``(0, 0, 1, 1)``,
      ``children = None``.
    * indices 1..K  — HAC leaves (saliency-peak clusters of patches).
    * indices K+1.. — HAC internal merge nodes; each has ``children``
      pointing at two earlier entries.

    Total length is ``2K`` (1 full-image + K leaves + K-1 internals).
    """
    full_image = RegionVector(
        box=(0.0, 0.0, 1.0, 1.0),
        vec=_l2_normalize(output.cls_vec.astype(np.float32, copy=False)),
        children=None,
        cell_mask=None,
        weight=0.0,
    )
    leaves = propose_leaves(output.patch_grid, output.patch_saliency, k=k)
    hac = build_hac_tree(
        leaves,
        alpha=alpha,
        patch_grid=output.patch_grid,
        patch_saliency=output.patch_saliency,
    )
    # The first K entries of `hac` are the leaves verbatim; entries K..2K-2
    # are the internal merge nodes (children indices are local to `hac`,
    # which corresponds to indices 1..2K-1 of the returned list once we
    # prepend `full_image`).  We rewrite the children indices to reference
    # positions in the final list (offset by +1 since full_image takes
    # position 0).
    out: list[RegionVector] = [full_image]
    for node in hac:
        if node.children is None:
            out.append(node)
        else:
            ci, cj = node.children
            out.append(
                RegionVector(
                    box=node.box,
                    vec=node.vec,
                    children=(ci + 1, cj + 1),
                    cell_mask=node.cell_mask,
                    weight=node.weight,
                )
            )
    return out
