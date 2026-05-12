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
and in :mod:`vtsearch.models.region_similarity`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    Resize-invariant by construction.  For the full-image node this is
    ``(0.0, 0.0, 1.0, 1.0)``.
    """

    vec: np.ndarray
    """L2-normalised vector for this region, shape ``(D,)``.

    Stored as **float16** in the dataset pickle and cast to float32 when
    read into RAM.  Construction code uses float32; the cast to float16
    happens at pickle time, not here.
    """

    children: Optional[tuple[int, int]] = None
    """Indices of the two children if this is an internal HAC node.

    ``None`` for leaves and for the CLS full-image node.  Encodes the merge
    structure of the agglomerative tree without us having to walk the list
    twice.
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
) -> Optional[PatchEmbedOutput]:
    """Turn a HuggingFace ViT ``ModelOutput`` into a :class:`PatchEmbedOutput`.

    Shared adapter for DINOv2 (``num_register_tokens=0``) and DINOv3
    (``num_register_tokens=4``).  Expects the token layout
    ``[CLS, R1..R_k, P1..P_N]`` and a square spatial patch grid.

    ``outputs`` must have ``last_hidden_state`` and ``attentions`` set —
    pass ``output_attentions=True`` to the model forward call.

    Returns ``None`` if the patch grid isn't square (the loader treats
    that as "no regions for this image", same as a forward-pass failure).
    """
    import torch  # noqa: PLC0415

    hidden = outputs.last_hidden_state[0]  # (T, D)
    attn = outputs.attentions[-1][0]  # (heads, T, T) — last block
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


def eupe_features_to_patch_output(features: dict) -> Optional[PatchEmbedOutput]:
    """Turn a :func:`facebookresearch/EUPE`-style ``forward_features`` dict
    into a :class:`PatchEmbedOutput`.

    EUPE's forward already separates CLS, storage tokens, and patch tokens
    into named keys (``x_norm_clstoken``, ``x_storage_tokens``,
    ``x_norm_patchtokens``), so we don't need a register-token slice — the
    storage tokens are already absent from ``x_norm_patchtokens``.

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

    cls = features["x_norm_clstoken"][0]  # (D,)
    patches = features["x_norm_patchtokens"][0]  # (N, D)
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


@dataclass
class _ProtoLeaf:
    """Intermediate leaf representation during clustering.

    Carries the cell-index set so we can compute the bounding box and the
    saliency-weighted vector after assignment is finalised.
    """

    cells: list[tuple[int, int]] = field(default_factory=list)
    """``(row, col)`` patch-grid indices that landed in this leaf."""

    seed: tuple[int, int] = (0, 0)
    """The peak cell this leaf is built around."""


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
        raise ValueError(
            f"patch_saliency must be ({height}, {width}); got {patch_saliency.shape}"
        )
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

        # Saliency-weighted-mean vector. Force a positive weight floor so a
        # cell with literally zero saliency still contributes its vector
        # (otherwise a flat-saliency image would produce a zero leaf vec).
        weights = np.maximum(patch_saliency[mask], 1e-8).astype(np.float32)
        weights = weights / weights.sum()
        vecs = patch_grid[mask].astype(np.float32, copy=False)
        leaf_vec = (vecs * weights[:, None]).sum(axis=0)
        leaf_vec = _l2_normalize(leaf_vec)

        leaves.append(RegionVector(box=box, vec=leaf_vec, children=None))

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
) -> list[RegionVector]:
    """Build a strict binary HAC tree on top of *leaves*.

    Returns a flat list of size ``2K - 1`` where the first ``K`` entries
    are the input leaves (verbatim) and the remaining ``K - 1`` entries
    are internal merge nodes, each with ``children`` pointing earlier in
    the list.

    The merge rule: at each step, find the pair of currently-live nodes
    with the highest :func:`_affinity` and merge them.  The merged node's
    vector is the saliency-balanced mean of the two children (here we use
    a flat 50/50 mean since we have no explicit weight per node — leaf
    saliency was already absorbed into the leaf vector), L2-normalised.
    The merged node's box is the union of the two child boxes.

    Complexity: ``O(K³)`` from the brute-force argmax inner loop.  For
    ``K ≤ 16`` this is negligible (<5 ms / image).
    """
    if not leaves:
        raise ValueError("leaves must be non-empty")

    nodes: list[RegionVector] = list(leaves)
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

        merged_vec = _l2_normalize((a.vec.astype(np.float32) + b.vec.astype(np.float32)) * 0.5)
        merged_box = _merge_boxes(a.box, b.box)
        nodes.append(RegionVector(box=merged_box, vec=merged_vec, children=(a_idx, b_idx)))
        new_idx = len(nodes) - 1
        live = [x for x in live if x not in (a_idx, b_idx)]
        live.append(new_idx)

    return nodes


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def to_fp16(regions: list[RegionVector]) -> list[RegionVector]:
    """Cast every region vector to float16 for pickling.

    Used by the loader pipeline to compress ``media["patch_regions"]``
    before it lands in the dataset pickle.  Vectors are rehydrated to
    float32 by callers that score them (similarity, MLP).
    """
    return [
        RegionVector(box=r.box, vec=r.vec.astype(np.float16, copy=False), children=r.children)
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
    )
    leaves = propose_leaves(output.patch_grid, output.patch_saliency, k=k)
    hac = build_hac_tree(leaves, alpha=alpha)
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
                )
            )
    return out
