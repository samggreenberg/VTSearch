"""Raw patch grids for image embedders (the MaxPatch region geometry).

Patch-based image encoders (DINOv2, DINOv3, EUPE) return a vector per spatial
patch plus a CLS token.  This module owns that raw output and the small amount
of geometry the vote/score paths need on top of it:

  * ``PatchEmbedOutput`` - the wire format the embedder hands us:
    CLS vector + per-patch grid + per-patch saliency.

  * ``nearest_patch_to_box`` - the Good region-vote rule: the single raw patch
    vector standing in for a user-drawn box.

  * ``patch_row_box`` - the inverse map, turning a winning **score row** back
    into the rectangle the UI outlines (row 0 = whole image, rows ``1..H*W`` =
    grid cells in row-major order).

There is no region *tree*.  The Max-Patch study
(``docs/experiments/2026-07-29-max-patch/REPORT.md``) measured the HAC region tree that
used to live here against tree-free raw patches over 23 scale-band Visual
Genome categories and the tree lost on both halves of the error at every scale
band, so ingest now stores only ``media["patch_grid"]`` and every vote / score
path works directly off it.  The tree survives solely as experiment-tier code
in :mod:`vtscore.eval.patch_styles`, which keeps its own node type.

This module is intentionally vector-only (numpy / no torch dependency beyond
the embedder-output adapters); it does *not* score patches, store them on disk,
or care about ``MediaEmbedder`` registration - those concerns live in the
loader pipeline, :mod:`vtscore.embedding.matrix`, and
:mod:`vtscore.training.region_similarity`.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Vector helpers
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

    ``outputs`` must have ``last_hidden_state`` and ``attentions`` set -
    pass ``output_attentions=True`` to the model forward call.

    *batch_index* selects which image in a batched forward to extract
    (default 0 preserves the single-image call sites).

    Returns ``None`` if the patch grid isn't square (the loader treats
    that as "no regions for this image", same as a forward-pass failure).
    """
    import torch  # noqa: PLC0415

    hidden = outputs.last_hidden_state[batch_index]  # (T, D)
    attn = outputs.attentions[-1][batch_index]  # (heads, T, T) - last block
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
    ``x_norm_patchtokens``), so we don't need a register-token slice - the
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
    # collapsing to a one-hot.
    #
    # The 4.0 is an inverse temperature applied directly to **cosine
    # magnitudes**, so how peaked this map comes out depends on the spread of
    # patch-to-CLS cosines in the embedder's space, which differs per embedder
    # (#3329 part 2 measured dinov3_patch as the least concentrated space in
    # its grid).  It was tuned in the caltech101_s sweep alongside the K and α
    # of the HAC region tree — the tree #2886 deleted — and nothing has read
    # ``patch_saliency`` since: ingest drops it (see
    # ``vtscore.datasets.stages.embedding._attach_patch_grid_to_media``) and no
    # other caller exists.  So the #3347 audit found the constant real but
    # inert, and left it rather than re-tuning a number with no consumer.
    # Anything that revives saliency-weighted pooling must re-derive this
    # per embedder rather than inherit 4.0.
    saliency = torch.softmax(sims * 4.0, dim=-1).reshape(side, side)

    return PatchEmbedOutput(
        cls_vec=_norm_torch(cls).astype(np.float32),
        patch_grid=_norm_torch(patch_grid).astype(np.float32),
        patch_saliency=saliency.detach().cpu().float().numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Vote attribution and row geometry (MaxPatch)
# ---------------------------------------------------------------------------


def nearest_patch_to_box(
    patch_grid: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    """Return the single patch vector spatially closest to a voted *box*.

    The Good region-vote rule: a vote that designated a box trains on **one raw
    patch vector** - the patch whose cell best stands in for the voted region -
    which is by construction one of the rows
    :func:`vtscore.embedding.matrix.media_score_rows` later scores the image
    over.  The pick is purely spatial:

    * among patches whose centers fall inside the (clamped) box, the one whose
      center is nearest the box center wins;
    * when no patch center falls inside (a box thinner than one cell), the
      patch whose center is nearest the box center wins outright.

    Both rules collapse to "the patch nearest the box center, preferring
    in-box patches", so a whole-image box picks the central patch and a tight
    box picks the patch under it.

    Parameters
    ----------
    patch_grid : ndarray, shape (H, W, D)
        Per-patch vectors as stored in ``media["patch_grid"]`` (float16 pickle
        dtype or float32; already L2-normalised).
    box : (x0, y0, x1, y1)
        Normalised image coordinates in ``[0, 1]``; swapped corners tolerated,
        out-of-range coordinates clamped; ``x`` runs along the grid columns,
        ``y`` along the rows.

    Returns
    -------
    ndarray, shape (D,), float32, L2-normalised.
    """
    if patch_grid.ndim != 3:
        raise ValueError(f"patch_grid must be (H, W, D); got shape {patch_grid.shape}")
    if len(box) != 4:
        raise ValueError(f"box must be a 4-tuple; got {box!r}")

    height, width, _ = patch_grid.shape
    x0, y0, x1, y1 = (float(v) for v in box)
    x_lo, x_hi = max(0.0, min(x0, x1)), min(1.0, max(x0, x1))
    y_lo, y_hi = max(0.0, min(y0, y1)), min(1.0, max(y0, y1))
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)

    col_centers = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    row_centers = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    inside_x = (col_centers >= x_lo) & (col_centers <= x_hi)
    inside_y = (row_centers >= y_lo) & (row_centers <= y_hi)
    inside = inside_y[:, None] & inside_x[None, :]

    dist2 = (col_centers[None, :] - cx) ** 2 + (row_centers[:, None] - cy) ** 2
    if inside.any():
        dist2 = np.where(inside, dist2, np.inf)
    flat_idx = int(np.argmin(dist2))
    row, col = divmod(flat_idx, width)
    return _l2_normalize(patch_grid[row, col].astype(np.float32, copy=False))


def patch_row_box(row_index: int, height: int, width: int) -> tuple[float, float, float, float]:
    """The rectangle a **score row** covers, in normalised image coordinates.

    Inverse of the row layout
    :func:`vtscore.embedding.matrix.media_score_rows` builds: row ``0`` is the
    image-level (CLS) vector and covers the whole image; rows ``1 .. H*W`` are
    the raw patch cells of an ``(H, W, D)`` grid in **row-major** order, so row
    ``1 + r*W + c`` is the cell at grid position ``(r, c)``.

    Used to turn the winning row of a max-pool back into the box the gallery /
    image viewer outlines as the best match.  Out-of-range indices clamp to the
    whole image rather than raising: a caller holding a stale winner should get
    a harmless overlay, not a 500.
    """
    if row_index <= 0 or height <= 0 or width <= 0 or row_index > height * width:
        return (0.0, 0.0, 1.0, 1.0)
    row, col = divmod(row_index - 1, width)
    return (col / width, row / height, (col + 1) / width, (row + 1) / height)
