"""Detection-style abstraction for the Max-Patch experiment.

The voting-iterations harness (:mod:`vtscore.eval.voting_iterations`) can run
each simulated detector under a named **detection style** - the bundle of rules
that decides (a) which vector a Good vote trains on, (b) which vector(s) a Bad
vote trains on, (c) how a trained MLP scores an image at inference, and (d) how
a cropped exemplar seeds the startup sort.  Three styles exist:

* ``whole_image`` - the classic single-vector pipeline (SigLIP et al.): every
  vote and every score uses the image-level embedding; region boxes are
  ignored.  The baseline arm.

* ``max_hac`` - the production patch pipeline: a Good region-vote snaps to the
  nearest HAC region-tree node (:func:`vtscore.media.patch_embed.snap_box_to_region`),
  a Bad vote floods the CLS node + HAC leaves as negatives
  (:func:`vtscore.detectors.training.bad_negative_vecs`), and an image scores
  by max-pooling the MLP over every region node - exactly what the live
  detector does on a patch dataset.

* ``max_patch`` - the HAC-free alternative under test: a Good region-vote
  trains on the **single raw patch** closest to the voted box
  (:func:`vtscore.media.patch_embed.nearest_patch_to_box`), a Bad vote floods
  the full-image vector + **every raw patch** of the image as negatives, and an
  image scores by max-pooling the MLP over the full-image vector plus all
  ``H x W`` raw patch vectors.  No region tree is consulted at any point.

Each style also maps a *query vector* (e.g. the full-image embedding of a
cropped exemplar) to per-image similarities for the Autopilot seed phase:
whole-image cosine, max-over-region-nodes cosine, and max-over-patches cosine
respectively.

**Every vector a style can train a vote on must also be a row that style
scores over.**  ``max_hac`` gets this for free: ``patch_regions[0]`` is the
CLS full-image node (``children=None``), so it is both flooded by
:func:`~vtscore.detectors.training.bad_negative_vecs` and pooled at inference,
and a *boxless* Good vote - which falls back to the image-level vector - trains
in a geometry inference actually evaluates.  ``max_patch`` originally scored raw
patches only, so a boxless Good vote trained on a vector that was never scored;
the classifier then separated "full-image-like" from "raw-patch-like" (every Bad
vote floods raw patches as negatives) and the calibrated threshold landed in a
gap the production score distribution never reaches - perfect ranking, zero FPR,
catastrophic FNR.  The full-image row in :meth:`MaxPatchStyle.score_rows` (and
its matching negative in :meth:`MaxPatchStyle.bad_vecs`) closes that hole.

Styles are **stateful per run**: :func:`resolve_style` returns a fresh instance
whose flattened score matrices are memoised per media-id set, so repeated
per-step scoring of the same test/sim split doesn't rebuild a multi-hundred-
thousand-row matrix 150 times.  Do not share one instance across datasets.

This is experiment-tier code: the production vote/score paths in
:mod:`vtscore.detectors.training` are the source of truth for ``max_hac``, and
this module reuses them directly rather than re-implementing.
"""

from __future__ import annotations

import os

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np
    import torch.nn as nn

from vtscore.embedding.media_vectors import media_embedding

#: Rows per forward-pass chunk when scoring a flattened patch matrix.  Patch
#: matrices are stored float16 (the pickle dtype) and upcast chunk-wise, so
#: peak float32 memory stays bounded regardless of dataset size.
_SCORE_CHUNK_ROWS = 65_536


def _unit(vec: "np.ndarray") -> "np.ndarray":
    import numpy as np  # noqa: PLC0415

    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _forward_sigmoid_chunked(model: "nn.Sequential", matrix: "np.ndarray") -> "np.ndarray":
    """Run ``sigmoid(model(matrix))`` in chunks; accepts a float16 or float32 matrix."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    out = np.empty(matrix.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, matrix.shape[0], _SCORE_CHUNK_ROWS):
            chunk = torch.from_numpy(np.ascontiguousarray(matrix[start : start + _SCORE_CHUNK_ROWS]))
            chunk = chunk.to(device=device, dtype=torch.float32)
            out[start : start + chunk.shape[0]] = torch.sigmoid(model(chunk)).squeeze(1).cpu().numpy()
    return out


def _segment_max(flat: "np.ndarray", seg_starts: "np.ndarray") -> "np.ndarray":
    import numpy as np  # noqa: PLC0415

    return np.maximum.reduceat(flat, seg_starts)


class WholeImageStyle:
    """Single-vector baseline: votes and scores use the image-level embedding."""

    name = "whole_image"

    def good_vec(self, media: dict[str, Any], box: Optional[tuple[float, float, float, float]]) -> "np.ndarray":
        return media_embedding(media)

    def bad_vecs(self, media: dict[str, Any]) -> list["np.ndarray"]:
        return [media_embedding(media)]

    def score_rows(self, media: dict[str, Any]) -> "np.ndarray":
        import numpy as np  # noqa: PLC0415

        return np.asarray(media_embedding(media), dtype=np.float32)[None, :]

    def score_media(self, model: "nn.Sequential", clips_dict: dict[int, dict[str, Any]]) -> dict[int, float]:
        import numpy as np  # noqa: PLC0415

        ids = sorted(clips_dict)
        if not ids:
            return {}
        matrix = np.stack([np.asarray(media_embedding(clips_dict[cid]), dtype=np.float32) for cid in ids])
        scores = _forward_sigmoid_chunked(model, matrix)
        return {cid: float(s) for cid, s in zip(ids, scores, strict=True)}

    def exemplar_sims(self, clips_dict: dict[int, dict[str, Any]], query_vec: "np.ndarray") -> dict[int, float]:
        import numpy as np  # noqa: PLC0415

        q = _unit(query_vec)
        ids = sorted(clips_dict)
        if not ids:
            return {}
        matrix = np.stack([_unit(media_embedding(clips_dict[cid])) for cid in ids])
        cos = matrix @ q
        return {cid: float(c) for cid, c in zip(ids, cos, strict=True)}


class _FlattenedStyle:
    """Shared max-pool machinery for the two patch styles.

    Subclasses provide :meth:`_rows_for_media` - the per-image stack of
    candidate vectors an image is max-pooled over (region-tree nodes for
    ``max_hac``, raw patches for ``max_patch``).  The flattened
    ``(rows, seg_starts, ids)`` arrays are memoised per media-id set: region
    and patch vectors never change during a run, only the MLP weights do.
    """

    name = "abstract"

    def __init__(self) -> None:
        self._matrix_cache: dict[frozenset[int], tuple[list[int], Any, Any]] = {}

    def _rows_for_media(self, media: dict[str, Any]) -> "np.ndarray":
        raise NotImplementedError

    def score_rows(self, media: dict[str, Any]) -> "np.ndarray":
        """The rows this style max-pools over when scoring *media* at inference.

        Public counterpart of :meth:`_rows_for_media`, upcast to float32.  The
        calibrator uses it to collapse each vote's bag in **inference**
        geometry rather than in the geometry it happened to train on - see
        :func:`vtscore.training.thresholds.compute_fold_orderings`.
        """
        import numpy as np  # noqa: PLC0415

        return np.asarray(self._rows_for_media(media), dtype=np.float32)

    def _flattened(self, clips_dict: dict[int, dict[str, Any]]) -> tuple[list[int], "np.ndarray", "np.ndarray"]:
        import numpy as np  # noqa: PLC0415

        key = frozenset(clips_dict)
        cached = self._matrix_cache.get(key)
        if cached is not None:
            return cached
        ids = sorted(clips_dict)
        blocks = [self._rows_for_media(clips_dict[cid]) for cid in ids]
        seg_starts = np.zeros(len(blocks), dtype=np.int64)
        np.cumsum([b.shape[0] for b in blocks[:-1]], out=seg_starts[1:])
        # Keep the flattened stack float16 (the pickle dtype) so a large
        # patch dataset doesn't double its memory here; the scorer upcasts
        # chunk-wise.
        matrix = np.concatenate(blocks, axis=0).astype(np.float16, copy=False)
        result = (ids, matrix, seg_starts)
        self._matrix_cache[key] = result
        return result

    def score_media(self, model: "nn.Sequential", clips_dict: dict[int, dict[str, Any]]) -> dict[int, float]:
        if not clips_dict:
            return {}
        ids, matrix, seg_starts = self._flattened(clips_dict)
        flat = _forward_sigmoid_chunked(model, matrix)
        pooled = _segment_max(flat, seg_starts)
        return {cid: float(s) for cid, s in zip(ids, pooled, strict=True)}

    def exemplar_sims(self, clips_dict: dict[int, dict[str, Any]], query_vec: "np.ndarray") -> dict[int, float]:
        import numpy as np  # noqa: PLC0415

        if not clips_dict:
            return {}
        q = _unit(query_vec)
        ids, matrix, seg_starts = self._flattened(clips_dict)
        flat = matrix.astype(np.float32, copy=False) @ q
        pooled = _segment_max(flat.astype(np.float64, copy=False), seg_starts)
        return {cid: float(s) for cid, s in zip(ids, pooled, strict=True)}


class MaxHacStyle(_FlattenedStyle):
    """The production patch pipeline: HAC snap / leaf flood / region max-pool."""

    name = "max_hac"

    def good_vec(self, media: dict[str, Any], box: Optional[tuple[float, float, float, float]]) -> "np.ndarray":
        from vtscore.detectors.training import pool_box_from_media  # noqa: PLC0415

        pooled = pool_box_from_media(media, box)
        return pooled if pooled is not None else media_embedding(media)

    def bad_vecs(self, media: dict[str, Any]) -> list["np.ndarray"]:
        from vtscore.detectors.training import bad_negative_vecs  # noqa: PLC0415

        return bad_negative_vecs(media)

    def _rows_for_media(self, media: dict[str, Any]) -> "np.ndarray":
        import numpy as np  # noqa: PLC0415

        regions = media.get("patch_regions")
        if regions:
            return np.stack([np.asarray(r.vec, dtype=np.float16) for r in regions])
        return np.asarray(media_embedding(media), dtype=np.float16)[None, :]


class MaxPatchStyle(_FlattenedStyle):
    """The HAC-free alternative: nearest patch / all-patch flood / patch max-pool."""

    name = "max_patch"

    def good_vec(self, media: dict[str, Any], box: Optional[tuple[float, float, float, float]]) -> "np.ndarray":
        import numpy as np  # noqa: PLC0415

        grid = media.get("patch_grid")
        if box is not None and grid is not None:
            from vtscore.media.patch_embed import nearest_patch_to_box  # noqa: PLC0415

            return nearest_patch_to_box(np.asarray(grid), box)
        # Image-level Good vote (or a grid-less media): the CLS/full-image
        # vector - the only image-level representative available.
        return media_embedding(media)

    def bad_vecs(self, media: dict[str, Any]) -> list["np.ndarray"]:
        """The full-image vector plus every raw patch, as negatives.

        The full-image row is included for the same reason ``max_hac`` floods
        the CLS node: a Bad vote asserts that *no* row of this image should
        score high, and :meth:`_rows_for_media` max-pools the full-image row at
        inference.  Leaving it out would hand every image an un-suppressed
        scoring row.
        """
        import numpy as np  # noqa: PLC0415

        grid = media.get("patch_grid")
        if grid is None:
            return [media_embedding(media)]
        flat = np.asarray(grid, dtype=np.float32).reshape(-1, np.asarray(grid).shape[-1])
        return [np.asarray(media_embedding(media), dtype=np.float32), *flat]

    def _rows_for_media(self, media: dict[str, Any]) -> "np.ndarray":
        """The full-image vector stacked above every raw patch.

        Row 0 is the image-level (CLS) vector - the ``max_hac`` tree carries the
        same node at ``patch_regions[0]``, and without it a boxless Good vote
        (:meth:`good_vec` with ``box=None``) would train on a vector this
        scorer never evaluates.  See the module docstring.
        """
        import numpy as np  # noqa: PLC0415

        cls_row = np.asarray(media_embedding(media), dtype=np.float16)[None, :]
        grid = media.get("patch_grid")
        if grid is None:
            return cls_row
        arr = np.asarray(grid, dtype=np.float16)
        return np.concatenate([cls_row, arr.reshape(-1, arr.shape[-1])], axis=0)


def build_patch_hac_tree(
    patch_grid: "np.ndarray",
    cls_vec: "Optional[np.ndarray]" = None,
    *,
    alpha: float = 0.5,
    pca_dims: "Optional[int]" = None,
) -> list:
    """Binary HAC tree with the **raw patches as leaves** - the MaxPatchHAC tree.

    Where the production tree (:func:`vtscore.media.patch_embed.build_region_tree`)
    K-means-pools patches into ~12 leaves *before* merging, this keeps every one
    of the ``H*W`` raw patches as its own leaf and agglomeratively merges them
    (blended cosine + spatial distance, average linkage) into progressively
    larger region nodes.  The tree therefore carries candidates at every scale
    from a single patch (which wins on small targets, like ``max_patch``) up to
    the whole image (which wins on large targets, like ``max_hac``) at only ~2x
    the node count of the raw patches (``2*H*W - 1`` tree nodes + the CLS node).

    Returns a :class:`~vtscore.media.patch_embed.RegionVector` list in the same
    layout convention as ``build_region_tree``: index 0 is the CLS whole-image
    node (when *cls_vec* is given), then the raw-patch leaves, then the internal
    merge nodes whose ``children`` index earlier entries in the list.  Internal
    node vectors are the L2-normalised **uniform** mean of their member patches
    (the experiment carries no per-patch saliency, so - unlike production -
    every patch counts equally).
    """
    import numpy as np  # noqa: PLC0415
    from scipy.cluster.hierarchy import linkage  # noqa: PLC0415
    from scipy.spatial.distance import squareform  # noqa: PLC0415

    from vtscore.media.patch_embed import RegionVector  # noqa: PLC0415

    grid = np.asarray(patch_grid, dtype=np.float32)
    height, width, dim = grid.shape
    n = height * width
    patches = grid.reshape(n, dim)
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    patches = patches / np.where(norms > 1e-12, norms, 1.0)

    rows, cols = np.divmod(np.arange(n), width)
    leaf_boxes = [
        (c / width, r / height, (c + 1) / width, (r + 1) / height) for r, c in zip(rows.tolist(), cols.tolist())
    ]

    if n > 1:
        # Optional PCA on the merge *order* only: decide affinities on cosines
        # in a per-image PCA-reduced space (denoised); stored node vecs stay
        # full-dim, so scoring is unchanged. pca_dims=None is the raw path.
        sim = patches
        if pca_dims:
            from vtscore.media.patch_embed import _fit_pca_projector  # noqa: PLC0415

            project = _fit_pca_projector(grid, int(pca_dims))
            if project is not None:
                sim = project(patches)  # one batched transform, not n per-vector calls
        cos_d = np.clip((1.0 - sim @ sim.T) * 0.5, 0.0, 1.0)
        centers = np.stack([(cols + 0.5) / width, (rows + 0.5) / height], axis=1).astype(np.float32)
        spatial = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1)) / np.sqrt(2.0)
        blended = alpha * cos_d + (1.0 - alpha) * spatial
        np.fill_diagonal(blended, 0.0)
        linkage_matrix = linkage(squareform(blended, checks=False), method="average")
    else:
        linkage_matrix = np.empty((0, 4))

    sums = [patches[i].copy() for i in range(n)]
    boxes = list(leaf_boxes)
    nodes = [
        RegionVector(box=leaf_boxes[i], vec=patches[i], children=None, cell_mask=None, weight=1.0) for i in range(n)
    ]
    for merge in linkage_matrix:
        a, b = int(merge[0]), int(merge[1])
        total = sums[a] + sums[b]
        sums.append(total)
        vec = total / max(float(np.linalg.norm(total)), 1e-12)
        ax0, ay0, ax1, ay1 = boxes[a]
        bx0, by0, bx1, by1 = boxes[b]
        box = (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))
        boxes.append(box)
        nodes.append(
            RegionVector(
                box=box,
                vec=vec.astype(np.float32),
                children=(a, b),
                cell_mask=None,
                weight=nodes[a].weight + nodes[b].weight,
            )
        )

    if cls_vec is None:
        return nodes
    full = RegionVector(
        box=(0.0, 0.0, 1.0, 1.0),
        vec=_unit(np.asarray(cls_vec, dtype=np.float32)),
        children=None,
        cell_mask=None,
        weight=0.0,
    )
    out = [full]
    for node in nodes:
        if node.children is None:
            out.append(node)
        else:
            ci, cj = node.children
            out.append(
                RegionVector(box=node.box, vec=node.vec, children=(ci + 1, cj + 1), cell_mask=None, weight=node.weight)
            )
    return out


class MaxPatchHacStyle(_FlattenedStyle):
    """Raw-patch-leaf HAC tree: multi-scale snap / all-node flood / all-node max-pool.

    The hybrid under test.  It builds a HAC tree whose leaves are the raw
    patches and merges them up a binary tree (:func:`build_patch_hac_tree`), so
    the tree carries candidates at every scale.  A Good region-vote **snaps to
    the tree node whose box best matches** the drawn box (multi-scale, like
    ``max_hac`` but over a raw-patch-leaved tree); a Bad vote floods **every
    tree node** as a negative - symmetric with inference, which max-pools the
    MLP over every node; an image scores by max-pooling over all nodes.  The
    per-image tree is memoised per media id (it depends only on the frozen
    ``patch_grid``), so the 150-step trajectory builds each tree once.
    """

    name = "max_patch_hac"

    def __init__(self) -> None:
        super().__init__()
        self._tree_cache: "dict[int, list]" = {}

    def _tree(self, media: dict[str, Any]) -> list:
        mid = int(media.get("id", id(media)))
        cached = self._tree_cache.get(mid)
        if cached is not None:
            return cached
        import numpy as np  # noqa: PLC0415

        grid = media.get("patch_grid")
        if grid is None:
            from vtscore.media.patch_embed import RegionVector  # noqa: PLC0415

            tree = [
                RegionVector(
                    box=(0.0, 0.0, 1.0, 1.0),
                    vec=_unit(np.asarray(media_embedding(media), dtype=np.float32)),
                    children=None,
                    cell_mask=None,
                    weight=0.0,
                )
            ]
        else:
            tree = build_patch_hac_tree(np.asarray(grid), media_embedding(media))
        self._tree_cache[mid] = tree
        return tree

    def good_vec(self, media: dict[str, Any], box: Optional[tuple[float, float, float, float]]) -> "np.ndarray":
        if box is not None and media.get("patch_grid") is not None:
            from vtscore.media.patch_embed import snap_box_to_region  # noqa: PLC0415

            snapped = snap_box_to_region(self._tree(media), box)
            if snapped is not None:
                return snapped
        return media_embedding(media)

    def bad_vecs(self, media: dict[str, Any]) -> list["np.ndarray"]:
        import numpy as np  # noqa: PLC0415

        return [np.asarray(node.vec, dtype=np.float32) for node in self._tree(media)]

    def _rows_for_media(self, media: dict[str, Any]) -> "np.ndarray":
        import numpy as np  # noqa: PLC0415

        return np.stack([np.asarray(node.vec, dtype=np.float16) for node in self._tree(media)])


class MaxPatchPcaHacStyle(MaxPatchHacStyle):
    """MaxPatchHAC with a PCA-denoised merge order.

    Identical to :class:`MaxPatchHacStyle` except the raw-patch HAC tree's merge
    *order* is decided on cosines in a per-image PCA space (``pca_dims``
    components) rather than the full 768-dim patch space — the option ported
    from the HAC-tree-improvements branch.  Only the tree *topology* changes;
    every stored node vector stays full-dim, so the scoring / vote / flood
    machinery is exactly :class:`MaxPatchHacStyle`'s.  ``MAXPATCH_PCA_DIMS``
    (default 32) sets the reduced dimensionality.
    """

    name = "max_patch_pca_hac"
    pca_dims = int(os.environ.get("MAXPATCH_PCA_DIMS", "32"))

    def _tree(self, media: dict[str, Any]) -> list:
        mid = int(media.get("id", id(media)))
        cached = self._tree_cache.get(mid)
        if cached is not None:
            return cached
        import numpy as np  # noqa: PLC0415

        grid = media.get("patch_grid")
        if grid is None:
            from vtscore.media.patch_embed import RegionVector  # noqa: PLC0415

            tree = [
                RegionVector(
                    box=(0.0, 0.0, 1.0, 1.0),
                    vec=_unit(np.asarray(media_embedding(media), dtype=np.float32)),
                    children=None,
                    cell_mask=None,
                    weight=0.0,
                )
            ]
        else:
            tree = build_patch_hac_tree(np.asarray(grid), media_embedding(media), pca_dims=self.pca_dims)
        self._tree_cache[mid] = tree
        return tree


#: Style-name registry.  Values are *classes*; :func:`resolve_style` returns a
#: fresh instance so per-run matrix memoisation never leaks across datasets.
STYLES: dict[str, type] = {
    WholeImageStyle.name: WholeImageStyle,
    MaxHacStyle.name: MaxHacStyle,
    MaxPatchStyle.name: MaxPatchStyle,
    MaxPatchHacStyle.name: MaxPatchHacStyle,
    MaxPatchPcaHacStyle.name: MaxPatchPcaHacStyle,
}


def resolve_style(name: str) -> Any:
    """Return a fresh style instance for *name*; raise ``KeyError`` on a typo."""
    try:
        cls = STYLES[name]
    except KeyError:
        raise KeyError(f"Unknown detection style {name!r}; available: {', '.join(sorted(STYLES))}") from None
    return cls()
