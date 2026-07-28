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
  **every raw patch** of the image as a negative, and an image scores by
  max-pooling the MLP over all ``H x W`` raw patch vectors.  No region tree is
  consulted at any point.

Each style also maps a *query vector* (e.g. the full-image embedding of a
cropped exemplar) to per-image similarities for the Autopilot seed phase:
whole-image cosine, max-over-region-nodes cosine, and max-over-patches cosine
respectively.

Styles are **stateful per run**: :func:`resolve_style` returns a fresh instance
whose flattened score matrices are memoised per media-id set, so repeated
per-step scoring of the same test/sim split doesn't rebuild a multi-hundred-
thousand-row matrix 150 times.  Do not share one instance across datasets.

This is experiment-tier code: the production vote/score paths in
:mod:`vtscore.detectors.training` are the source of truth for ``max_hac``, and
this module reuses them directly rather than re-implementing.
"""

from __future__ import annotations

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
        import numpy as np  # noqa: PLC0415

        grid = media.get("patch_grid")
        if grid is None:
            return [media_embedding(media)]
        flat = np.asarray(grid, dtype=np.float32).reshape(-1, np.asarray(grid).shape[-1])
        return list(flat)

    def _rows_for_media(self, media: dict[str, Any]) -> "np.ndarray":
        import numpy as np  # noqa: PLC0415

        grid = media.get("patch_grid")
        if grid is None:
            return np.asarray(media_embedding(media), dtype=np.float16)[None, :]
        arr = np.asarray(grid, dtype=np.float16)
        return arr.reshape(-1, arr.shape[-1])


#: Style-name registry.  Values are *classes*; :func:`resolve_style` returns a
#: fresh instance so per-run matrix memoisation never leaks across datasets.
STYLES: dict[str, type] = {
    WholeImageStyle.name: WholeImageStyle,
    MaxHacStyle.name: MaxHacStyle,
    MaxPatchStyle.name: MaxPatchStyle,
}


def resolve_style(name: str) -> Any:
    """Return a fresh style instance for *name*; raise ``KeyError`` on a typo."""
    try:
        cls = STYLES[name]
    except KeyError:
        raise KeyError(f"Unknown detection style {name!r}; available: {', '.join(sorted(STYLES))}") from None
    return cls()
