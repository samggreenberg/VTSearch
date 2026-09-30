"""The tiled Stage 1 and the verification cache for structural document search (#3928).

On a ``sift_vlad_doc`` dataset every page carries ``media["tile_vectors"]``
(:mod:`vtscore.media.structural_tiles`). There, Stage 1 is not the detector
head's page-VLAD ranking. A page scores its best (query, tile) cosine, and the
queries are:

* on the vote path: each Good vote's box VLAD (the mark it boxed), or all the
  Good page's own tiles when the vote has no box;
* on the example path: each uploaded crop's VLAD.

Bads do not enter Stage 1 (#4169; the #3928 M3 gate measured an SVM over tile
rows 0.11 AP below max over queries at 50,000 pages).

The shortlist Stage 2 verifies grows to :func:`tiled_top_k`, and
:class:`VerificationCache` keeps each (template, page) fit across retrains, so
a vote costs one new template's pass, not the whole shortlist again (M4: ~1 s a
template for 1,000 pages on a GPU).

Library-tier. Every entry point is a no-op unless the snapshot carries tiles.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from vtscore.media.structural import MatchStats, StructuralFeatures, StructuralMatcher

_log = logging.getLogger(__name__)

#: Shortlist Stage 2 verifies when the tiled Stage 1 ran and a GPU does the matching.
#: At 50,000 pages it keeps 97% of exhaustive AP after 10 votes (#3928 M3).
TILED_TOP_K = 1000
#: The same without CUDA, where a template costs ~6 s per 1,000 pages instead of ~1 s
#: (#3928 M4).
TILED_TOP_K_CPU = 500
#: Rows per matmul chunk when scoring tiles, bounding the float32 working set.
_CHUNK_ROWS = 262_144
#: Cached verification fits kept before the cache starts over.
_CACHE_LIMIT = 500_000


def snapshot_has_tiles(snap: dict[Any, dict]) -> bool:
    """True iff any media in *snap* carries ``tile_vectors``."""
    return any(m.get("tile_vectors") is not None for m in snap.values())


def _cuda() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - no torch or no driver is simply "no GPU"
        return False


def tiled_top_k(n_pages: int) -> int:
    """How many Stage-1 pages Stage 2 verifies after a tiled Stage 1."""
    return max(0, min(n_pages, TILED_TOP_K if _cuda() else TILED_TOP_K_CPU))


# --------------------------------------------------------------------------
# The stacked tile matrix
# --------------------------------------------------------------------------

_MATRIX_CACHE: dict[tuple, tuple[list[Any], np.ndarray, np.ndarray]] = {}


def _tile_matrix(snap: dict[Any, dict]) -> tuple[list[Any], np.ndarray, np.ndarray]:
    """``(ids, tiles, starts)`` over the snapshot's tiled pages, stacked once per page set.

    Keyed on the identities of the pages' :class:`TileVectors` objects, which
    persist across requests on one loaded dataset and change on a reload, so a
    vote reuses the matrix and a reload rebuilds it.
    """
    ids = [mid for mid, m in snap.items() if m.get("tile_vectors") is not None]
    key = (len(ids), hash(tuple(id(snap[mid]["tile_vectors"]) for mid in ids)))
    hit = _MATRIX_CACHE.get(key)
    if hit is not None:
        return hit
    tiles = [snap[mid]["tile_vectors"] for mid in ids]
    counts = np.array([t.count for t in tiles], dtype=np.int64)
    matrix = np.concatenate([t.vectors for t in tiles]).astype(np.float16, copy=False)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    _MATRIX_CACHE.clear()  # one dataset at a time: the matrix is large
    entry = (ids, matrix, starts)
    _MATRIX_CACHE[key] = entry
    return entry


def _page_scores(matrix: np.ndarray, starts: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Each page's best (query, tile) cosine, chunked so float32 never holds the whole matrix."""
    q = np.asarray(queries, dtype=np.float32).T  # (dim, Q)
    best = np.empty(matrix.shape[0], dtype=np.float32)
    for lo in range(0, matrix.shape[0], _CHUNK_ROWS):
        block = matrix[lo : lo + _CHUNK_ROWS].astype(np.float32) @ q
        best[lo : lo + _CHUNK_ROWS] = block.max(axis=1)
    return np.maximum.reduceat(best, starts)


def tiled_stage1(snap: dict[Any, dict], queries: np.ndarray, score_key: str = "score") -> list[dict]:
    """Every media in *snap*, best Stage-1 score first; pages without tiles come last at 0."""
    ids, matrix, starts = _tile_matrix(snap)
    scores = _page_scores(matrix, starts, queries) if ids else np.zeros(0, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    out = [{"id": ids[i], score_key: round(float(scores[i]), 4)} for i in order]
    tiled = set(ids)
    out.extend({"id": mid, score_key: 0.0} for mid in snap if mid not in tiled)
    return out


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------


def vote_queries(
    good_votes: Any,
    feature_snap: dict[Any, dict],
    region_boxes: dict[Any, tuple[float, float, float, float]],
) -> Optional[np.ndarray]:
    """Stage-1 queries from the Good votes, projected like the tiles, or ``None``.

    ``None`` when the cached projection is missing (a WARNING names the fix) or
    no Good vote yields a query; the caller then keeps its own Stage 1.
    """
    from vtscore.media.structural_tiles import (  # noqa: PLC0415
        TileProjectionMissing,
        box_vlad,
        load_tile_projection,
        project_query,
        tile_vectors,
    )

    try:
        projection = load_tile_projection()
    except TileProjectionMissing as exc:
        _log.warning("tiled Stage 1 skipped: %s", exc)
        return None
    rows: list[np.ndarray] = []
    for cid in good_votes:
        media = feature_snap.get(cid)
        if media is None:
            continue
        feats = media.get("local_features")
        if not isinstance(feats, StructuralFeatures) or feats.count == 0:
            continue
        box = region_boxes.get(cid)
        if box is not None and (raw := box_vlad(feats, box)) is not None:
            rows.append(project_query(projection, raw))
            continue
        # No box: the page's own tiles, so whichever part of it the mark is on can match.
        stored = media.get("tile_vectors")
        tiles = stored if stored is not None else tile_vectors(feats, projection)
        rows.append(np.asarray(tiles.vectors, dtype=np.float32))
    return np.concatenate(rows) if rows else None


def example_queries(templates: Sequence[StructuralFeatures]) -> Optional[np.ndarray]:
    """Stage-1 queries from uploaded crops: each crop's VLAD, projected; ``None`` without a projection."""
    from vtscore.media.structural import aggregate_vlad, load_vlad_codebook  # noqa: PLC0415
    from vtscore.media.structural_tiles import (  # noqa: PLC0415
        TileProjectionMissing,
        load_tile_projection,
        project_query,
    )

    try:
        projection = load_tile_projection()
    except TileProjectionMissing as exc:
        _log.warning("tiled Stage 1 skipped: %s", exc)
        return None
    raws = [aggregate_vlad(t.descriptors_f32(), load_vlad_codebook()) for t in templates if t.count > 0]
    return project_query(projection, np.stack(raws)) if raws else None


# --------------------------------------------------------------------------
# Verification cache
# --------------------------------------------------------------------------


class VerificationCache:
    """(template, page) :class:`MatchStats` kept across retrains of one detector.

    In memory only, on the :class:`~vtscore.state.DetectorContext`, like the
    detector head. An entry records the identity of the page's features object
    it was computed on, so a dataset reload or switch misses rather than returning
    a fit for different features.
    """

    def __init__(self) -> None:
        self._fits: dict[tuple[Any, Any], tuple[int, MatchStats]] = {}

    def __len__(self) -> int:
        return len(self._fits)

    def best_many(
        self,
        templates: Sequence[tuple[Any, StructuralFeatures]],
        candidates: Sequence[tuple[Any, StructuralFeatures]],
        matcher: StructuralMatcher,
    ) -> list[MatchStats]:
        """Max-over-templates :class:`MatchStats` per candidate, verifying only what is not cached.

        *templates* and *candidates* are ``(key, features)`` pairs. A template
        key must identify the template's content (media, box, features object);
        a candidate key is its media id.
        """
        if len(self._fits) > _CACHE_LIMIT:
            self._fits.clear()
        verify_many = getattr(matcher, "verify_many", None)
        for tkey, tpl in templates:
            todo = [
                (ckey, feats)
                for ckey, feats in candidates
                if (hit := self._fits.get((tkey, ckey))) is None or hit[0] != id(feats)
            ]
            if not todo:
                continue
            feats_list = [f for _, f in todo]
            fits = verify_many(tpl, feats_list) if verify_many else [matcher.verify(tpl, f) for f in feats_list]
            for (ckey, feats), stats in zip(todo, fits):
                self._fits[(tkey, ckey)] = (id(feats), stats)
        out: list[MatchStats] = []
        for ckey, _feats in candidates:
            best, best_key = MatchStats(), (False, -1, -1.0)
            for tkey, _tpl in templates:
                stats = self._fits[(tkey, ckey)][1]
                key = (stats.model_ok, stats.inlier_count, stats.inlier_ratio)
                if key > best_key:
                    best, best_key = stats, key
            out.append(best)
        return out
