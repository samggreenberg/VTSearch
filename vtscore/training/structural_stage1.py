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

from vtscore.media.structural import DEFAULT_MIN_INLIERS, MatchStats, StructuralFeatures, StructuralMatcher

_log = logging.getLogger(__name__)

#: Shortlist Stage 2 verifies when the tiled Stage 1 ran and a GPU does the matching.
#: 2,000 by the owner's choice (2026-10-01) from #4391's arms at 50,000 pages: +0.02 AP
#: with votes and +0.05 before any vote over 1,000, for 2.4 s a vote (p90 3.1 s).
#: 4,000 gained a little more for 4.5 s (p90 6.1 s, over the 5 s budget).
TILED_TOP_K = 2000
#: The same without CUDA, where a template costs ~6 s per 1,000 pages instead of
#: ~1 s (#3928 M4), so the shortlist is half as long.
TILED_TOP_K_CPU = 1000
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


#: How the shortlist is sized (#4391, pre-registered arms): ``"fixed"`` is the
#: shipped K; ``"adaptive"`` verifies further blocks while the last one keeps
#: passing the gate (:func:`should_extend`); ``"cap"`` always verifies
#: :data:`TILED_K_CAP`, the ceiling ``"adaptive"`` can reach.
K_POLICY = "fixed"
#: The most pages ``"adaptive"`` / ``"cap"`` verify.
TILED_K_CAP = 4000
#: Pages added per extension.
EXTEND_STEP = 1000
#: The tail of the verified shortlist that decides an extension ...
EXTEND_WINDOW = 250
#: ... and the fraction of it that must pass the gate.
EXTEND_RATE = 0.10
#: The shortlist the last re-rank verified (read by the FullMarks replay).
LAST_TOP_K = 0


def tiled_top_k(n_pages: int) -> int:
    """How many Stage-1 pages Stage 2 verifies first after a tiled Stage 1."""
    base = TILED_K_CAP if K_POLICY == "cap" else (TILED_TOP_K if _cuda() else TILED_TOP_K_CPU)
    return max(0, min(n_pages, base))


def should_extend(stage1_ids: Sequence[Any], verified: dict[Any, float], top_k: int) -> bool:
    """Whether ``"adaptive"`` verifies the next block: the shortlist's tail still passes the gate.

    *stage1_ids* is the Stage-1 order and *verified* each verified page's gate
    score. When at least :data:`EXTEND_RATE` of the last :data:`EXTEND_WINDOW`
    shortlisted pages pass (>= 0.5), the class probably continues past K.
    """
    if K_POLICY != "adaptive" or top_k >= min(len(stage1_ids), TILED_K_CAP):
        return False
    window = stage1_ids[max(0, top_k - EXTEND_WINDOW) : top_k]
    if not window:
        return False
    passed = sum(1 for mid in window if verified.get(mid, 0.0) >= 0.5)
    return passed >= EXTEND_RATE * len(window)


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
    _GPU_CACHE.clear()  # its device copy goes with it (the key is the matrix's id)
    entry = (ids, matrix, starts)
    _MATRIX_CACHE[key] = entry
    return entry


#: The GPU copy of the cached tile matrix: ``(matrix key, fp16 tiles, page index per tile)``.
_GPU_CACHE: dict[str, Any] = {}


def _gpu_page_scores(matrix: np.ndarray, starts: np.ndarray, queries: np.ndarray) -> Optional[np.ndarray]:
    """:func:`_page_scores` on the GPU, or ``None`` when CUDA is absent or short of memory.

    The fp16 tile matrix is copied to the device once per cached matrix and kept
    there. Each chunk is widened to float32 on the device before the matmul, so
    the scores equal the CPU path's. At 50,000 pages this is milliseconds against
    ~3.5 s on the CPU (#4391), which was most of a vote's retrain.
    """
    if not _cuda():
        return None
    import torch  # noqa: PLC0415

    try:
        key = (id(matrix), matrix.shape)
        if _GPU_CACHE.get("key") != key:
            _GPU_CACHE.clear()
            # A new matrix is a new page set: the cached candidate descriptors (#4469) belong to the old one.
            from vtscore.media.structural import release_device_descriptors  # noqa: PLC0415

            release_device_descriptors()
            # Hand the previous matrix (and anything else cached but unused) back to the device first:
            # memory PyTorch's allocator holds does not count as free, and on a 32 GB card that
            # alone failed this check from the second matrix on (#4170).
            torch.cuda.empty_cache()
            free, _total = torch.cuda.mem_get_info()
            if matrix.nbytes * 3 > free:  # the copy, plus room for a float32 chunk and the models
                _log.warning(
                    "tiled Stage 1: %.1f GiB tile matrix needs %.1f GiB of GPU memory, %.1f GiB free; "
                    "scoring on the CPU",
                    matrix.nbytes / 2**30,
                    matrix.nbytes * 3 / 2**30,
                    free / 2**30,
                )
                return None
            counts = np.diff(np.append(starts, matrix.shape[0]))
            _GPU_CACHE.update(
                key=key,
                tiles=torch.from_numpy(matrix).to("cuda"),
                page=torch.from_numpy(np.repeat(np.arange(len(starts)), counts)).to("cuda"),
                pages=len(starts),
            )
        tiles, page = _GPU_CACHE["tiles"], _GPU_CACHE["page"]
        q = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32)).to("cuda").T  # (dim, Q)
        best = torch.empty(tiles.shape[0], dtype=torch.float32, device="cuda")
        for lo in range(0, tiles.shape[0], _CHUNK_ROWS):
            best[lo : lo + _CHUNK_ROWS] = (tiles[lo : lo + _CHUNK_ROWS].float() @ q).amax(dim=1)
        out = torch.full((_GPU_CACHE["pages"],), float("-inf"), device="cuda")
        out.scatter_reduce_(0, page, best, reduce="amax", include_self=True)
        return out.cpu().numpy()
    except Exception:  # noqa: BLE001 - any device failure falls back to the CPU path
        _log.warning("tiled Stage 1: GPU scoring failed; scoring on the CPU", exc_info=True)
        _GPU_CACHE.clear()
        return None


def _page_scores(matrix: np.ndarray, starts: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Each page's best (query, tile) cosine, chunked so float32 never holds the whole matrix."""
    on_gpu = _gpu_page_scores(matrix, starts, queries)
    if on_gpu is not None:
        return on_gpu
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
    box_templates: Optional[dict[Any, StructuralFeatures]] = None,
) -> Optional[np.ndarray]:
    """Stage-1 queries from the Good votes, projected like the tiles, or ``None``.

    ``None`` when the cached projection is missing (a WARNING names the fix) or
    no Good vote yields a query; the caller then keeps its own Stage 1.

    *box_templates* maps a boxed Good to the template Stage 2 verifies with. When
    given, its query is that template's VLAD, so a stop-list that prunes the
    template prunes the query too (#4170 / #4180).
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
        tpl = (box_templates or {}).get(cid) if box is not None else None
        if tpl is not None and tpl.count > 0:
            from vtscore.media.structural import aggregate_vlad, load_vlad_codebook  # noqa: PLC0415

            rows.append(project_query(projection, aggregate_vlad(tpl.descriptors_f32(), load_vlad_codebook())))
            continue
        if box is not None and (raw := box_vlad(feats, box)) is not None:
            rows.append(project_query(projection, raw))
            continue
        if media.get("seeded_example"):
            # A crop seeded as a Good (#4170's --seed-crop whole): the whole crop is the mark, so it
            # queries as one VLAD, as the example sort does. Its tiles would be fragments of the mark.
            # The app's seeded examples carry no local_features today, so only the review sets this.
            from vtscore.media.structural import aggregate_vlad, load_vlad_codebook  # noqa: PLC0415

            rows.append(project_query(projection, aggregate_vlad(feats.descriptors_f32(), load_vlad_codebook())))
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

    def fit(self, template_key: Any, candidate_key: Any) -> Optional[MatchStats]:
        """The cached fit of one template against one page, or ``None`` if never verified."""
        hit = self._fits.get((template_key, candidate_key))
        return None if hit is None else hit[1]

    def best_many(
        self,
        templates: Sequence[tuple[Any, StructuralFeatures]],
        candidates: Sequence[tuple[Any, StructuralFeatures]],
        matcher: StructuralMatcher,
        *,
        parents: Optional[dict[Any, tuple[Any, StructuralFeatures]]] = None,
    ) -> list[MatchStats]:
        """Max-over-templates :class:`MatchStats` per candidate, verifying only what is not cached.

        *templates* and *candidates* are ``(key, features)`` pairs. A template
        key must identify the template's content (media, box, features object);
        a candidate key is its media id.

        *parents* maps a pruned template's key to its unpruned ``(key, features)``
        (#4432). Pruning only drops template descriptors, and matching runs from
        template to page, so a pruned template cannot fit a page its parent fails
        on: it is verified only where the parent clears the 8-inlier gate, and
        scores no fit elsewhere.
        """
        if len(self._fits) > _CACHE_LIMIT:
            self._fits.clear()
        parents = parents or {}
        for tkey, tpl in templates:
            todo = self._uncached(tkey, candidates)
            parent = parents.get(tkey)
            if todo and parent is not None:
                pkey, ptpl = parent
                self._verify(pkey, ptpl, self._uncached(pkey, todo), matcher)
                passing = []
                for ckey, feats in todo:
                    pstats = self._fits[(pkey, ckey)][1]
                    if pstats.model_ok and pstats.inlier_count >= DEFAULT_MIN_INLIERS:
                        passing.append((ckey, feats))
                    else:
                        self._fits[(tkey, ckey)] = (id(feats), MatchStats())
                todo = passing
            self._verify(tkey, tpl, todo, matcher)
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

    def _uncached(
        self, tkey: Any, candidates: Sequence[tuple[Any, StructuralFeatures]]
    ) -> list[tuple[Any, StructuralFeatures]]:
        """The *candidates* with no fit for *tkey* on their current features."""
        return [
            (ckey, feats)
            for ckey, feats in candidates
            if (hit := self._fits.get((tkey, ckey))) is None or hit[0] != id(feats)
        ]

    def _verify(
        self,
        tkey: Any,
        tpl: StructuralFeatures,
        todo: Sequence[tuple[Any, StructuralFeatures]],
        matcher: StructuralMatcher,
    ) -> None:
        """Verify *tpl* against each of *todo* and keep the fits."""
        if not todo:
            return
        verify_many = getattr(matcher, "verify_many", None)
        feats_list = [f for _, f in todo]
        fits = verify_many(tpl, feats_list) if verify_many else [matcher.verify(tpl, f) for f in feats_list]
        for (ckey, feats), stats in zip(todo, fits):
            self._fits[(tkey, ckey)] = (id(feats), stats)
