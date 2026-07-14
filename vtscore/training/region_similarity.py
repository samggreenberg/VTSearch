"""Region-aware cosine similarity scoring.

Single entry point for routes that need to score the loaded media set
against a query vector.  Handles the two cases transparently:

* **Legacy single-vector media** (SigLIP, CLIP, etc.) - exactly the
  same fast vectorised numpy path as before.  No ``best_region`` info
  is returned because there is none.

* **Patch-region media** (DINOv2, DINOv3, EUPE) - for each media,
  cosine-similarity against every ``RegionVector`` in
  ``media["patch_regions"]`` (full-image, leaves, HAC internals) and
  return the **max**.  Also returns the winning region's box so the
  gallery card can outline it.

The dispatch is per-loaded-snapshot: if at least one media in the
snapshot has ``patch_regions``, we take the region-aware path for the
whole snapshot; otherwise we fall back to the fast legacy path.  This
keeps zero overhead on existing SigLIP datasets.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-media scoring
# ---------------------------------------------------------------------------


def score_against_query(
    media: dict,
    query_vec: np.ndarray,
    embedder_name: Optional[str] = None,
) -> tuple[float, Optional[tuple[float, float, float, float]]]:
    """Return ``(max_cosine_similarity, best_region_box)`` for *media*.

    *query_vec* **must already be L2-normalized**, as are all stored media
    and region vectors (every embedding is normalized once at ingest - see
    :mod:`vtscore.embedding.normalize`).  Cosine similarity therefore reduces
    to a plain dot product, with no per-comparison normalization.  Callers
    obtain a unit query from :meth:`MediaEmbedder.embed_text`,
    :meth:`~MediaEmbedder.embed_media`, or a stored media embedding - all of
    which are already unit-norm.

    For patch-region media, we score every region vector and return the max
    along with that region's bounding box.  For single-vector media we score
    the full-image vector and return ``(score, (0, 0, 1, 1))``.  *embedder_name*
    selects which bound embedder's full-image vector to use (the region path is
    always against the patch embedder that owns ``patch_regions``); ``None``
    reads the media's primary vector.  A zero/empty query, or a missing
    embedding, yields ``(0.0, None)``.
    """
    if float(np.linalg.norm(query_vec)) == 0:
        return 0.0, None

    regions = media.get("patch_regions")
    if regions:
        best_score = -1.0
        best_box: Optional[tuple[float, float, float, float]] = None
        for r in regions:
            v = np.asarray(r.vec, dtype=np.float32)
            sim = float(v @ query_vec)
            if sim > best_score:
                best_score = sim
                best_box = r.box
        return (best_score if best_box is not None else 0.0), best_box

    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    emb = media_embedding(media, embedder_name)
    if emb is None:
        return 0.0, None
    emb_arr = np.asarray(emb, dtype=np.float32)
    sim = float(emb_arr @ query_vec)
    return sim, (0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Snapshot-level scoring (route-side dispatch)
# ---------------------------------------------------------------------------


def _snapshot_has_patch_regions(snap: dict[Any, dict]) -> bool:
    """True iff any media in *snap* carries a populated ``patch_regions``."""
    for m in snap.values():
        if m.get("patch_regions"):
            return True
    return False


def cosine_sort_with_boxes(
    snap: dict[Any, dict],
    query_vec: np.ndarray,
    embedder_name: Optional[str] = None,
    *,
    region_aware: Optional[bool] = None,
) -> tuple[list[dict], list[float]]:
    """Score every media in *snap* against *query_vec*, return per-media dicts.

    Each result dict has ``id``, ``similarity``, and (only when the
    snapshot is scored region-aware) ``best_region`` - a 4-tuple
    ``(x0, y0, x1, y1)`` in normalised image coordinates.

    *embedder_name* selects which bound embedder's vectors to score against
    (``None`` = the media's primary vector); for a single-embedder dataset
    the routing layer collapses this to the cached primary path.

    *region_aware* gates the per-region max-pool path.  Region vectors
    (``patch_regions``) belong to the dataset's **patch** embedder, so the
    region path is valid only when *query_vec* was embedded by that same
    embedder.  Callers pass ``region_aware=True`` when the resolved
    *embedder_name* is the dataset's patch embedder (cosine example sort,
    text sort on a patch-capable embedder), and ``region_aware=False`` for a
    text query against a separate text embedder on a dual-embedder dataset.
    ``None`` (the default) means "use regions if the snapshot has any" - the
    legacy single-embedder behaviour.

    Returns ``(results, raw_similarities)`` - *results* sorted descending
    by similarity, *raw_similarities* in the original snapshot order
    (used by GMM threshold computation).

    Performance: zero overhead for legacy single-vector snapshots - we
    detect them via :func:`_snapshot_has_patch_regions` and take the
    fast vectorised numpy path.  Patch-region snapshots flatten every
    ``(media, region)`` pair into one ``(R, D)`` matrix (cached on the
    dataset context) and score them with a single matvec + segmented
    max-pool, so the whole snapshot is one BLAS call rather than N·K
    interpreter round-trips (K ~ the per-image region count, typically 23).
    """
    if not snap:
        return [], []

    use_regions = _snapshot_has_patch_regions(snap) if region_aware is None else region_aware
    if use_regions and _snapshot_has_patch_regions(snap):
        # Vectorised mirror of the MLP scoring path (``_score_all_media``):
        # flatten every (media, region) pair into one (R, D) matrix, take a
        # single matvec against the query, then segment-max-pool back down to
        # one score + winning region per media.  Replaces the per-media,
        # per-region Python loop that made this O(N·K) in interpreter
        # round-trips on a user-facing sort.
        from vtscore.embedding.matrix import get_region_matrix_for_snap, segmented_max_pool  # noqa: PLC0415

        # A zero/degenerate query dots to 0 everywhere; preserve the old
        # ``score_against_query`` behaviour of scoring 0 with no best_region.
        if float(np.linalg.norm(query_vec)) == 0:
            zero_results: list[dict[str, Any]] = [{"id": cid, "similarity": 0.0} for cid in snap]
            return zero_results, [0.0] * len(snap)

        all_ids, region_matrix, media_index_per_row, region_index_per_row = get_region_matrix_for_snap(snap)
        if not all_ids:
            return [], []
        # float64 matches the MLP path's max-pool dtype and keeps round(.,4)
        # stable; region rows and the query are both unit-norm, so the matvec
        # is the cosine similarity of each region against the query.
        flat_sims = (region_matrix @ query_vec).astype(np.float64, copy=False)
        scores, best_region = segmented_max_pool(
            flat_sims, media_index_per_row, region_index_per_row, len(all_ids)
        )
        region_results: list[dict[str, Any]] = []
        for cid, sim, bri in zip(all_ids, scores, best_region, strict=True):
            entry: dict[str, Any] = {"id": cid, "similarity": round(sim, 4)}
            regions = snap[cid].get("patch_regions")
            if regions and 0 <= bri < len(regions):
                entry["best_region"] = list(regions[bri].box)
            else:
                # Region-less media in a region-aware snapshot: the winning
                # "region" is the full-image vector, so its box is the whole
                # image - matching the legacy per-media path's (0, 0, 1, 1).
                entry["best_region"] = [0.0, 0.0, 1.0, 1.0]
            region_results.append(entry)
        region_results.sort(key=lambda x: x["similarity"], reverse=True)
        return region_results, scores

    # Fast path: pure single-vector cosine via one matrix-vector product.
    # Both the stored embeddings and *query_vec* are unit-norm (normalized
    # once at ingest - see vtscore.embedding.normalize), so cosine is just
    # the dot product; no per-row normalization is needed.  A zero stored
    # embedding (left at norm 0 by l2_normalize) dots to 0.0, preserving the
    # old "zero-norm scores 0" behaviour.  The matrix is reused from the
    # active DatasetContext cache when available.
    from vtscore.embedding.matrix import get_embedding_matrix_for_snap

    all_ids, all_embs = get_embedding_matrix_for_snap(snap, embedder_name)
    similarities = np.dot(all_embs, query_vec)
    sims_list = similarities.tolist()
    results = [{"id": cid, "similarity": round(float(sim), 4)} for cid, sim in zip(all_ids, similarities, strict=True)]
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results, sims_list
