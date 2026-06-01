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
    along with that region's bounding box.  For legacy single-vector media,
    we score ``media["embedding"]`` and return ``(score, (0, 0, 1, 1))``.  A
    zero/empty query, or a missing embedding, yields ``(0.0, None)``.
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

    emb = media.get("embedding")
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
) -> tuple[list[dict], list[float]]:
    """Score every media in *snap* against *query_vec*, return per-media dicts.

    Each result dict has ``id``, ``similarity``, and (only when the active
    snapshot is patch-region-aware) ``best_region`` - a 4-tuple
    ``(x0, y0, x1, y1)`` in normalised image coordinates.

    Returns ``(results, raw_similarities)`` - *results* sorted descending
    by similarity, *raw_similarities* in the original snapshot order
    (used by GMM threshold computation).

    Performance: zero overhead for legacy single-vector snapshots - we
    detect them via :func:`_snapshot_has_patch_regions` and take the
    fast vectorised numpy path.  Patch-region snapshots iterate per-
    media and are O(N · K) where K is the per-image region count
    (typically 23).
    """
    if not snap:
        return [], []

    if _snapshot_has_patch_regions(snap):
        results: list[dict] = []
        sims: list[float] = []
        for cid, m in snap.items():
            sim, box = score_against_query(m, query_vec)
            entry = {"id": cid, "similarity": round(sim, 4)}
            if box is not None:
                entry["best_region"] = list(box)
            results.append(entry)
            sims.append(sim)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results, sims

    # Fast path: pure single-vector cosine via one matrix-vector product.
    # Both the stored embeddings and *query_vec* are unit-norm (normalized
    # once at ingest - see vtscore.embedding.normalize), so cosine is just
    # the dot product; no per-row normalization is needed.  A zero stored
    # embedding (left at norm 0 by l2_normalize) dots to 0.0, preserving the
    # old "zero-norm scores 0" behaviour.  The matrix is reused from the
    # active DatasetContext cache when available.
    from vtscore.embedding.matrix import get_embedding_matrix_for_snap

    all_ids, all_embs = get_embedding_matrix_for_snap(snap)
    similarities = np.dot(all_embs, query_vec)
    sims_list = similarities.tolist()
    results = [{"id": cid, "similarity": round(float(sim), 4)} for cid, sim in zip(all_ids, similarities, strict=True)]
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results, sims_list
