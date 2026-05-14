"""Region-aware cosine similarity scoring.

Single entry point for routes that need to score the loaded media set
against a query vector.  Handles the two cases transparently:

* **Legacy single-vector media** (SigLIP, CLIP, etc.) — exactly the
  same fast vectorised numpy path as before.  No ``best_region`` info
  is returned because there is none.

* **Patch-region media** (DINOv2, DINOv3, EUPE) — for each media,
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

    *query_vec* is the embedded query (text or example image), L2-norm
    irrelevant — we cosine-normalise here.

    For patch-region media, we score every region vector and return the
    max along with that region's bounding box.  For legacy single-vector
    media, we score ``media["embedding"]`` and return
    ``(score, (0, 0, 1, 1))``.  Missing / zero-norm embeddings yield
    ``(0.0, None)``.
    """
    q_norm = float(np.linalg.norm(query_vec))
    if q_norm == 0:
        return 0.0, None

    regions = media.get("patch_regions")
    if regions:
        best_score = -1.0
        best_box: Optional[tuple[float, float, float, float]] = None
        for r in regions:
            v = np.asarray(r.vec, dtype=np.float32)
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0:
                continue
            sim = float(v @ query_vec) / (q_norm * v_norm)
            if sim > best_score:
                best_score = sim
                best_box = r.box
        return (best_score if best_box is not None else 0.0), best_box

    emb = media.get("embedding")
    if emb is None:
        return 0.0, None
    emb_arr = np.asarray(emb, dtype=np.float32)
    e_norm = float(np.linalg.norm(emb_arr))
    if e_norm == 0:
        return 0.0, None
    sim = float(emb_arr @ query_vec) / (q_norm * e_norm)
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
    snapshot is patch-region-aware) ``best_region`` — a 4-tuple
    ``(x0, y0, x1, y1)`` in normalised image coordinates.

    Returns ``(results, raw_similarities)`` — *results* sorted descending
    by similarity, *raw_similarities* in the original snapshot order
    (used by GMM threshold computation).

    Performance: zero overhead for legacy single-vector snapshots — we
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

    # Fast path: pure single-vector cosine — same arithmetic as the
    # original _cosine_sort, retained verbatim so SigLIP / CLIP /
    # SigLIP2 datasets see zero overhead from this refactor.  The matrix
    # is reused from the active DatasetContext cache when available.
    from vtsearch.models.embedding_matrix import get_embedding_matrix_for_snap

    all_ids, all_embs = get_embedding_matrix_for_snap(snap)
    q_norm = np.linalg.norm(query_vec)
    emb_norms = np.linalg.norm(all_embs, axis=1)
    norm_products = emb_norms * q_norm
    safe_norms = np.where(norm_products == 0, 1.0, norm_products)
    similarities = np.dot(all_embs, query_vec) / safe_norms
    similarities = np.where(norm_products == 0, 0.0, similarities)
    sims_list = similarities.tolist()
    results = [
        {"id": cid, "similarity": round(float(sim), 4)}
        for cid, sim in zip(all_ids, similarities)
    ]
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results, sims_list
