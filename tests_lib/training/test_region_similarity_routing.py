"""Embedder routing through region-aware cosine similarity (Phase 2b.4).

``cosine_sort_with_boxes`` / ``score_against_query`` gained an
``embedder_name`` (which bound embedder's vectors to score against) and a
``region_aware`` gate (the per-patch max-pool is valid only when the query
was embedded by the embedder that owns ``patch_grid``).  These cover the
dual-embedder dataset where a text query must score against the text
embedder's full-image vectors even though the medias also carry a *patch*
embedder's patch grids.
"""

from __future__ import annotations


import numpy as np

from vtscore.training.region_similarity import cosine_sort_with_boxes, score_against_query


def _basis(i: int) -> np.ndarray:
    return np.eye(4, dtype=np.float32)[i]


def _dual_snap() -> dict[int, dict]:
    """Two medias with distinct text (siglip) vectors + a patch (dinov3) tree.

    media 1's siglip vector is e1, media 2's is e2; both share a dinov3
    full-image vector e0 and one region vector e3.
    """
    snap: dict[int, dict] = {}
    for cid in (1, 2):
        snap[cid] = {
            "id": cid,
            "embedder": "dinov3_patch",  # the recorded primary
            "embedding": _basis(0),
            "embeddings": {"siglip": _basis(cid), "dinov3_patch": _basis(0)},
            "patch_grid": _basis(3)[None, None, :].astype(np.float16),
        }
    return snap


class TestScoreAgainstQueryEmbedderName:
    def test_named_embedder_selects_that_full_image_vector(self):
        media = _dual_snap()[1]
        # Drop the patch grid to exercise the single-vector branch.
        media = {k: v for k, v in media.items() if k != "patch_grid"}
        # siglip vector is e1 -> matches a query of e1, not the dinov3 e0.
        sim, box = score_against_query(media, _basis(1), "siglip")
        assert sim == 1.0
        assert box == (0.0, 0.0, 1.0, 1.0)
        sim0, _ = score_against_query(media, _basis(0), "siglip")
        assert sim0 == 0.0

    def test_default_uses_primary_vector(self):
        media = {
            "embedder": "dinov3_patch",
            "embedding": _basis(0),
            "embeddings": {"siglip": _basis(1), "dinov3_patch": _basis(0)},
        }
        sim, _ = score_against_query(media, _basis(0))
        assert sim == 1.0


class TestCosineSortRouting:
    def test_text_query_scores_against_text_embedder_no_regions(self):
        snap = _dual_snap()
        # Text query e1 matches media 1's siglip vector; region path suppressed.
        results, _sims = cosine_sort_with_boxes(snap, _basis(1), "siglip", region_aware=False)
        assert results[0]["id"] == 1
        assert results[0]["similarity"] == 1.0
        assert "best_region" not in results[0]

    def test_patch_query_uses_region_path(self):
        snap = _dual_snap()
        results, _sims = cosine_sort_with_boxes(snap, _basis(3), "dinov3_patch", region_aware=True)
        assert results[0]["similarity"] == 1.0
        assert "best_region" in results[0]

    def test_region_aware_none_defaults_to_snapshot_detection(self):
        # Legacy single-embedder behaviour: regions present -> region path.
        snap = _dual_snap()
        results, _sims = cosine_sort_with_boxes(snap, _basis(3))
        assert "best_region" in results[0]
