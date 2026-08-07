"""``train_and_threshold``'s population estimator must see inference geometry.

The safe threshold this function returns is later compared against the
per-media scores ``score_media_with_model`` produces - which, on a patch
dataset, are **region max-pooled** (~24 region rows collapsed to their max).
Fitting on the one image-level vector per media instead put the fitted
component means systematically below the distribution the cut is applied to
(the region max is >= the whole-image region's own score), biasing the cut low
and over-including on region-voting detectors.

These tests pin that the fold-anchored mixture is realized on exactly the
scores inference computes: the pooled distribution on a region dataset, and the
unchanged embedding-matrix distribution on a plain single-vector dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vtscore.training.thresholds as thresholds_mod
from vtscore.detectors.training import _score_all_media, train_and_threshold
from vtscore.embedding.matrix import get_embedding_matrix_for_snap

DIM = 8
N_MEDIA = 20
# Media ids well clear of the active context's own ids: the embedding- and
# region-matrix caches key on the sorted id list, so a snap that coincidentally
# reuses the context's ids would be served the context's cached matrix.
ID_BASE = 5000


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)


@pytest.fixture
def captured_gmm_scores(monkeypatch):
    """Record the final score distribution the cut is realized on."""
    seen: list[list[float]] = []
    real = thresholds_mod.fit_fold_anchored_cut

    def _spy(fold_haystacks, fold_orderings, final_scores, **kwargs):
        seen.append([float(s) for s in final_scores])
        return real(fold_haystacks, fold_orderings, final_scores, **kwargs)

    monkeypatch.setattr(thresholds_mod, "fit_fold_anchored_cut", _spy)
    return seen


def _labels(rng: np.random.Generator, good_proto: np.ndarray, bad_proto: np.ndarray):
    """Eight labels (>= the ramp floor) clustered around two prototypes."""
    X_list, y_list = [], []
    for _ in range(4):
        X_list.append(_unit(good_proto + 0.1 * rng.standard_normal(DIM)))
        y_list.append(1.0)
        X_list.append(_unit(bad_proto + 0.1 * rng.standard_normal(DIM)))
        y_list.append(0.0)
    return X_list, y_list


def _image_level_scores(model, snap, embedder_name):
    """Score one image-level vector per media - the pre-fix GMM input."""
    all_ids, embs = get_embedding_matrix_for_snap(snap, embedder_name)
    with torch.no_grad():
        logits = model(torch.from_numpy(embs).to(next(model.parameters()).device))
        return all_ids, torch.sigmoid(logits).squeeze(-1).cpu().numpy().astype(np.float64)


class TestRegionDatasetFitsPooledScores:
    def _snap(self, rng: np.random.Generator, good_proto: np.ndarray) -> dict[int, dict]:
        """Media whose score rows are ``[image-level row, a good-like patch, ...]``.

        Row 0 of the MaxPatch stack is the media's own image-level vector, so
        the pooled score is by construction >= the image-level score - exactly
        the one-sided bias the old fit suffered from.
        """
        snap: dict[int, dict] = {}
        for cid in range(ID_BASE + 1, ID_BASE + N_MEDIA + 1):
            image_vec = _unit(rng.standard_normal(DIM))
            hot_vec = _unit(good_proto + 0.4 * rng.standard_normal(DIM))
            snap[cid] = {
                "id": cid,
                "media_type": "image",
                "embedder": "dinov3_patch",
                "embeddings": {"dinov3_patch": image_vec},
                # Score rows are [image_vec, hot_vec, ...]: a 1x2 patch grid
                # whose first cell is the "hot" region and whose second repeats
                # the image vector, so the max-pool has something to find.
                "patch_grid": np.stack([hot_vec, image_vec])[None, :, :].astype(np.float16),
            }
        return snap

    def test_gmm_sees_region_max_pooled_scores(self, captured_gmm_scores):
        rng = np.random.default_rng(11)
        good_proto = _unit(rng.standard_normal(DIM))
        bad_proto = _unit(rng.standard_normal(DIM))
        snap = self._snap(rng, good_proto)
        X_list, y_list = _labels(rng, good_proto, bad_proto)

        model, _threshold = train_and_threshold(X_list, y_list, snap=snap, embedder_name="dinov3_patch")

        assert len(captured_gmm_scores) == 1
        fitted = np.asarray(captured_gmm_scores[0], dtype=np.float64)

        # What inference will actually compare against the threshold.
        pooled_ids, pooled, _best = _score_all_media(model, snap, "dinov3_patch")
        np.testing.assert_allclose(fitted, np.asarray(pooled, dtype=np.float64), rtol=0, atol=0)

        # ...and it is a genuinely different distribution from the image-level
        # one the GMM used to be fitted on: never lower, sometimes higher.
        # The tolerance is float16, not float64: the flattened score matrix is
        # stored in the patch grid's own dtype, so row 0 is the fp16-rounded
        # image vector while ``_image_level_scores`` reads the float32 one.
        image_ids, image_level = _image_level_scores(model, snap, "dinov3_patch")
        assert image_ids == pooled_ids
        assert np.all(fitted >= image_level - 1e-3)
        assert np.any(fitted > image_level + 1e-6), (
            "the region max-pool must lift at least one media's score above its "
            "image-level score, otherwise this fixture proves nothing"
        )


class TestPlainDatasetUnchanged:
    def test_gmm_still_sees_the_embedding_matrix_scores(self, captured_gmm_scores):
        """A single-vector dataset takes ``_score_all_media``'s matrix fallback,
        so the fitted distribution is byte-for-byte the pre-fix one."""
        rng = np.random.default_rng(13)
        good_proto = _unit(rng.standard_normal(DIM))
        bad_proto = _unit(rng.standard_normal(DIM))
        snap = {
            cid: {
                "id": cid,
                "media_type": "audio",
                "embedder": "test",
                "embeddings": {"test": _unit(rng.standard_normal(DIM))},
            }
            for cid in range(ID_BASE + 1, ID_BASE + N_MEDIA + 1)
        }
        X_list, y_list = _labels(rng, good_proto, bad_proto)

        model, _threshold = train_and_threshold(X_list, y_list, snap=snap)

        assert len(captured_gmm_scores) == 1
        fitted = np.asarray(captured_gmm_scores[0], dtype=np.float64)
        _ids, image_level = _image_level_scores(model, snap, None)
        np.testing.assert_allclose(fitted, image_level, rtol=0, atol=1e-9)
