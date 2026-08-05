"""``train_and_threshold``'s safe-threshold GMM must be fitted on inference geometry.

The blended threshold this function returns is later compared against the
per-media scores ``score_media_with_model`` produces - which, on a patch
dataset, are **region max-pooled** (~24 region rows collapsed to their max).
Fitting the GMM on the one image-level vector per media instead put the fitted
component means systematically below the distribution the cut is applied to
(the region max is >= the whole-image region's own score), biasing the midpoint
low and over-including on region-voting detectors.

These tests pin that the GMM sees exactly the scores inference computes: the
pooled distribution on a region dataset, and the unchanged embedding-matrix
distribution on a plain single-vector dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vtscore.state as core_state
import vtscore.training as training_pkg
from vtscore.detectors.training import _score_all_media, train_and_threshold
from vtscore.embedding.matrix import get_embedding_matrix_for_snap
from vtscore.media.patch_embed import RegionVector

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
    """Turn safe-thresholds on and record the scores the GMM blend is fitted on."""
    monkeypatch.setattr(core_state, "get_safe_thresholds", lambda: True)

    seen: list[list[float]] = []
    real = training_pkg.calculate_safe_threshold

    def _spy(xcal_threshold, all_scores, ctx, **kwargs):
        seen.append([float(s) for s in all_scores])
        return real(xcal_threshold, all_scores, ctx, **kwargs)

    monkeypatch.setattr(training_pkg, "calculate_safe_threshold", _spy)
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
        """Media whose region tree is ``[whole-image root, a good-like sub-region]``.

        The root region carries the media's own image-level vector, so the
        pooled score is by construction >= the image-level score - exactly the
        one-sided bias the old fit suffered from.
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
                "patch_regions": [
                    RegionVector(box=(0.0, 0.0, 1.0, 1.0), vec=image_vec),
                    RegionVector(box=(0.25, 0.25, 0.75, 0.75), vec=hot_vec),
                ],
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
        image_ids, image_level = _image_level_scores(model, snap, "dinov3_patch")
        assert image_ids == pooled_ids
        assert np.all(fitted >= image_level - 1e-9)
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
