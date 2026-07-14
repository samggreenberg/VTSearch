"""Tests for the parametrised geometric verification models."""

import numpy as np
import pytest

from vtscore.media.structural_geometry import fit_model, fit_scale_translation


@pytest.fixture()
def st_correspondences():
    """60 correspondences under dst = 0.37*src + t, 30% corrupted to outliers."""
    rng = np.random.RandomState(0)
    src = rng.rand(60, 2).astype(np.float32)
    s_true, t_true = 0.37, np.array([0.21, -0.05])
    dst = (s_true * src + t_true).astype(np.float32)
    dst[:18] = rng.rand(18, 2)
    return src, dst, s_true


def test_fit_scale_translation_recovers_model(st_correspondences):
    src, dst, s_true = st_correspondences
    model, mask = fit_scale_translation(src, dst)
    assert model is not None
    assert mask is not None
    assert abs(model[0, 0] - s_true) < 1e-3
    assert model[0, 1] == 0.0 and model[1, 0] == 0.0  # no rotation terms
    assert mask.sum() >= 42  # the uncorrupted correspondences
    assert mask[:18].sum() < 3  # outliers rejected


def test_fit_model_scale_translation_stats(st_correspondences):
    src, dst, _ = st_correspondences
    stats = fit_model(src, dst, 60, "scale_translation")
    assert stats.model_ok
    assert stats.inlier_count >= 40
    assert stats.tentative_count == 60
    assert stats.inlier_box is not None
    assert not stats.reflection


def test_scale_translation_rejects_rotation():
    """The 3-DoF model must NOT explain a rotated pattern; 4-DoF must."""
    rng = np.random.RandomState(0)
    src = rng.rand(60, 2).astype(np.float32)
    th = np.deg2rad(35)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = (0.8 * src @ rot.T + np.array([0.2, 0.1])).astype(np.float32)
    st3 = fit_model(src, dst, 60, "scale_translation")
    st4 = fit_model(src, dst, 60, "similarity")
    assert st4.model_ok and st4.inlier_count >= 50
    assert st3.inlier_count < st4.inlier_count / 2


def test_negative_scale_flagged_as_reflection():
    """A 180-degree rotation looks like s < 0 to the no-rotation model."""
    rng = np.random.RandomState(2)
    src = rng.rand(40, 2).astype(np.float32)
    dst = (-0.5 * src + np.array([0.9, 0.9])).astype(np.float32)
    stats = fit_model(src, dst, 40, "scale_translation")
    assert not stats.model_ok
    assert stats.reflection


def test_similarity_matches_production_estimator():
    """The 'similarity' branch must agree with cv2 on clean data."""
    rng = np.random.RandomState(1)
    src = rng.rand(50, 2).astype(np.float32)
    th = np.deg2rad(10)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = (1.3 * src @ rot.T + np.array([0.05, -0.02])).astype(np.float32)
    stats = fit_model(src, dst, 50, "similarity")
    assert stats.model_ok
    assert stats.inlier_count >= 48
    assert abs(stats.scale - 1.3) < 0.02


def test_degenerate_inputs():
    empty = np.zeros((0, 2), np.float32)
    assert not fit_model(empty, empty, 0, "scale_translation").model_ok
    one = np.array([[0.5, 0.5]], np.float32)
    assert not fit_model(one, one, 1, "scale_translation").model_ok
    # coincident points cannot pin a scale
    coin = np.tile(np.array([[0.3, 0.3]], np.float32), (10, 1))
    stats = fit_model(coin, coin, 10, "scale_translation")
    assert not stats.model_ok


def test_unknown_model_raises():
    pts = np.zeros((5, 2), np.float32)
    with pytest.raises(ValueError, match="unknown geometric model"):
        fit_model(pts, pts, 5, "homography")
