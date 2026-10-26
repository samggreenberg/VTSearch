"""imagekit.face.face_align: the similarity fit, norm_crop and the Align transform."""

from __future__ import annotations

import math

import numpy as np
import pytest

from imagekit.face.face_align import ARCFACE_TEMPLATE, alignment_matrix, norm_crop, similarity_transform
from imagekit.util import Sample, apply_transforms, build_transforms


def _apply(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    return homogeneous @ matrix.T


class TestSimilarityTransform:
    def test_recovers_a_known_similarity(self):
        angle, scale, tx, ty = math.radians(30), 1.7, 5.0, -3.0
        a, b = scale * math.cos(angle), scale * math.sin(angle)
        truth = np.array([[a, -b, tx], [b, a, ty]])
        src = np.random.default_rng(42).uniform(0, 100, (5, 2))
        dst = _apply(truth, src)
        np.testing.assert_allclose(similarity_transform(src, dst), truth, atol=1e-4)

    def test_refuses_mismatched_sets(self):
        with pytest.raises(ValueError):
            similarity_transform(np.zeros((5, 2)), np.zeros((4, 2)))


class TestNormCrop:
    def test_landmarks_on_the_template_give_the_identity_warp(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        crop = norm_crop(img, ARCFACE_TEMPLATE)
        assert crop.shape == (112, 112, 3)
        assert np.abs(crop.astype(int) - img.astype(int)).max() <= 1

    def test_a_shifted_scaled_face_lands_on_the_template(self):
        # Landmarks at 2x the template, offset by (100, 50): the crop is that region, halved.
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
        landmarks = ARCFACE_TEMPLATE * 2 + np.array([100, 50], np.float32)
        crop = norm_crop(img, landmarks)
        matrix = alignment_matrix(landmarks)
        np.testing.assert_allclose(_apply(matrix, landmarks), ARCFACE_TEMPLATE, atol=1e-3)
        # The crop's centre pixel comes from the source point the inverse warp names.
        inv = np.vstack([matrix, [0, 0, 1]])
        src = np.linalg.inv(inv) @ np.array([56.0, 56.0, 1.0])
        sx, sy = int(round(src[0])), int(round(src[1]))
        region = img[sy - 2 : sy + 3, sx - 2 : sx + 3].astype(int)
        assert np.abs(region.mean(axis=(0, 1)) - crop[54:59, 54:59].astype(int).mean(axis=(0, 1))).max() < 40

    def test_missing_landmarks_fall_back_to_the_box(self):
        img = np.zeros((100, 100, 3), np.uint8)
        img[20:40, 30:50] = 255
        crop = norm_crop(img, None, bbox=(30, 20, 50, 40))
        assert crop.shape == (112, 112, 3) and crop.min() == 255
        assert norm_crop(img, np.full((5, 2), np.nan, np.float32), bbox=(0, 0, 10, 10)).max() == 0

    def test_no_landmarks_and_no_box_resizes_the_whole_image(self):
        img = np.full((30, 60, 3), 9, np.uint8)
        assert norm_crop(img, None, image_size=16).shape == (16, 16, 3)

    def test_fallback_error_raises(self):
        with pytest.raises(ValueError, match="landmarks"):
            norm_crop(np.zeros((10, 10, 3), np.uint8), None, bbox=(0, 0, 5, 5), fallback="error")
        with pytest.raises(ValueError, match="fallback"):
            norm_crop(np.zeros((10, 10, 3), np.uint8), None, fallback="bogus")

    def test_non_square_crop(self):
        img = np.zeros((200, 200, 3), np.uint8)
        assert norm_crop(img, ARCFACE_TEMPLATE, height=96, width=80).shape == (96, 80, 3)


class TestAlignTransform:
    def test_align_reads_landmarks_and_box_from_the_sample(self):
        img = np.zeros((100, 100, 3), np.uint8)
        img[10:30, 10:30] = 200
        chain = build_transforms([{"name": "Align", "args": {"height": 112, "width": 112, "fallback": "crop"}}])
        out = apply_transforms(chain, Sample(img, extra={"landmarks": None, "bbox": (10, 10, 30, 30)}))
        assert out.image.shape == (112, 112, 3) and out.image.min() == 200

    def test_bad_fallback_is_refused_at_build_time(self):
        with pytest.raises(ValueError):
            build_transforms([{"name": "Align", "args": {"fallback": "nope"}}])
