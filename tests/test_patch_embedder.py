"""Tests for the patch-region pipeline.

Covers the pure-numpy parts end-to-end (no model weights / no GPU
required) and verifies the integration shapes that the loader,
similarity helper, and MLP scoring code rely on.

The DINOv3 / EUPE forward passes are not exercised here — those need
real weights and live in tests/test_gpu.py.  This file focuses on
the algorithmic + integration layers that are reachable without a
model load.
"""

from __future__ import annotations

import io
import pickle
from unittest.mock import MagicMock

import numpy as np
import torch

from vtsearch.models.patch_regions import (
    PatchEmbedOutput,
    RegionVector,
    build_hac_tree,
    build_region_tree,
    eupe_features_to_patch_output,
    hf_vit_to_patch_output,
    propose_leaves,
    to_fp16,
)
from vtsearch.models.region_similarity import (
    cosine_sort_with_boxes,
    score_against_query,
)


# ---------------------------------------------------------------------------
# Fixtures: hand-crafted PatchEmbedOutputs
# ---------------------------------------------------------------------------


def _normed(rng: np.random.Generator, shape) -> np.ndarray:
    """Random L2-normalised float32 array of *shape*."""
    v = rng.standard_normal(shape).astype(np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(min=1e-12)


def _make_output(h: int = 4, w: int = 4, d: int = 8, seed: int = 0) -> PatchEmbedOutput:
    rng = np.random.default_rng(seed)
    cls = _normed(rng, (d,))
    grid = _normed(rng, (h, w, d))
    sal = np.abs(rng.standard_normal((h, w))).astype(np.float32)
    sal = sal / sal.sum()
    return PatchEmbedOutput(cls_vec=cls, patch_grid=grid, patch_saliency=sal)


# ---------------------------------------------------------------------------
# RegionVector + pickle round-trip
# ---------------------------------------------------------------------------


class TestRegionVector:
    def test_construct_minimal(self):
        v = RegionVector(box=(0.0, 0.0, 1.0, 1.0), vec=np.zeros(8, dtype=np.float32))
        assert v.children is None
        assert v.box == (0.0, 0.0, 1.0, 1.0)

    def test_construct_with_children(self):
        v = RegionVector(box=(0.0, 0.0, 0.5, 0.5), vec=np.zeros(8, dtype=np.float32), children=(1, 2))
        assert v.children == (1, 2)

    def test_pickle_round_trip_fp16(self):
        """fp16-cast vectors round-trip through pickle.dump / load identically.

        This is what the loader stores in dataset pickles.
        """
        original = RegionVector(
            box=(0.1, 0.2, 0.7, 0.8),
            vec=np.array([0.1, -0.3, 0.5, 0.7], dtype=np.float16),
            children=(3, 4),
        )
        buf = io.BytesIO()
        pickle.dump(original, buf)
        buf.seek(0)
        restored: RegionVector = pickle.load(buf)
        assert restored.box == original.box
        assert restored.children == original.children
        assert restored.vec.dtype == np.float16
        np.testing.assert_array_equal(restored.vec, original.vec)


# ---------------------------------------------------------------------------
# propose_leaves
# ---------------------------------------------------------------------------


class TestProposeLeaves:
    def test_exact_k_leaves_returned(self):
        out = _make_output()
        for k in (1, 4, 16):
            leaves = propose_leaves(out.patch_grid, out.patch_saliency, k=k)
            assert len(leaves) == k

    def test_leaves_have_unit_norm_vectors(self):
        out = _make_output()
        leaves = propose_leaves(out.patch_grid, out.patch_saliency, k=4)
        for leaf in leaves:
            assert leaf.children is None
            np.testing.assert_allclose(np.linalg.norm(leaf.vec), 1.0, atol=1e-5)

    def test_leaves_have_normalised_boxes(self):
        out = _make_output(h=8, w=8, d=8)
        leaves = propose_leaves(out.patch_grid, out.patch_saliency, k=6)
        for leaf in leaves:
            x0, y0, x1, y1 = leaf.box
            assert 0.0 <= x0 < x1 <= 1.0
            assert 0.0 <= y0 < y1 <= 1.0

    def test_rejects_invalid_inputs(self):
        out = _make_output()
        # Non-3D patch grid
        with pytest_raises(ValueError):
            propose_leaves(np.zeros((4, 4)), out.patch_saliency, k=4)
        # Saliency shape mismatch
        with pytest_raises(ValueError):
            propose_leaves(out.patch_grid, np.zeros((3, 3)), k=4)
        # k out of range
        with pytest_raises(ValueError):
            propose_leaves(out.patch_grid, out.patch_saliency, k=0)
        with pytest_raises(ValueError):
            propose_leaves(out.patch_grid, out.patch_saliency, k=99)


def pytest_raises(exc):
    """Tiny helper so we don't have to import pytest at module top."""
    import pytest  # noqa: PLC0415

    return pytest.raises(exc)


# ---------------------------------------------------------------------------
# build_hac_tree
# ---------------------------------------------------------------------------


class TestBuildHacTree:
    def test_strict_binary_tree_has_2K_minus_1_nodes(self):
        out = _make_output()
        for k in (2, 4, 8, 12):
            leaves = propose_leaves(out.patch_grid, out.patch_saliency, k=k)
            tree = build_hac_tree(leaves, alpha=0.5)
            assert len(tree) == 2 * k - 1

    def test_internals_have_well_formed_children(self):
        out = _make_output()
        leaves = propose_leaves(out.patch_grid, out.patch_saliency, k=6)
        tree = build_hac_tree(leaves, alpha=0.5)
        for i, node in enumerate(tree):
            if node.children is not None:
                ci, cj = node.children
                assert 0 <= ci < i, f"child {ci} of node {i} not earlier"
                assert 0 <= cj < i, f"child {cj} of node {i} not earlier"
                assert ci != cj

    def test_internals_unit_norm(self):
        out = _make_output()
        leaves = propose_leaves(out.patch_grid, out.patch_saliency, k=4)
        tree = build_hac_tree(leaves, alpha=0.5)
        for node in tree:
            np.testing.assert_allclose(np.linalg.norm(node.vec), 1.0, atol=1e-5)

    def test_alpha_extremes_dont_crash(self):
        out = _make_output()
        leaves = propose_leaves(out.patch_grid, out.patch_saliency, k=4)
        for alpha in (0.0, 0.5, 1.0):
            tree = build_hac_tree(leaves, alpha=alpha)
            assert len(tree) == 7  # 2*4-1


# ---------------------------------------------------------------------------
# build_region_tree
# ---------------------------------------------------------------------------


class TestBuildRegionTree:
    def test_total_count_is_2K(self):
        out = _make_output()
        for k in (4, 8, 12):
            tree = build_region_tree(out, k=k, alpha=0.5)
            # 1 full-image + K leaves + K-1 internals = 2K
            assert len(tree) == 2 * k

    def test_full_image_is_index_0(self):
        out = _make_output()
        tree = build_region_tree(out, k=4, alpha=0.5)
        assert tree[0].box == (0.0, 0.0, 1.0, 1.0)
        assert tree[0].children is None
        # CLS vector L2-normalised
        np.testing.assert_allclose(np.linalg.norm(tree[0].vec), 1.0, atol=1e-5)

    def test_children_indices_offset_for_full_image(self):
        """HAC builder returns children indices into its own list; build_region_tree
        prepends the full-image node and shifts every internal's children by 1."""
        out = _make_output()
        tree = build_region_tree(out, k=4, alpha=0.5)
        for i, node in enumerate(tree):
            if node.children is not None:
                ci, cj = node.children
                # No internal references the full-image node as a child.
                assert ci > 0 and cj > 0
                # Children point earlier in the final list.
                assert ci < i and cj < i


# ---------------------------------------------------------------------------
# to_fp16
# ---------------------------------------------------------------------------


class TestToFp16:
    def test_casts_vec_dtype(self):
        out = _make_output()
        regions = build_region_tree(out, k=4, alpha=0.5)
        casted = to_fp16(regions)
        for r in casted:
            assert r.vec.dtype == np.float16

    def test_preserves_box_and_children(self):
        out = _make_output()
        regions = build_region_tree(out, k=4, alpha=0.5)
        casted = to_fp16(regions)
        for orig, cast in zip(regions, casted):
            assert cast.box == orig.box
            assert cast.children == orig.children


# ---------------------------------------------------------------------------
# Adapters: HF ViT and EUPE
# ---------------------------------------------------------------------------


class TestHfVitAdapter:
    def test_no_register_tokens_dinov2_layout(self):
        """1 CLS + 16 patches → 4×4 grid."""
        torch.manual_seed(0)
        T = 1 + 16  # CLS + 16 patches (4x4 grid)
        D = 32
        H = 6
        last_hidden = torch.randn(1, T, D)
        attentions = (torch.randn(1, H, T, T).softmax(dim=-1),)
        outputs = MagicMock(last_hidden_state=last_hidden, attentions=attentions)

        out = hf_vit_to_patch_output(outputs, num_register_tokens=0)
        assert out is not None
        assert out.cls_vec.shape == (D,)
        assert out.patch_grid.shape == (4, 4, D)
        assert out.patch_saliency.shape == (4, 4)
        np.testing.assert_allclose(out.patch_saliency.sum(), 1.0, atol=1e-5)

    def test_with_register_tokens_dinov3_layout(self):
        """1 CLS + 4 registers + 196 patches → 14×14 grid (DINOv3 ViT-B/16 @ 224²)."""
        torch.manual_seed(1)
        registers = 4
        patches = 196
        T = 1 + registers + patches
        D = 768
        H = 12
        last_hidden = torch.randn(1, T, D)
        attentions = (torch.randn(1, H, T, T).softmax(dim=-1),)
        outputs = MagicMock(last_hidden_state=last_hidden, attentions=attentions)

        out = hf_vit_to_patch_output(outputs, num_register_tokens=registers)
        assert out is not None
        assert out.cls_vec.shape == (D,)
        assert out.patch_grid.shape == (14, 14, D)
        assert out.patch_saliency.shape == (14, 14)
        np.testing.assert_allclose(out.patch_saliency.sum(), 1.0, atol=1e-5)

    def test_non_square_patch_count_returns_none(self):
        """A non-square patch count (would happen on a misconfigured image
        size) is rejected cleanly so the loader treats it as "no regions"."""
        T = 1 + 17  # 17 patches → not square
        D = 8
        last_hidden = torch.randn(1, T, D)
        attentions = (torch.randn(1, 4, T, T).softmax(dim=-1),)
        outputs = MagicMock(last_hidden_state=last_hidden, attentions=attentions)
        assert hf_vit_to_patch_output(outputs, num_register_tokens=0) is None


class TestEupeAdapter:
    def test_layout_eupe_vitb16(self):
        """EUPE forward_features dict: CLS / storage / patches."""
        torch.manual_seed(2)
        D = 768
        cls = torch.randn(1, D)
        storage = torch.randn(1, 4, D)
        patches = torch.randn(1, 196, D)
        features = {
            "x_norm_clstoken": cls,
            "x_storage_tokens": storage,
            "x_norm_patchtokens": patches,
            "x_prenorm": None,
            "masks": None,
        }
        out = eupe_features_to_patch_output(features)
        assert out is not None
        assert out.cls_vec.shape == (D,)
        assert out.patch_grid.shape == (14, 14, D)
        assert out.patch_saliency.shape == (14, 14)
        np.testing.assert_allclose(out.patch_saliency.sum(), 1.0, atol=1e-5)

    def test_non_square_returns_none(self):
        D = 8
        features = {
            "x_norm_clstoken": torch.randn(1, D),
            "x_storage_tokens": torch.randn(1, 0, D),
            "x_norm_patchtokens": torch.randn(1, 17, D),
        }
        assert eupe_features_to_patch_output(features) is None


# ---------------------------------------------------------------------------
# Capability flags on real embedder classes
# ---------------------------------------------------------------------------


class TestEmbedderCapabilities:
    def test_dinov2_supports_patch_regions(self):
        from vtsearch.media.image.embedder_dinov2 import ImageDinov2Embedder

        e = ImageDinov2Embedder()
        assert e.supports_patch_regions is True
        assert e.supports_text is False
        assert e.license_notice is None

    def test_dinov3_supports_patch_regions(self):
        from vtsearch.media.image.embedder_dinov3 import ImageDinov3Embedder

        e = ImageDinov3Embedder()
        assert e.supports_patch_regions is True
        assert e.supports_text is False
        assert e.license_notice is None

    def test_eupe_supports_patch_regions_and_carries_license_notice(self):
        from vtsearch.media.image.embedder_eupe import ImageEupeEmbedder

        e = ImageEupeEmbedder()
        assert e.supports_patch_regions is True
        assert e.supports_text is False
        assert isinstance(e.license_notice, str)
        assert "noncommercial" in e.license_notice.lower()

    def test_siglip_does_not_support_patch_regions(self):
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        e = ImageSiglipEmbedder()
        assert e.supports_patch_regions is False
        assert e.license_notice is None

    def test_default_patch_forward_returns_none(self):
        """Single-vector embedders inherit the ABC default and return None."""
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        e = ImageSiglipEmbedder()
        assert e.patch_forward({"media_path": "/nonexistent.jpg"}) is None


# ---------------------------------------------------------------------------
# Region-aware similarity scoring
# ---------------------------------------------------------------------------


class TestScoreAgainstQuery:
    def test_legacy_single_vector_returns_full_image_box(self):
        media = {"embedding": np.array([1.0, 0.0], dtype=np.float32)}
        q = np.array([1.0, 0.0], dtype=np.float32)
        score, box = score_against_query(media, q)
        assert box == (0.0, 0.0, 1.0, 1.0)
        np.testing.assert_allclose(score, 1.0, atol=1e-5)

    def test_zero_query_norm_returns_zero(self):
        media = {"embedding": np.array([1.0, 0.0], dtype=np.float32)}
        score, box = score_against_query(media, np.zeros(2, dtype=np.float32))
        assert score == 0.0
        assert box is None

    def test_patch_regions_return_winning_box(self):
        # Two regions: one matches the query exactly, the other is orthogonal.
        rv_match = RegionVector(
            box=(0.1, 0.2, 0.4, 0.5),
            vec=np.array([1.0, 0.0], dtype=np.float32),
        )
        rv_other = RegionVector(
            box=(0.6, 0.7, 0.9, 0.95),
            vec=np.array([0.0, 1.0], dtype=np.float32),
        )
        media = {
            "patch_regions": [rv_match, rv_other],
            "embedding": np.array([0.5, 0.5], dtype=np.float32),
        }
        q = np.array([1.0, 0.0], dtype=np.float32)
        score, box = score_against_query(media, q)
        np.testing.assert_allclose(score, 1.0, atol=1e-5)
        assert box == (0.1, 0.2, 0.4, 0.5)


class TestCosineSortWithBoxes:
    def test_legacy_path_no_best_region_field(self):
        """Snapshot of single-vector media → no best_region field on results
        (preserves SigLIP API shape exactly)."""
        snap = {
            1: {"embedding": np.array([1.0, 0.0], dtype=np.float32)},
            2: {"embedding": np.array([0.0, 1.0], dtype=np.float32)},
        }
        q = np.array([1.0, 0.0], dtype=np.float32)
        results, sims = cosine_sort_with_boxes(snap, q)
        assert len(results) == 2
        for r in results:
            assert "best_region" not in r
        # Highest similarity sorted first.
        assert results[0]["id"] == 1

    def test_patch_path_emits_best_region_field(self):
        rv = RegionVector(
            box=(0.0, 0.0, 0.5, 0.5),
            vec=np.array([1.0, 0.0], dtype=np.float32),
        )
        snap = {
            1: {
                "patch_regions": [rv],
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
            },
            2: {
                "embedding": np.array([0.0, 1.0], dtype=np.float32),
            },
        }
        q = np.array([1.0, 0.0], dtype=np.float32)
        results, _ = cosine_sort_with_boxes(snap, q)
        # Mixed snapshot with at least one patch-region media — region path
        # taken for the whole snapshot, every result gets best_region.
        for r in results:
            assert "best_region" in r


# ---------------------------------------------------------------------------
# v1 vote semantics — sanity check
# ---------------------------------------------------------------------------


class TestVoteSemanticsV1:
    """V1 records both Good and Bad votes against the whole image — there is
    no region-vote API yet.  This pins that down by verifying the vote
    endpoint persists no region_box on the LabeledElement.
    """

    def test_labeled_element_has_no_region_box_field_in_v1(self):
        """``LabeledElement`` should not (yet) carry a ``region_box`` field.

        When phase-2 region voting lands this assertion can be inverted to
        require the field with a default of None.  Today its absence is the
        contract.
        """
        from dataclasses import fields

        from vtsearch.datasets.labelset import LabeledElement

        names = {f.name for f in fields(LabeledElement)}
        assert "region_box" not in names, (
            "LabeledElement gained a region_box field — phase 2 has begun "
            "and this v1-only test should be replaced by a phase-2 contract test."
        )
