"""Tests for the patch pipeline (MaxPatch geometry).

Covers the pure-numpy parts end-to-end (no model weights / no GPU
required) and verifies the integration shapes that the loader,
similarity helper, and MLP scoring code rely on.

Ingest stores only the raw ``(H, W, D)`` ``patch_grid``; there is no region
tree (#2886).  The score-row stack every path shares - image-level vector at
row 0, then the raw patches row-major - is
:func:`vtscore.embedding.matrix.media_score_rows`.

The DINOv3 / EUPE forward passes are not exercised here; those need
real weights and live in tests/test_gpu.py.  This file focuses on
the algorithmic + integration layers that are reachable without a
model load.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from vtscore.embedding.media_vectors import media_embedding
from vtscore.media.patch_embed import (
    PatchEmbedOutput,
    eupe_features_to_patch_output,
    hf_vit_to_patch_output,
    nearest_patch_to_box,
)
from vtscore.training.region_similarity import (
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
# fp16 storage / fp32 compute rank stability
# ---------------------------------------------------------------------------


def _make_region_batch(
    num_media: int,
    grid_side: int,
    dim: int,
    seed: int,
) -> dict[int, dict]:
    """Build a batch of media dicts with an fp32-stored ``patch_grid``.

    Each patch vector is a fresh L2-normalised draw with no shared structure
    between media or cells, which is the worst case for cosine ranking
    stability (no built-in margin between competing rows).
    """
    rng = np.random.default_rng(seed)
    batch: dict[int, dict] = {}
    for i in range(num_media):
        grid = rng.standard_normal((grid_side, grid_side, dim)).astype(np.float32)
        grid /= np.linalg.norm(grid, axis=-1, keepdims=True).clip(min=1e-12)
        batch[i] = {
            "patch_grid": grid,
            "embedder": "siglip",
            "embeddings": {"siglip": grid[0, 0].copy()},
        }
    return batch


def _to_fp16_batch(snap_fp32: dict[int, dict]) -> dict[int, dict]:
    """Mirror of *snap_fp32* with the patch grid cast to fp16 (storage mode)."""
    return {
        cid: {
            "patch_grid": m["patch_grid"].astype(np.float16),
            "embedder": "siglip",
            "embeddings": {"siglip": media_embedding(m)},
        }
        for cid, m in snap_fp32.items()
    }


class TestFp16Fp32RankStability:
    """fp16 storage must not flip max-over-row cosine ranks vs. fp32.

    Patch vectors are stored fp16 to keep the dataset (and the flattened
    scoring matrix) manageable, but are cast back to fp32 at compute time.
    This test pins the claim that the fp16 round-trip is cheap *and* faithful:
    the cosine ranking under fp16 storage matches fp32 storage outside of a
    tiny quantization-noise tie band.

    MaxPatch made this pin *more* load-bearing, not less: the score stack went
    from ~24 pooled region nodes to 1 + H*W raw patches, so there are ~8x as
    many quantised rows competing for each image's max.
    """

    # Production embedders are all 768-dim ViT-Bs.  A 5x5 grid keeps the test
    # fast while still putting 26 quantised rows per image into the max.
    _DIM = 768
    _GRID_SIDE = 5
    _NUM_MEDIA = 50
    _NUM_QUERIES = 20

    def test_max_score_difference_below_quantization_noise(self):
        """Per-media row-max score differs by less than 1e-2 under fp16 storage.

        Empirically fp16 quantization of 768-dim unit vectors costs ~1e-3 in
        cosine sim; 1e-2 is a generous ceiling that catches any catastrophic
        regression (e.g. dropping a normalisation, accidentally storing
        non-normalised fp16, etc.).
        """
        snap_fp32 = _make_region_batch(self._NUM_MEDIA, self._GRID_SIDE, self._DIM, seed=0)
        snap_fp16 = _to_fp16_batch(snap_fp32)
        rng = np.random.default_rng(42)

        max_diff = 0.0
        for _ in range(self._NUM_QUERIES):
            q = rng.standard_normal(self._DIM).astype(np.float32)
            q = q / np.linalg.norm(q)
            for cid in snap_fp32:
                s32, _ = score_against_query(snap_fp32[cid], q)
                s16, _ = score_against_query(snap_fp16[cid], q)
                max_diff = max(max_diff, abs(s32 - s16))

        assert max_diff < 1e-2, (
            f"fp16 vs fp32 row-max score diff = {max_diff:.4g}; "
            "expected < 1e-2; investigate whether fp16 storage is dropping "
            "normalisation or vectors are no longer near-unit-norm"
        )

    def test_no_rank_flips_outside_noise_band(self):
        """Pairs of media whose fp32 scores differ by more than the fp16 noise
        floor keep the same relative order under fp16 storage.

        Pairs *inside* the noise band (|Δscore| ≤ 5e-3) are allowed to swap;
        that's an unavoidable consequence of fp16 quantization and matches
        what "rank doesn't flip in any meaningful sense" means in practice.
        """
        snap_fp32 = _make_region_batch(self._NUM_MEDIA, self._GRID_SIDE, self._DIM, seed=1)
        snap_fp16 = _to_fp16_batch(snap_fp32)
        rng = np.random.default_rng(7)
        noise_band = 5e-3

        flips: list[tuple[int, int, float, float]] = []
        for _ in range(self._NUM_QUERIES):
            q = rng.standard_normal(self._DIM).astype(np.float32)
            q = q / np.linalg.norm(q)
            scores32 = {cid: score_against_query(m, q)[0] for cid, m in snap_fp32.items()}
            scores16 = {cid: score_against_query(m, q)[0] for cid, m in snap_fp16.items()}
            ids = list(scores32.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    diff32 = scores32[a] - scores32[b]
                    if abs(diff32) <= noise_band:
                        continue
                    diff16 = scores16[a] - scores16[b]
                    if (diff32 > 0) != (diff16 > 0):
                        flips.append((a, b, diff32, diff16))

        assert not flips, (
            f"{len(flips)} non-tie rank flip(s) between fp16 and fp32 storage; "
            f"first: media {flips[0][0]} vs {flips[0][1]}, "
            f"fp32 Δ={flips[0][2]:.4g}, fp16 Δ={flips[0][3]:.4g}"
        )

    def test_top_result_preserved(self):
        """The #1 result under fp32 storage is also the #1 result under fp16,
        across every random query; this is the assertion users actually feel."""
        snap_fp32 = _make_region_batch(self._NUM_MEDIA, self._GRID_SIDE, self._DIM, seed=2)
        snap_fp16 = _to_fp16_batch(snap_fp32)
        rng = np.random.default_rng(99)

        for _ in range(self._NUM_QUERIES):
            q = rng.standard_normal(self._DIM).astype(np.float32)
            q = q / np.linalg.norm(q)
            top32 = cosine_sort_with_boxes(snap_fp32, q)[0][0]["id"]
            top16 = cosine_sort_with_boxes(snap_fp16, q)[0][0]["id"]
            assert top32 == top16, f"Top result flipped between storage modes: fp32→{top32}, fp16→{top16}"


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
    def test_dinov2_patch_supports_patch_regions(self):
        from vtscore.media.image.embedder_dinov2_patch import ImageDinov2PatchEmbedder

        e = ImageDinov2PatchEmbedder()
        assert e.supports_patch_regions is True
        assert e.supports_text is False
        assert e.license_notice is None

    def test_dinov2_single_does_not_support_patch_regions(self):
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        e = ImageDinov2SingleEmbedder()
        assert e.supports_patch_regions is False

    def test_dinov3_patch_supports_patch_regions(self):
        from vtscore.media.image.embedder_dinov3_patch import ImageDinov3PatchEmbedder

        e = ImageDinov3PatchEmbedder()
        assert e.supports_patch_regions is True
        assert e.supports_text is False
        assert e.license_notice is None

    def test_dinov3_single_does_not_support_patch_regions(self):
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        e = ImageDinov3SingleEmbedder()
        assert e.supports_patch_regions is False

    def test_eupe_patch_supports_patch_regions_and_carries_license_notice(self):
        from vtscore.media.image.embedder_eupe_patch import ImageEupePatchEmbedder

        e = ImageEupePatchEmbedder()
        assert e.supports_patch_regions is True
        assert e.supports_text is False
        assert isinstance(e.license_notice, str)
        assert "noncommercial" in e.license_notice.lower()

    def test_eupe_single_carries_license_notice(self):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        e = ImageEupeSingleEmbedder()
        assert e.supports_patch_regions is False
        assert isinstance(e.license_notice, str)
        assert "noncommercial" in e.license_notice.lower()

    def test_siglip_does_not_support_patch_regions(self):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        e = ImageSiglipEmbedder()
        assert e.supports_patch_regions is False
        assert e.license_notice is None

    def test_default_patch_forward_returns_none(self):
        """Single-vector embedders inherit the ABC default and return None."""
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        e = ImageSiglipEmbedder()
        assert e.patch_forward({"media_path": "/nonexistent.jpg"}) is None


class TestDinoAttentionBackend:
    """Patch-capable DINO loaders must request the eager attention backend.

    The patch variants read CLS→patch attention out of the forward pass
    (``output_attentions=True``).  The SDPA backend silently drops those
    tensors, leaving an empty ``attentions`` tuple that ``attentions[-1]``
    then indexes into and raises ``IndexError``.  The single-vector variants
    never read attentions, so they must keep the faster default (no
    ``attn_implementation`` override).
    """

    def _captured_load_kwargs(self, embedder, shared_module):
        """Drive ``_load_models_impl`` with the network/weight load stubbed and
        return the kwargs passed to the model ``from_pretrained`` call."""
        from contextlib import contextmanager
        from unittest.mock import patch

        @contextmanager
        def _noop_cm(*args, **kwargs):
            yield

        calls = []

        def fake_load(load_fn, *args, **kwargs):
            kwargs.pop("on_progress", None)
            calls.append(kwargs)
            # First call loads the model, second loads the processor.  Return a
            # lightweight stand-in that supports the chained .to(...).eval().
            stub = MagicMock()
            stub.to.return_value = stub
            return stub

        with (
            patch.object(shared_module, "load_pretrained_local_first", side_effect=fake_load),
            patch.object(shared_module, "embedder_load_setup", return_value=None),
            patch.object(shared_module, "timed_progress", _noop_cm),
            patch.object(shared_module, "intercept_tqdm_progress", _noop_cm),
            patch.object(shared_module, "intercept_weight_loading_progress", _noop_cm),
            patch.object(shared_module, "hf_token", return_value=None),
        ):
            embedder._load_models_impl()
        # The model load is the first call; the processor load is the second.
        return calls[0]

    def test_dinov3_patch_requests_eager_attention(self):
        from vtscore.media.image import _dinov3_shared
        from vtscore.media.image.embedder_dinov3_patch import ImageDinov3PatchEmbedder

        kwargs = self._captured_load_kwargs(ImageDinov3PatchEmbedder(), _dinov3_shared)
        assert kwargs.get("attn_implementation") == "eager"

    def test_dinov3_single_does_not_force_attention_backend(self):
        from vtscore.media.image import _dinov3_shared
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        kwargs = self._captured_load_kwargs(ImageDinov3SingleEmbedder(), _dinov3_shared)
        assert "attn_implementation" not in kwargs

    def test_dinov2_patch_requests_eager_attention(self):
        from vtscore.media.image import _dinov2_shared
        from vtscore.media.image.embedder_dinov2_patch import ImageDinov2PatchEmbedder

        kwargs = self._captured_load_kwargs(ImageDinov2PatchEmbedder(), _dinov2_shared)
        assert kwargs.get("attn_implementation") == "eager"

    def test_dinov2_single_does_not_force_attention_backend(self):
        from vtscore.media.image import _dinov2_shared
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        kwargs = self._captured_load_kwargs(ImageDinov2SingleEmbedder(), _dinov2_shared)
        assert "attn_implementation" not in kwargs


# ---------------------------------------------------------------------------
# Region-aware similarity scoring
# ---------------------------------------------------------------------------


class TestScoreAgainstQuery:
    def test_legacy_single_vector_returns_full_image_box(self):
        media = {"embedder": "siglip", "embeddings": {"siglip": np.array([1.0, 0.0], dtype=np.float32)}}
        q = np.array([1.0, 0.0], dtype=np.float32)
        score, box = score_against_query(media, q)
        assert box == (0.0, 0.0, 1.0, 1.0)
        np.testing.assert_allclose(score, 1.0, atol=1e-5)

    def test_zero_query_norm_returns_zero(self):
        media = {"embedder": "siglip", "embeddings": {"siglip": np.array([1.0, 0.0], dtype=np.float32)}}
        score, box = score_against_query(media, np.zeros(2, dtype=np.float32))
        assert score == 0.0
        assert box is None

    def test_patch_grid_returns_winning_cell_box(self):
        # A 1x2 grid: the right-hand cell matches the query exactly, the left
        # one is orthogonal - and so is the image-level (row 0) vector.
        media = {
            "patch_grid": np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32),
            "embedder": "siglip",
            "embeddings": {"siglip": np.array([0.0, 1.0], dtype=np.float32)},
        }
        q = np.array([1.0, 0.0], dtype=np.float32)
        score, box = score_against_query(media, q)
        np.testing.assert_allclose(score, 1.0, atol=1e-5)
        # Row 2 = grid cell (0, 1) - the right half of a 1x2 grid.
        assert box == (0.5, 0.0, 1.0, 1.0)

    def test_whole_image_row_can_win(self):
        """Row 0 is the image-level vector, so a query matching it wins the
        whole-image box - the geometry that keeps a boxless Good vote honest."""
        media = {
            "patch_grid": np.array([[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32),
            "embedder": "siglip",
            "embeddings": {"siglip": np.array([1.0, 0.0], dtype=np.float32)},
        }
        score, box = score_against_query(media, np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(score, 1.0, atol=1e-5)
        assert box == (0.0, 0.0, 1.0, 1.0)


class TestCosineSortWithBoxes:
    def test_legacy_path_no_best_region_field(self):
        """Snapshot of single-vector media → no best_region field on results
        (preserves SigLIP API shape exactly)."""
        snap = {
            1: {"embedder": "siglip", "embeddings": {"siglip": np.array([1.0, 0.0], dtype=np.float32)}},
            2: {"embedder": "siglip", "embeddings": {"siglip": np.array([0.0, 1.0], dtype=np.float32)}},
        }
        q = np.array([1.0, 0.0], dtype=np.float32)
        results, sims = cosine_sort_with_boxes(snap, q)
        assert len(results) == 2
        for r in results:
            assert "best_region" not in r
        # Highest similarity sorted first.
        assert results[0]["id"] == 1

    def test_patch_path_emits_best_region_field(self):
        snap = {
            1: {
                "patch_grid": np.array([[[1.0, 0.0]]], dtype=np.float32),
                "embedder": "siglip",
                "embeddings": {"siglip": np.array([1.0, 0.0], dtype=np.float32)},
            },
            2: {
                "embedder": "siglip",
                "embeddings": {"siglip": np.array([0.0, 1.0], dtype=np.float32)},
            },
        }
        q = np.array([1.0, 0.0], dtype=np.float32)
        results, _ = cosine_sort_with_boxes(snap, q)
        # Mixed snapshot with at least one patch-grid media; the patch path is
        # taken for the whole snapshot, so every result gets best_region.
        for r in results:
            assert "best_region" in r


# ---------------------------------------------------------------------------
# v2: LabeledElement.region_box data-model contract
# ---------------------------------------------------------------------------


class TestRegionBoxOnLabeledElement:
    """v2 region voting attaches a normalised ``(x0, y0, x1, y1)`` box to a
    yes-vote when the user drew a region.  This class pins the data-model
    contract: the field exists, defaults to ``None`` (image-level), and
    round-trips through dict serialisation including JSON's tuple→list
    coercion.

    Replaces the v1 ``test_labeled_element_has_no_region_box_field_in_v1``
    absence check.  Vote-endpoint wiring (yes-vote accepts region_box,
    no-vote rejects it) and on-the-fly vote-vector pooling are separate
    v2 work items.
    """

    def test_region_box_defaults_to_none(self):
        from vtscore.datasets.labelset import LabeledElement

        el = LabeledElement(md5="abc", label="good")
        assert el.region_box is None

    def test_region_box_omitted_from_dict_when_none(self):
        """Image-level votes don't emit ``region_box`` so the exported JSON
        stays a strict superset of the v1 format for legacy consumers."""
        from vtscore.datasets.labelset import LabeledElement

        el = LabeledElement(md5="abc", label="good")
        assert "region_box" not in el.to_dict()

    def test_region_box_round_trips_through_dict(self):
        from vtscore.datasets.labelset import LabeledElement

        original = LabeledElement(
            md5="abc",
            label="good",
            region_box=(0.1, 0.2, 0.7, 0.8),
        )
        restored = LabeledElement.from_dict(original.to_dict())
        assert restored.region_box == (0.1, 0.2, 0.7, 0.8)

    def test_region_box_accepts_list_from_json(self):
        """JSON encoders turn tuples into lists; ``from_dict`` must accept
        a list and coerce back to a 4-tuple of floats so the dataclass
        invariant holds regardless of the dict source."""
        from vtscore.datasets.labelset import LabeledElement

        d = {"md5": "abc", "label": "good", "region_box": [0.0, 0.25, 0.5, 1.0]}
        el = LabeledElement.from_dict(d)
        assert isinstance(el.region_box, tuple)
        assert el.region_box == (0.0, 0.25, 0.5, 1.0)
        assert all(isinstance(v, float) for v in el.region_box)

    def test_region_box_survives_labelset_round_trip(self):
        from vtscore.datasets.labelset import LabeledElement, LabelSet

        ls = LabelSet(
            [
                LabeledElement(md5="a", label="good", region_box=(0.1, 0.2, 0.3, 0.4)),
                LabeledElement(md5="b", label="good"),
                LabeledElement(md5="c", label="bad"),
            ]
        )
        restored = LabelSet.from_dict(ls.to_dict())
        assert restored.elements[0].region_box == (0.1, 0.2, 0.3, 0.4)
        assert restored.elements[1].region_box is None
        assert restored.elements[2].region_box is None

    def test_region_box_survives_merge(self):
        """A region_box on the first occurrence of a key is preserved through
        ``LabelSet.merge``; the merge already keeps the first entry's
        position, so its region annotation should ride along with it."""
        from vtscore.datasets.labelset import LabeledElement, LabelSet

        a = LabelSet(
            [
                LabeledElement(md5="x", label="good", region_box=(0.1, 0.2, 0.3, 0.4)),
            ]
        )
        b = LabelSet(
            [
                LabeledElement(md5="y", label="good"),
            ]
        )
        merged = a.merge(b)
        by_md5 = {e.md5: e for e in merged.elements}
        assert by_md5["x"].region_box == (0.1, 0.2, 0.3, 0.4)
        assert by_md5["y"].region_box is None


# ---------------------------------------------------------------------------
# v2: vote endpoint, label export, and region-aware training wiring
# ---------------------------------------------------------------------------


class TestVoteEndpointRegionBox:
    """``POST /api/medias/<id>/vote`` accepts an optional ``region_box`` on
    yes-votes and rejects it on no-votes.  Patch-embedder v2 step 3.
    """

    def test_good_vote_with_region_box_persists_in_state(self, client):
        from vtsearch.state import (
            good_votes,
            vote_region_boxes,
        )

        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.7, 0.8]},
        )
        assert resp.status_code == 200
        assert 1 in good_votes
        assert vote_region_boxes[1] == (0.1, 0.2, 0.7, 0.8)

    def test_votes_response_exposes_good_region_boxes(self, client):
        # A region good-vote's box surfaces in GET /api/votes so the Good pile
        # can request a cropped thumbnail of just the voted region.
        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.7, 0.8]},
        )
        votes = client.get("/api/votes")
        assert votes.status_code == 200
        boxes = votes.get_json()["good_region_boxes"]
        assert boxes["1"] == [0.1, 0.2, 0.7, 0.8]

    def test_votes_response_drops_region_box_on_revote_bad(self, client):
        # Re-voting the item bad clears its region box from the votes payload.
        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.7, 0.8]},
        )
        client.post("/api/medias/1/vote", json={"target": "bad"})
        votes = client.get("/api/votes")
        assert "1" not in votes.get_json()["good_region_boxes"]

    def test_good_vote_without_region_box_omits_from_state(self, client):
        from vtsearch.state import (
            good_votes,
            vote_region_boxes,
        )

        resp = client.post("/api/medias/1/vote", json={"target": "good"})
        assert resp.status_code == 200
        assert 1 in good_votes
        assert 1 not in vote_region_boxes

    def test_bad_vote_with_region_box_rejected(self, client):
        """No-votes are always image-level by design; sending a region_box
        with a bad-vote is a client bug and the endpoint refuses to silently
        drop it.  See the patch-embedder v2 interaction-design notes."""
        from vtsearch.state import (
            bad_votes,
            vote_region_boxes,
        )

        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "bad", "region_box": [0.1, 0.2, 0.7, 0.8]},
        )
        assert resp.status_code == 400
        assert 1 not in bad_votes
        assert 1 not in vote_region_boxes

    def test_bad_vote_without_region_box_unchanged(self, client):
        from vtsearch.state import (
            bad_votes,
            vote_region_boxes,
        )

        resp = client.post("/api/medias/1/vote", json={"target": "bad"})
        assert resp.status_code == 200
        assert 1 in bad_votes
        assert 1 not in vote_region_boxes

    def test_malformed_region_box_rejected(self, client):
        # The marshmallow schema only declares ``region_box`` as a List of
        # Floats; the length check (==4) is done by ``_parse_region_box``
        # in the handler, so a 3-element list passes the schema and is
        # rejected by the handler with a 400 + standard ``message`` envelope.
        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.7]},
        )
        assert resp.status_code == 400
        assert "region_box" in resp.get_json()["message"]

    def test_out_of_range_region_box_rejected(self, client):
        # Range check ([0, 1]) is done by ``_parse_region_box`` in the
        # handler → 400 + standard ``message`` envelope.
        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.7, 1.5]},
        )
        assert resp.status_code == 400

    def test_non_numeric_region_box_rejected(self, client):
        # ``["a", "b", "c", "d"]`` fails the marshmallow Float coercion →
        # schema-level 422 with the per-field ``errors`` envelope.
        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": ["a", "b", "c", "d"]},
        )
        assert resp.status_code == 422

    def test_unvote_clears_region_box(self, client):
        """target=none removes the vote AND any region_box, so a subsequent
        fresh yes-vote isn't tagged with a stale annotation."""
        from vtsearch.state import (
            good_votes,
            vote_region_boxes,
        )

        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.7, 0.8]},
        )
        assert 1 in vote_region_boxes
        client.post("/api/medias/1/vote", json={"target": "none"})
        assert 1 not in good_votes
        assert 1 not in vote_region_boxes

    def test_idempotent_re_vote_preserves_region_box(self, client):
        """Re-sending target=good without region_box on an already-good media
        is idempotent; region_box stays unchanged.  This is intentional: an
        absent ``region_box`` on an idempotent call must not silently wipe a
        previously-recorded one (H1 idempotency rule)."""
        from vtsearch.state import vote_region_boxes

        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.7, 0.8]},
        )
        assert vote_region_boxes[1] == (0.1, 0.2, 0.7, 0.8)
        client.post("/api/medias/1/vote", json={"target": "good"})
        assert vote_region_boxes[1] == (0.1, 0.2, 0.7, 0.8)

    def test_switch_good_to_bad_clears_region_box(self, client):
        """A region annotation belongs to a yes-vote; flipping the same
        media to a no-vote drops it."""
        from vtsearch.state import (
            bad_votes,
            good_votes,
            vote_region_boxes,
        )

        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.7, 0.8]},
        )
        assert 1 in vote_region_boxes
        client.post("/api/medias/1/vote", json={"target": "bad"})
        assert 1 not in good_votes
        assert 1 in bad_votes
        assert 1 not in vote_region_boxes

    def test_replacing_region_box_via_unvote_then_revote(self, client):
        """Un-voting clears the region_box; re-voting good with a new box
        stores the new value."""
        from vtsearch.state import vote_region_boxes

        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.4, 0.4]},
        )
        client.post("/api/medias/1/vote", json={"target": "none"})
        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.3, 0.3, 0.9, 0.9]},
        )
        assert vote_region_boxes[1] == (0.3, 0.3, 0.9, 0.9)

    def test_idempotent_revote_with_new_region_box_replaces_in_place(self, client):
        """Drawing a new box on an already-good media replaces the previous
        annotation without going through un-vote, since the user explicitly
        sent a new ``region_box``.  An idempotent re-vote *without* a
        region_box leaves the existing one alone (so a stale-view tab can't
        wipe a region a different tab just set (H1 idempotency)."""
        from vtsearch.state import vote_region_boxes

        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.4, 0.4]},
        )
        # New box on an already-good media → replace in place.
        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.3, 0.3, 0.9, 0.9]},
        )
        assert vote_region_boxes[1] == (0.3, 0.3, 0.9, 0.9)

        # Idempotent re-vote without a region_box → existing box preserved.
        client.post("/api/medias/1/vote", json={"target": "good"})
        assert vote_region_boxes[1] == (0.3, 0.3, 0.9, 0.9)


class TestLabelExportRegionBox:
    """``GET /api/labels/export`` emits ``region_box`` on yes-votes that
    carried one and never on no-votes.  Patch-embedder v2 step 5
    (round-trip of step 1's data-model contract through the API layer).
    """

    def test_export_emits_region_box_on_good_vote(self, client):
        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.7, 0.8]},
        )
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        labels = resp.get_json()["labels"]
        assert len(labels) == 1
        assert labels[0]["label"] == "good"
        assert labels[0]["region_box"] == [0.1, 0.2, 0.7, 0.8]

    def test_export_omits_region_box_on_plain_good_vote(self, client):
        client.post("/api/medias/1/vote", json={"target": "good"})
        resp = client.get("/api/labels/export")
        labels = resp.get_json()["labels"]
        assert len(labels) == 1
        assert "region_box" not in labels[0]

    def test_export_never_emits_region_box_on_bad_vote(self, client):
        client.post("/api/medias/1/vote", json={"target": "bad"})
        resp = client.get("/api/labels/export")
        labels = resp.get_json()["labels"]
        assert len(labels) == 1
        assert labels[0]["label"] == "bad"
        assert "region_box" not in labels[0]

    def test_export_with_mixed_votes(self, client):
        """Region-annotated good, plain good, and a bad in the same export."""
        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.1, 0.2, 0.3, 0.4]},
        )
        client.post("/api/medias/2/vote", json={"target": "good"})
        client.post("/api/medias/3/vote", json={"target": "bad"})
        resp = client.get("/api/labels/export")
        labels = resp.get_json()["labels"]
        by_md5 = {e["md5"]: e for e in labels}
        # Look up which md5 is which media id via the test medias.
        from vtsearch.state import medias

        rb_md5 = medias[1]["md5"]
        plain_md5 = medias[2]["md5"]
        bad_md5 = medias[3]["md5"]
        assert by_md5[rb_md5]["region_box"] == [0.1, 0.2, 0.3, 0.4]
        assert "region_box" not in by_md5[plain_md5]
        assert "region_box" not in by_md5[bad_md5]


class TestLabelImportRegionBox:
    """``POST /api/labels/import`` round-trips ``region_box`` from the
    serialised LabelSet into in-memory state, so a sync_from / explicit
    import recovers the user's region annotations.
    """

    def test_import_restores_region_box_on_good(self, client):
        from vtsearch.state import (
            good_votes,
            medias,
            vote_region_boxes,
        )

        md5 = medias[1]["md5"]
        resp = client.post(
            "/api/labels/import",
            json={"labels": [{"md5": md5, "label": "good", "region_box": [0.2, 0.3, 0.5, 0.6]}]},
        )
        assert resp.status_code == 200
        assert 1 in good_votes
        assert vote_region_boxes[1] == (0.2, 0.3, 0.5, 0.6)

    def test_import_ignores_region_box_on_bad(self, client):
        from vtsearch.state import (
            bad_votes,
            medias,
            vote_region_boxes,
        )

        md5 = medias[1]["md5"]
        # A no-vote in an imported labelset cannot carry a box (LabeledElement
        # would never have one for a "bad" label in a well-formed export).
        # If a malformed import does include one, the importer must not
        # propagate it.
        resp = client.post(
            "/api/labels/import",
            json={"labels": [{"md5": md5, "label": "bad", "region_box": [0.0, 0.0, 1.0, 1.0]}]},
        )
        assert resp.status_code == 200
        assert 1 in bad_votes
        assert 1 not in vote_region_boxes


class TestRegionAwareTraining:
    """``train_and_score`` and ``populate_label_embeddings`` derive the
    training vector on-the-fly when an element carries a ``region_box``: the
    raw patch of ``media["patch_grid"]`` nearest the box, which is by
    construction one of the rows the scorer max-pools.
    """

    def _media_with_patch_grid(self, grid_value: float, cid: int) -> dict:
        """Build a synthetic image media dict with a one-axis patch grid.

        Each cell is a distinct axis-aligned unit vector, so it's easy to
        check which cells got pooled by inspecting the resulting vector.
        """
        h, w, d = 4, 4, 16
        n = h * w
        flat = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            flat[i, i] = 1.0
        grid = flat.reshape(h, w, d)
        # Image-level CLS embedding: a different unit vector so we can tell
        # the two paths apart in assertions.
        cls = np.zeros(d, dtype=np.float32)
        cls[15] = grid_value  # mostly axis 15
        cls[14] = (1.0 - grid_value**2) ** 0.5  # keep unit length
        return {
            "id": cid,
            "md5": f"md5-{cid:04x}",
            "media_type": "image",
            "embedder": "siglip",
            "embeddings": {"siglip": cls},
            "patch_grid": grid,
        }

    def test_train_and_score_uses_nearest_patch_when_grid_present(self):
        """A yes-vote with region_box trains on the raw patch nearest the box,
        not the CLS embedding."""
        from vtscore.detectors.training import _training_vec_for_vote

        media = self._media_with_patch_grid(0.99, cid=42)
        box = (0.0, 0.0, 0.5, 0.5)  # top-left quadrant of a 4x4 grid

        vec = _training_vec_for_vote(media, box)
        expected = nearest_patch_to_box(media["patch_grid"], box)
        np.testing.assert_array_equal(vec, expected)
        # And the CLS vector is *not* what we got.
        assert not np.array_equal(vec, media_embedding(media))

    def test_training_vector_is_one_of_the_scored_rows(self):
        """The train/score invariant: the Good vote's vector must appear in the
        stack the scorer max-pools, or the classifier learns to separate the
        training geometry from the scoring geometry (perfect ranking, FNR 1)."""
        from vtscore.detectors.training import _training_vec_for_vote
        from vtscore.embedding.matrix import media_score_rows

        media = self._media_with_patch_grid(0.99, cid=7)
        rows = media_score_rows(media, "siglip")
        for box in [(0.02, 0.02, 0.48, 0.48), (0.5, 0.5, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), None]:
            vec = _training_vec_for_vote(media, box)
            assert any(np.allclose(row, vec, atol=1e-6) for row in rows), f"box {box} trains outside the score stack"

    def test_train_and_score_falls_back_to_cls_without_patch_grid(self):
        """Legacy / single-vector datasets have no ``patch_grid``; even with
        a stashed region_box, training must use the full-image CLS vector."""
        from vtscore.detectors.training import _training_vec_for_vote

        media = {
            "id": 1,
            "md5": "abc",
            "media_type": "image",
            "embedder": "siglip",
            "embeddings": {"siglip": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
        }
        vec = _training_vec_for_vote(media, (0.1, 0.2, 0.7, 0.8))
        np.testing.assert_array_equal(vec, media_embedding(media))

    def test_train_and_score_falls_back_to_cls_when_no_region_box(self):
        from vtscore.detectors.training import _training_vec_for_vote

        media = self._media_with_patch_grid(0.99, cid=1)
        vec = _training_vec_for_vote(media, region_box=None)
        np.testing.assert_array_equal(vec, media_embedding(media))

    def _register_synthetic_image(self, cid: int, grid_value: float = 0.99) -> dict:
        """Insert a synthetic image media into the active dataset context.

        ``populate_label_embeddings`` uses ``resolve_current_dataset_cid``
        internally, which looks up via the global ``snapshot_medias()``.
        Registering the media in the active dataset makes that lookup
        succeed by md5.  Conftest's ``reset_state`` wipes this between
        tests so there's no cross-test bleed.
        """
        from vtsearch.state import medias

        media = self._media_with_patch_grid(grid_value, cid=cid)
        medias[cid] = media
        return media

    def test_populate_label_embeddings_pools_when_region_box_set(self):
        """``populate_label_embeddings`` caches pooled vectors for elements
        whose ``region_box`` is set and whose source media has a
        ``patch_grid``.  The cache value matches ``nearest_patch_to_box``
        on the same grid + box."""
        from vtscore.datasets.labelset import LabeledElement, LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        cid = 9001
        media = self._register_synthetic_image(cid)
        # Element matches the media by md5.
        elem = LabeledElement(md5=media["md5"], label="good", region_box=(0.0, 0.0, 0.5, 0.5))
        ls = LabelSet([elem])

        det_ctx = DetectorContext("d1")
        snap = {cid: media}
        populate_label_embeddings(det_ctx, ls, media_type="image", snap=snap)

        eid = stable_element_id(elem)
        assert eid in det_ctx.label_embeddings
        expected = nearest_patch_to_box(media["patch_grid"], (0.0, 0.0, 0.5, 0.5))
        np.testing.assert_allclose(det_ctx.label_embeddings[eid], expected, atol=1e-6)
        # And it's NOT the CLS embedding.
        assert not np.allclose(det_ctx.label_embeddings[eid], media_embedding(media))

    def test_populate_label_embeddings_repools_when_region_box_set(self):
        """Region-voted elements re-pool on every call so region_box edits
        propagate without an explicit cache invalidation.  Image-level
        elements keep their cached vector across calls (fast path)."""
        from vtscore.datasets.labelset import LabeledElement, LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        cid = 9002
        media = self._register_synthetic_image(cid)
        elem = LabeledElement(md5=media["md5"], label="good", region_box=(0.0, 0.0, 0.5, 0.5))
        ls = LabelSet([elem])
        det_ctx = DetectorContext("d1")
        snap = {cid: media}

        populate_label_embeddings(det_ctx, ls, media_type="image", snap=snap)
        eid = stable_element_id(elem)
        first = np.array(det_ctx.label_embeddings[eid], copy=True)

        # Simulate a region-box edit by mutating the element in place
        # and re-running.  The cache *must* refresh.
        elem.region_box = (0.5, 0.5, 1.0, 1.0)  # bottom-right quadrant
        populate_label_embeddings(det_ctx, ls, media_type="image", snap=snap)
        second = det_ctx.label_embeddings[eid]
        expected_second = nearest_patch_to_box(media["patch_grid"], (0.5, 0.5, 1.0, 1.0))
        np.testing.assert_allclose(second, expected_second, atol=1e-6)
        # The two patch vectors must be different (top-left vs bottom-right).
        assert not np.allclose(first, second)

    def test_populate_label_embeddings_keeps_cached_when_no_region_box(self):
        """Plain image-level elements stay cached across calls; the fast
        path for non-region datasets is preserved."""
        from vtscore.datasets.labelset import LabeledElement, LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        cid = 9003
        media = self._register_synthetic_image(cid)
        elem = LabeledElement(md5=media["md5"], label="good")  # no region_box
        ls = LabelSet([elem])
        det_ctx = DetectorContext("d1")
        snap = {cid: media}

        populate_label_embeddings(det_ctx, ls, media_type="image", snap=snap)
        eid = stable_element_id(elem)
        # Pre-set a sentinel and verify it survives a second pass (cache hit).
        sentinel = np.full_like(det_ctx.label_embeddings[eid], 7.0)
        det_ctx.label_embeddings[eid] = sentinel
        populate_label_embeddings(det_ctx, ls, media_type="image", snap=snap)
        np.testing.assert_array_equal(det_ctx.label_embeddings[eid], sentinel)

    def test_populate_label_embeddings_invalidates_when_region_box_removed(self):
        """Logical-bug-audit M4: when a labelset element loses its ``region_box``
        (e.g. the user flipped good→bad on a previously region-voted media, or
        un-voted/re-voted without a region), the cache must NOT keep returning
        the previously-pooled region vector.  The next pass should produce the
        image-level CLS embedding instead."""
        from vtscore.datasets.labelset import LabeledElement, LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        cid = 9004
        media = self._register_synthetic_image(cid)
        # First pass: element carries a region_box → cached as a pooled vector.
        elem = LabeledElement(md5=media["md5"], label="good", region_box=(0.0, 0.0, 0.5, 0.5))
        ls = LabelSet([elem])
        det_ctx = DetectorContext("d1")
        snap = {cid: media}

        populate_label_embeddings(det_ctx, ls, media_type="image", snap=snap)
        eid = stable_element_id(elem)
        pooled = np.array(det_ctx.label_embeddings[eid], copy=True)
        # Sanity: the cached vector is the pooled box, not the CLS embedding.
        assert not np.allclose(pooled, media_embedding(media))
        assert det_ctx.label_embedding_regions[eid] == (0.0, 0.0, 0.5, 0.5)

        # Simulate the bug-triggering edit: same element identity (same eid)
        # but the user removed the region_box (e.g. flipped to bad, then back
        # to good without a region; bad votes never carry a region_box).
        elem.region_box = None
        populate_label_embeddings(det_ctx, ls, media_type="image", snap=snap)
        # The cached vector must now be the image-level CLS embedding, not
        # the stale top-left pooled vector.
        np.testing.assert_array_equal(det_ctx.label_embeddings[eid], media_embedding(media))
        assert det_ctx.label_embedding_regions[eid] is None

    def test_populate_label_embeddings_invalidates_when_region_box_added(self):
        """Mirror of the M4 fix in the opposite direction: an image-level
        cached entry must be re-pooled when the element gains a region_box.
        (This already worked before M4 because the original cache check
        gated on ``elem.region_box is None``, but the explicit test pins the
        behaviour now that the cache check also consults the region cache.)"""
        from vtscore.datasets.labelset import LabeledElement, LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        cid = 9005
        media = self._register_synthetic_image(cid)
        elem = LabeledElement(md5=media["md5"], label="good")  # no region_box
        ls = LabelSet([elem])
        det_ctx = DetectorContext("d1")
        snap = {cid: media}

        populate_label_embeddings(det_ctx, ls, media_type="image", snap=snap)
        eid = stable_element_id(elem)
        assert det_ctx.label_embedding_regions[eid] is None
        np.testing.assert_array_equal(det_ctx.label_embeddings[eid], media_embedding(media))

        # User adds a region annotation to the element.
        elem.region_box = (0.5, 0.5, 1.0, 1.0)
        populate_label_embeddings(det_ctx, ls, media_type="image", snap=snap)
        expected = nearest_patch_to_box(media["patch_grid"], (0.5, 0.5, 1.0, 1.0))
        np.testing.assert_allclose(det_ctx.label_embeddings[eid], expected, atol=1e-6)
        assert det_ctx.label_embedding_regions[eid] == (0.5, 0.5, 1.0, 1.0)


class TestRegionAwareTrainingCrossDataset:
    """Cross-dataset region-vote path: when a labelset element carries a
    ``region_box`` and its source file is not in the active dataset's snap,
    ``_embed_one`` should resolve the file, run ``patch_forward`` on it,
    and take the raw patch under the box via :func:`nearest_patch_to_box`; the
    user's region intent isn't silently downgraded to a full-image embedding
    (bug H2).

    Falls back (with a warning) to the image-level embedding only when the
    chosen embedder can't produce a patch grid.
    """

    def _make_grid(self, h: int = 4, w: int = 4, d: int = 16) -> np.ndarray:
        """Axis-aligned unit-vector grid: cell (r, c) holds e_{r*w + c}.

        Lets the test assert exactly which cells got pooled by inspecting
        the resulting vector's argmax pattern.
        """
        n = h * w
        flat = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            flat[i, i] = 1.0
        return flat.reshape(h, w, d)

    def _patch_capable_stub(self, grid: np.ndarray):
        """A stub embedder whose ``patch_forward`` returns *grid*."""
        from vtscore.media.patch_embed import PatchEmbedOutput

        cls = np.zeros(grid.shape[-1], dtype=np.float32)
        cls[-1] = 1.0  # arbitrary unit vector, distinct from any cell

        class _PatchStub:
            name = "patch-stub"
            supports_patch_regions = True

            def patch_forward(self, _media):
                saliency = np.ones(grid.shape[:2], dtype=np.float32)
                saliency /= saliency.sum()
                return PatchEmbedOutput(cls_vec=cls, patch_grid=grid, patch_saliency=saliency)

        return _PatchStub(), cls

    def _single_vector_stub(self, dim: int = 16):
        """Stub embedder that does NOT support patch regions; ``embed_media``
        returns a sentinel vector so the fallback can be detected."""
        sentinel = np.zeros(dim, dtype=np.float32)
        sentinel[0] = 1.0

        class _SingleStub:
            name = "single-stub"
            supports_patch_regions = False

            def embed_media(self, _media):
                return sentinel

        return _SingleStub(), sentinel

    def _wire_resolution(
        self,
        monkeypatch,
        tmp_path,
        embedder,
        filename: str = "missing.png",
    ):
        """Make ``resolve_file_context`` yield a real-looking path and route
        ``get_embedder`` / ``embedders_for_type`` / ``embed_file`` to *embedder*.

        Returns the resolved path so the test can assert it was used.
        """
        from contextlib import contextmanager
        from pathlib import Path

        import vtscore.detectors.resolver as resolver_mod
        import vtscore.media as media_mod

        fake = Path(tmp_path) / filename
        fake.write_bytes(b"")  # exists on disk so resolvers don't second-guess

        @contextmanager
        def _fake_ctx(_origin, _origin_name="", _filename=""):
            yield fake

        monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_ctx)
        monkeypatch.setattr(media_mod, "get_embedder", lambda _name: embedder)
        monkeypatch.setattr(media_mod, "embedders_for_type", lambda _mt: [embedder])

        # Image-level fallback inside ``embed_file`` re-looks-up the
        # embedder; stub it directly so the fallback path produces the
        # single-stub sentinel rather than a real model load.
        def _fake_embed_file(_path, _media_type, _embedder_name=""):
            return embedder.embed_media({"media_path": str(fake)})

        monkeypatch.setattr(resolver_mod, "embed_file", _fake_embed_file)
        return fake

    def _cross_dataset_elem(self, *, region_box, md5="ff" * 16, origin_name="missing.png"):
        """Build an element whose md5 won't match any active media so the
        in-dataset path skips and we hit ``_embed_one``."""
        from vtscore.datasets.labelset import LabeledElement

        return LabeledElement(
            md5=md5,
            label="good",
            origin={"importer": "server_folder", "params": {"folder": "/nowhere"}},
            origin_name=origin_name,
            filename=origin_name,
            region_box=region_box,
        )

    def test_cross_dataset_region_vote_uses_pooled_vector_when_embedder_supports_patches(self, monkeypatch, tmp_path):
        """An element with ``region_box`` whose file isn't in the active
        snap resolves via ``patch_forward`` + ``nearest_patch_to_box`` instead
        of falling back to the full-image embedding."""
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        grid = self._make_grid()
        stub, _cls = self._patch_capable_stub(grid)
        self._wire_resolution(monkeypatch, tmp_path, stub)

        box = (0.0, 0.0, 0.5, 0.5)
        elem = self._cross_dataset_elem(region_box=box)
        ls = LabelSet([elem])
        det_ctx = DetectorContext("d1")
        # Empty snap → in-dataset path skipped; ``_embed_one`` is invoked.
        populate_label_embeddings(det_ctx, ls, media_type="image", snap={})

        eid = stable_element_id(elem)
        assert eid in det_ctx.label_embeddings
        expected = nearest_patch_to_box(grid, box)
        np.testing.assert_allclose(det_ctx.label_embeddings[eid], expected, atol=1e-6)

    def test_cross_dataset_region_vote_box_change_reflects_in_pooled_vector(self, monkeypatch, tmp_path):
        """Two boxes over disjoint quadrants of the same patch grid must
        produce two visibly different cached vectors; proves the region intent
        really threads through the cross-dataset path."""
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        grid = self._make_grid()
        stub, _ = self._patch_capable_stub(grid)
        self._wire_resolution(monkeypatch, tmp_path, stub)

        top_left = (0.0, 0.0, 0.5, 0.5)
        bot_right = (0.5, 0.5, 1.0, 1.0)

        elem = self._cross_dataset_elem(region_box=top_left)
        ls = LabelSet([elem])
        det_ctx = DetectorContext("d1")

        populate_label_embeddings(det_ctx, ls, media_type="image", snap={})
        eid = stable_element_id(elem)
        first = np.array(det_ctx.label_embeddings[eid], copy=True)

        elem.region_box = bot_right
        populate_label_embeddings(det_ctx, ls, media_type="image", snap={})
        second = det_ctx.label_embeddings[eid]

        np.testing.assert_allclose(second, nearest_patch_to_box(grid, bot_right), atol=1e-6)
        assert not np.allclose(first, second), "Region edit must re-resolve, not reuse the cached vector"

    def test_cross_dataset_region_vote_falls_back_with_warning_for_single_vector_embedder(
        self, monkeypatch, tmp_path, caplog
    ):
        """Legacy single-vector embedders can't produce a patch grid.  The
        element still trains (we don't drop the vote), but the cached
        embedding is the full-image vector and a warning surfaces the
        silent downgrade."""
        import logging

        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        stub, sentinel = self._single_vector_stub()
        self._wire_resolution(monkeypatch, tmp_path, stub)

        elem = self._cross_dataset_elem(region_box=(0.0, 0.0, 0.5, 0.5))
        ls = LabelSet([elem])
        det_ctx = DetectorContext("d1")

        with caplog.at_level(logging.WARNING, logger="vtscore.detectors.labelset_training"):
            populate_label_embeddings(det_ctx, ls, media_type="image", snap={})

        eid = stable_element_id(elem)
        np.testing.assert_array_equal(det_ctx.label_embeddings[eid], sentinel)
        assert any("region_box" in r.message and "patch regions" in r.message for r in caplog.records), (
            f"Expected a region-downgrade warning; got: {[r.message for r in caplog.records]}"
        )

    def test_cross_dataset_region_vote_returns_none_when_embedder_patch_forward_fails(self, monkeypatch, tmp_path):
        """``patch_forward`` returning ``None`` (failed decode, etc.)
        downgrades to the image-level fallback rather than skipping the
        element entirely, keeping the vote in the training set."""
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        sentinel = np.zeros(16, dtype=np.float32)
        sentinel[5] = 1.0

        class _FlakyPatchStub:
            name = "flaky-patch-stub"
            supports_patch_regions = True

            def patch_forward(self, _media):
                return None  # simulates a failed forward pass

            def embed_media(self, _media):
                return sentinel

        self._wire_resolution(monkeypatch, tmp_path, _FlakyPatchStub())

        elem = self._cross_dataset_elem(region_box=(0.0, 0.0, 0.5, 0.5))
        ls = LabelSet([elem])
        det_ctx = DetectorContext("d1")
        populate_label_embeddings(det_ctx, ls, media_type="image", snap={})

        eid = stable_element_id(elem)
        np.testing.assert_array_equal(det_ctx.label_embeddings[eid], sentinel)

    def test_cross_dataset_no_region_box_unchanged(self, monkeypatch, tmp_path):
        """Image-level cross-dataset elements (no ``region_box``) must hit
        the existing ``embed_file`` path; patch_forward should not even
        be called.  Guards against the new path firing spuriously."""
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        sentinel = np.zeros(16, dtype=np.float32)
        sentinel[7] = 1.0
        patch_calls = []

        class _ObservingPatchStub:
            name = "observing-patch-stub"
            supports_patch_regions = True

            def patch_forward(self, media):
                patch_calls.append(media)
                raise AssertionError("patch_forward must not be called for image-level votes")

            def embed_media(self, _media):
                return sentinel

        self._wire_resolution(monkeypatch, tmp_path, _ObservingPatchStub())

        elem = self._cross_dataset_elem(region_box=None)
        ls = LabelSet([elem])
        det_ctx = DetectorContext("d1")
        populate_label_embeddings(det_ctx, ls, media_type="image", snap={})

        eid = stable_element_id(elem)
        np.testing.assert_array_equal(det_ctx.label_embeddings[eid], sentinel)
        assert patch_calls == []


class TestBadVoteRegionFlooding:
    """A Bad vote on a patch media floods the image's whole score-row stack -
    the image-level vector plus every raw patch - so the max-pool can't surface
    any look-alike sub-region of a rejected image.  Legacy datasets contribute
    one image-level negative, unchanged.
    """

    @staticmethod
    def _patch_media(cid, d=8):
        rng = np.random.default_rng(cid)

        def _unit(v):
            v = np.asarray(v, dtype=np.float32)
            return v / (np.linalg.norm(v) + 1e-8)

        cls = _unit(rng.standard_normal(d))
        # A 1x2 grid: two raw patches, so a Bad vote floods 1 + 2 = 3 rows.
        patch1 = _unit(rng.standard_normal(d))
        patch2 = _unit(rng.standard_normal(d))
        return {
            "id": cid,
            "md5": f"md5-{cid:04x}",
            "media_type": "image",
            "embedder": "dinov3_patch",
            "embeddings": {"dinov3_patch": cls},
            "patch_grid": np.stack([patch1, patch2])[None, :, :],
        }, [cls, patch1, patch2]

    def test_bad_negative_vecs_returns_image_vector_plus_every_patch(self):
        from vtscore.detectors.training import bad_negative_vecs

        media, expected_rows = self._patch_media(1)
        vecs = bad_negative_vecs(media, "dinov3_patch")
        assert len(vecs) == 3  # image-level row + 2 raw patches
        for got, exp in zip(vecs, expected_rows):
            np.testing.assert_allclose(got, exp, atol=1e-6)

    def test_flood_is_exactly_the_scoring_stack(self):
        """#2731's flood/score gap is closed: MaxPatch has no internal nodes,
        so every row inference max-pools is a row a Bad vote trains down."""
        from vtscore.detectors.training import bad_negative_vecs, inference_score_rows

        media, _ = self._patch_media(3)
        np.testing.assert_array_equal(
            np.stack(bad_negative_vecs(media, "dinov3_patch")),
            inference_score_rows(media, "dinov3_patch"),
        )

    def test_bad_negative_vecs_legacy_media_single_image_vector(self):
        from vtscore.detectors.training import bad_negative_vecs

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        media = {"id": 1, "media_type": "image", "embedder": "siglip", "embeddings": {"siglip": vec}}
        vecs = bad_negative_vecs(media, "siglip")
        assert len(vecs) == 1
        np.testing.assert_array_equal(vecs[0], vec)

    def test_build_vote_xy_floods_bad_patch_vote_with_grouped_rows(self):
        from vtscore.detectors.training import _build_vote_xy

        good_media, _ = self._patch_media(1)
        bad_media, _ = self._patch_media(2)
        clips = {1: good_media, 2: bad_media}

        # No region box on the good vote, so it trains image-level (1 row).
        X, y, groups, score_rows = _build_vote_xy(clips, {1: None}, {2: None}, {}, "dinov3_patch")
        # 1 good row + 3 flooded bad rows, all sharing one bag id.
        assert y == [1.0, 0.0, 0.0, 0.0]
        assert groups == [("g", 1), ("b", 2), ("b", 2), ("b", 2)]
        assert len(X) == 4
        # Both bags carry the *scoring* stack (image row + every patch) so
        # calibration collapses them the way inference does - not 1 row
        # against 3.
        assert set(score_rows) == {("g", 1), ("b", 2)}
        assert score_rows[("g", 1)].shape == (3, 8)
        assert score_rows[("b", 2)].shape == (3, 8)

    def test_per_bag_weights_balance_bag_not_rows(self):
        from vtscore.training.thresholds import _per_bag_fit_weights

        # 1 good vote, 1 bad bag flooded into 3 rows.
        y = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        groups = [("g", 1), ("b", 2), ("b", 2), ("b", 2)]
        w = _per_bag_fit_weights(y, groups)
        # good weight = n_bad_bags / n_good = 1/1 = 1; each bad row = 1/3.
        np.testing.assert_allclose(w, [1.0, 1 / 3, 1 / 3, 1 / 3], atol=1e-6)
        # Total good mass == total bad mass (balanced at the bag level).
        assert abs(w[y == 1].sum() - w[y == 0].sum()) < 1e-6

    def test_flood_context_noop_when_every_bag_is_one_row(self):
        from vtscore.detectors.training import _flood_context

        X = [np.zeros(4, dtype=np.float32) for _ in range(3)]
        y = [1.0, 0.0, 0.0]
        groups = [("g", 1), ("b", 2), ("b", 3)]  # no multi-row bag
        n_votes, cal_groups, sample_weights = _flood_context(X, y, groups)
        assert n_votes == 3
        assert cal_groups is None  # unflooded -> legacy row-wise path
        assert sample_weights is None

    def test_build_xy_from_labelset_floods_bad_from_neg_cache(self):
        from vtscore.datasets.labelset import LabeledElement, LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import build_xy_from_labelset
        from vtscore.state.core import DetectorContext

        good = LabeledElement(md5="g", label="good")
        bad = LabeledElement(md5="b", label="bad")
        det = DetectorContext("d1")
        gid, bid = stable_element_id(good), stable_element_id(bad)
        det.label_embeddings[gid] = np.array([1, 0, 0, 0], dtype=np.float32)
        det.label_embeddings[bid] = np.array([0, 1, 0, 0], dtype=np.float32)  # ignored when flooded
        flooded = [np.array([0, 0, 1, 0], dtype=np.float32), np.array([0, 0, 0, 1], dtype=np.float32)]
        det.label_negative_regions[bid] = flooded

        X, y, groups, _score_rows = build_xy_from_labelset(det, LabelSet([good, bad]))
        assert y == [1.0, 0.0, 0.0]
        assert groups == [("g", gid), ("b", bid), ("b", bid)]
        np.testing.assert_array_equal(X[1], flooded[0])
        np.testing.assert_array_equal(X[2], flooded[1])


class TestBagAwareCalibration:
    """``compute_fold_orderings(groups=...)`` splits by bag and max-pools each
    calibration group to one score - so a Bad bag's flooded patches never
    straddle the Train/Calibrate boundary and score as one image.
    """

    def test_grouped_calibration_keeps_bags_together_and_maxpools(self):
        from vtscore.training.thresholds import compute_fold_orderings

        rng = np.random.default_rng(0)
        # 3 good votes (singletons) + 3 bad bags (3 rows each) = 6 bags, 12 rows.
        X, y, groups = [], [], []
        for g in range(3):
            X.append(rng.standard_normal(4).astype(np.float32))
            y.append(1.0)
            groups.append(("g", g))
        for b in range(3):
            for _ in range(3):
                X.append(rng.standard_normal(4).astype(np.float32))
                y.append(0.0)
                groups.append(("b", b))

        orderings, fallback = compute_fold_orderings(
            X, y, 4, rng=np.random.RandomState(42), calibrate_count=2, hidden_dim=8, groups=groups
        )
        assert fallback is None
        assert len(orderings) == 2
        for scores, labels in orderings:
            # One score per calibration GROUP, not per row: a fold holding
            # k cal groups yields exactly k scores/labels.
            assert len(scores) == len(labels)
            assert all(lbl in (0.0, 1.0) for lbl in labels)
            # No fold has more entries than there are bags.
            assert len(scores) <= 6
