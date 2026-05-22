"""Tests for the batched ``_embed_media_bulk_impl`` overrides on image embedders.

The overrides themselves are model-free — they delegate to the shared
helper in ``vtscore.media.image._image_bulk`` and a per-embedder
``_forward_pil_batch`` callable.  These tests stub the model + processor
so we can assert:

* The bulk path decodes each file path to a PIL image and batches the
  GPU forward in chunks of ``embed_batch_size``.
* Per-image PIL-decode failures don't kill the batch.
* The returned list aligns position-for-position with the input list.
* The same plumbing works for ``patch_forward_bulk`` on the patch
  embedders.

No model weights are loaded — these tests run on CPU in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    Image.new("RGB", (8, 8), color=color).save(path)


def _media(path: Path) -> dict:
    return {"media_path": str(path), "origin": None, "origin_name": path.name}


# ---------------------------------------------------------------------------
# resolve_embed_batch_size
# ---------------------------------------------------------------------------


class TestResolveEmbedBatchSize:
    def test_default(self, monkeypatch):
        from vtscore.media.embedder import (
            DEFAULT_EMBED_BATCH_SIZE,
            resolve_embed_batch_size,
        )

        monkeypatch.delenv("VTSEARCH_EMBED_BATCH_SIZE", raising=False)
        assert resolve_embed_batch_size() == DEFAULT_EMBED_BATCH_SIZE

    def test_env_override(self, monkeypatch):
        from vtscore.media.embedder import resolve_embed_batch_size

        monkeypatch.setenv("VTSEARCH_EMBED_BATCH_SIZE", "7")
        assert resolve_embed_batch_size() == 7

    def test_invalid_env_falls_back(self, monkeypatch):
        from vtscore.media.embedder import (
            DEFAULT_EMBED_BATCH_SIZE,
            resolve_embed_batch_size,
        )

        monkeypatch.setenv("VTSEARCH_EMBED_BATCH_SIZE", "abc")
        assert resolve_embed_batch_size() == DEFAULT_EMBED_BATCH_SIZE

    def test_nonpositive_env_falls_back(self, monkeypatch):
        from vtscore.media.embedder import (
            DEFAULT_EMBED_BATCH_SIZE,
            resolve_embed_batch_size,
        )

        monkeypatch.setenv("VTSEARCH_EMBED_BATCH_SIZE", "0")
        assert resolve_embed_batch_size() == DEFAULT_EMBED_BATCH_SIZE


# ---------------------------------------------------------------------------
# Shared bulk_embed_image_files helper
# ---------------------------------------------------------------------------


class TestBulkEmbedImageFilesHelper:
    """The shared helper in ``vtscore.media.image._image_bulk``."""

    def test_chunks_into_batch_size_groups(self, tmp_path):
        from vtscore.media.image._image_bulk import bulk_embed_image_files

        paths = []
        for i in range(5):
            p = tmp_path / f"img_{i}.png"
            _write_image(p)
            paths.append(p)

        medias = [_media(p) for p in paths]

        seen_batch_sizes: list[int] = []

        def fake_forward(images: list[Image.Image]) -> np.ndarray:
            seen_batch_sizes.append(len(images))
            # 1-d "embedding" = batch position so we can verify slot alignment
            return np.arange(len(images), dtype=np.float32).reshape(-1, 1)

        out = bulk_embed_image_files(
            medias,
            forward_pil_batch=fake_forward,
            batch_size=2,
            on_progress=lambda *a, **kw: None,
            label="test",
        )

        # 5 items at batch_size=2 → 3 chunks (2, 2, 1).
        assert seen_batch_sizes == [2, 2, 1]
        assert len(out) == 5
        assert all(v is not None for v in out)

    def test_pil_decode_failure_keeps_others_in_batch(self, tmp_path):
        from vtscore.media.image._image_bulk import bulk_embed_image_files

        good = tmp_path / "good.png"
        _write_image(good)
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not a real image")

        medias = [_media(good), _media(bad), _media(good)]

        def fake_forward(images: list[Image.Image]) -> np.ndarray:
            # Only the two good images should reach the forward.
            assert len(images) == 2
            return np.ones((len(images), 3), dtype=np.float32)

        out = bulk_embed_image_files(
            medias,
            forward_pil_batch=fake_forward,
            batch_size=10,
            on_progress=lambda *a, **kw: None,
            label="test",
        )

        assert out[0] is not None
        assert out[1] is None  # decode failed
        assert out[2] is not None

    def test_forward_exception_drops_batch_but_continues(self, tmp_path):
        from vtscore.media.image._image_bulk import bulk_embed_image_files

        # 4 images, batch_size=2 → 2 chunks; first chunk's forward raises.
        paths = []
        for i in range(4):
            p = tmp_path / f"img_{i}.png"
            _write_image(p)
            paths.append(p)
        medias = [_media(p) for p in paths]

        calls = {"n": 0}

        def fake_forward(images: list[Image.Image]) -> np.ndarray:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated OOM")
            return np.ones((len(images), 2), dtype=np.float32)

        out = bulk_embed_image_files(
            medias,
            forward_pil_batch=fake_forward,
            batch_size=2,
            on_progress=lambda *a, **kw: None,
            label="test",
        )

        assert out[0] is None
        assert out[1] is None
        assert out[2] is not None
        assert out[3] is not None

    def test_empty_input(self):
        from vtscore.media.image._image_bulk import bulk_embed_image_files

        out = bulk_embed_image_files(
            [],
            forward_pil_batch=lambda _imgs: np.zeros((0, 1), dtype=np.float32),
            batch_size=4,
            on_progress=lambda *a, **kw: None,
            label="test",
        )
        assert out == []

    def test_progress_emitted_per_batch(self, tmp_path):
        from vtscore.media.image._image_bulk import bulk_embed_image_files

        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            _write_image(p)
            paths.append(p)
        medias = [_media(p) for p in paths]

        events: list[tuple[str, int, int]] = []

        def on_progress(status, msg, cur, tot):
            events.append((status, cur, tot))

        def fake_forward(images: list[Image.Image]) -> np.ndarray:
            return np.zeros((len(images), 1), dtype=np.float32)

        bulk_embed_image_files(
            medias,
            forward_pil_batch=fake_forward,
            batch_size=2,
            on_progress=on_progress,
            label="test",
        )

        # batch_size=2 over 3 items → 2 batches → 2 progress ticks.
        assert events == [("embedding", 2, 3), ("embedding", 3, 3)]


# ---------------------------------------------------------------------------
# Concrete embedder overrides — wired up but model-free.
# ---------------------------------------------------------------------------


def _stub_image_embedder(emb, dim: int = 3):
    """Install minimal mocks so the bulk override can run end-to-end."""

    fake_processor = mock.MagicMock()
    # Processor return type just needs `.items()` returning (name, tensor) pairs;
    # we replace the forward callable in the embedder so the processor output
    # doesn't actually flow into a model.
    fake_processor.return_value = {}
    emb._processor = fake_processor

    fake_model = mock.MagicMock()
    emb._model = fake_model

    # Replace the per-class _forward_pil_batch so we never touch torch.
    def fake_forward(images):
        # Position-based vectors so we can assert slot alignment.
        return np.stack([np.full(dim, float(i), dtype=np.float32) for i in range(len(images))])

    emb._forward_pil_batch = fake_forward


class TestSiglipBulkOverride:
    def test_bulk_returns_per_input_vector(self, tmp_path):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        _stub_image_embedder(emb)

        paths = [tmp_path / f"img_{i}.png" for i in range(3)]
        for p in paths:
            _write_image(p)

        out = emb.embed_media_bulk([_media(p) for p in paths])

        assert len(out) == 3
        assert all(v is not None for v in out)
        # Position-based vectors confirm slot alignment.
        assert out[0][0] == 0.0  # pyright: ignore[reportOptionalSubscript]
        assert out[1][0] == 1.0  # pyright: ignore[reportOptionalSubscript]
        assert out[2][0] == 2.0  # pyright: ignore[reportOptionalSubscript]

    def test_routes_through_bulk_helper(self, tmp_path):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        _stub_image_embedder(emb)

        p = tmp_path / "img.png"
        _write_image(p)

        with mock.patch(
            "vtscore.media.image.embedder_siglip.bulk_embed_image_files",
            wraps=__import__(
                "vtscore.media.image._image_bulk", fromlist=["bulk_embed_image_files"]
            ).bulk_embed_image_files,
        ) as wrapped:
            emb.embed_media_bulk([_media(p)])

        assert wrapped.call_count == 1
        kwargs = wrapped.call_args.kwargs
        assert kwargs["label"] == "SigLIP"


class TestClipBulkOverride:
    def test_bulk_returns_per_input_vector(self, tmp_path):
        from vtscore.media.image.embedder_clip import ImageClipEmbedder

        emb = ImageClipEmbedder()
        _stub_image_embedder(emb)

        paths = [tmp_path / f"img_{i}.png" for i in range(4)]
        for p in paths:
            _write_image(p)

        out = emb.embed_media_bulk([_media(p) for p in paths])

        assert len(out) == 4
        assert all(v is not None for v in out)


class TestSiglip2BulkOverride:
    def test_bulk_returns_per_input_vector(self, tmp_path):
        from vtscore.media.image.embedder_siglip2 import ImageSiglip2Embedder

        emb = ImageSiglip2Embedder()
        _stub_image_embedder(emb)

        p = tmp_path / "img.png"
        _write_image(p)
        out = emb.embed_media_bulk([_media(p)])
        assert len(out) == 1 and out[0] is not None


class TestDinov2BulkOverride:
    def test_bulk_returns_per_input_vector(self, tmp_path):
        from vtscore.media.image.embedder_dinov2_single import (
            ImageDinov2SingleEmbedder,
        )

        emb = ImageDinov2SingleEmbedder()
        _stub_image_embedder(emb)

        paths = [tmp_path / f"img_{i}.png" for i in range(2)]
        for p in paths:
            _write_image(p)

        out = emb.embed_media_bulk([_media(p) for p in paths])
        assert all(v is not None for v in out)
        assert [v[0] for v in out if v is not None] == [0.0, 1.0]


class TestDinov3BulkOverride:
    def test_bulk_returns_per_input_vector(self, tmp_path):
        from vtscore.media.image.embedder_dinov3_single import (
            ImageDinov3SingleEmbedder,
        )

        emb = ImageDinov3SingleEmbedder()
        _stub_image_embedder(emb)

        paths = [tmp_path / f"img_{i}.png" for i in range(2)]
        for p in paths:
            _write_image(p)

        out = emb.embed_media_bulk([_media(p) for p in paths])
        assert len(out) == 2 and all(v is not None for v in out)


class TestEupeBulkOverride:
    def test_bulk_returns_per_input_vector(self, tmp_path):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        emb = ImageEupeSingleEmbedder()
        emb._model = mock.MagicMock()
        emb._preprocess = mock.MagicMock()
        # EUPE's bulk goes through _forward_pil_batch as well — same shortcut.

        def fake_forward(images):
            return np.stack([np.full(3, float(i), dtype=np.float32) for i in range(len(images))])

        emb._forward_pil_batch = fake_forward

        paths = [tmp_path / f"img_{i}.png" for i in range(3)]
        for p in paths:
            _write_image(p)

        out = emb.embed_media_bulk([_media(p) for p in paths])
        assert all(v is not None for v in out)
        assert [v[0] for v in out if v is not None] == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# patch_forward_bulk
# ---------------------------------------------------------------------------


class TestPatchForwardBulkDefault:
    """The ABC's default impl loops :meth:`patch_forward` per item."""

    def test_default_loops_per_item(self):
        from vtscore.media.embedder import MediaEmbedder

        class _Stub(MediaEmbedder):
            @property
            def name(self):
                return "stub"

            @property
            def media_type_id(self):
                return "image"

            @property
            def supports_patch_regions(self):
                return True

            def _load_models_impl(self):
                self._model = True

            def _embed_media_impl(self, media):
                return None

            def _patch_forward_impl(self, media):
                return {"id": media["id"]}

        emb = _Stub()
        emb._model = True
        out = emb.patch_forward_bulk([{"id": 1}, {"id": 2}, {"id": 3}])
        assert out == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_default_emits_progress(self):
        from vtscore.media.embedder import MediaEmbedder

        class _Stub(MediaEmbedder):
            @property
            def name(self):
                return "stub"

            @property
            def media_type_id(self):
                return "image"

            @property
            def supports_patch_regions(self):
                return True

            def _load_models_impl(self):
                self._model = True

            def _embed_media_impl(self, media):
                return None

            def _patch_forward_impl(self, media):
                return None

        emb = _Stub()
        emb._model = True
        events: list[tuple] = []
        emb._on_progress = lambda s, m, c, t: events.append((s, c, t))
        emb.patch_forward_bulk([{}, {}])

        assert events == [("embedding", 1, 2), ("embedding", 2, 2)]


class TestPatchForwardBulkOverrides:
    """The DINOv2/v3 + EUPE patch embedders override the bulk hook."""

    def test_dinov3_patch_routes_through_helper(self, tmp_path):
        from vtscore.media.image.embedder_dinov3_patch import (
            ImageDinov3PatchEmbedder,
        )

        emb = ImageDinov3PatchEmbedder()
        emb._model = mock.MagicMock()
        emb._processor = mock.MagicMock()

        seen: list[int] = []

        def fake_patch_forward(images):
            seen.append(len(images))
            return [{"i": i} for i, _ in enumerate(images)]

        emb._patch_forward_pil_batch = fake_patch_forward  # pyright: ignore[reportAttributeAccessIssue]

        paths = [tmp_path / f"img_{i}.png" for i in range(3)]
        for p in paths:
            _write_image(p)

        out = emb.patch_forward_bulk([_media(p) for p in paths])
        assert seen == [3]  # one batch, all three
        assert len(out) == 3
        assert all(o is not None for o in out)

    def test_dinov2_patch_routes_through_helper(self, tmp_path):
        from vtscore.media.image.embedder_dinov2_patch import (
            ImageDinov2PatchEmbedder,
        )

        emb = ImageDinov2PatchEmbedder()
        emb._model = mock.MagicMock()
        emb._processor = mock.MagicMock()

        def fake_patch_forward(images):
            return [{"i": i} for i, _ in enumerate(images)]

        emb._patch_forward_pil_batch = fake_patch_forward  # pyright: ignore[reportAttributeAccessIssue]

        p = tmp_path / "img.png"
        _write_image(p)
        out = emb.patch_forward_bulk([_media(p)])
        assert len(out) == 1
        assert out[0] == {"i": 0}


# ---------------------------------------------------------------------------
# embed_missing routes patch-region embedders through patch_forward_bulk
# ---------------------------------------------------------------------------


class TestEmbedMissingRoutesToPatchForwardBulk:
    def test_embed_missing_calls_patch_forward_bulk_once(self):
        from vtscore.datasets.load_pipeline import embed_missing

        emb = mock.MagicMock()
        emb.name = "fake_patch"
        emb._model = True
        emb._on_progress = lambda *a, **kw: None
        emb.supports_patch_regions = True
        emb.embed_media_bulk.side_effect = lambda medias: [np.zeros(768, dtype=np.float32) for _ in medias]

        patch_outputs = [mock.MagicMock(patch_grid=np.zeros((4, 4, 768), dtype=np.float32)) for _ in range(3)]
        emb.patch_forward_bulk.return_value = patch_outputs

        medias = {i: {"media_type": "image", "embedding": None, "media_path": f"/tmp/img_{i}.png"} for i in range(1, 4)}

        with (
            mock.patch("vtscore.media.embedders_for_type", return_value=[emb]),
            mock.patch(
                "vtscore.media.patch_embed.build_region_tree", return_value=np.zeros((23, 768), dtype=np.float32)
            ),
            mock.patch("vtscore.media.patch_embed.to_fp16", side_effect=lambda x: x.astype(np.float16)),
        ):
            embed_missing(medias)

        assert emb.patch_forward_bulk.call_count == 1
        sent = emb.patch_forward_bulk.call_args.args[0]
        assert len(sent) == 3
        for m in medias.values():
            assert m.get("patch_regions") is not None
            assert m.get("patch_grid") is not None
