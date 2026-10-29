"""Tests for the batched ``_embed_media_bulk_impl`` overrides on image embedders.

The overrides themselves are model-free; they delegate to the shared
helper in ``vtscore.media.image._image_bulk`` and a per-embedder
``_forward_pil_batch`` callable.  These tests stub the model + processor
so we can assert:

* The bulk path decodes each file path to a PIL image and batches the
  GPU forward in chunks of ``embed_batch_size``.
* Per-image PIL-decode failures don't kill the batch.
* The returned list aligns position-for-position with the input list.
* The decode is threaded and runs one batch *ahead* of the forward, so
  the GPU is not idle while PIL works, and the overlap does not disturb
  slot alignment.
* The same plumbing works for ``patch_forward_bulk`` on the patch
  embedders.

No model weights are loaded; these tests run on CPU in milliseconds.
"""

from __future__ import annotations

import threading
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
# Threaded, one-batch-ahead decode
# ---------------------------------------------------------------------------


def _record_decode_threads(monkeypatch, module) -> set[str]:
    """Patch ``_load_pil`` to record which thread each decode ran on."""
    threads: set[str] = set()
    lock = threading.Lock()
    real_load = module._load_pil

    def recording_load(source):
        with lock:
            threads.add(threading.current_thread().name)
        return real_load(source)

    monkeypatch.setattr(module, "_load_pil", recording_load)
    return threads


class TestDecodePrefetch:
    """The decode overlaps the forward instead of blocking in front of it."""

    def test_next_batch_decodes_during_the_current_forward(self, tmp_path, monkeypatch):
        """The whole point: the GPU forward and the next decode run together.

        The forward for batch 0 blocks until *every* image has been decoded.
        With the decode inline that can never happen - batch 1 is only read
        after the forward returns - so a serial implementation fails on the
        wait timeout rather than by an assertion on timing.
        """
        from vtscore.media.image import _image_bulk

        paths = [tmp_path / f"img_{i}.png" for i in range(4)]
        for p in paths:
            _write_image(p)
        medias = [_media(p) for p in paths]

        monkeypatch.setenv("VTSEARCH_DECODE_WORKERS", "4")

        all_decoded = threading.Event()
        seen = 0
        lock = threading.Lock()
        real_load = _image_bulk._load_pil

        def counting_load(source):
            img = real_load(source)
            nonlocal seen
            with lock:
                seen += 1
                if seen == len(paths):
                    all_decoded.set()
            return img

        monkeypatch.setattr(_image_bulk, "_load_pil", counting_load)

        overlapped: list[bool] = []

        def fake_forward(images):
            overlapped.append(all_decoded.wait(timeout=10))
            return np.zeros((len(images), 1), dtype=np.float32)

        out = _image_bulk.bulk_embed_image_files(
            medias,
            forward_pil_batch=fake_forward,
            batch_size=2,
            on_progress=lambda *a, **kw: None,
            label="test",
        )

        # Batch 1 finished decoding while batch 0's forward was still running.
        assert overlapped[0] is True
        assert all(v is not None for v in out)

    def test_decode_runs_off_the_calling_thread(self, tmp_path, monkeypatch):
        from vtscore.media.image import _image_bulk

        paths = [tmp_path / f"img_{i}.png" for i in range(3)]
        for p in paths:
            _write_image(p)

        monkeypatch.setenv("VTSEARCH_DECODE_WORKERS", "2")
        threads = _record_decode_threads(monkeypatch, _image_bulk)

        _image_bulk.bulk_embed_image_files(
            [_media(p) for p in paths],
            forward_pil_batch=lambda imgs: np.zeros((len(imgs), 1), dtype=np.float32),
            batch_size=2,
            on_progress=lambda *a, **kw: None,
            label="test",
        )

        assert threading.current_thread().name not in threads

    def test_zero_workers_decodes_inline(self, tmp_path, monkeypatch):
        """``VTSEARCH_DECODE_WORKERS=0`` is the escape hatch back to serial."""
        from vtscore.media.image import _image_bulk

        paths = [tmp_path / f"img_{i}.png" for i in range(3)]
        for p in paths:
            _write_image(p)

        monkeypatch.setenv("VTSEARCH_DECODE_WORKERS", "0")
        threads = _record_decode_threads(monkeypatch, _image_bulk)

        out = _image_bulk.bulk_embed_image_files(
            [_media(p) for p in paths],
            forward_pil_batch=lambda imgs: np.zeros((len(imgs), 1), dtype=np.float32),
            batch_size=2,
            on_progress=lambda *a, **kw: None,
            label="test",
        )

        assert threads == {threading.current_thread().name}
        assert all(v is not None for v in out)

    def test_out_of_order_completion_keeps_slot_alignment(self, tmp_path, monkeypatch):
        """A slow first decode must not shuffle images against their slots.

        Image *i* is a solid grey of level ``40 * i``, and the fake forward
        reads that level back out, so a mis-ordered batch is visible in the
        output rather than merely possible. Image 0's decode is held until a
        later one finishes, which forces the completion order to differ from
        the submission order.
        """
        from vtscore.media.image import _image_bulk

        levels = [0, 40, 80, 120]
        paths = []
        for i, level in enumerate(levels):
            p = tmp_path / f"img_{i}.png"
            _write_image(p, color=(level, level, level))
            paths.append(p)

        monkeypatch.setenv("VTSEARCH_DECODE_WORKERS", "4")

        released = threading.Event()
        real_load = _image_bulk._load_pil

        def staggered_load(source):
            first = Path(str(source)).name == "img_0.png"
            if first:
                # Blocks until one of the later decodes lands, so image 0's
                # future is the last to resolve.
                released.wait(timeout=10)
            img = real_load(source)
            if not first:
                released.set()
            return img

        monkeypatch.setattr(_image_bulk, "_load_pil", staggered_load)

        def fake_forward(images):
            return np.array([[float(im.getpixel((0, 0))[0])] for im in images], dtype=np.float32)

        out = _image_bulk.bulk_embed_image_files(
            [_media(p) for p in paths],
            forward_pil_batch=fake_forward,
            batch_size=4,
            on_progress=lambda *a, **kw: None,
            label="test",
        )

        assert [float(v[0]) for v in out if v is not None] == [float(x) for x in levels]

    def test_threaded_decode_failure_still_lands_in_its_own_slot(self, tmp_path, monkeypatch):
        from vtscore.media.image import _image_bulk

        good_a = tmp_path / "a.png"
        good_b = tmp_path / "b.png"
        _write_image(good_a)
        _write_image(good_b)
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not a real image")

        monkeypatch.setenv("VTSEARCH_DECODE_WORKERS", "4")

        out = _image_bulk.bulk_embed_image_files(
            [_media(good_a), _media(bad), _media(good_b)],
            forward_pil_batch=lambda imgs: np.ones((len(imgs), 2), dtype=np.float32),
            batch_size=4,
            on_progress=lambda *a, **kw: None,
            label="test",
        )

        assert [v is None for v in out] == [False, True, False]

    def test_patch_forward_bulk_also_prefetches(self, tmp_path, monkeypatch):
        from vtscore.media.image import _image_bulk

        paths = [tmp_path / f"img_{i}.png" for i in range(3)]
        for p in paths:
            _write_image(p)

        monkeypatch.setenv("VTSEARCH_DECODE_WORKERS", "2")
        threads = _record_decode_threads(monkeypatch, _image_bulk)

        out = _image_bulk.bulk_patch_forward_image_files(
            [_media(p) for p in paths],
            forward_pil_batch=lambda imgs: [{"i": i} for i, _ in enumerate(imgs)],
            batch_size=2,
            on_progress=lambda *a, **kw: None,
            label="test",
        )

        assert threading.current_thread().name not in threads
        assert len(out) == 3 and all(o is not None for o in out)


# ---------------------------------------------------------------------------
# Concrete embedder overrides: wired up but model-free.
# ---------------------------------------------------------------------------


def _stub_image_embedder(emb, dim: int = 8):
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
        # One-hot per slot so the slot index is recoverable via argmax even
        # after the base wrapper L2-normalizes the result (a constant-
        # magnitude vector would collapse to one shared direction).
        out = np.zeros((len(images), dim), dtype=np.float32)
        for i in range(len(images)):
            out[i, i] = 1.0
        return out

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
        # One-hot per slot confirms slot alignment (argmax survives L2-norm).
        assert [int(np.argmax(v)) for v in out if v is not None] == [0, 1, 2]

    def test_routes_through_bulk_helper(self, tmp_path):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        _stub_image_embedder(emb)

        p = tmp_path / "img.png"
        _write_image(p)

        with mock.patch(
            "vtscore.media.image._cross_modal_shared.bulk_embed_image_files",
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
        assert [int(np.argmax(v)) for v in out if v is not None] == [0, 1]


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
        # EUPE's bulk goes through _forward_pil_batch as well; same shortcut.

        def fake_forward(images):
            # One-hot per slot so argmax recovers the slot after L2-norm.
            out = np.zeros((len(images), 3), dtype=np.float32)
            for i in range(len(images)):
                out[i, i] = 1.0
            return out

        emb._forward_pil_batch = fake_forward

        paths = [tmp_path / f"img_{i}.png" for i in range(3)]
        for p in paths:
            _write_image(p)

        out = emb.embed_media_bulk([_media(p) for p in paths])
        assert all(v is not None for v in out)
        assert [int(np.argmax(v)) for v in out if v is not None] == [0, 1, 2]


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
        from vtscore.datasets.stages.embedding import embed_missing

        emb = mock.MagicMock()
        emb.name = "fake_patch"
        emb._model = True
        emb._on_progress = lambda *a, **kw: None
        emb.supports_patch_regions = True
        emb.embed_media_bulk.side_effect = lambda medias: [np.zeros(768, dtype=np.float32) for _ in medias]

        patch_outputs = [mock.MagicMock(patch_grid=np.zeros((4, 4, 768), dtype=np.float32)) for _ in range(3)]
        emb.patch_forward_bulk.return_value = patch_outputs

        medias = {i: {"media_type": "image", "embeddings": {}, "media_path": f"/tmp/img_{i}.png"} for i in range(1, 4)}

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias)

        assert emb.patch_forward_bulk.call_count == 1
        sent = emb.patch_forward_bulk.call_args.args[0]
        assert len(sent) == 3
        for m in medias.values():
            assert m.get("patch_grid") is not None
            # The HAC tree ingest used to build alongside it is gone (#2886).
            assert m.get("patch_regions") is None

    def test_embed_missing_backfills_patch_grid_for_preembedded(self):
        """Already-embedded images that lack a patch grid still get the patch
        pass.

        Regression: the patch-region pass used to be gated on the image-level
        ``missing`` set, so a dataset that arrived already-embedded (pickle /
        content-vector importer) but without ``patch_grid`` never ran the
        patch pass.  The best-match highlight then had no region to draw even
        though the embedder reported ``supports_patch_regions``.
        """
        from vtscore.datasets.stages.embedding import embed_missing

        emb = mock.MagicMock()
        emb.name = "fake_patch"
        emb._model = True
        emb._on_progress = lambda *a, **kw: None
        emb.supports_patch_regions = True

        patch_outputs = [mock.MagicMock(patch_grid=np.zeros((4, 4, 768), dtype=np.float32)) for _ in range(2)]
        emb.patch_forward_bulk.return_value = patch_outputs

        # Every media already carries an embedding (nothing in the "missing"
        # set) but no patch_grid yet.
        medias = {
            i: {
                "media_type": "image",
                "embeddings": {"fake_patch": np.zeros(768, dtype=np.float32)},
                "embedder": "fake_patch",
                "media_path": f"/tmp/img_{i}.png",
            }
            for i in range(1, 3)
        }

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias)

        # No image-level embedding work (all were embedded already)...
        emb.embed_media_bulk.assert_not_called()
        # ...but the patch pass still ran over the region-less images.
        assert emb.patch_forward_bulk.call_count == 1
        sent = emb.patch_forward_bulk.call_args.args[0]
        assert len(sent) == 2
        for m in medias.values():
            assert m.get("patch_grid") is not None
            # The HAC tree ingest used to build alongside it is gone (#2886).
            assert m.get("patch_regions") is None

    def test_backfill_resolves_stored_embedder_over_single_vector_default(self):
        """Reloading a patch dataset with no explicit embedder still back-fills.

        Regression: ``embed_missing`` resolved the embedder from the requested
        name or the *media-type default*, which for images is the single-vector
        SigLIP embedder (``supports_patch_regions=False``).  A pre-embedded
        patch dataset (cached pickle reload / content-vector importer) loaded
        with ``embedder_name=""`` therefore resolved to SigLIP and skipped the
        patch-region back-fill entirely - so region voting and the best-match
        highlight had no region data, and nothing rendered.  The resolver now
        falls back to the embedder the media were embedded with (stored on each
        media dict) before the media-type default.
        """
        from vtscore.datasets.stages.embedding import embed_missing

        # The patch embedder the media were originally embedded with...
        patch_emb = mock.MagicMock()
        patch_emb.name = "fake_patch"
        patch_emb._model = True
        patch_emb._on_progress = lambda *a, **kw: None
        patch_emb.supports_patch_regions = True
        patch_emb.patch_forward_bulk.return_value = [
            mock.MagicMock(patch_grid=np.zeros((4, 4, 768), dtype=np.float32)) for _ in range(2)
        ]

        # ...and the single-vector embedder that is the media-type default.
        default_single = mock.MagicMock()
        default_single.name = "fake_single"
        default_single.supports_patch_regions = False

        medias = {
            i: {
                "media_type": "image",
                "embeddings": {"fake_patch": np.zeros(768, dtype=np.float32)},
                "embedder": "fake_patch",
                "media_path": f"/tmp/img_{i}.png",
            }
            for i in range(1, 3)
        }

        def _get_embedder(name):
            if name == "fake_patch":
                return patch_emb
            raise KeyError(name)

        with (
            mock.patch("vtscore.media.get_embedder", side_effect=_get_embedder),
            mock.patch("vtscore.media.embedders_for_type", return_value=[default_single]),
        ):
            # No explicit embedder: must resolve to the stored "fake_patch",
            # not the single-vector default.
            embed_missing(medias, embedder_name="")

        # The single-vector default never got a look-in for the patch pass.
        default_single.patch_forward_bulk.assert_not_called()
        assert patch_emb.patch_forward_bulk.call_count == 1
        for m in medias.values():
            assert m.get("patch_grid") is not None
            # The HAC tree ingest used to build alongside it is gone (#2886).
            assert m.get("patch_regions") is None
