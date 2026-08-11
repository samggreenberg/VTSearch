"""Tests for lazy clips: reference (thin) datasets that store clip recipes.

A reference import keeps a ``media_path`` reference instead of copying bytes
(Phase 1).  Phase 2 (this module) makes *clippers* honour that contract: when a
reference parent is tiled/sliced, the derived clips store no ``media_bytes`` of
their own - just the source path plus the clip boundaries in ``origin.params``
- and reproduce their bytes on demand via ``_resolve_media_bytes``.

See "Reference (no-copy) imports and lazy clips" in ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vtscore.datasets.clipper_chain import _content_hash
from vtscore.datasets.stages import clipper as clipper_stage
from vtscore.media.audio.clipper import _wav_slice
from vtscore.media.lazy_clip import _ByteBoundedLRU, clip_recipe, lazy_clip_bytes

from tests_lib.helpers import make_png_bytes, make_wav_bytes


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_wav(tmp_path: Path, name: str = "src.wav", duration: float = 1.0) -> Path:
    p = tmp_path / name
    p.write_bytes(make_wav_bytes(frequency=440.0, duration=duration))
    return p


def _thin_audio_media(path: Path) -> dict[str, Any]:
    """Build a thin (reference) audio media - no bytes, just a path."""
    return {
        "id": 1,
        "media_type": "audio",
        "media_bytes": None,
        "media_string": None,
        "media_path": str(path),
        "duration": 0,
        "file_size": path.stat().st_size,
        "md5": "deadbeef",
        "filename": path.name,
        "category": "custom",
        "origin": {"importer": "server_folder", "params": {"path": str(path.parent)}},
        "origin_name": path.name,
    }


def _thin_image_media(path: Path) -> dict[str, Any]:
    return {
        "id": 1,
        "media_type": "image",
        "media_bytes": None,
        "media_string": None,
        "media_path": str(path),
        "duration": 0,
        "file_size": path.stat().st_size,
        "md5": "deadbeef",
        "filename": path.name,
        "category": "custom",
        "origin": {"importer": "server_folder", "params": {"path": str(path.parent)}},
        "origin_name": path.name,
    }


@pytest.fixture
def _stub_clip_fixup(monkeypatch):
    """Skip the per-clip MD5/embedding/thumbnail fixup (no embedder needed).

    The lazy-clip mechanism under test (hydrate -> clip -> re-lazify -> resolve)
    is independent of embedding, so we stub the fixup the same way the existing
    clipper-chain tests do.
    """
    monkeypatch.setattr(clipper_stage, "_fixup_clip_md5_and_embeddings", lambda *a, **k: None)
    monkeypatch.setattr(clipper_stage, "_regenerate_clip_thumbnails", lambda *a, **k: None)


def _relazify(clips: dict) -> None:
    from vtscore.state import DatasetContext  # noqa: PLC0415

    ctx = DatasetContext("_test_lazy_clips")
    ctx.medias = clips
    clipper_stage._relazify_reference_clips_stage(ctx)


# ---------------------------------------------------------------------------
# clip_recipe
# ---------------------------------------------------------------------------


class TestClipRecipe:
    def test_audio_recipe_from_origin_params(self):
        media = {"media_type": "audio", "origin": {"params": {"clip_start": "0.5", "clip_end": "1.5"}}}
        assert clip_recipe(media) == ("audio", 0.5, 1.5)

    def test_image_recipe_from_origin_params_string(self):
        media = {"media_type": "image", "origin": {"params": {"clip_box": "0,0,16,16"}}}
        assert clip_recipe(media) == ("image", (0, 0, 16, 16))

    def test_image_recipe_from_list(self):
        media = {"media_type": "image", "origin": {"params": {"clip_box": [1, 2, 3, 4]}}}
        assert clip_recipe(media) == ("image", (1, 2, 3, 4))

    def test_no_recipe_without_boundaries(self):
        media = {"media_type": "audio", "origin": {"params": {}}}
        assert clip_recipe(media) is None

    def test_text_and_video_never_lazy(self):
        assert clip_recipe({"media_type": "text", "origin": {"params": {"clip_index": "2"}}}) is None
        assert clip_recipe({"media_type": "video", "origin": {"params": {"clip_start": "0", "clip_end": "1"}}}) is None

    def test_malformed_box_is_none(self):
        media = {"media_type": "image", "origin": {"params": {"clip_box": "0,0,16"}}}
        assert clip_recipe(media) is None


# ---------------------------------------------------------------------------
# lazy_clip_bytes
# ---------------------------------------------------------------------------


class TestLazyClipBytes:
    def test_audio_slice_matches_wav_slice(self, tmp_path):
        src = _write_wav(tmp_path, duration=1.0)
        media = {
            "media_type": "audio",
            "media_path": str(src),
            "origin": {"params": {"clip_start": "0.2", "clip_end": "0.6"}},
        }
        got = lazy_clip_bytes(media)
        expected = _wav_slice(src.read_bytes(), 0.2, 0.6)
        assert got == expected
        # And the slice is genuinely shorter than the source.
        with wave.open(__import__("io").BytesIO(got), "rb") as wf:
            assert wf.getnframes() < wave.open(str(src), "rb").getnframes()

    def test_image_crop(self, tmp_path):
        src = tmp_path / "src.png"
        src.write_bytes(make_png_bytes(width=32, height=32, color=(10, 20, 30)))
        media = {
            "media_type": "image",
            "media_path": str(src),
            "origin": {"params": {"clip_box": "0,0,16,16"}},
        }
        got = lazy_clip_bytes(media)
        assert got is not None
        from PIL import Image
        import io as _io

        with Image.open(_io.BytesIO(got)) as img:
            assert img.size == (16, 16)

    def test_no_recipe_returns_none(self, tmp_path):
        src = _write_wav(tmp_path)
        media = {"media_type": "audio", "media_path": str(src), "origin": {"params": {}}}
        assert lazy_clip_bytes(media) is None

    def test_missing_source_returns_none(self, tmp_path):
        media = {
            "media_type": "audio",
            "media_path": str(tmp_path / "nope.wav"),
            "origin": {"params": {"clip_start": "0", "clip_end": "0.5"}},
        }
        assert lazy_clip_bytes(media) is None

    def test_cache_returns_identical_object(self, tmp_path):
        src = _write_wav(tmp_path, duration=1.0)
        media = {
            "media_type": "audio",
            "media_path": str(src),
            "origin": {"params": {"clip_start": "0.1", "clip_end": "0.4"}},
        }
        first = lazy_clip_bytes(media)
        second = lazy_clip_bytes(media)
        assert first is second  # served from the process-scoped cache


# ---------------------------------------------------------------------------
# _resolve_media_bytes integration (serve path)
# ---------------------------------------------------------------------------


class TestResolveMediaBytes:
    def test_audio_media_type_resolves_lazy_clip(self, tmp_path):
        import vtscore.media as media_registry

        src = _write_wav(tmp_path, duration=1.0)
        media = {
            "media_type": "audio",
            "media_bytes": None,
            "media_path": str(src),
            "origin": {"params": {"clip_start": "0.3", "clip_end": "0.7"}},
        }
        resolved = media_registry.get("audio")._resolve_media_bytes(media)
        assert resolved == _wav_slice(src.read_bytes(), 0.3, 0.7)

    def test_inline_bytes_take_precedence(self, tmp_path):
        import vtscore.media as media_registry

        src = _write_wav(tmp_path)
        media = {
            "media_type": "audio",
            "media_bytes": b"inline",
            "media_path": str(src),
            "origin": {"params": {"clip_start": "0", "clip_end": "0.1"}},
        }
        assert media_registry.get("audio")._resolve_media_bytes(media) == b"inline"

    def test_whole_file_thin_media_unaffected(self, tmp_path):
        import vtscore.media as media_registry

        src = _write_wav(tmp_path)
        media = {"media_type": "audio", "media_bytes": None, "media_path": str(src), "origin": {"params": {}}}
        assert media_registry.get("audio")._resolve_media_bytes(media) == src.read_bytes()


# ---------------------------------------------------------------------------
# End-to-end: clipper stage produces lazy clips for reference parents
# ---------------------------------------------------------------------------


class TestClipperStageLazyClips:
    def test_thin_audio_tiling_produces_lazy_clips(self, tmp_path, _stub_clip_fixup):
        src = _write_wav(tmp_path, duration=1.2)
        clips = {1: _thin_audio_media(src)}

        clipper_stage._apply_clipper(clips, "sound_tiling", {"duration": 0.3, "min_overlap": 0.0})
        assert len(clips) > 1  # actually tiled

        # Before re-lazify: clips carry the marker + materialized bytes.
        for clip in clips.values():
            assert clip.get("_lazy_source") == str(src)
            assert clip["media_bytes"] is not None

        _relazify(clips)

        for clip in clips.values():
            assert "_lazy_source" not in clip
            assert clip["media_bytes"] is None
            assert clip["media_path"] == str(src)
            params = clip["origin"]["params"]
            assert "clip_start" in params and "clip_end" in params
            # Resolution reproduces the exact slice.
            import vtscore.media as media_registry

            resolved = media_registry.get("audio")._resolve_media_bytes(clip)
            expected = _wav_slice(src.read_bytes(), float(params["clip_start"]), float(params["clip_end"]))
            assert resolved == expected

    def test_thin_image_tiling_produces_lazy_clips(self, tmp_path, _stub_clip_fixup):
        src = tmp_path / "src.png"
        src.write_bytes(make_png_bytes(width=16, height=48, color=(5, 10, 15)))
        clips = {1: _thin_image_media(src)}

        clipper_stage._apply_clipper(clips, "image_tiling", None)
        assert len(clips) > 1

        _relazify(clips)

        import vtscore.media as media_registry

        for clip in clips.values():
            assert clip["media_bytes"] is None
            assert clip["media_path"] == str(src)
            assert "clip_box" in clip["origin"]["params"]
            resolved = media_registry.get("image")._resolve_media_bytes(clip)
            assert resolved is not None
            from PIL import Image
            import io as _io

            with Image.open(_io.BytesIO(resolved)) as img:
                assert img.size == (16, 16)  # square tile from the 16x48 source

    def test_full_mode_audio_not_relazified(self, tmp_path, _stub_clip_fixup):
        """A non-reference parent (bytes present) keeps materialized clip bytes."""
        src = _write_wav(tmp_path, duration=1.2)
        media = _thin_audio_media(src)
        media["media_bytes"] = src.read_bytes()  # full mode
        clips = {1: media}

        clipper_stage._apply_clipper(clips, "sound_tiling", {"duration": 0.3, "min_overlap": 0.0})
        assert len(clips) > 1
        for clip in clips.values():
            assert "_lazy_source" not in clip  # never marked
            assert clip["media_bytes"] is not None

        _relazify(clips)  # no markers -> no-op
        for clip in clips.values():
            assert clip["media_bytes"] is not None

    def test_converter_chain_not_lazified(self, tmp_path):
        """A chain containing a converter changes media type, so its output is
        no longer a slice of the source file; such parents must NOT be
        hydrated/marked for re-lazification."""
        src = _write_wav(tmp_path)
        clips = {1: _thin_audio_media(src)}
        steps = [{"kind": "converter", "name": "audio2image", "params": {}}]
        marked = clipper_stage._hydrate_reference_parents(clips, steps)
        assert marked is False
        assert "_lazy_source" not in clips[1]


# ---------------------------------------------------------------------------
# Pickle round-trip: lazy clips survive save -> reopen and still resolve
# ---------------------------------------------------------------------------


class TestPickleRoundTrip:
    def test_lazy_clip_survives_round_trip(self, tmp_path, _stub_clip_fixup):
        from vtscore.datasets.loader import export_dataset_to_file, load_dataset_from_pickle

        src = _write_wav(tmp_path, duration=1.2)
        clips = {1: _thin_audio_media(src)}
        clipper_stage._apply_clipper(clips, "sound_tiling", {"duration": 0.3, "min_overlap": 0.0})
        _relazify(clips)

        # Attach embeddings so the loader keeps the entries (a clip with no
        # usable vector is dropped on load).
        rng = np.random.default_rng(0)
        for clip in clips.values():
            clip["embeddings"] = {"stub": rng.standard_normal(8).astype(np.float32)}
            clip["embedder"] = "stub"

        pkl = tmp_path / "ds.pkl"
        pkl.write_bytes(export_dataset_to_file(clips, media_type="audio"))

        reopened: dict = {}
        load_dataset_from_pickle(pkl, reopened)
        assert len(reopened) == len(clips)

        import vtscore.media as media_registry

        for clip in reopened.values():
            assert clip["media_bytes"] is None
            assert clip["media_path"] == str(src)
            params = clip["origin"]["params"]
            resolved = media_registry.get("audio")._resolve_media_bytes(clip)
            expected = _wav_slice(src.read_bytes(), float(params["clip_start"]), float(params["clip_end"]))
            assert resolved == expected


# ---------------------------------------------------------------------------
# Phase 2b: lazy converter output
# ---------------------------------------------------------------------------


class _FakeDoc2Image:
    """A deterministic stand-in for a real converter (no heavy deps).

    Produces one image output per "page", whose bytes are derived from the
    source bytes so a test can predict the exact output for any out_index.
    """

    name = "doc2img"
    source_type = "document"
    target_type = "image"

    def __init__(self, n_pages: int = 3) -> None:
        self.n_pages = n_pages

    def convert_normalized(self, media: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
        src = media.get("media_bytes") or b""
        n = int(params.get("pages", self.n_pages))
        return [
            {"filename": f"page_{i}.png", "media_bytes": src + f"|page{i}".encode(), "duration": 0} for i in range(n)
        ]


def _page_bytes(source: bytes, idx: int) -> bytes:
    return source + f"|page{idx}".encode()


def _converter_media(
    source_path: Path,
    *,
    out_index: int,
    n_out: int,
    content_hash: str | None,
    params: dict[str, Any] | None = None,
    name: str = "doc2img",
    target_type: str = "image",
) -> dict[str, Any]:
    """Build a byte-less reference-converted media (the re-lazified form)."""
    p: dict[str, Any] = {
        "converter": name,
        "source_file": source_path.name,
        "converter_out_index": str(out_index),
        "converter_n_out": str(n_out),
    }
    if content_hash is not None:
        p["converter_content_hash"] = content_hash
    for k, v in (params or {}).items():
        p[f"converter_param_{k}"] = str(v)
    return {
        "media_type": target_type,
        "media_bytes": None,
        "media_string": None,
        "media_path": str(source_path),
        "filename": f"{source_path.stem}_page.png",
        "origin": {"importer": "converter", "params": p},
    }


@pytest.fixture
def _fake_converter(monkeypatch):
    fake = _FakeDoc2Image(n_pages=3)
    monkeypatch.setattr("vtscore.converters.get_converter", lambda name: fake if name == "doc2img" else None)
    return fake


class TestConverterRecipe:
    def test_recipe_recognised_by_converter_param(self):
        media = {
            "media_type": "image",
            "origin": {"params": {"converter": "doc2img", "converter_out_index": "1", "converter_n_out": "3"}},
        }
        recipe = clip_recipe(media)
        assert recipe is not None
        assert recipe[0] == "converter"
        assert recipe[1] == "doc2img"
        assert recipe[3] == 1  # out_index coerced to int

    def test_converter_params_reconstructed(self):
        media = {
            "media_type": "text",
            "origin": {
                "params": {
                    "converter": "doc2img",
                    "converter_out_index": "0",
                    "converter_param_pages": "5",
                    "converter_param_dpi": "150",
                }
            },
        }
        recipe = clip_recipe(media)
        assert recipe is not None
        assert dict(recipe[2]) == {"pages": "5", "dpi": "150"}

    def test_no_disambiguator_is_not_lazy(self):
        # A converter origin with neither out_index nor content_hash can't be
        # resolved back to a specific output, so it is treated as not-lazy.
        media = {"media_type": "image", "origin": {"params": {"converter": "doc2img"}}}
        assert clip_recipe(media) is None

    def test_converter_branch_beats_image_clip_branch(self):
        # A converted image carries media_type=image but is NOT an image clip
        # (no clip_box); the converter branch must win.
        media = {
            "media_type": "image",
            "origin": {"params": {"converter": "doc2img", "converter_out_index": "2", "converter_n_out": "3"}},
        }
        recipe = clip_recipe(media)
        assert recipe is not None
        assert recipe[0] == "converter"


class TestLazyConverterBytes:
    def test_resolves_correct_page(self, tmp_path, _fake_converter):
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"PDFBYTES")
        ch = _content_hash({"media_bytes": _page_bytes(b"PDFBYTES", 1)})
        media = _converter_media(src, out_index=1, n_out=3, content_hash=ch, params={"pages": 3})
        assert lazy_clip_bytes(media) == _page_bytes(b"PDFBYTES", 1)

    def test_resolve_via_media_type(self, tmp_path, _fake_converter):
        import vtscore.media as media_registry

        src = tmp_path / "doc2.pdf"
        src.write_bytes(b"OTHER")
        ch = _content_hash({"media_bytes": _page_bytes(b"OTHER", 0)})
        media = _converter_media(src, out_index=0, n_out=3, content_hash=ch, params={"pages": 3})
        # _resolve_media_bytes consults lazy_clip before the whole-file read,
        # so the serve path gets the rendered page, not the source PDF.
        resolved = media_registry.get("image")._resolve_media_bytes(media)
        assert resolved == _page_bytes(b"OTHER", 0)

    def test_content_hash_survives_output_count_drift(self, tmp_path, monkeypatch):
        # Recorded with n_out=3; converter now yields 5 outputs. The content
        # hash still pins the right page (drift falls back to content match).
        fake = _FakeDoc2Image(n_pages=5)
        monkeypatch.setattr("vtscore.converters.get_converter", lambda name: fake if name == "doc2img" else None)
        src = tmp_path / "drift.pdf"
        src.write_bytes(b"DRIFT")
        ch = _content_hash({"media_bytes": _page_bytes(b"DRIFT", 2)})
        media = _converter_media(src, out_index=2, n_out=3, content_hash=ch, params={"pages": 5})
        assert lazy_clip_bytes(media) == _page_bytes(b"DRIFT", 2)

    def test_refuses_to_guess_on_drift_without_hash(self, tmp_path, monkeypatch):
        # No content hash + output-count drift => positional pick is unsafe,
        # so resolution returns None rather than the wrong page.
        fake = _FakeDoc2Image(n_pages=5)
        monkeypatch.setattr("vtscore.converters.get_converter", lambda name: fake if name == "doc2img" else None)
        src = tmp_path / "drift2.pdf"
        src.write_bytes(b"DRIFT2")
        media = _converter_media(src, out_index=2, n_out=3, content_hash=None, params={"pages": 5})
        assert lazy_clip_bytes(media) is None

    def test_unknown_converter_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("vtscore.converters.get_converter", lambda name: None)
        src = tmp_path / "x.pdf"
        src.write_bytes(b"X")
        media = _converter_media(src, out_index=0, n_out=1, content_hash=None)
        assert lazy_clip_bytes(media) is None

    def test_cache_returns_identical_object(self, tmp_path, _fake_converter):
        src = tmp_path / "cache.pdf"
        src.write_bytes(b"CACHEME")
        ch = _content_hash({"media_bytes": _page_bytes(b"CACHEME", 0)})
        media = _converter_media(src, out_index=0, n_out=3, content_hash=ch, params={"pages": 3})
        first = lazy_clip_bytes(media)
        second = lazy_clip_bytes(media)
        assert first is second  # served from the byte-bounded converter cache


class TestRelazifyConverterOutput:
    def test_relazify_strips_converted_bytes(self, tmp_path, _fake_converter):
        """A converter output tagged with _lazy_source is stripped to a
        reference by the re-lazify stage, then resolves on demand."""
        import vtscore.media as media_registry

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"PDFBYTES")
        ch = _content_hash({"media_bytes": _page_bytes(b"PDFBYTES", 2)})
        media = _converter_media(src, out_index=2, n_out=3, content_hash=ch, params={"pages": 3})
        # Simulate the pre-relazify state: bytes present + marker set.
        media["media_bytes"] = _page_bytes(b"PDFBYTES", 2)
        media["_lazy_source"] = str(src)

        _relazify({1: media})

        assert "_lazy_source" not in media
        assert media["media_bytes"] is None
        assert media["media_path"] == str(src)
        resolved = media_registry.get("image")._resolve_media_bytes(media)
        assert resolved == _page_bytes(b"PDFBYTES", 2)


class TestByteBoundedLRU:
    def test_evicts_oldest_until_under_ceiling(self):
        cache = _ByteBoundedLRU(max_bytes=10)
        cache.put(("a",), b"xxxxx")  # 5 bytes
        cache.put(("b",), b"yyyyy")  # 10 total
        assert cache.get(("a",)) == b"xxxxx"
        cache.put(("c",), b"zzzzz")  # 15 -> evict oldest (b, since a was just touched)
        assert cache.get(("b",)) is None
        assert cache.get(("a",)) == b"xxxxx"
        assert cache.get(("c",)) == b"zzzzz"

    def test_keeps_single_oversized_entry(self):
        cache = _ByteBoundedLRU(max_bytes=4)
        big = b"0123456789"  # 10 bytes > ceiling
        cache.put(("big",), big)
        # The just-inserted entry is kept so the immediate fetch hits.
        assert cache.get(("big",)) == big
        # A later insert evicts the oversized entry.
        cache.put(("small",), b"ab")
        assert cache.get(("big",)) is None
        assert cache.get(("small",)) == b"ab"

    def test_update_existing_key_adjusts_total(self):
        cache = _ByteBoundedLRU(max_bytes=10)
        cache.put(("k",), b"xxxxx")
        cache.put(("k",), b"yy")  # replace; total should be 2, not 7
        cache.put(("j",), b"zzzzzzz")  # 2 + 7 = 9 <= 10, both fit
        assert cache.get(("k",)) == b"yy"
        assert cache.get(("j",)) == b"zzzzzzz"
