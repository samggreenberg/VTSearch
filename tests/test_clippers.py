"""Tests for the MediaClipper base class and all built-in clippers."""

import io
import wave

import pytest

from vtsearch.audio import generate_wav
from vtsearch.media.base import MediaClipper


# ---------------------------------------------------------------------------
# MediaClipper ABC
# ---------------------------------------------------------------------------


class TestMediaClipperABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            MediaClipper()

    def test_to_dict_on_concrete(self):
        from vtsearch.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        d = c.to_dict()
        assert d == {"name": "sound_default", "media_type": "audio"}


# ---------------------------------------------------------------------------
# SoundDefaultClipper
# ---------------------------------------------------------------------------


class TestSoundDefaultClipper:
    def test_returns_media_unchanged(self):
        from vtsearch.media.audio.clipper import SoundDefaultClipper

        media = {"id": 1, "type": "audio", "media_bytes": b"fake", "duration": 3.0}
        result = SoundDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtsearch.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        assert c.name == "sound_default"
        assert c.media_type == "audio"
        assert isinstance(c, MediaClipper)


# ---------------------------------------------------------------------------
# SoundTilingClipper
# ---------------------------------------------------------------------------


class TestSoundTilingClipper:
    def test_identity(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        assert c.name == "sound_tiling_2.0s"
        assert c.media_type == "audio"
        assert c.duration == 2.0
        assert isinstance(c, MediaClipper)

    def test_rejects_non_positive_duration(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        with pytest.raises(ValueError):
            SoundTilingClipper(0)
        with pytest.raises(ValueError):
            SoundTilingClipper(-1)

    def test_short_audio_returned_unchanged(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        wav = generate_wav(440, 1.5)
        media = {"id": 1, "type": "audio", "media_bytes": wav, "duration": 1.5}
        result = SoundTilingClipper(2.0).clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_tiles_longer_audio(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        wav = generate_wav(440, 5.0)
        media = {"id": 1, "type": "audio", "media_bytes": wav, "duration": 5.0}
        result = SoundTilingClipper(2.0).clip(media)
        # 5.0 / 2.0 = 2.5 → ceil → 3 tiles
        assert len(result) == 3
        for idx, tile in enumerate(result):
            assert tile["clip_index"] == idx
            assert "clip_start" in tile
            assert "clip_end" in tile
            assert tile["duration"] == pytest.approx(2.0, abs=0.01)
            # Each tile should be valid WAV bytes
            with wave.open(io.BytesIO(tile["media_bytes"]), "rb") as wf:
                assert wf.getframerate() == 48000

    def test_9_5s_produces_five_2s_tiles(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        wav = generate_wav(440, 9.5)
        media = {"id": 1, "type": "audio", "media_bytes": wav, "duration": 9.5}
        result = SoundTilingClipper(2.0).clip(media)
        # 9.5 / 2.0 = 4.75 → round → 5 tiles
        assert len(result) == 5
        # First tile starts at 0
        assert result[0]["clip_start"] == pytest.approx(0.0)
        # Last tile ends at 9.5
        assert result[-1]["clip_end"] == pytest.approx(9.5, abs=0.01)

    def test_no_media_bytes_returns_unchanged(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        media = {"id": 1, "type": "audio", "duration": 10.0}
        result = SoundTilingClipper(2.0).clip(media)
        assert result == [media]

    def test_to_dict_includes_duration(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(3.5)
        d = c.to_dict()
        assert d["name"] == "sound_tiling_3.5s"
        assert d["media_type"] == "audio"
        assert d["duration"] == 3.5


# ---------------------------------------------------------------------------
# VideoDefaultClipper
# ---------------------------------------------------------------------------


class TestVideoDefaultClipper:
    def test_returns_media_unchanged(self):
        from vtsearch.media.video.clipper import VideoDefaultClipper

        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 10.0}
        result = VideoDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtsearch.media.video.clipper import VideoDefaultClipper

        c = VideoDefaultClipper()
        assert c.name == "video_default"
        assert c.media_type == "video"
        assert isinstance(c, MediaClipper)


# ---------------------------------------------------------------------------
# VideoTilingClipper
# ---------------------------------------------------------------------------


class TestVideoTilingClipper:
    def test_identity(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(2.0)
        assert c.name == "video_tiling_2.0s"
        assert c.media_type == "video"
        assert c.duration == 2.0
        assert isinstance(c, MediaClipper)

    def test_rejects_non_positive_duration(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        with pytest.raises(ValueError):
            VideoTilingClipper(0)
        with pytest.raises(ValueError):
            VideoTilingClipper(-1)

    def test_short_video_returned_unchanged(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 1.5}
        result = VideoTilingClipper(2.0).clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_9_5s_produces_five_2s_tiles(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 9.5}
        result = VideoTilingClipper(2.0).clip(media)
        assert len(result) == 5
        assert result[0]["clip_start"] == pytest.approx(0.0)
        assert result[-1]["clip_end"] == pytest.approx(9.5, abs=0.01)
        for idx, tile in enumerate(result):
            assert tile["clip_index"] == idx
            assert tile["duration"] == pytest.approx(2.0)

    def test_exact_multiple_duration(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 10.0}
        result = VideoTilingClipper(2.0).clip(media)
        assert len(result) == 5
        # No overlap when duration is exact multiple
        assert result[0]["clip_start"] == pytest.approx(0.0)
        assert result[-1]["clip_end"] == pytest.approx(10.0)

    def test_to_dict_includes_duration(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(3.5)
        d = c.to_dict()
        assert d["name"] == "video_tiling_3.5s"
        assert d["media_type"] == "video"
        assert d["duration"] == 3.5

    def test_zero_duration_video_returned_unchanged(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 0}
        result = VideoTilingClipper(2.0).clip(media)
        assert result == [media]


# ---------------------------------------------------------------------------
# ImageDefaultClipper
# ---------------------------------------------------------------------------


class TestImageDefaultClipper:
    def test_returns_media_unchanged(self):
        from vtsearch.media.image.clipper import ImageDefaultClipper

        media = {"id": 1, "type": "image", "media_bytes": b"fake", "width": 100, "height": 100}
        result = ImageDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtsearch.media.image.clipper import ImageDefaultClipper

        c = ImageDefaultClipper()
        assert c.name == "image_default"
        assert c.media_type == "image"
        assert isinstance(c, MediaClipper)


# ---------------------------------------------------------------------------
# ImageTilingClipper
# ---------------------------------------------------------------------------


def _make_image_bytes(width, height, fmt="PNG"):
    """Helper to create a simple solid-colour image as bytes."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestImageTilingClipper:
    def test_identity(self):
        from vtsearch.media.image.clipper import ImageTilingClipper

        c = ImageTilingClipper()
        assert c.name == "image_tiling"
        assert c.media_type == "image"
        assert isinstance(c, MediaClipper)

    def test_square_image_returned_unchanged(self):
        from vtsearch.media.image.clipper import ImageTilingClipper

        img_bytes = _make_image_bytes(100, 100)
        media = {"id": 1, "type": "image", "media_bytes": img_bytes, "width": 100, "height": 100}
        result = ImageTilingClipper().clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_portrait_image_tiled_vertically(self):
        from PIL import Image

        from vtsearch.media.image.clipper import ImageTilingClipper

        # 100 x 250 image → tile_size = 100, ceil(250/100) = 3 tiles
        img_bytes = _make_image_bytes(100, 250)
        media = {"id": 1, "type": "image", "media_bytes": img_bytes, "width": 100, "height": 250}
        result = ImageTilingClipper().clip(media)
        assert len(result) == 3
        for tile in result:
            assert tile["width"] == 100
            assert tile["height"] == 100
            img = Image.open(io.BytesIO(tile["media_bytes"]))
            assert img.size == (100, 100)
            assert "clip_index" in tile
            assert "clip_box" in tile

    def test_landscape_image_tiled_horizontally(self):
        from PIL import Image

        from vtsearch.media.image.clipper import ImageTilingClipper

        # 300 x 100 image → tile_size = 100, 300/100 = 3 → 3 tiles
        img_bytes = _make_image_bytes(300, 100)
        media = {"id": 1, "type": "image", "media_bytes": img_bytes, "width": 300, "height": 100}
        result = ImageTilingClipper().clip(media)
        assert len(result) == 3
        for tile in result:
            assert tile["width"] == 100
            assert tile["height"] == 100
            img = Image.open(io.BytesIO(tile["media_bytes"]))
            assert img.size == (100, 100)

    def test_8_5_by_11_produces_two_tiles(self):
        """An 8.5x11 (scaled to 85x110) yields two 85x85 tiles."""
        from PIL import Image

        from vtsearch.media.image.clipper import ImageTilingClipper

        # 85 x 110 portrait: tile_size = 85, ceil(110/85) = 2 tiles
        img_bytes = _make_image_bytes(85, 110)
        media = {"id": 1, "type": "image", "media_bytes": img_bytes, "width": 85, "height": 110}
        result = ImageTilingClipper().clip(media)
        assert len(result) == 2
        # First tile at top, second at bottom
        assert result[0]["clip_box"][1] == 0  # y=0
        assert result[1]["clip_box"][3] == 110  # y2 = height
        for tile in result:
            assert tile["width"] == 85
            assert tile["height"] == 85
            img = Image.open(io.BytesIO(tile["media_bytes"]))
            assert img.size == (85, 85)

    def test_no_media_bytes_returns_unchanged(self):
        from vtsearch.media.image.clipper import ImageTilingClipper

        media = {"id": 1, "type": "image", "width": 100, "height": 200}
        result = ImageTilingClipper().clip(media)
        assert result == [media]

    def test_missing_dimensions_returns_unchanged(self):
        from vtsearch.media.image.clipper import ImageTilingClipper

        media = {"id": 1, "type": "image", "media_bytes": b"fake"}
        result = ImageTilingClipper().clip(media)
        assert result == [media]


# ---------------------------------------------------------------------------
# TextDefaultClipper
# ---------------------------------------------------------------------------


class TestTextDefaultClipper:
    def test_returns_media_unchanged(self):
        from vtsearch.media.text.clipper import TextDefaultClipper

        media = {"id": 1, "type": "paragraph", "media_string": "Hello world."}
        result = TextDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtsearch.media.text.clipper import TextDefaultClipper

        c = TextDefaultClipper()
        assert c.name == "text_default"
        assert c.media_type == "paragraph"
        assert isinstance(c, MediaClipper)


# ---------------------------------------------------------------------------
# TextSentenceClipper
# ---------------------------------------------------------------------------


class TestTextSentenceClipper:
    def test_identity(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        c = TextSentenceClipper()
        assert c.name == "text_sentence"
        assert c.media_type == "paragraph"
        assert isinstance(c, MediaClipper)

    def test_single_sentence_unchanged(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        media = {"id": 1, "type": "paragraph", "media_string": "Hello world."}
        result = TextSentenceClipper().clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_splits_multiple_sentences(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        text = "First sentence. Second sentence. Third one!"
        media = {"id": 1, "type": "paragraph", "media_string": text, "word_count": 7, "character_count": len(text)}
        result = TextSentenceClipper().clip(media)
        assert len(result) == 3
        assert result[0]["media_string"] == "First sentence."
        assert result[1]["media_string"] == "Second sentence."
        assert result[2]["media_string"] == "Third one!"
        for idx, tile in enumerate(result):
            assert tile["clip_index"] == idx
            assert tile["word_count"] == len(tile["media_string"].split())
            assert tile["character_count"] == len(tile["media_string"])

    def test_question_and_exclamation(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        text = "Is this a test? Yes it is! Great."
        media = {"id": 1, "type": "paragraph", "media_string": text}
        result = TextSentenceClipper().clip(media)
        assert len(result) == 3
        assert result[0]["media_string"] == "Is this a test?"
        assert result[1]["media_string"] == "Yes it is!"
        assert result[2]["media_string"] == "Great."

    def test_empty_string_returns_unchanged(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        media = {"id": 1, "type": "paragraph", "media_string": ""}
        result = TextSentenceClipper().clip(media)
        assert result == [media]

    def test_no_media_string_returns_unchanged(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        media = {"id": 1, "type": "paragraph"}
        result = TextSentenceClipper().clip(media)
        assert result == [media]


# ---------------------------------------------------------------------------
# DocumentDefaultClipper
# ---------------------------------------------------------------------------


class TestDocumentDefaultClipper:
    def test_returns_media_unchanged(self):
        from vtsearch.media.document.clipper import DocumentDefaultClipper

        media = {"id": 1, "type": "document", "media_bytes": b"fake-pdf"}
        result = DocumentDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtsearch.media.document.clipper import DocumentDefaultClipper

        c = DocumentDefaultClipper()
        assert c.name == "document_default"
        assert c.media_type == "document"
        assert isinstance(c, MediaClipper)


# ---------------------------------------------------------------------------
# Clipper Registry
# ---------------------------------------------------------------------------


class TestClipperRegistry:
    def test_all_clippers_returns_list(self):
        from vtsearch.media import all_clippers

        clippers = all_clippers()
        assert isinstance(clippers, list)
        assert len(clippers) >= 9  # 5 defaults + 4 tiling/sentence

    def test_all_clippers_dict_returns_dicts(self):
        from vtsearch.media import all_clippers_dict

        dicts = all_clippers_dict()
        assert all(isinstance(d, dict) for d in dicts)
        names = [d["name"] for d in dicts]
        assert "sound_default" in names
        assert "image_default" in names
        assert "text_default" in names
        assert "video_default" in names
        assert "document_default" in names

    def test_get_clipper(self):
        from vtsearch.media import get_clipper

        c = get_clipper("sound_default")
        assert c.name == "sound_default"

    def test_get_clipper_unknown_raises(self):
        from vtsearch.media import get_clipper

        with pytest.raises(KeyError):
            get_clipper("nonexistent_clipper")

    def test_clippers_for_type(self):
        from vtsearch.media import clippers_for_type

        audio_clippers = clippers_for_type("audio")
        assert len(audio_clippers) >= 2
        names = [c.name for c in audio_clippers]
        assert "sound_default" in names
        assert "sound_tiling_2.0s" in names

    def test_clippers_for_type_image(self):
        from vtsearch.media import clippers_for_type

        image_clippers = clippers_for_type("image")
        assert len(image_clippers) >= 2
        names = [c.name for c in image_clippers]
        assert "image_default" in names
        assert "image_tiling" in names

    def test_clippers_for_type_paragraph(self):
        from vtsearch.media import clippers_for_type

        text_clippers = clippers_for_type("paragraph")
        names = [c.name for c in text_clippers]
        assert "text_default" in names
        assert "text_sentence" in names

    def test_clippers_for_type_video(self):
        from vtsearch.media import clippers_for_type

        video_clippers = clippers_for_type("video")
        names = [c.name for c in video_clippers]
        assert "video_default" in names
        assert "video_tiling_2.0s" in names

    def test_clippers_for_type_document(self):
        from vtsearch.media import clippers_for_type

        doc_clippers = clippers_for_type("document")
        assert len(doc_clippers) >= 1
        names = [c.name for c in doc_clippers]
        assert "document_default" in names

    def test_every_media_type_has_default_clipper(self):
        from vtsearch.media import all_types, clippers_for_type

        for mt in all_types():
            clippers = clippers_for_type(mt.type_id)
            assert len(clippers) >= 1, f"No clippers for {mt.type_id}"
            names = [c.name for c in clippers]
            assert any("default" in n for n in names), (
                f"No default clipper for {mt.type_id}"
            )


# ---------------------------------------------------------------------------
# Clippers API endpoint
# ---------------------------------------------------------------------------


class TestClippersApiEndpoint:
    def test_list_all_clippers(self, client):
        resp = client.get("/api/clippers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "clippers" in data
        names = [c["name"] for c in data["clippers"]]
        assert "sound_default" in names
        assert "document_default" in names

    def test_filter_by_type_id(self, client):
        resp = client.get("/api/clippers?media_type=audio")
        assert resp.status_code == 200
        data = resp.get_json()
        clippers = data["clippers"]
        assert all(c["media_type"] == "audio" for c in clippers)
        names = [c["name"] for c in clippers]
        assert "sound_default" in names
        assert "image_default" not in names

    def test_filter_by_folder_name(self, client):
        resp = client.get("/api/clippers?media_type=images")
        assert resp.status_code == 200
        data = resp.get_json()
        clippers = data["clippers"]
        assert all(c["media_type"] == "image" for c in clippers)
        names = [c["name"] for c in clippers]
        assert "image_default" in names

    def test_filter_by_document(self, client):
        resp = client.get("/api/clippers?media_type=document")
        assert resp.status_code == 200
        data = resp.get_json()
        clippers = data["clippers"]
        assert len(clippers) >= 1
        assert all(c["media_type"] == "document" for c in clippers)


# ---------------------------------------------------------------------------
# Apply clipper helper
# ---------------------------------------------------------------------------


class TestApplyClipper:
    def test_apply_clipper_noop_for_empty_name(self):
        from vtsearch.routes.datasets_loading import _apply_clipper

        clips = {1: {"id": 1, "type": "audio", "origin": {"importer": "test", "params": {}}}}
        _apply_clipper(clips, "")
        assert len(clips) == 1

    def test_apply_clipper_unknown_name_noop(self):
        from vtsearch.routes.datasets_loading import _apply_clipper

        clips = {1: {"id": 1, "type": "audio", "origin": {"importer": "test", "params": {}}}}
        _apply_clipper(clips, "nonexistent_clipper")
        assert len(clips) == 1

    def test_apply_default_clipper_passthrough(self):
        from vtsearch.routes.datasets_loading import _apply_clipper

        media = {"id": 1, "type": "audio", "media_bytes": b"fake", "origin": {"importer": "test", "params": {}}}
        clips = {1: media}
        _apply_clipper(clips, "sound_default")
        assert len(clips) == 1
        assert clips[1]["origin"]["params"]["clipper"] == "sound_default"

    def test_apply_clipper_annotates_origin(self):
        from vtsearch.routes.datasets_loading import _apply_clipper

        media = {
            "id": 1,
            "type": "paragraph",
            "media_string": "First sentence. Second sentence.",
            "word_count": 4,
            "character_count": 32,
            "origin": {"importer": "folder", "params": {"path": "/data"}},
        }
        clips = {1: media}
        _apply_clipper(clips, "text_sentence")
        assert len(clips) == 2
        # Check origins include clipper
        for c in clips.values():
            assert c["origin"]["params"]["clipper"] == "text_sentence"
        # Check fresh IDs assigned
        assert set(clips.keys()) == {1, 2}
        # Check clip_index is set on clipped items
        assert clips[1].get("clip_index") is not None or clips[2].get("clip_index") is not None


# ---------------------------------------------------------------------------
# Dataset registry clipper column
# ---------------------------------------------------------------------------


class TestDatasetRegistryClipperColumn:
    def test_registry_includes_clipper(self, client):
        from vtsearch.datasets.registry import register_dataset

        register_dataset(
            name="clip-ds",
            media_type="audio",
            num_items=10,
            pkl_path="/tmp/clip.pkl",
            clipper="sound_tiling_2.0s",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["clipper"] == "sound_tiling_2.0s"

    def test_registry_clipper_defaults_to_empty(self, client):
        from vtsearch.datasets.registry import register_dataset

        register_dataset(
            name="no-clip",
            media_type="audio",
            num_items=5,
            pkl_path="/tmp/noclip.pkl",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["clipper"] == ""
