"""Tests for the MediaClipper base class and all built-in clippers."""

import io
import wave

import pytest

from vtsearch.utils.audio_generator import generate_wav
from vtsearch.media.clipper import MediaClipper


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
        assert d == {
            "name": "sound_default",
            "display_name": "Default",
            "description": "Import each audio file as-is, without splitting.",
            "media_type": "audio",
        }

    def test_display_name_default(self):
        from vtsearch.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        assert c.display_name == "Default"

    def test_display_name_tiling(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        assert c.display_name == "Tiling"

    def test_display_name_video_scene(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper()
        assert c.display_name == "Scene"

    def test_creation_questions_defaults_to_parameters(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        assert c.creation_questions == c.parameters
        assert len(c.creation_questions) == 2

    def test_creation_questions_empty_for_default_clipper(self):
        from vtsearch.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        assert c.creation_questions == []
        assert c.creation_questions == c.parameters

    def test_to_dict_includes_creation_questions_when_present(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        d = c.to_dict()
        assert "creation_questions" in d
        assert len(d["creation_questions"]) == 2
        assert d["creation_questions"][0]["key"] == "duration"

    def test_to_dict_omits_creation_questions_when_empty(self):
        from vtsearch.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        d = c.to_dict()
        assert "creation_questions" not in d


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
        assert c.name == "sound_tiling"
        assert c.media_type == "audio"
        assert c.duration == 2.0
        assert c.min_overlap == 0.0
        assert isinstance(c, MediaClipper)

    def test_rejects_non_positive_duration(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        with pytest.raises(ValueError):
            SoundTilingClipper(0)
        with pytest.raises(ValueError):
            SoundTilingClipper(-1)

    def test_rejects_negative_min_overlap(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        with pytest.raises(ValueError):
            SoundTilingClipper(2.0, min_overlap=-0.1)

    def test_rejects_min_overlap_ge_duration(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        with pytest.raises(ValueError):
            SoundTilingClipper(2.0, min_overlap=2.0)
        with pytest.raises(ValueError):
            SoundTilingClipper(2.0, min_overlap=3.0)

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

    def test_to_dict_includes_duration_and_min_overlap(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(3.5)
        d = c.to_dict()
        assert d["name"] == "sound_tiling"
        assert d["display_name"] == "Tiling"
        assert d["media_type"] == "audio"
        assert d["duration"] == 3.5
        assert d["min_overlap"] == 0.0

    def test_min_overlap_produces_more_tiles(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        wav = generate_wav(440, 10.0)
        media = {"id": 1, "type": "audio", "media_bytes": wav, "duration": 10.0}
        # Without overlap: ceil(10/2) = 5 tiles
        result_no_overlap = SoundTilingClipper(2.0, min_overlap=0.0).clip(media)
        assert len(result_no_overlap) == 5
        # With 1.0s min overlap: max_stride = 1.0, ceil((10-2)/1)+1 = 9 tiles
        result_with_overlap = SoundTilingClipper(2.0, min_overlap=1.0).clip(media)
        assert len(result_with_overlap) == 9
        # Verify all tiles are 2s
        for tile in result_with_overlap:
            assert tile["duration"] == pytest.approx(2.0, abs=0.01)
        # First starts at 0, last ends at 10
        assert result_with_overlap[0]["clip_start"] == pytest.approx(0.0)
        assert result_with_overlap[-1]["clip_end"] == pytest.approx(10.0, abs=0.01)
        # Verify actual overlap >= 1.0 between consecutive tiles
        for i in range(len(result_with_overlap) - 1):
            overlap = result_with_overlap[i]["clip_end"] - result_with_overlap[i + 1]["clip_start"]
            assert overlap >= 1.0 - 0.01


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
        assert c.name == "video_tiling"
        assert c.media_type == "video"
        assert c.duration == 2.0
        assert c.min_overlap == 0.0
        assert isinstance(c, MediaClipper)

    def test_rejects_non_positive_duration(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        with pytest.raises(ValueError):
            VideoTilingClipper(0)
        with pytest.raises(ValueError):
            VideoTilingClipper(-1)

    def test_rejects_negative_min_overlap(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        with pytest.raises(ValueError):
            VideoTilingClipper(2.0, min_overlap=-0.1)

    def test_rejects_min_overlap_ge_duration(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        with pytest.raises(ValueError):
            VideoTilingClipper(2.0, min_overlap=2.0)
        with pytest.raises(ValueError):
            VideoTilingClipper(2.0, min_overlap=3.0)

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

    def test_to_dict_includes_duration_and_min_overlap(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(3.5)
        d = c.to_dict()
        assert d["name"] == "video_tiling"
        assert d["display_name"] == "Tiling"
        assert d["media_type"] == "video"
        assert d["duration"] == 3.5
        assert d["min_overlap"] == 0.0

    def test_zero_duration_video_returned_unchanged(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 0}
        result = VideoTilingClipper(2.0).clip(media)
        assert result == [media]

    def test_min_overlap_produces_more_tiles(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 10.0}
        # Without overlap: ceil(10/2) = 5 tiles
        result_no_overlap = VideoTilingClipper(2.0, min_overlap=0.0).clip(media)
        assert len(result_no_overlap) == 5
        # With 1.0s min overlap: max_stride = 1.0, ceil((10-2)/1)+1 = 9 tiles
        result_with_overlap = VideoTilingClipper(2.0, min_overlap=1.0).clip(media)
        assert len(result_with_overlap) == 9
        for tile in result_with_overlap:
            assert tile["duration"] == pytest.approx(2.0)
        assert result_with_overlap[0]["clip_start"] == pytest.approx(0.0)
        assert result_with_overlap[-1]["clip_end"] == pytest.approx(10.0)
        # Verify actual overlap >= 1.0 between consecutive tiles
        for i in range(len(result_with_overlap) - 1):
            overlap = result_with_overlap[i]["clip_end"] - result_with_overlap[i + 1]["clip_start"]
            assert overlap >= 1.0 - 0.01


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

        media = {"id": 1, "type": "text", "media_string": "Hello world."}
        result = TextDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtsearch.media.text.clipper import TextDefaultClipper

        c = TextDefaultClipper()
        assert c.name == "text_default"
        assert c.media_type == "text"
        assert isinstance(c, MediaClipper)


# ---------------------------------------------------------------------------
# TextSentenceClipper
# ---------------------------------------------------------------------------


class TestTextSentenceClipper:
    def test_identity(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        c = TextSentenceClipper()
        assert c.name == "text_sentence"
        assert c.media_type == "text"
        assert isinstance(c, MediaClipper)

    def test_single_sentence_unchanged(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        media = {"id": 1, "type": "text", "media_string": "Hello world."}
        result = TextSentenceClipper().clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_splits_multiple_sentences(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        text = "First sentence. Second sentence. Third one!"
        media = {"id": 1, "type": "text", "media_string": text, "word_count": 7, "character_count": len(text)}
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
        media = {"id": 1, "type": "text", "media_string": text}
        result = TextSentenceClipper().clip(media)
        assert len(result) == 3
        assert result[0]["media_string"] == "Is this a test?"
        assert result[1]["media_string"] == "Yes it is!"
        assert result[2]["media_string"] == "Great."

    def test_empty_string_returns_unchanged(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        media = {"id": 1, "type": "text", "media_string": ""}
        result = TextSentenceClipper().clip(media)
        assert result == [media]

    def test_no_media_string_returns_unchanged(self):
        from vtsearch.media.text.clipper import TextSentenceClipper

        media = {"id": 1, "type": "text"}
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
# VideoSceneClipper
# ---------------------------------------------------------------------------


class TestVideoSceneClipper:
    def test_identity(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper()
        assert c.name == "video_scene"
        assert c.media_type == "video"
        assert c.threshold == 0.3
        assert c.min_scene_duration == 1.0
        assert isinstance(c, MediaClipper)

    def test_custom_params(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper(threshold=0.5, min_scene_duration=2.0)
        assert c.threshold == 0.5
        assert c.min_scene_duration == 2.0

    def test_rejects_invalid_threshold(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        with pytest.raises(ValueError):
            VideoSceneClipper(threshold=-0.1)
        with pytest.raises(ValueError):
            VideoSceneClipper(threshold=1.1)

    def test_rejects_non_positive_min_scene_duration(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        with pytest.raises(ValueError):
            VideoSceneClipper(min_scene_duration=0)
        with pytest.raises(ValueError):
            VideoSceneClipper(min_scene_duration=-1)

    def test_zero_duration_returns_unchanged(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        media = {"id": 1, "type": "video", "duration": 0}
        result = VideoSceneClipper().clip(media)
        assert result == [media]

    def test_no_media_bytes_or_path_returns_unchanged(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        media = {"id": 1, "type": "video", "duration": 10.0}
        result = VideoSceneClipper().clip(media)
        assert result == [media]

    def test_to_dict_includes_params(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper(threshold=0.4, min_scene_duration=1.5)
        d = c.to_dict()
        assert d["name"] == "video_scene"
        assert d["media_type"] == "video"
        assert d["threshold"] == 0.4
        assert d["min_scene_duration"] == 1.5

    def test_detect_scene_boundaries_no_cv2(self, monkeypatch):
        """When OpenCV is not available, clip returns the media unchanged."""
        import builtins

        from vtsearch.media.video.clipper import VideoSceneClipper

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("no cv2")
            return real_import(name, *args, **kwargs)

        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 10.0}
        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = VideoSceneClipper().clip(media)
        assert result == [media]

    def test_detect_scene_boundaries_helper_empty(self, monkeypatch):
        """When _detect_scene_boundaries returns no cuts, media is unchanged."""
        from vtsearch.media.video import clipper as clipper_mod
        from vtsearch.media.video.clipper import VideoSceneClipper

        monkeypatch.setattr(clipper_mod, "_detect_scene_boundaries", lambda *a, **kw: [])
        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 10.0}
        result = VideoSceneClipper().clip(media)
        assert result == [media]

    def test_splits_at_detected_boundaries(self, monkeypatch):
        """When boundaries are found, the clipper produces the right scenes."""
        from vtsearch.media.video import clipper as clipper_mod
        from vtsearch.media.video.clipper import VideoSceneClipper

        # Simulate two scene boundaries at 3.0s and 7.0s in a 10s video.
        monkeypatch.setattr(clipper_mod, "_detect_scene_boundaries", lambda *a, **kw: [3.0, 7.0])
        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 10.0}
        result = VideoSceneClipper().clip(media)

        assert len(result) == 3

        # Scene 0: [0, 3)
        assert result[0]["clip_start"] == pytest.approx(0.0)
        assert result[0]["clip_end"] == pytest.approx(3.0)
        assert result[0]["duration"] == pytest.approx(3.0)
        assert result[0]["clip_index"] == 0
        assert result[0]["scene_index"] == 0

        # Scene 1: [3, 7)
        assert result[1]["clip_start"] == pytest.approx(3.0)
        assert result[1]["clip_end"] == pytest.approx(7.0)
        assert result[1]["duration"] == pytest.approx(4.0)
        assert result[1]["clip_index"] == 1
        assert result[1]["scene_index"] == 1

        # Scene 2: [7, 10)
        assert result[2]["clip_start"] == pytest.approx(7.0)
        assert result[2]["clip_end"] == pytest.approx(10.0)
        assert result[2]["duration"] == pytest.approx(3.0)
        assert result[2]["clip_index"] == 2
        assert result[2]["scene_index"] == 2

    def test_single_boundary_produces_two_scenes(self, monkeypatch):
        from vtsearch.media.video import clipper as clipper_mod
        from vtsearch.media.video.clipper import VideoSceneClipper

        monkeypatch.setattr(clipper_mod, "_detect_scene_boundaries", lambda *a, **kw: [5.0])
        media = {"id": 1, "type": "video", "media_bytes": b"fake", "duration": 8.0}
        result = VideoSceneClipper().clip(media)
        assert len(result) == 2
        assert result[0]["clip_start"] == pytest.approx(0.0)
        assert result[0]["clip_end"] == pytest.approx(5.0)
        assert result[1]["clip_start"] == pytest.approx(5.0)
        assert result[1]["clip_end"] == pytest.approx(8.0)

    def test_media_path_used_when_available(self, monkeypatch, tmp_path):
        """When media_path exists, it's used instead of writing a temp file."""
        from vtsearch.media.video import clipper as clipper_mod
        from vtsearch.media.video.clipper import VideoSceneClipper

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video data")

        paths_seen = []

        def mock_detect(video_path, threshold, min_scene_duration):
            paths_seen.append(video_path)
            return [2.0]

        monkeypatch.setattr(clipper_mod, "_detect_scene_boundaries", mock_detect)

        media = {"id": 1, "type": "video", "media_path": str(video_file), "duration": 5.0}
        result = VideoSceneClipper().clip(media)
        assert len(result) == 2
        assert paths_seen[0] == str(video_file)


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
        # All dicts should have display_name
        for d in dicts:
            assert "display_name" in d

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
        assert "sound_tiling" in names

    def test_clippers_for_type_image(self):
        from vtsearch.media import clippers_for_type

        image_clippers = clippers_for_type("image")
        assert len(image_clippers) >= 2
        names = [c.name for c in image_clippers]
        assert "image_default" in names
        assert "image_tiling" in names

    def test_clippers_for_type_paragraph(self):
        from vtsearch.media import clippers_for_type

        text_clippers = clippers_for_type("text")
        names = [c.name for c in text_clippers]
        assert "text_default" in names
        assert "text_sentence" in names

    def test_clippers_for_type_video(self):
        from vtsearch.media import clippers_for_type

        video_clippers = clippers_for_type("video")
        names = [c.name for c in video_clippers]
        assert "video_default" in names
        assert "video_tiling" in names

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
        # All entries should include display_name
        for c in data["clippers"]:
            assert "display_name" in c

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
        resp = client.get("/api/clippers?media_type=image")
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

    def test_creation_questions_in_api_response(self, client):
        resp = client.get("/api/clippers?media_type=audio")
        assert resp.status_code == 200
        data = resp.get_json()
        clippers = data["clippers"]
        tiling = next(c for c in clippers if c["name"] == "sound_tiling")
        assert "creation_questions" in tiling
        assert len(tiling["creation_questions"]) == 2
        keys = [q["key"] for q in tiling["creation_questions"]]
        assert "duration" in keys
        # Default clipper should not have creation_questions
        default = next(c for c in clippers if c["name"] == "sound_default")
        assert "creation_questions" not in default


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
            "type": "text",
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
            clipper="sound_tiling",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["clipper"] == "Tiling"

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

    def test_registry_default_clipper_shows_dash(self, client):
        from vtsearch.datasets.registry import register_dataset

        register_dataset(
            name="default-clip",
            media_type="image",
            num_items=3,
            pkl_path="/tmp/defclip.pkl",
            clipper="image_default",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["clipper"] == "-"


# ---------------------------------------------------------------------------
# Clipper parameters
# ---------------------------------------------------------------------------


class TestClipperParameters:
    """Test the parameters property and with_params method on clippers."""

    def test_default_clipper_has_no_parameters(self):
        from vtsearch.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        assert c.parameters == []

    def test_default_clipper_with_params_returns_self(self):
        from vtsearch.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        assert c.with_params({"anything": 42}) is c

    def test_sound_tiling_parameters(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        params = c.parameters
        assert len(params) == 2
        assert params[0]["key"] == "duration"
        assert params[0]["type"] == "number"
        assert params[0]["default"] == 2.0
        assert params[0]["min"] == 0.1
        assert params[1]["key"] == "min_overlap"
        assert params[1]["type"] == "number"
        assert params[1]["default"] == 0.0
        assert params[1]["min"] == 0

    def test_sound_tiling_with_params(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        c2 = c.with_params({"duration": 5.0})
        assert isinstance(c2, SoundTilingClipper)
        assert c2.duration == 5.0
        assert c2.min_overlap == 0.0
        assert c2 is not c
        assert c.duration == 2.0  # original unchanged

    def test_sound_tiling_with_params_overlap(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        c2 = c.with_params({"duration": 5.0, "min_overlap": 1.0})
        assert c2.duration == 5.0
        assert c2.min_overlap == 1.0

    def test_sound_tiling_with_params_ignores_unknown_keys(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        c2 = c.with_params({"unknown_key": 99})
        assert c2.duration == 2.0  # falls back to current value

    def test_video_tiling_parameters(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(2.0)
        params = c.parameters
        assert len(params) == 2
        assert params[0]["key"] == "duration"
        assert params[0]["default"] == 2.0
        assert params[1]["key"] == "min_overlap"
        assert params[1]["default"] == 0.0

    def test_video_tiling_with_params(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(2.0)
        c2 = c.with_params({"duration": 10.0})
        assert isinstance(c2, VideoTilingClipper)
        assert c2.duration == 10.0
        assert c2.min_overlap == 0.0
        assert c.duration == 2.0

    def test_video_tiling_with_params_overlap(self):
        from vtsearch.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(2.0)
        c2 = c.with_params({"duration": 10.0, "min_overlap": 2.0})
        assert c2.duration == 10.0
        assert c2.min_overlap == 2.0

    def test_video_scene_parameters(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper()
        params = c.parameters
        assert len(params) == 2
        keys = [p["key"] for p in params]
        assert "threshold" in keys
        assert "min_scene_duration" in keys
        # Check defaults match constructor defaults
        thresh_param = next(p for p in params if p["key"] == "threshold")
        assert thresh_param["default"] == 0.3
        min_dur_param = next(p for p in params if p["key"] == "min_scene_duration")
        assert min_dur_param["default"] == 1.0

    def test_video_scene_with_params(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper()
        c2 = c.with_params({"threshold": 0.5, "min_scene_duration": 2.5})
        assert isinstance(c2, VideoSceneClipper)
        assert c2.threshold == 0.5
        assert c2.min_scene_duration == 2.5
        assert c.threshold == 0.3  # original unchanged

    def test_video_scene_with_partial_params(self):
        from vtsearch.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper(threshold=0.4, min_scene_duration=1.5)
        c2 = c.with_params({"threshold": 0.6})
        assert c2.threshold == 0.6
        assert c2.min_scene_duration == 1.5  # kept from original

    def test_to_dict_includes_parameters(self):
        from vtsearch.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        d = c.to_dict()
        assert "parameters" in d
        assert len(d["parameters"]) == 2
        assert d["parameters"][0]["key"] == "duration"
        assert d["parameters"][1]["key"] == "min_overlap"

    def test_to_dict_no_parameters_for_default_clipper(self):
        from vtsearch.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        d = c.to_dict()
        assert "parameters" not in d


class TestClipperParametersApi:
    """Test that the /api/clippers endpoint returns parameter info."""

    def test_clippers_api_includes_parameters(self, client):
        resp = client.get("/api/clippers?media_type=audio")
        assert resp.status_code == 200
        data = resp.get_json()
        clippers = data["clippers"]
        # sound_tiling should have parameters
        tiling = next(c for c in clippers if c["name"].startswith("sound_tiling"))
        assert "parameters" in tiling
        assert len(tiling["parameters"]) == 2
        assert tiling["parameters"][0]["key"] == "duration"
        assert tiling["parameters"][1]["key"] == "min_overlap"

    def test_default_clipper_has_no_parameters_in_api(self, client):
        resp = client.get("/api/clippers?media_type=audio")
        data = resp.get_json()
        default = next(c for c in data["clippers"] if c["name"] == "sound_default")
        assert "parameters" not in default

    def test_video_scene_clipper_in_registry(self, client):
        resp = client.get("/api/clippers?media_type=video")
        data = resp.get_json()
        names = [c["name"] for c in data["clippers"]]
        assert "video_scene" in names
        scene = next(c for c in data["clippers"] if c["name"] == "video_scene")
        assert "parameters" in scene
        keys = [p["key"] for p in scene["parameters"]]
        assert "threshold" in keys
        assert "min_scene_duration" in keys


class TestApplyClipperWithParams:
    """Test _apply_clipper with custom clipper_params."""

    def test_apply_clipper_with_custom_duration(self):
        from vtsearch.utils.audio_generator import generate_wav
        from vtsearch.routes.datasets_loading import _apply_clipper

        # Generate a 10s audio clip
        wav = generate_wav(440, 10.0)
        media = {
            "id": 1,
            "type": "audio",
            "media_bytes": wav,
            "duration": 10.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        # With default 2s duration: ceil(10/2) = 5 tiles
        _apply_clipper(clips, "sound_tiling")
        assert len(clips) == 5

    def test_apply_clipper_with_overridden_duration(self):
        from vtsearch.utils.audio_generator import generate_wav
        from vtsearch.routes.datasets_loading import _apply_clipper

        wav = generate_wav(440, 10.0)
        media = {
            "id": 1,
            "type": "audio",
            "media_bytes": wav,
            "duration": 10.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        # Override to 5s duration: ceil(10/5) = 2 tiles
        _apply_clipper(clips, "sound_tiling", {"duration": 5.0})
        assert len(clips) == 2

    def test_apply_clipper_params_none_uses_defaults(self):
        from vtsearch.utils.audio_generator import generate_wav
        from vtsearch.routes.datasets_loading import _apply_clipper

        wav = generate_wav(440, 10.0)
        media = {
            "id": 1,
            "type": "audio",
            "media_bytes": wav,
            "duration": 10.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        _apply_clipper(clips, "sound_tiling", None)
        assert len(clips) == 5  # default 2s → 5 tiles

    def test_apply_clipper_with_min_overlap(self):
        from vtsearch.utils.audio_generator import generate_wav
        from vtsearch.routes.datasets_loading import _apply_clipper

        wav = generate_wav(440, 10.0)
        media = {
            "id": 1,
            "type": "audio",
            "media_bytes": wav,
            "duration": 10.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        # 2s clips with 1s min overlap: max_stride=1, ceil((10-2)/1)+1 = 9 tiles
        _apply_clipper(clips, "sound_tiling", {"duration": 2.0, "min_overlap": 1.0})
        assert len(clips) == 9
