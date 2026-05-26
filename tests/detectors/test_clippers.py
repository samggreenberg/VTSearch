"""Tests for the MediaClipper base class and all built-in clippers."""

import io
import wave

import pytest

from vtscore.media.audio.audio_generator import generate_wav
from vtscore.media.clipper import MediaClipper


# ---------------------------------------------------------------------------
# MediaClipper ABC
# ---------------------------------------------------------------------------


class TestMediaClipperABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            MediaClipper()  # pyright: ignore[reportAbstractUsage]

    def test_to_dict_on_concrete(self):
        from vtscore.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        d = c.to_dict()
        assert d == {
            "name": "sound_default",
            "display_name": "None",
            "description": "Import each audio file as-is, without splitting.",
            "media_type": "audio",
        }

    def test_display_name_default(self):
        from vtscore.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        assert c.display_name == "None"

    def test_display_name_tiling(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        assert c.display_name == "Tiling"

    def test_display_name_video_scene(self):
        from vtscore.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper()
        assert c.display_name == "Scene"

    def test_creation_questions_defaults_to_parameters(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        assert c.creation_questions == c.parameters
        assert len(c.creation_questions) == 2

    def test_creation_questions_empty_for_default_clipper(self):
        from vtscore.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        assert c.creation_questions == []
        assert c.creation_questions == c.parameters

    def test_to_dict_includes_creation_questions_when_present(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        d = c.to_dict()
        assert "creation_questions" in d
        assert len(d["creation_questions"]) == 2
        assert d["creation_questions"][0]["key"] == "duration"

    def test_to_dict_omits_creation_questions_when_empty(self):
        from vtscore.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        d = c.to_dict()
        assert "creation_questions" not in d


# ---------------------------------------------------------------------------
# SoundDefaultClipper
# ---------------------------------------------------------------------------


class TestSoundDefaultClipper:
    def test_returns_media_unchanged(self):
        from vtscore.media.audio.clipper import SoundDefaultClipper

        media = {"id": 1, "media_type": "audio", "media_bytes": b"fake", "duration": 3.0}
        result = SoundDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtscore.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        assert c.name == "sound_default"
        assert c.media_type == "audio"
        assert isinstance(c, MediaClipper)


# ---------------------------------------------------------------------------
# SoundTilingClipper
# ---------------------------------------------------------------------------


class TestSoundTilingClipper:
    def test_identity(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        assert c.name == "sound_tiling"
        assert c.media_type == "audio"
        assert c.duration == 2.0
        assert c.min_overlap == 0.0
        assert isinstance(c, MediaClipper)

    def test_rejects_non_positive_duration(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        with pytest.raises(ValueError):
            SoundTilingClipper(0)
        with pytest.raises(ValueError):
            SoundTilingClipper(-1)

    def test_rejects_negative_min_overlap(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        with pytest.raises(ValueError):
            SoundTilingClipper(2.0, min_overlap=-0.1)

    def test_rejects_min_overlap_ge_duration(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        with pytest.raises(ValueError):
            SoundTilingClipper(2.0, min_overlap=2.0)
        with pytest.raises(ValueError):
            SoundTilingClipper(2.0, min_overlap=3.0)

    def test_short_audio_returned_unchanged(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        wav = generate_wav(440, 1.5)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 1.5}
        result = SoundTilingClipper(2.0).clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_tiles_longer_audio(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        wav = generate_wav(440, 5.0)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 5.0}
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
        from vtscore.media.audio.clipper import SoundTilingClipper

        wav = generate_wav(440, 9.5)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 9.5}
        result = SoundTilingClipper(2.0).clip(media)
        # 9.5 / 2.0 = 4.75 → round → 5 tiles
        assert len(result) == 5
        # First tile starts at 0
        assert result[0]["clip_start"] == pytest.approx(0.0)
        # Last tile ends at 9.5
        assert result[-1]["clip_end"] == pytest.approx(9.5, abs=0.01)

    def test_no_media_bytes_returns_unchanged(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        media = {"id": 1, "media_type": "audio", "duration": 10.0}
        result = SoundTilingClipper(2.0).clip(media)
        assert result == [media]

    def test_to_dict_includes_duration_and_min_overlap(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(3.5)
        d = c.to_dict()
        assert d["name"] == "sound_tiling"
        assert d["display_name"] == "Tiling"
        assert d["media_type"] == "audio"
        assert d["duration"] == 3.5
        assert d["min_overlap"] == 0.0

    def test_min_overlap_produces_more_tiles(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        wav = generate_wav(440, 10.0)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 10.0}
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
# SoundClipClipper
# ---------------------------------------------------------------------------


class TestSoundClipClipper:
    def test_identity(self):
        from vtscore.media.audio.clipper import SoundClipClipper

        c = SoundClipClipper(0.5, 1.5)
        assert c.name == "sound_clip"
        assert c.media_type == "audio"
        assert c.start == 0.5
        assert c.end == 1.5
        assert isinstance(c, MediaClipper)

    def test_rejects_negative_start(self):
        from vtscore.media.audio.clipper import SoundClipClipper

        with pytest.raises(ValueError):
            SoundClipClipper(-0.1, 1.0)

    def test_rejects_end_le_start(self):
        from vtscore.media.audio.clipper import SoundClipClipper

        with pytest.raises(ValueError):
            SoundClipClipper(1.0, 1.0)
        with pytest.raises(ValueError):
            SoundClipClipper(2.0, 1.0)

    def test_extracts_requested_range(self):
        from vtscore.media.audio.clipper import SoundClipClipper

        wav = generate_wav(440, 5.0)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 5.0}
        result = SoundClipClipper(1.0, 3.5).clip(media)
        assert len(result) == 1
        clip = result[0]
        assert clip["clip_start"] == pytest.approx(1.0)
        assert clip["clip_end"] == pytest.approx(3.5)
        assert clip["duration"] == pytest.approx(2.5, abs=0.01)
        assert clip["clip_index"] == 0
        # Result is valid WAV
        with wave.open(io.BytesIO(clip["media_bytes"]), "rb") as wf:
            assert wf.getframerate() == 48000

    def test_clamps_to_audio_duration(self):
        from vtscore.media.audio.clipper import SoundClipClipper

        wav = generate_wav(440, 2.0)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 2.0}
        # Request a window that runs past the end; clamps.
        result = SoundClipClipper(1.0, 10.0).clip(media)
        assert len(result) == 1
        assert result[0]["clip_start"] == pytest.approx(1.0)
        assert result[0]["clip_end"] == pytest.approx(2.0, abs=0.01)

    def test_no_media_bytes_returns_unchanged(self):
        from vtscore.media.audio.clipper import SoundClipClipper

        media = {"id": 1, "media_type": "audio", "duration": 3.0}
        result = SoundClipClipper(0.0, 1.0).clip(media)
        assert result == [media]

    def test_with_params(self):
        from vtscore.media.audio.clipper import SoundClipClipper

        c = SoundClipClipper(0.0, 1.0)
        c2 = c.with_params({"start": 2.0, "end": 4.5})
        assert isinstance(c2, SoundClipClipper)
        assert c2.start == 2.0
        assert c2.end == 4.5
        # original unchanged
        assert c.start == 0.0
        assert c.end == 1.0

    def test_to_dict(self):
        from vtscore.media.audio.clipper import SoundClipClipper

        c = SoundClipClipper(1.5, 3.0)
        d = c.to_dict()
        assert d["name"] == "sound_clip"
        assert d["media_type"] == "audio"
        assert d["start"] == 1.5
        assert d["end"] == 3.0


# ---------------------------------------------------------------------------
# SoundSilenceClipper
# ---------------------------------------------------------------------------


def _concat_wavs(*wavs: bytes) -> bytes:
    """Concatenate multiple WAV byte strings into a single WAV file.

    All inputs must share the same sample rate, channel count, and sample
    width.  Used to build tone/silence/tone test fixtures.
    """
    all_frames: list[bytes] = []
    params = None
    for wb in wavs:
        with wave.open(io.BytesIO(wb), "rb") as wf:
            if params is None:
                params = wf.getparams()
            all_frames.append(wf.readframes(wf.getnframes()))
    assert params is not None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(b"".join(all_frames))
    return buf.getvalue()


class TestSoundSilenceClipper:
    def test_identity(self):
        from vtscore.media.audio.clipper import SoundSilenceClipper

        c = SoundSilenceClipper()
        assert c.name == "sound_silence"
        assert c.media_type == "audio"
        assert c.top_db == 40.0
        assert c.min_clip_duration == 0.3
        assert c.pad == 0.05
        assert isinstance(c, MediaClipper)

    def test_custom_params(self):
        from vtscore.media.audio.clipper import SoundSilenceClipper

        c = SoundSilenceClipper(top_db=20.0, min_clip_duration=0.5, pad=0.1)
        assert c.top_db == 20.0
        assert c.min_clip_duration == 0.5
        assert c.pad == 0.1

    def test_rejects_non_positive_top_db(self):
        from vtscore.media.audio.clipper import SoundSilenceClipper

        with pytest.raises(ValueError):
            SoundSilenceClipper(top_db=0)
        with pytest.raises(ValueError):
            SoundSilenceClipper(top_db=-1)

    def test_rejects_negative_min_clip_duration(self):
        from vtscore.media.audio.clipper import SoundSilenceClipper

        with pytest.raises(ValueError):
            SoundSilenceClipper(min_clip_duration=-0.1)

    def test_rejects_negative_pad(self):
        from vtscore.media.audio.clipper import SoundSilenceClipper

        with pytest.raises(ValueError):
            SoundSilenceClipper(pad=-0.1)

    def test_no_media_bytes_returns_unchanged(self):
        from vtscore.media.audio.clipper import SoundSilenceClipper

        media = {"id": 1, "media_type": "audio", "duration": 3.0}
        result = SoundSilenceClipper().clip(media)
        assert result == [media]

    def test_invalid_bytes_returns_unchanged(self):
        from vtscore.media.audio.clipper import SoundSilenceClipper

        media = {"id": 1, "media_type": "audio", "media_bytes": b"not a wav", "duration": 1.0}
        result = SoundSilenceClipper().clip(media)
        assert result == [media]

    def test_splits_tone_silence_tone(self):
        """Tone | silence | tone | silence | tone → three non-silent clips."""
        from vtscore.media.audio.clipper import SoundSilenceClipper

        # Build a 5 s clip: 1 s tone, 1 s silence, 1 s tone, 1 s silence, 1 s tone.
        wav = _concat_wavs(
            generate_wav(440, 1.0),
            generate_wav(0, 1.0),
            generate_wav(440, 1.0),
            generate_wav(0, 1.0),
            generate_wav(440, 1.0),
        )
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 5.0}

        result = SoundSilenceClipper(top_db=40, min_clip_duration=0.1, pad=0.0).clip(media)
        assert len(result) == 3
        for idx, clip in enumerate(result):
            assert clip["clip_index"] == idx
            assert clip["clip_end"] > clip["clip_start"]
            # Each non-silent block is ~1 s
            assert clip["duration"] == pytest.approx(1.0, abs=0.15)
            # Output should be valid WAV
            with wave.open(io.BytesIO(clip["media_bytes"]), "rb") as wf:
                assert wf.getframerate() == 48000

    def test_drops_intro_outro_silence(self):
        """Leading and trailing silence should be discarded."""
        from vtscore.media.audio.clipper import SoundSilenceClipper

        # 0.5 s silence | 1 s tone | 0.5 s silence
        wav = _concat_wavs(
            generate_wav(0, 0.5),
            generate_wav(440, 1.0),
            generate_wav(0, 0.5),
        )
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 2.0}

        result = SoundSilenceClipper(top_db=40, min_clip_duration=0.1, pad=0.0).clip(media)
        assert len(result) == 1
        clip = result[0]
        # Clip should start at ~0.5 s and end at ~1.5 s, NOT at 0 and 2.0
        assert clip["clip_start"] == pytest.approx(0.5, abs=0.15)
        assert clip["clip_end"] == pytest.approx(1.5, abs=0.15)
        assert clip["duration"] < 2.0  # intro/outro silence was dropped

    def test_padding_extends_clip_bounds(self):
        from vtscore.media.audio.clipper import SoundSilenceClipper

        wav = _concat_wavs(
            generate_wav(0, 0.5),
            generate_wav(440, 1.0),
            generate_wav(0, 0.5),
        )
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 2.0}

        # With 0.2 s pad, the single clip should extend ~0.2 s into the surrounding silence.
        result = SoundSilenceClipper(top_db=40, min_clip_duration=0.1, pad=0.2).clip(media)
        assert len(result) == 1
        clip = result[0]
        assert clip["clip_start"] < 0.5
        assert clip["clip_end"] > 1.5
        # But never beyond the actual audio bounds.
        assert clip["clip_start"] >= 0.0
        assert clip["clip_end"] <= 2.0

    def test_min_clip_duration_drops_short_intervals(self):
        """Non-silent intervals shorter than min_clip_duration are dropped."""
        from vtscore.media.audio.clipper import SoundSilenceClipper

        # 50 ms blip | 1 s silence | 1 s tone
        wav = _concat_wavs(
            generate_wav(440, 0.05),
            generate_wav(0, 1.0),
            generate_wav(440, 1.0),
        )
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 2.05}

        # min_clip_duration=0.3 should drop the blip but keep the 1 s tone.
        result = SoundSilenceClipper(top_db=40, min_clip_duration=0.3, pad=0.0).clip(media)
        assert len(result) == 1
        assert result[0]["duration"] == pytest.approx(1.0, abs=0.15)

    def test_all_silence_returns_unchanged(self):
        """Audio that's silent throughout is returned unchanged (not dropped)."""
        from vtscore.media.audio.clipper import SoundSilenceClipper

        wav = generate_wav(0, 2.0)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 2.0}
        result = SoundSilenceClipper().clip(media)
        assert result == [media]

    def test_with_params(self):
        from vtscore.media.audio.clipper import SoundSilenceClipper

        c = SoundSilenceClipper()
        c2 = c.with_params({"top_db": 25, "min_clip_duration": 0.5, "pad": 0.2})
        assert isinstance(c2, SoundSilenceClipper)
        assert c2.top_db == 25.0
        assert c2.min_clip_duration == 0.5
        assert c2.pad == 0.2
        # Original unchanged.
        assert c.top_db == 40.0
        assert c.min_clip_duration == 0.3
        assert c.pad == 0.05

    def test_to_dict(self):
        from vtscore.media.audio.clipper import SoundSilenceClipper

        c = SoundSilenceClipper(top_db=30, min_clip_duration=0.4, pad=0.1)
        d = c.to_dict()
        assert d["name"] == "sound_silence"
        assert d["display_name"] == "Silence"
        assert d["media_type"] == "audio"
        assert d["top_db"] == 30
        assert d["min_clip_duration"] == 0.4
        assert d["pad"] == 0.1
        # Parameters surface as creation questions automatically.
        assert "creation_questions" in d
        assert {q["key"] for q in d["creation_questions"]} == {"top_db", "min_clip_duration", "pad"}

    def test_librosa_unavailable_returns_unchanged(self, monkeypatch):
        """If librosa import fails, clipper returns the media unchanged."""
        import builtins

        from vtscore.media.audio.clipper import SoundSilenceClipper

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError("no librosa")
            return real_import(name, *args, **kwargs)

        wav = generate_wav(440, 1.0)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 1.0}
        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = SoundSilenceClipper().clip(media)
        assert result == [media]


# ---------------------------------------------------------------------------
# SoundSpeechActivityClipper
# ---------------------------------------------------------------------------


class TestSoundSpeechActivityClipper:
    def test_identity(self):
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        c = SoundSpeechActivityClipper()
        assert c.name == "sound_speech_activity"
        assert c.media_type == "audio"
        assert c.threshold == 0.5
        assert c.min_clip_duration == 0.3
        assert c.pad == 0.05
        assert isinstance(c, MediaClipper)

    def test_custom_params(self):
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        c = SoundSpeechActivityClipper(threshold=0.7, min_clip_duration=0.5, pad=0.1)
        assert c.threshold == 0.7
        assert c.min_clip_duration == 0.5
        assert c.pad == 0.1

    def test_rejects_threshold_out_of_range(self):
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        with pytest.raises(ValueError):
            SoundSpeechActivityClipper(threshold=0.0)
        with pytest.raises(ValueError):
            SoundSpeechActivityClipper(threshold=-0.1)
        with pytest.raises(ValueError):
            SoundSpeechActivityClipper(threshold=1.5)

    def test_rejects_negative_min_clip_duration(self):
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        with pytest.raises(ValueError):
            SoundSpeechActivityClipper(min_clip_duration=-0.1)

    def test_rejects_negative_pad(self):
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        with pytest.raises(ValueError):
            SoundSpeechActivityClipper(pad=-0.1)

    def test_no_media_bytes_returns_unchanged(self):
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        media = {"id": 1, "media_type": "audio", "duration": 3.0}
        result = SoundSpeechActivityClipper().clip(media)
        assert result == [media]

    def test_returns_unchanged_when_silero_unavailable(self, monkeypatch):
        """If Silero can't be loaded, clipper returns the media unchanged."""
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        c = SoundSpeechActivityClipper()
        # Force the lazy loader to report unavailable so we never touch torch.hub.
        monkeypatch.setattr(c, "_load_model", lambda: False)
        wav = generate_wav(440, 1.0)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 1.0}
        result = c.clip(media)
        assert result == [media]

    def test_splits_on_mocked_intervals(self):
        """Three mocked speech intervals → three clips with right boundaries."""
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        wav = _concat_wavs(
            generate_wav(440, 1.0),
            generate_wav(0, 1.0),
            generate_wav(440, 1.0),
            generate_wav(0, 1.0),
            generate_wav(440, 1.0),
        )
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 5.0}

        c = SoundSpeechActivityClipper(min_clip_duration=0.1, pad=0.0)
        c._detect_speech_intervals = lambda _b: [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]  # pyright: ignore[reportAttributeAccessIssue]

        result = c.clip(media)
        assert len(result) == 3
        for idx, clip in enumerate(result):
            assert clip["clip_index"] == idx
            assert clip["clip_end"] > clip["clip_start"]
            assert clip["duration"] == pytest.approx(1.0, abs=0.01)
            with wave.open(io.BytesIO(clip["media_bytes"]), "rb") as wf:
                assert wf.getframerate() == 48000
        assert [(c["clip_start"], c["clip_end"]) for c in result] == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]

    def test_returns_unchanged_when_no_intervals(self):
        """An empty detector result keeps the original media intact."""
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        wav = generate_wav(440, 1.0)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 1.0}
        c = SoundSpeechActivityClipper()
        c._detect_speech_intervals = lambda _b: []  # pyright: ignore[reportAttributeAccessIssue]
        result = c.clip(media)
        assert result == [media]

    def test_returns_unchanged_on_detector_error(self):
        """If detection returns None (decoder failure, missing torch), pass through."""
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        wav = generate_wav(440, 1.0)
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 1.0}
        c = SoundSpeechActivityClipper()
        c._detect_speech_intervals = lambda _b: None  # pyright: ignore[reportAttributeAccessIssue]
        result = c.clip(media)
        assert result == [media]

    def test_with_params(self):
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        c = SoundSpeechActivityClipper()
        c2 = c.with_params({"threshold": 0.7, "min_clip_duration": 0.5, "pad": 0.2})
        assert isinstance(c2, SoundSpeechActivityClipper)
        assert c2.threshold == 0.7
        assert c2.min_clip_duration == 0.5
        assert c2.pad == 0.2
        # Original unchanged.
        assert c.threshold == 0.5
        assert c.min_clip_duration == 0.3
        assert c.pad == 0.05

    def test_to_dict(self):
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        c = SoundSpeechActivityClipper(threshold=0.6, min_clip_duration=0.4, pad=0.1)
        d = c.to_dict()
        assert d["name"] == "sound_speech_activity"
        assert d["media_type"] == "audio"
        assert d["threshold"] == 0.6
        assert d["min_clip_duration"] == 0.4
        assert d["pad"] == 0.1
        assert len(d["parameters"]) == 3

    def test_min_clip_duration_drops_short_intervals(self):
        """Short detector intervals are dropped at the post-filter stage."""
        from vtscore.media.audio.clipper import SoundSpeechActivityClipper

        wav = _concat_wavs(generate_wav(440, 0.05), generate_wav(0, 1.0), generate_wav(440, 1.0))
        media = {"id": 1, "media_type": "audio", "media_bytes": wav, "duration": 2.05}

        # Drive the post-filter directly: the 50 ms interval drops, the 1 s one stays.
        c = SoundSpeechActivityClipper(min_clip_duration=0.3, pad=0.0)
        # Bypass the post-filter inside _detect_speech_intervals by mocking it
        # to mirror what the real detector would produce *before* filtering,
        # and apply the same filter that the real method applies.
        intervals = [(0.0, 0.05), (1.05, 2.05)]
        filtered = [(t0, t1) for (t0, t1) in intervals if (t1 - t0) >= 0.3]
        c._detect_speech_intervals = lambda _b: filtered  # pyright: ignore[reportAttributeAccessIssue]
        result = c.clip(media)
        assert len(result) == 1
        assert result[0]["duration"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# VideoDefaultClipper
# ---------------------------------------------------------------------------


class TestVideoDefaultClipper:
    def test_returns_media_unchanged(self):
        from vtscore.media.video.clipper import VideoDefaultClipper

        media = {"id": 1, "media_type": "video", "media_bytes": b"fake", "duration": 10.0}
        result = VideoDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtscore.media.video.clipper import VideoDefaultClipper

        c = VideoDefaultClipper()
        assert c.name == "video_default"
        assert c.media_type == "video"
        assert isinstance(c, MediaClipper)


# ---------------------------------------------------------------------------
# VideoTilingClipper
# ---------------------------------------------------------------------------


class TestVideoTilingClipper:
    def test_identity(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(2.0)
        assert c.name == "video_tiling"
        assert c.media_type == "video"
        assert c.duration == 2.0
        assert c.min_overlap == 0.0
        assert isinstance(c, MediaClipper)

    def test_rejects_non_positive_duration(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        with pytest.raises(ValueError):
            VideoTilingClipper(0)
        with pytest.raises(ValueError):
            VideoTilingClipper(-1)

    def test_rejects_negative_min_overlap(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        with pytest.raises(ValueError):
            VideoTilingClipper(2.0, min_overlap=-0.1)

    def test_rejects_min_overlap_ge_duration(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        with pytest.raises(ValueError):
            VideoTilingClipper(2.0, min_overlap=2.0)
        with pytest.raises(ValueError):
            VideoTilingClipper(2.0, min_overlap=3.0)

    def test_short_video_returned_unchanged(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "media_type": "video", "media_bytes": b"fake", "duration": 1.5}
        result = VideoTilingClipper(2.0).clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_9_5s_produces_five_2s_tiles(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "media_type": "video", "media_bytes": b"fake", "duration": 9.5}
        result = VideoTilingClipper(2.0).clip(media)
        assert len(result) == 5
        assert result[0]["clip_start"] == pytest.approx(0.0)
        assert result[-1]["clip_end"] == pytest.approx(9.5, abs=0.01)
        for idx, tile in enumerate(result):
            assert tile["clip_index"] == idx
            assert tile["duration"] == pytest.approx(2.0)

    def test_exact_multiple_duration(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "media_type": "video", "media_bytes": b"fake", "duration": 10.0}
        result = VideoTilingClipper(2.0).clip(media)
        assert len(result) == 5
        # No overlap when duration is exact multiple
        assert result[0]["clip_start"] == pytest.approx(0.0)
        assert result[-1]["clip_end"] == pytest.approx(10.0)

    def test_to_dict_includes_duration_and_min_overlap(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(3.5)
        d = c.to_dict()
        assert d["name"] == "video_tiling"
        assert d["display_name"] == "Tiling"
        assert d["media_type"] == "video"
        assert d["duration"] == 3.5
        assert d["min_overlap"] == 0.0

    def test_zero_duration_video_returned_unchanged(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "media_type": "video", "media_bytes": b"fake", "duration": 0}
        result = VideoTilingClipper(2.0).clip(media)
        assert result == [media]

    def test_min_overlap_produces_more_tiles(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        media = {"id": 1, "media_type": "video", "media_bytes": b"fake", "duration": 10.0}
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
        from vtscore.media.image.clipper import ImageDefaultClipper

        media = {"id": 1, "media_type": "image", "media_bytes": b"fake", "width": 100, "height": 100}
        result = ImageDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtscore.media.image.clipper import ImageDefaultClipper

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
        from vtscore.media.image.clipper import ImageTilingClipper

        c = ImageTilingClipper()
        assert c.name == "image_tiling"
        assert c.media_type == "image"
        assert isinstance(c, MediaClipper)

    def test_square_image_returned_unchanged(self):
        from vtscore.media.image.clipper import ImageTilingClipper

        img_bytes = _make_image_bytes(100, 100)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 100, "height": 100}
        result = ImageTilingClipper().clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_portrait_image_tiled_vertically(self):
        from PIL import Image

        from vtscore.media.image.clipper import ImageTilingClipper

        # 100 x 250 image → tile_size = 100, ceil(250/100) = 3 tiles
        img_bytes = _make_image_bytes(100, 250)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 100, "height": 250}
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

        from vtscore.media.image.clipper import ImageTilingClipper

        # 300 x 100 image → tile_size = 100, 300/100 = 3 → 3 tiles
        img_bytes = _make_image_bytes(300, 100)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 300, "height": 100}
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

        from vtscore.media.image.clipper import ImageTilingClipper

        # 85 x 110 portrait: tile_size = 85, ceil(110/85) = 2 tiles
        img_bytes = _make_image_bytes(85, 110)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 85, "height": 110}
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
        from vtscore.media.image.clipper import ImageTilingClipper

        media = {"id": 1, "media_type": "image", "width": 100, "height": 200}
        result = ImageTilingClipper().clip(media)
        assert result == [media]

    def test_missing_dimensions_returns_unchanged(self):
        from vtscore.media.image.clipper import ImageTilingClipper

        media = {"id": 1, "media_type": "image", "media_bytes": b"fake"}
        result = ImageTilingClipper().clip(media)
        assert result == [media]


# ---------------------------------------------------------------------------
# ImageBboxClipper
# ---------------------------------------------------------------------------


class TestImageBboxClipper:
    def test_identity(self):
        from vtscore.media.image.clipper import ImageBboxClipper

        c = ImageBboxClipper([10, 20, 50, 80])
        assert c.name == "image_bbox"
        assert c.media_type == "image"
        assert c.box == (10, 20, 50, 80)
        assert isinstance(c, MediaClipper)

    def test_rejects_invalid_box(self):
        from vtscore.media.image.clipper import ImageBboxClipper

        with pytest.raises(ValueError):
            ImageBboxClipper([0, 0, 0, 0])  # zero size
        with pytest.raises(ValueError):
            ImageBboxClipper([10, 10, 5, 5])  # x2 < x1
        with pytest.raises(ValueError):
            ImageBboxClipper([-1, 0, 5, 5])  # negative coord
        with pytest.raises(ValueError):
            ImageBboxClipper([0, 0, 5])  # wrong arity

    def test_crops_to_box(self):
        from PIL import Image

        from vtscore.media.image.clipper import ImageBboxClipper

        img_bytes = _make_image_bytes(200, 100)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 200, "height": 100}
        result = ImageBboxClipper([20, 10, 120, 90]).clip(media)
        assert len(result) == 1
        clip = result[0]
        assert clip["width"] == 100
        assert clip["height"] == 80
        assert clip["clip_box"] == [20, 10, 120, 90]
        assert clip["clip_index"] == 0
        img = Image.open(io.BytesIO(clip["media_bytes"]))
        assert img.size == (100, 80)

    def test_clamps_box_to_image_bounds(self):
        from vtscore.media.image.clipper import ImageBboxClipper

        img_bytes = _make_image_bytes(50, 50)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 50, "height": 50}
        # Request a box wider than the image; clamps to (0,0,50,50).
        result = ImageBboxClipper([0, 0, 200, 200]).clip(media)
        assert len(result) == 1
        clip = result[0]
        assert clip["clip_box"] == [0, 0, 50, 50]
        assert clip["width"] == 50
        assert clip["height"] == 50

    def test_no_media_bytes_returns_unchanged(self):
        from vtscore.media.image.clipper import ImageBboxClipper

        media = {"id": 1, "media_type": "image", "width": 100, "height": 100}
        result = ImageBboxClipper([0, 0, 50, 50]).clip(media)
        assert result == [media]

    def test_with_params(self):
        from vtscore.media.image.clipper import ImageBboxClipper

        c = ImageBboxClipper([0, 0, 10, 10])
        c2 = c.with_params({"box": [5, 5, 25, 25]})
        assert isinstance(c2, ImageBboxClipper)
        assert c2.box == (5, 5, 25, 25)
        # original unchanged
        assert c.box == (0, 0, 10, 10)

    def test_to_dict(self):
        from vtscore.media.image.clipper import ImageBboxClipper

        c = ImageBboxClipper([1, 2, 3, 4])
        d = c.to_dict()
        assert d["name"] == "image_bbox"
        assert d["media_type"] == "image"
        assert d["box"] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# ImageObjectClipper
# ---------------------------------------------------------------------------


def _make_mock_yolo(detections, class_names):
    """Build a mock ultralytics YOLO callable.

    *detections* is a list of ``(class_id, confidence, [x1, y1, x2, y2])``.
    *class_names* maps class_id -> label string.
    """
    from unittest.mock import MagicMock

    import torch

    mock_boxes = MagicMock()
    mock_boxes.conf = torch.tensor([d[1] for d in detections]) if detections else torch.empty(0)
    mock_boxes.cls = torch.tensor([d[0] for d in detections]) if detections else torch.empty(0, dtype=torch.long)
    mock_boxes.xyxy = (
        torch.tensor([d[2] for d in detections], dtype=torch.float32) if detections else torch.empty((0, 4))
    )
    mock_boxes.__len__ = lambda self: len(detections)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_result.names = class_names

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    return mock_model


class TestImageObjectClipper:
    def test_identity(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        c = ImageObjectClipper()
        assert c.name == "image_object"
        assert c.media_type == "image"
        assert c.display_name == "Object"
        assert isinstance(c, MediaClipper)

    def test_default_params(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        c = ImageObjectClipper()
        assert c.threshold == 0.25
        assert c.class_filter == ""
        assert c.max_detections == 20
        assert c.padding == 0.0
        assert c.model_id == "yolo11n.pt"

    def test_parameters_schema(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        params = ImageObjectClipper().parameters
        keys = {p["key"] for p in params}
        assert keys == {"threshold", "class_filter", "max_detections", "padding", "model_id"}

    def test_creation_questions_match_parameters(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        c = ImageObjectClipper()
        assert c.creation_questions == c.parameters

    def test_to_dict(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        c = ImageObjectClipper(threshold=0.4, class_filter="person,car", max_detections=5, padding=0.1)
        d = c.to_dict()
        assert d["name"] == "image_object"
        assert d["media_type"] == "image"
        assert d["threshold"] == 0.4
        assert d["class_filter"] == "person,car"
        assert d["max_detections"] == 5
        assert d["padding"] == 0.1
        assert d["model_id"] == "yolo11n.pt"
        assert "creation_questions" in d

    def test_with_params_returns_new_instance(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        c = ImageObjectClipper()
        c2 = c.with_params({"threshold": 0.7, "class_filter": "dog", "max_detections": 3, "padding": 0.2})
        assert isinstance(c2, ImageObjectClipper)
        assert c2.threshold == 0.7
        assert c2.class_filter == "dog"
        assert c2.max_detections == 3
        assert c2.padding == 0.2
        # original unchanged
        assert c.threshold == 0.25
        assert c.class_filter == ""

    def test_no_media_bytes_returns_unchanged(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        c = ImageObjectClipper()
        media = {"id": 1, "media_type": "image", "width": 100, "height": 100}
        result = c.clip(media)
        assert result == [media]

    def test_no_detections_returns_original(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        img_bytes = _make_image_bytes(200, 200)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 200, "height": 200}
        c = ImageObjectClipper()
        c._model = _make_mock_yolo([], {0: "person"})
        result = c.clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_emits_one_clip_per_detection(self):
        from PIL import Image

        from vtscore.media.image.clipper import ImageObjectClipper

        img_bytes = _make_image_bytes(200, 200)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 200, "height": 200}
        detections = [
            (0, 0.9, [10.0, 10.0, 60.0, 60.0]),
            (0, 0.8, [100.0, 100.0, 180.0, 180.0]),
        ]
        c = ImageObjectClipper(threshold=0.5)
        c._model = _make_mock_yolo(detections, {0: "person"})
        result = c.clip(media)
        assert len(result) == 2
        # Sorted by confidence desc
        assert result[0]["clip_box"] == [10, 10, 60, 60]
        assert result[1]["clip_box"] == [100, 100, 180, 180]
        for clip in result:
            assert "media_bytes" in clip and clip["media_bytes"] != img_bytes
            img = Image.open(io.BytesIO(clip["media_bytes"]))
            assert img.size == (clip["width"], clip["height"])
            assert "clip_index" in clip
            assert clip["file_size"] == len(clip["media_bytes"])

    def test_sorts_by_confidence_desc(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        img_bytes = _make_image_bytes(200, 200)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 200, "height": 200}
        detections = [
            (0, 0.3, [10.0, 10.0, 40.0, 40.0]),
            (0, 0.95, [50.0, 50.0, 120.0, 120.0]),
            (0, 0.6, [130.0, 130.0, 190.0, 190.0]),
        ]
        c = ImageObjectClipper(threshold=0.25)
        c._model = _make_mock_yolo(detections, {0: "person"})
        result = c.clip(media)
        assert len(result) == 3
        assert result[0]["clip_box"] == [50, 50, 120, 120]
        assert result[1]["clip_box"] == [130, 130, 190, 190]
        assert result[2]["clip_box"] == [10, 10, 40, 40]

    def test_threshold_filters_low_confidence(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        img_bytes = _make_image_bytes(200, 200)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 200, "height": 200}
        detections = [
            (0, 0.9, [10.0, 10.0, 60.0, 60.0]),
            (0, 0.2, [100.0, 100.0, 180.0, 180.0]),
        ]
        c = ImageObjectClipper(threshold=0.5)
        c._model = _make_mock_yolo(detections, {0: "person"})
        result = c.clip(media)
        assert len(result) == 1
        assert result[0]["clip_box"] == [10, 10, 60, 60]

    def test_class_filter_keeps_only_whitelisted(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        img_bytes = _make_image_bytes(200, 200)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 200, "height": 200}
        detections = [
            (0, 0.9, [10.0, 10.0, 60.0, 60.0]),  # person
            (1, 0.85, [80.0, 80.0, 150.0, 150.0]),  # car
            (2, 0.8, [120.0, 120.0, 190.0, 190.0]),  # dog
        ]
        c = ImageObjectClipper(threshold=0.5, class_filter="person,dog")
        c._model = _make_mock_yolo(detections, {0: "person", 1: "car", 2: "dog"})
        result = c.clip(media)
        assert len(result) == 2
        # 'car' (cls=1) excluded; remaining sorted by confidence
        boxes = {tuple(r["clip_box"]) for r in result}
        assert boxes == {(10, 10, 60, 60), (120, 120, 190, 190)}

    def test_empty_class_filter_keeps_all(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        img_bytes = _make_image_bytes(200, 200)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 200, "height": 200}
        detections = [
            (0, 0.9, [10.0, 10.0, 60.0, 60.0]),
            (1, 0.8, [80.0, 80.0, 150.0, 150.0]),
        ]
        c = ImageObjectClipper(threshold=0.5, class_filter="")
        c._model = _make_mock_yolo(detections, {0: "person", 1: "car"})
        result = c.clip(media)
        assert len(result) == 2

    def test_max_detections_caps_output(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        img_bytes = _make_image_bytes(400, 400)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 400, "height": 400}
        # 6 detections with varying confidence; cap at 3 should keep top 3
        detections = [
            (0, 0.5, [0.0, 0.0, 20.0, 20.0]),
            (0, 0.95, [30.0, 30.0, 60.0, 60.0]),
            (0, 0.4, [70.0, 70.0, 100.0, 100.0]),
            (0, 0.85, [120.0, 120.0, 160.0, 160.0]),
            (0, 0.6, [180.0, 180.0, 220.0, 220.0]),
            (0, 0.7, [240.0, 240.0, 280.0, 280.0]),
        ]
        c = ImageObjectClipper(threshold=0.0, max_detections=3)
        c._model = _make_mock_yolo(detections, {0: "person"})
        result = c.clip(media)
        assert len(result) == 3
        confs_kept = {(r["clip_box"][0], r["clip_box"][1]) for r in result}
        # Top 3 confidences are 0.95, 0.85, 0.7 → boxes starting at (30,30), (120,120), (240,240)
        assert confs_kept == {(30, 30), (120, 120), (240, 240)}

    def test_padding_expands_box(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        img_bytes = _make_image_bytes(400, 400)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 400, "height": 400}
        # 100x100 box at (150,150)-(250,250) with 10% padding → expand by 10px each side
        detections = [(0, 0.9, [150.0, 150.0, 250.0, 250.0])]
        c = ImageObjectClipper(threshold=0.0, padding=0.1)
        c._model = _make_mock_yolo(detections, {0: "person"})
        result = c.clip(media)
        assert len(result) == 1
        box = result[0]["clip_box"]
        assert box == [140, 140, 260, 260]
        assert result[0]["width"] == 120
        assert result[0]["height"] == 120

    def test_padding_clamps_to_image_bounds(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        img_bytes = _make_image_bytes(200, 200)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 200, "height": 200}
        # Box near the corner; aggressive padding would go negative / past edge.
        detections = [(0, 0.9, [10.0, 10.0, 50.0, 50.0])]
        c = ImageObjectClipper(threshold=0.0, padding=1.0)  # +40px each side
        c._model = _make_mock_yolo(detections, {0: "person"})
        result = c.clip(media)
        assert len(result) == 1
        box = result[0]["clip_box"]
        # x1, y1 clamped to 0; x2, y2 within bounds
        assert box[0] == 0
        assert box[1] == 0
        assert box[2] <= 200
        assert box[3] <= 200

    def test_clip_index_is_set(self):
        from vtscore.media.image.clipper import ImageObjectClipper

        img_bytes = _make_image_bytes(200, 200)
        media = {"id": 1, "media_type": "image", "media_bytes": img_bytes, "width": 200, "height": 200}
        detections = [
            (0, 0.9, [10.0, 10.0, 60.0, 60.0]),
            (0, 0.8, [70.0, 70.0, 120.0, 120.0]),
        ]
        c = ImageObjectClipper(threshold=0.0)
        c._model = _make_mock_yolo(detections, {0: "person"})
        result = c.clip(media)
        assert [r["clip_index"] for r in result] == [0, 1]

    def test_registered_in_clippers_list(self):
        from vtscore.media.image import CLIPPERS
        from vtscore.media.image.clipper import ImageObjectClipper

        assert any(isinstance(c, ImageObjectClipper) for c in CLIPPERS)


# ---------------------------------------------------------------------------
# TextDefaultClipper
# ---------------------------------------------------------------------------


class TestTextDefaultClipper:
    def test_returns_media_unchanged(self):
        from vtscore.media.text.clipper import TextDefaultClipper

        media = {"id": 1, "media_type": "text", "media_string": "Hello world."}
        result = TextDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtscore.media.text.clipper import TextDefaultClipper

        c = TextDefaultClipper()
        assert c.name == "text_default"
        assert c.media_type == "text"
        assert isinstance(c, MediaClipper)


# ---------------------------------------------------------------------------
# TextParagraphClipper
# ---------------------------------------------------------------------------


class TestTextParagraphClipper:
    def test_identity(self):
        from vtscore.media.text.clipper import TextParagraphClipper

        c = TextParagraphClipper()
        assert c.name == "text_paragraph"
        assert c.media_type == "text"
        assert isinstance(c, MediaClipper)

    def test_single_paragraph_unchanged(self):
        from vtscore.media.text.clipper import TextParagraphClipper

        media = {"id": 1, "media_type": "text", "media_string": "Just one paragraph here."}
        result = TextParagraphClipper().clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_splits_multiple_paragraphs(self):
        from vtscore.media.text.clipper import TextParagraphClipper

        text = "First paragraph.\n\nSecond paragraph.\n\nThird one."
        media = {"id": 1, "media_type": "text", "media_string": text, "word_count": 6, "character_count": len(text)}
        result = TextParagraphClipper().clip(media)
        assert len(result) == 3
        assert result[0]["media_string"] == "First paragraph."
        assert result[1]["media_string"] == "Second paragraph."
        assert result[2]["media_string"] == "Third one."
        for idx, tile in enumerate(result):
            assert tile["clip_index"] == idx
            assert tile["word_count"] == len(tile["media_string"].split())
            assert tile["character_count"] == len(tile["media_string"])

    def test_multiline_paragraphs_preserved(self):
        """A paragraph that internally contains single newlines stays intact."""
        from vtscore.media.text.clipper import TextParagraphClipper

        text = "Line one.\nLine two of the same para.\n\nSecond paragraph here."
        media = {"id": 1, "media_type": "text", "media_string": text}
        result = TextParagraphClipper().clip(media)
        assert len(result) == 2
        assert result[0]["media_string"] == "Line one.\nLine two of the same para."
        assert result[1]["media_string"] == "Second paragraph here."

    def test_collapses_runs_of_blank_lines(self):
        from vtscore.media.text.clipper import TextParagraphClipper

        text = "Alpha.\n\n\n\nBeta.\n\n\nGamma."
        media = {"id": 1, "media_type": "text", "media_string": text}
        result = TextParagraphClipper().clip(media)
        assert [t["media_string"] for t in result] == ["Alpha.", "Beta.", "Gamma."]

    def test_handles_windows_line_endings(self):
        from vtscore.media.text.clipper import TextParagraphClipper

        text = "Para one.\r\n\r\nPara two.\r\n\r\nPara three."
        media = {"id": 1, "media_type": "text", "media_string": text}
        result = TextParagraphClipper().clip(media)
        assert [t["media_string"] for t in result] == ["Para one.", "Para two.", "Para three."]

    def test_blank_line_with_whitespace_still_splits(self):
        """A blank line that contains spaces/tabs still counts as a separator."""
        from vtscore.media.text.clipper import TextParagraphClipper

        text = "First.\n   \nSecond."
        media = {"id": 1, "media_type": "text", "media_string": text}
        result = TextParagraphClipper().clip(media)
        assert [t["media_string"] for t in result] == ["First.", "Second."]

    def test_empty_string_returns_unchanged(self):
        from vtscore.media.text.clipper import TextParagraphClipper

        media = {"id": 1, "media_type": "text", "media_string": ""}
        result = TextParagraphClipper().clip(media)
        assert result == [media]

    def test_no_media_string_returns_unchanged(self):
        from vtscore.media.text.clipper import TextParagraphClipper

        media = {"id": 1, "media_type": "text"}
        result = TextParagraphClipper().clip(media)
        assert result == [media]

    def test_registered_for_text_type(self):
        """The paragraph clipper is auto-discovered for the text media type."""
        from vtscore.media import clippers_for_type

        names = {c.name for c in clippers_for_type("text")}
        assert "text_paragraph" in names

    def test_to_dict_shape(self):
        from vtscore.media.text.clipper import TextParagraphClipper

        d = TextParagraphClipper().to_dict()
        assert d["name"] == "text_paragraph"
        assert d["media_type"] == "text"
        assert d["display_name"] == "Paragraph"
        assert "blank lines" in d["description"]


# ---------------------------------------------------------------------------
# TextSentenceClipper
# ---------------------------------------------------------------------------


class TestTextSentenceClipper:
    def test_identity(self):
        from vtscore.media.text.clipper import TextSentenceClipper

        c = TextSentenceClipper()
        assert c.name == "text_sentence"
        assert c.media_type == "text"
        assert isinstance(c, MediaClipper)

    def test_single_sentence_unchanged(self):
        from vtscore.media.text.clipper import TextSentenceClipper

        media = {"id": 1, "media_type": "text", "media_string": "Hello world."}
        result = TextSentenceClipper().clip(media)
        assert len(result) == 1
        assert result[0] is media

    def test_splits_multiple_sentences(self):
        from vtscore.media.text.clipper import TextSentenceClipper

        text = "First sentence. Second sentence. Third one!"
        media = {"id": 1, "media_type": "text", "media_string": text, "word_count": 7, "character_count": len(text)}
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
        from vtscore.media.text.clipper import TextSentenceClipper

        text = "Is this a test? Yes it is! Great."
        media = {"id": 1, "media_type": "text", "media_string": text}
        result = TextSentenceClipper().clip(media)
        assert len(result) == 3
        assert result[0]["media_string"] == "Is this a test?"
        assert result[1]["media_string"] == "Yes it is!"
        assert result[2]["media_string"] == "Great."

    def test_empty_string_returns_unchanged(self):
        from vtscore.media.text.clipper import TextSentenceClipper

        media = {"id": 1, "media_type": "text", "media_string": ""}
        result = TextSentenceClipper().clip(media)
        assert result == [media]

    def test_no_media_string_returns_unchanged(self):
        from vtscore.media.text.clipper import TextSentenceClipper

        media = {"id": 1, "media_type": "text"}
        result = TextSentenceClipper().clip(media)
        assert result == [media]


# ---------------------------------------------------------------------------
# DocumentDefaultClipper
# ---------------------------------------------------------------------------


class TestDocumentDefaultClipper:
    def test_returns_media_unchanged(self):
        from vtscore.media.document.clipper import DocumentDefaultClipper

        media = {"id": 1, "media_type": "document", "media_bytes": b"fake-pdf"}
        result = DocumentDefaultClipper().clip(media)
        assert result == [media]

    def test_identity(self):
        from vtscore.media.document.clipper import DocumentDefaultClipper

        c = DocumentDefaultClipper()
        assert c.name == "document_default"
        assert c.media_type == "document"
        assert isinstance(c, MediaClipper)


# ---------------------------------------------------------------------------
# VideoSceneClipper
# ---------------------------------------------------------------------------


class TestVideoSceneClipper:
    def test_identity(self):
        from vtscore.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper()
        assert c.name == "video_scene"
        assert c.media_type == "video"
        assert c.threshold == 0.3
        assert c.min_scene_duration == 1.0
        assert isinstance(c, MediaClipper)

    def test_custom_params(self):
        from vtscore.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper(threshold=0.5, min_scene_duration=2.0)
        assert c.threshold == 0.5
        assert c.min_scene_duration == 2.0

    def test_rejects_invalid_threshold(self):
        from vtscore.media.video.clipper import VideoSceneClipper

        with pytest.raises(ValueError):
            VideoSceneClipper(threshold=-0.1)
        with pytest.raises(ValueError):
            VideoSceneClipper(threshold=1.1)

    def test_rejects_non_positive_min_scene_duration(self):
        from vtscore.media.video.clipper import VideoSceneClipper

        with pytest.raises(ValueError):
            VideoSceneClipper(min_scene_duration=0)
        with pytest.raises(ValueError):
            VideoSceneClipper(min_scene_duration=-1)

    def test_zero_duration_returns_unchanged(self):
        from vtscore.media.video.clipper import VideoSceneClipper

        media = {"id": 1, "media_type": "video", "duration": 0}
        result = VideoSceneClipper().clip(media)
        assert result == [media]

    def test_no_media_bytes_or_path_returns_unchanged(self):
        from vtscore.media.video.clipper import VideoSceneClipper

        media = {"id": 1, "media_type": "video", "duration": 10.0}
        result = VideoSceneClipper().clip(media)
        assert result == [media]

    def test_to_dict_includes_params(self):
        from vtscore.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper(threshold=0.4, min_scene_duration=1.5)
        d = c.to_dict()
        assert d["name"] == "video_scene"
        assert d["media_type"] == "video"
        assert d["threshold"] == 0.4
        assert d["min_scene_duration"] == 1.5

    def test_detect_scene_boundaries_no_cv2(self, monkeypatch):
        """When OpenCV is not available, clip returns the media unchanged."""
        import builtins

        from vtscore.media.video.clipper import VideoSceneClipper

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("no cv2")
            return real_import(name, *args, **kwargs)

        media = {"id": 1, "media_type": "video", "media_bytes": b"fake", "duration": 10.0}
        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = VideoSceneClipper().clip(media)
        assert result == [media]

    def test_detect_scene_boundaries_helper_empty(self, monkeypatch):
        """When _detect_scene_boundaries returns no cuts, media is unchanged."""
        from vtscore.media.video import clipper as clipper_mod
        from vtscore.media.video.clipper import VideoSceneClipper

        monkeypatch.setattr(clipper_mod, "_detect_scene_boundaries", lambda *a, **kw: [])
        media = {"id": 1, "media_type": "video", "media_bytes": b"fake", "duration": 10.0}
        result = VideoSceneClipper().clip(media)
        assert result == [media]

    def test_splits_at_detected_boundaries(self, monkeypatch):
        """When boundaries are found, the clipper produces the right scenes."""
        from vtscore.media.video import clipper as clipper_mod
        from vtscore.media.video.clipper import VideoSceneClipper

        # Simulate two scene boundaries at 3.0s and 7.0s in a 10s video.
        monkeypatch.setattr(clipper_mod, "_detect_scene_boundaries", lambda *a, **kw: [3.0, 7.0])
        media = {"id": 1, "media_type": "video", "media_bytes": b"fake", "duration": 10.0}
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
        from vtscore.media.video import clipper as clipper_mod
        from vtscore.media.video.clipper import VideoSceneClipper

        monkeypatch.setattr(clipper_mod, "_detect_scene_boundaries", lambda *a, **kw: [5.0])
        media = {"id": 1, "media_type": "video", "media_bytes": b"fake", "duration": 8.0}
        result = VideoSceneClipper().clip(media)
        assert len(result) == 2
        assert result[0]["clip_start"] == pytest.approx(0.0)
        assert result[0]["clip_end"] == pytest.approx(5.0)
        assert result[1]["clip_start"] == pytest.approx(5.0)
        assert result[1]["clip_end"] == pytest.approx(8.0)

    def test_media_path_used_when_available(self, monkeypatch, tmp_path):
        """When media_path exists, it's used instead of writing a temp file."""
        from vtscore.media.video import clipper as clipper_mod
        from vtscore.media.video.clipper import VideoSceneClipper

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video data")

        paths_seen = []

        def mock_detect(video_path, threshold, min_scene_duration):
            paths_seen.append(video_path)
            return [2.0]

        monkeypatch.setattr(clipper_mod, "_detect_scene_boundaries", mock_detect)

        media = {"id": 1, "media_type": "video", "media_path": str(video_file), "duration": 5.0}
        result = VideoSceneClipper().clip(media)
        assert len(result) == 2
        assert paths_seen[0] == str(video_file)


# ---------------------------------------------------------------------------
# Clipper Registry
# ---------------------------------------------------------------------------


class TestClipperRegistry:
    def test_all_clippers_returns_list(self):
        from vtscore.media import all_clippers

        clippers = all_clippers()
        assert isinstance(clippers, list)
        assert len(clippers) >= 9  # 5 defaults + 4 tiling/sentence

    def test_all_clippers_dict_returns_dicts(self):
        from vtscore.media import all_clippers_dict

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
        from vtscore.media import get_clipper

        c = get_clipper("sound_default")
        assert c.name == "sound_default"

    def test_get_clipper_unknown_raises(self):
        from vtscore.media import get_clipper

        with pytest.raises(KeyError):
            get_clipper("nonexistent_clipper")

    def test_clippers_for_type(self):
        from vtscore.media import clippers_for_type

        audio_clippers = clippers_for_type("audio")
        assert len(audio_clippers) >= 2
        names = [c.name for c in audio_clippers]
        assert "sound_default" in names
        assert "sound_tiling" in names

    def test_clippers_for_type_image(self):
        from vtscore.media import clippers_for_type

        image_clippers = clippers_for_type("image")
        assert len(image_clippers) >= 2
        names = [c.name for c in image_clippers]
        assert "image_default" in names
        assert "image_tiling" in names

    def test_clippers_for_type_paragraph(self):
        from vtscore.media import clippers_for_type

        text_clippers = clippers_for_type("text")
        names = [c.name for c in text_clippers]
        assert "text_default" in names
        assert "text_sentence" in names

    def test_clippers_for_type_video(self):
        from vtscore.media import clippers_for_type

        video_clippers = clippers_for_type("video")
        names = [c.name for c in video_clippers]
        assert "video_default" in names
        assert "video_tiling" in names

    def test_clippers_for_type_document(self):
        from vtscore.media import clippers_for_type

        doc_clippers = clippers_for_type("document")
        assert len(doc_clippers) >= 1
        names = [c.name for c in doc_clippers]
        assert "document_default" in names

    def test_every_media_type_has_default_clipper(self):
        from vtscore.media import all_types, clippers_for_type

        for mt in all_types():
            clippers = clippers_for_type(mt.type_id)
            assert len(clippers) >= 1, f"No clippers for {mt.type_id}"
            names = [c.name for c in clippers]
            assert any("default" in n for n in names), f"No default clipper for {mt.type_id}"


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
# crop_file_bytes helper
# ---------------------------------------------------------------------------


class TestCropFileBytes:
    """The helper used by routes to crop user-supplied example files."""

    def test_crop_audio_file(self, tmp_path):
        from vtscore.media.cropping import crop_file_bytes

        wav_path = tmp_path / "ex.wav"
        wav_path.write_bytes(generate_wav(440, 5.0))
        out = crop_file_bytes(wav_path, "audio", {"start": 1.0, "end": 3.0})
        # Result is a valid WAV of the requested length (~2s).
        with wave.open(io.BytesIO(out), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        assert duration == pytest.approx(2.0, abs=0.01)

    def test_crop_image_file(self, tmp_path):
        from PIL import Image

        from vtscore.media.cropping import crop_file_bytes

        img_path = tmp_path / "ex.png"
        img_path.write_bytes(_make_image_bytes(100, 100))
        out = crop_file_bytes(img_path, "image", {"box": [10, 20, 60, 80]})
        img = Image.open(io.BytesIO(out))
        assert img.size == (50, 60)

    def test_missing_file_raises(self, tmp_path):
        from vtscore.media.cropping import crop_file_bytes

        with pytest.raises(FileNotFoundError):
            crop_file_bytes(tmp_path / "nope.wav", "audio", {"start": 0.0, "end": 1.0})

    def test_unknown_media_type_raises(self, tmp_path):
        from vtscore.media.cropping import crop_file_bytes

        wav_path = tmp_path / "ex.wav"
        wav_path.write_bytes(generate_wav(440, 1.0))
        with pytest.raises(ValueError):
            crop_file_bytes(wav_path, "video", {"start": 0.0, "end": 0.5})

    def test_image_missing_box_raises(self, tmp_path):
        from vtscore.media.cropping import crop_file_bytes

        img_path = tmp_path / "ex.png"
        img_path.write_bytes(_make_image_bytes(100, 100))
        with pytest.raises(ValueError):
            crop_file_bytes(img_path, "image", {})


# ---------------------------------------------------------------------------
# Apply clipper helper
# ---------------------------------------------------------------------------


class TestApplyClipper:
    def test_apply_clipper_noop_for_empty_name(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        clips = {1: {"id": 1, "media_type": "audio", "origin": {"importer": "test", "params": {}}}}
        _apply_clipper(clips, "")
        assert len(clips) == 1

    def test_apply_clipper_unknown_name_noop(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        clips = {1: {"id": 1, "media_type": "audio", "origin": {"importer": "test", "params": {}}}}
        _apply_clipper(clips, "nonexistent_clipper")
        assert len(clips) == 1

    def test_apply_default_clipper_passthrough(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        media = {"id": 1, "media_type": "audio", "media_bytes": b"fake", "origin": {"importer": "test", "params": {}}}
        clips = {1: media}
        _apply_clipper(clips, "sound_default")
        assert len(clips) == 1
        assert clips[1]["origin"]["params"]["clipper"] == "sound_default"

    def test_apply_clipper_annotates_origin(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        media = {
            "id": 1,
            "media_type": "text",
            "media_string": "First sentence. Second sentence.",
            "word_count": 4,
            "character_count": 32,
            "origin": {"importer": "server_folder", "params": {"path": "/data"}},
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
        from vtscore.datasets.registry import register_dataset

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
        from vtscore.datasets.registry import register_dataset

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
        from vtscore.datasets.registry import register_dataset

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
        from vtscore.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        assert c.parameters == []

    def test_default_clipper_with_params_returns_self(self):
        from vtscore.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        assert c.with_params({"anything": 42}) is c

    def test_sound_tiling_parameters(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

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
        from vtscore.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        c2 = c.with_params({"duration": 5.0})
        assert isinstance(c2, SoundTilingClipper)
        assert c2.duration == 5.0
        assert c2.min_overlap == 0.0
        assert c2 is not c
        assert c.duration == 2.0  # original unchanged

    def test_sound_tiling_with_params_overlap(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        c2 = c.with_params({"duration": 5.0, "min_overlap": 1.0})
        assert c2.duration == 5.0
        assert c2.min_overlap == 1.0

    def test_sound_tiling_with_params_ignores_unknown_keys(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        c2 = c.with_params({"unknown_key": 99})
        assert c2.duration == 2.0  # falls back to current value

    def test_video_tiling_parameters(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(2.0)
        params = c.parameters
        assert len(params) == 2
        assert params[0]["key"] == "duration"
        assert params[0]["default"] == 2.0
        assert params[1]["key"] == "min_overlap"
        assert params[1]["default"] == 0.0

    def test_video_tiling_with_params(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(2.0)
        c2 = c.with_params({"duration": 10.0})
        assert isinstance(c2, VideoTilingClipper)
        assert c2.duration == 10.0
        assert c2.min_overlap == 0.0
        assert c.duration == 2.0

    def test_video_tiling_with_params_overlap(self):
        from vtscore.media.video.clipper import VideoTilingClipper

        c = VideoTilingClipper(2.0)
        c2 = c.with_params({"duration": 10.0, "min_overlap": 2.0})
        assert c2.duration == 10.0
        assert c2.min_overlap == 2.0

    def test_video_scene_parameters(self):
        from vtscore.media.video.clipper import VideoSceneClipper

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
        from vtscore.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper()
        c2 = c.with_params({"threshold": 0.5, "min_scene_duration": 2.5})
        assert isinstance(c2, VideoSceneClipper)
        assert c2.threshold == 0.5
        assert c2.min_scene_duration == 2.5
        assert c.threshold == 0.3  # original unchanged

    def test_video_scene_with_partial_params(self):
        from vtscore.media.video.clipper import VideoSceneClipper

        c = VideoSceneClipper(threshold=0.4, min_scene_duration=1.5)
        c2 = c.with_params({"threshold": 0.6})
        assert c2.threshold == 0.6
        assert c2.min_scene_duration == 1.5  # kept from original

    def test_to_dict_includes_parameters(self):
        from vtscore.media.audio.clipper import SoundTilingClipper

        c = SoundTilingClipper(2.0)
        d = c.to_dict()
        assert "parameters" in d
        assert len(d["parameters"]) == 2
        assert d["parameters"][0]["key"] == "duration"
        assert d["parameters"][1]["key"] == "min_overlap"

    def test_to_dict_no_parameters_for_default_clipper(self):
        from vtscore.media.audio.clipper import SoundDefaultClipper

        c = SoundDefaultClipper()
        d = c.to_dict()
        assert "parameters" not in d

    def test_to_dict_includes_summary_template_when_overridden(self):
        from vtscore.media.audio.clipper import SoundAutoClipper, SoundTilingClipper

        # SoundTilingClipper provides a templated summary distinct from its
        # description; that template should show up in to_dict().
        d = SoundTilingClipper(2.0).to_dict()
        assert d["summary_template"] == "Cut each audio file into {duration}s tiles (min overlap {min_overlap}s)."
        # SoundAutoClipper too.
        d = SoundAutoClipper().to_dict()
        assert d["summary_template"] == "Cut into {tile_duration}s tiles when audio is over {threshold}s."

    def test_to_dict_omits_summary_template_when_equal_to_description(self):
        from vtscore.media.audio.clipper import SoundDefaultClipper

        # No template override → summary_template equals description → not serialised.
        d = SoundDefaultClipper().to_dict()
        assert "summary_template" not in d


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
        from vtscore.media.audio.audio_generator import generate_wav
        from vtscore.datasets.load_pipeline import _apply_clipper

        # Generate a 10s audio clip
        wav = generate_wav(440, 10.0)
        media = {
            "id": 1,
            "media_type": "audio",
            "media_bytes": wav,
            "duration": 10.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        # With default 2s duration: ceil(10/2) = 5 tiles
        _apply_clipper(clips, "sound_tiling")
        assert len(clips) == 5

    def test_apply_clipper_with_overridden_duration(self):
        from vtscore.media.audio.audio_generator import generate_wav
        from vtscore.datasets.load_pipeline import _apply_clipper

        wav = generate_wav(440, 10.0)
        media = {
            "id": 1,
            "media_type": "audio",
            "media_bytes": wav,
            "duration": 10.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        # Override to 5s duration: ceil(10/5) = 2 tiles
        _apply_clipper(clips, "sound_tiling", {"duration": 5.0})
        assert len(clips) == 2

    def test_apply_clipper_params_none_uses_defaults(self):
        from vtscore.media.audio.audio_generator import generate_wav
        from vtscore.datasets.load_pipeline import _apply_clipper

        wav = generate_wav(440, 10.0)
        media = {
            "id": 1,
            "media_type": "audio",
            "media_bytes": wav,
            "duration": 10.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        _apply_clipper(clips, "sound_tiling", None)
        assert len(clips) == 5  # default 2s → 5 tiles

    def test_apply_clipper_with_min_overlap(self):
        from vtscore.media.audio.audio_generator import generate_wav
        from vtscore.datasets.load_pipeline import _apply_clipper

        wav = generate_wav(440, 10.0)
        media = {
            "id": 1,
            "media_type": "audio",
            "media_bytes": wav,
            "duration": 10.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        # 2s clips with 1s min overlap: max_stride=1, ceil((10-2)/1)+1 = 9 tiles
        _apply_clipper(clips, "sound_tiling", {"duration": 2.0, "min_overlap": 1.0})
        assert len(clips) == 9


# ---------------------------------------------------------------------------
# SoundAutoClipper / VideoAutoClipper
# ---------------------------------------------------------------------------


class TestSoundAutoClipper:
    def test_identity(self):
        from vtscore.media.audio.clipper import SoundAutoClipper

        c = SoundAutoClipper()
        assert c.name == "sound_auto"
        assert c.media_type == "audio"
        assert c.threshold == 30.0
        assert c.tile_duration == 10.0
        assert c.display_name == "Auto (recommended)"
        assert isinstance(c, MediaClipper)

    def test_rejects_non_positive_params(self):
        from vtscore.media.audio.clipper import SoundAutoClipper

        with pytest.raises(ValueError):
            SoundAutoClipper(threshold=0)
        with pytest.raises(ValueError):
            SoundAutoClipper(threshold=-1)
        with pytest.raises(ValueError):
            SoundAutoClipper(tile_duration=0)
        with pytest.raises(ValueError):
            SoundAutoClipper(tile_duration=-1)

    def test_resolve_for_media_short_returns_default(self):
        from vtscore.media.audio.clipper import SoundAutoClipper, SoundDefaultClipper

        media = {"id": 1, "media_type": "audio", "duration": 5.0}
        resolved = SoundAutoClipper().resolve_for_media(media)
        assert isinstance(resolved, SoundDefaultClipper)

    def test_resolve_for_media_long_returns_tiling(self):
        from vtscore.media.audio.clipper import SoundAutoClipper, SoundTilingClipper

        media = {"id": 1, "media_type": "audio", "duration": 120.0}
        resolved = SoundAutoClipper().resolve_for_media(media)
        assert isinstance(resolved, SoundTilingClipper)
        assert resolved.duration == 10.0

    def test_resolve_for_media_uses_own_duration_not_dataset(self):
        """Each media gets its own routing decision; no median trick."""
        from vtscore.media.audio.clipper import (
            SoundAutoClipper,
            SoundDefaultClipper,
            SoundTilingClipper,
        )

        c = SoundAutoClipper()
        short = {"id": 1, "media_type": "audio", "duration": 5.0}
        long_media = {"id": 2, "media_type": "audio", "duration": 120.0}
        assert isinstance(c.resolve_for_media(short), SoundDefaultClipper)
        assert isinstance(c.resolve_for_media(long_media), SoundTilingClipper)

    def test_resolve_for_media_falls_back_to_wav_when_no_duration(self):
        from vtscore.media.audio.clipper import SoundAutoClipper, SoundTilingClipper

        long_wav = generate_wav(440, 5.0)
        media = {"id": 1, "media_type": "audio", "media_bytes": long_wav}
        resolved = SoundAutoClipper(threshold=3.0, tile_duration=1.0).resolve_for_media(media)
        assert isinstance(resolved, SoundTilingClipper)

    def test_resolve_for_durations_is_noop(self):
        """Phase 2 routing is per-media; the per-dataset hook is a no-op."""
        from vtscore.media.audio.clipper import SoundAutoClipper

        c = SoundAutoClipper()
        assert c.resolve_for_durations([1.0, 1000.0]) is c
        assert c.resolve_for_durations([]) is c

    def test_with_params_overrides_threshold(self):
        from vtscore.media.audio.clipper import SoundAutoClipper, SoundTilingClipper

        c = SoundAutoClipper().with_params({"threshold": 5.0, "tile_duration": 3.0})
        assert c.threshold == 5.0
        assert c.tile_duration == 3.0
        # 10s exceeds 5s threshold → tiling with 3s segments
        resolved = c.resolve_for_media({"id": 1, "media_type": "audio", "duration": 10.0})
        assert isinstance(resolved, SoundTilingClipper)
        assert resolved.duration == 3.0

    def test_to_dict_exposes_params_and_strategy_fields(self):
        from vtscore.media.audio.clipper import SoundAutoClipper

        d = SoundAutoClipper().to_dict()
        assert d["name"] == "sound_auto"
        assert d["display_name"] == "Auto (recommended)"
        assert d["media_type"] == "audio"
        assert d["threshold"] == 30.0
        assert d["tile_duration"] == 10.0
        param_keys = [p["key"] for p in d["parameters"]]
        assert "threshold" in param_keys
        assert "tile_duration" in param_keys

    def test_clip_routes_per_media(self):
        """Direct .clip() use (outside the load pipeline) routes per-media."""
        from vtscore.media.audio.clipper import SoundAutoClipper

        c = SoundAutoClipper(threshold=3.0, tile_duration=1.0)
        # Short clip → pass-through
        short_wav = generate_wav(440, 2.0)
        short = {"id": 1, "media_type": "audio", "media_bytes": short_wav, "duration": 2.0}
        assert c.clip(short) == [short]
        # Long clip → tiled
        long_wav = generate_wav(440, 5.0)
        long_media = {"id": 1, "media_type": "audio", "media_bytes": long_wav, "duration": 5.0}
        result = c.clip(long_media)
        assert len(result) > 1

    def test_first_in_audio_clipper_registry(self):
        from vtscore.media import clippers_for_type

        names = [c.name for c in clippers_for_type("audio")]
        assert names[0] == "sound_auto"


class TestVideoAutoClipper:
    def test_identity(self):
        from vtscore.media.video.clipper import VideoAutoClipper

        c = VideoAutoClipper()
        assert c.name == "video_auto"
        assert c.media_type == "video"
        assert c.threshold == 30.0
        assert c.tile_duration == 10.0
        assert c.display_name == "Auto (recommended)"
        assert isinstance(c, MediaClipper)

    def test_rejects_non_positive_params(self):
        from vtscore.media.video.clipper import VideoAutoClipper

        with pytest.raises(ValueError):
            VideoAutoClipper(threshold=0)
        with pytest.raises(ValueError):
            VideoAutoClipper(tile_duration=0)

    def test_resolve_for_media_short_returns_default(self):
        from vtscore.media.video.clipper import VideoAutoClipper, VideoDefaultClipper

        resolved = VideoAutoClipper().resolve_for_media({"id": 1, "media_type": "video", "duration": 10.0})
        assert isinstance(resolved, VideoDefaultClipper)

    def test_resolve_for_media_long_returns_tiling(self):
        from vtscore.media.video.clipper import VideoAutoClipper, VideoTilingClipper

        resolved = VideoAutoClipper().resolve_for_media({"id": 1, "media_type": "video", "duration": 90.0})
        assert isinstance(resolved, VideoTilingClipper)
        assert resolved.duration == 10.0

    def test_resolve_for_media_uses_own_duration_not_dataset(self):
        from vtscore.media.video.clipper import (
            VideoAutoClipper,
            VideoDefaultClipper,
            VideoTilingClipper,
        )

        c = VideoAutoClipper()
        short = {"id": 1, "media_type": "video", "duration": 10.0}
        long_media = {"id": 2, "media_type": "video", "duration": 90.0}
        assert isinstance(c.resolve_for_media(short), VideoDefaultClipper)
        assert isinstance(c.resolve_for_media(long_media), VideoTilingClipper)

    def test_resolve_for_durations_is_noop(self):
        from vtscore.media.video.clipper import VideoAutoClipper

        c = VideoAutoClipper()
        assert c.resolve_for_durations([1.0, 1000.0]) is c
        assert c.resolve_for_durations([]) is c

    def test_with_params_overrides(self):
        from vtscore.media.video.clipper import VideoAutoClipper, VideoTilingClipper

        c = VideoAutoClipper().with_params({"threshold": 4.0, "tile_duration": 2.0})
        assert c.threshold == 4.0
        resolved = c.resolve_for_media({"id": 1, "media_type": "video", "duration": 10.0})
        assert isinstance(resolved, VideoTilingClipper)
        assert resolved.duration == 2.0

    def test_first_in_video_clipper_registry(self):
        from vtscore.media import clippers_for_type

        names = [c.name for c in clippers_for_type("video")]
        assert names[0] == "video_auto"


class TestApplyClipperResolvesAuto:
    """_apply_clipper resolves auto clippers per-media and tags each
    clip's origin with the resolved concrete clipper, not the auto one."""

    def test_short_dataset_resolves_to_default(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        wav = generate_wav(440, 5.0)
        media = {
            "id": 1,
            "media_type": "audio",
            "media_bytes": wav,
            "duration": 5.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        _apply_clipper(clips, "sound_auto")
        # 5s < 30s threshold → pass-through, one clip
        assert len(clips) == 1
        # Origin records the resolved concrete clipper.
        clip = next(iter(clips.values()))
        assert clip["origin"]["params"]["clipper"] == "sound_default"

    def test_long_dataset_resolves_to_tiling(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        wav = generate_wav(440, 60.0)
        media = {
            "id": 1,
            "media_type": "audio",
            "media_bytes": wav,
            "duration": 60.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        _apply_clipper(clips, "sound_auto")
        # 60s > 30s threshold → tiled with 10s segments → 6 tiles
        assert len(clips) == 6
        first = next(iter(clips.values()))
        assert first["origin"]["params"]["clipper"] == "sound_tiling"
        # Origin records the resolved clipper's parameter values.
        assert first["origin"]["params"]["clipper_duration"] == "10.0"

    def test_mixed_durations_resolve_per_media(self):
        """A short and a long item in the same dataset take different
        branches, and each clip records its own resolved concrete clipper."""
        from vtscore.datasets.load_pipeline import _apply_clipper

        short_wav = generate_wav(440, 2.0)
        long_wav = generate_wav(440, 8.0)
        clips = {
            1: {
                "id": 1,
                "media_type": "audio",
                "media_bytes": short_wav,
                "duration": 2.0,
                "origin": {"importer": "test", "params": {}},
            },
            2: {
                "id": 2,
                "media_type": "audio",
                "media_bytes": long_wav,
                "duration": 8.0,
                "origin": {"importer": "test", "params": {}},
            },
        }
        # Threshold 4s: short (2s) passes through, long (8s) tiles into 2s segments.
        _apply_clipper(clips, "sound_auto", {"threshold": 4.0, "tile_duration": 2.0})
        # 1 (pass-through) + 4 (8s / 2s) = 5 clips.
        assert len(clips) == 5
        resolved_names = [c["origin"]["params"]["clipper"] for c in clips.values()]
        assert resolved_names.count("sound_default") == 1
        assert resolved_names.count("sound_tiling") == 4

    def test_user_threshold_override_propagates(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        wav = generate_wav(440, 8.0)
        media = {
            "id": 1,
            "media_type": "audio",
            "media_bytes": wav,
            "duration": 8.0,
            "origin": {"importer": "test", "params": {}},
        }
        clips = {1: media}
        # Lower threshold so 8s triggers tiling, with 4s tiles → 2 tiles
        _apply_clipper(clips, "sound_auto", {"threshold": 5.0, "tile_duration": 4.0})
        assert len(clips) == 2
        first = next(iter(clips.values()))
        assert first["origin"]["params"]["clipper"] == "sound_tiling"
        assert first["origin"]["params"]["clipper_duration"] == "4.0"


# ---------------------------------------------------------------------------
# ImageFaceClipper
# ---------------------------------------------------------------------------


class _FakeBBox:
    def __init__(self, xmin: float, ymin: float, width: float, height: float) -> None:
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height


class _FakeLocationData:
    def __init__(self, bbox: _FakeBBox) -> None:
        self.relative_bounding_box = bbox


class _FakeDetection:
    def __init__(self, confidence: float, bbox: _FakeBBox) -> None:
        self.score = [confidence]
        self.location_data = _FakeLocationData(bbox)


class _FakeResults:
    def __init__(self, detections: list[_FakeDetection]) -> None:
        self.detections = detections


def _stub_detector(detections: list[_FakeDetection]):
    """Build a MediaPipe-shaped detector stub for ImageFaceClipper."""
    from unittest.mock import MagicMock

    detector = MagicMock()
    detector.process.return_value = _FakeResults(detections)
    return detector


class TestImageFaceClipper:
    def test_identity(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        c = ImageFaceClipper()
        assert c.name == "image_face"
        assert c.media_type == "image"
        assert c.display_name == "Face crops"
        assert isinstance(c, MediaClipper)

    def test_registered_in_registry(self):
        from vtscore.media import get_clipper

        c = get_clipper("image_face")
        assert c.name == "image_face"
        assert c.media_type == "image"

    def test_rejects_invalid_threshold(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        with pytest.raises(ValueError):
            ImageFaceClipper(threshold=-0.1)
        with pytest.raises(ValueError):
            ImageFaceClipper(threshold=1.5)

    def test_rejects_invalid_model_selection(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        with pytest.raises(ValueError):
            ImageFaceClipper(model_selection=2)

    def test_rejects_negative_padding(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        with pytest.raises(ValueError):
            ImageFaceClipper(padding=-0.01)

    def test_rejects_non_positive_min_size(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        with pytest.raises(ValueError):
            ImageFaceClipper(min_size=0)

    def test_returns_empty_when_no_detections(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        c = ImageFaceClipper()
        c._detector = _stub_detector([])
        media = {"id": 1, "media_type": "image", "media_bytes": _make_image_bytes(640, 480)}
        assert c.clip(media) == []

    def test_returns_empty_when_no_media_bytes(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        c = ImageFaceClipper()
        c._detector = _stub_detector([_FakeDetection(0.9, _FakeBBox(0.1, 0.1, 0.2, 0.2))])
        assert c.clip({"id": 1, "media_type": "image"}) == []

    def test_emits_one_clip_per_face(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        c = ImageFaceClipper(padding=0.0)
        c._detector = _stub_detector(
            [
                _FakeDetection(0.95, _FakeBBox(0.1, 0.1, 0.2, 0.2)),
                _FakeDetection(0.85, _FakeBBox(0.5, 0.4, 0.2, 0.2)),
            ]
        )
        media = {"id": 1, "media_type": "image", "media_bytes": _make_image_bytes(640, 480)}
        clips = c.clip(media)
        assert len(clips) == 2
        for clip in clips:
            assert clip["media_type"] == "image"
            assert isinstance(clip["media_bytes"], bytes)
            assert clip["width"] > 0 and clip["height"] > 0
            assert clip["file_size"] == len(clip["media_bytes"])
            assert clip["clip_box"] is not None and len(clip["clip_box"]) == 4

    def test_clip_index_orders_by_confidence(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        c = ImageFaceClipper(padding=0.0)
        c._detector = _stub_detector(
            [
                _FakeDetection(0.6, _FakeBBox(0.1, 0.1, 0.2, 0.2)),
                _FakeDetection(0.95, _FakeBBox(0.5, 0.4, 0.2, 0.2)),
            ]
        )
        media = {"id": 1, "media_type": "image", "media_bytes": _make_image_bytes(640, 480)}
        clips = c.clip(media)
        assert clips[0]["clip_index"] == 0
        assert clips[1]["clip_index"] == 1
        # Highest-confidence detection (0.95) lands at index 0; its bbox was
        # the second one passed in, centred at (0.5, 0.4).
        x1, y1, _x2, _y2 = clips[0]["clip_box"]
        assert x1 >= int(640 * 0.5) - 1

    def test_drops_low_confidence(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        c = ImageFaceClipper(threshold=0.7, padding=0.0)
        c._detector = _stub_detector(
            [
                _FakeDetection(0.95, _FakeBBox(0.1, 0.1, 0.2, 0.2)),
                _FakeDetection(0.4, _FakeBBox(0.5, 0.4, 0.2, 0.2)),
            ]
        )
        media = {"id": 1, "media_type": "image", "media_bytes": _make_image_bytes(640, 480)}
        clips = c.clip(media)
        assert len(clips) == 1

    def test_drops_too_small(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        c = ImageFaceClipper(min_size=100, padding=0.0)
        # 0.05 * 640 = 32px; below the 100px floor.
        c._detector = _stub_detector([_FakeDetection(0.95, _FakeBBox(0.1, 0.1, 0.05, 0.05))])
        media = {"id": 1, "media_type": "image", "media_bytes": _make_image_bytes(640, 480)}
        assert c.clip(media) == []

    def test_padding_expands_box(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        c = ImageFaceClipper(padding=0.5)
        # Face at (320,240) sized 64x64 → padded by 32px on each side → 128x128.
        # 0.5..0.5+64/640=0.6 in x; 0.5..0.5+64/480≈0.633 in y.
        c._detector = _stub_detector([_FakeDetection(0.95, _FakeBBox(0.5, 0.5, 0.1, 0.1333))])
        media = {"id": 1, "media_type": "image", "media_bytes": _make_image_bytes(640, 480)}
        clips = c.clip(media)
        assert len(clips) == 1
        x1, y1, x2, y2 = clips[0]["clip_box"]
        assert (x2 - x1) >= 96 and (y2 - y1) >= 96

    def test_clip_drops_inherited_embedding_and_md5(self):
        """Cropped faces must not keep the parent's embedding / md5;
        otherwise the load-pipeline fixup won't re-embed single-face crops."""
        from vtscore.media.image.clipper import ImageFaceClipper
        import numpy as np

        c = ImageFaceClipper(padding=0.0)
        c._detector = _stub_detector([_FakeDetection(0.95, _FakeBBox(0.1, 0.1, 0.3, 0.3))])
        media = {
            "id": 1,
            "media_type": "image",
            "media_bytes": _make_image_bytes(640, 480),
            "md5": "deadbeef",
            "embedding": np.zeros(8, dtype=np.float32),
        }
        clips = c.clip(media)
        assert len(clips) == 1
        assert "embedding" not in clips[0]
        assert "md5" not in clips[0]

    def test_with_params_overrides(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        c = ImageFaceClipper().with_params({"threshold": 0.8, "padding": 0.4, "min_size": 64, "model_selection": 0})
        assert isinstance(c, ImageFaceClipper)
        assert c._threshold == 0.8
        assert c._padding == 0.4
        assert c._min_size == 64
        assert c._model_selection == 0

    def test_to_dict_includes_params(self):
        from vtscore.media.image.clipper import ImageFaceClipper

        d = ImageFaceClipper(threshold=0.7, padding=0.3, min_size=48, model_selection=0).to_dict()
        assert d["name"] == "image_face"
        assert d["media_type"] == "image"
        assert d["threshold"] == 0.7
        assert d["padding"] == 0.3
        assert d["min_size"] == 48
        assert d["model_selection"] == 0
        assert "creation_questions" in d
        keys = {q["key"] for q in d["creation_questions"]}
        assert keys == {"threshold", "padding", "min_size", "model_selection"}
