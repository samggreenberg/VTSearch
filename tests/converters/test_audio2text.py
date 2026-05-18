"""Tests for the audio→text (Whisper ASR) converter."""

from __future__ import annotations

import io
import sys
import types
import wave
from unittest.mock import patch

import numpy as np


def _make_silent_wav(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Build a tiny mono 16-bit PCM WAV — content doesn't matter (Whisper is mocked)."""
    samples = np.zeros(int(duration_s * sample_rate), dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _fake_whisper_module(transcribe_result: dict, load_model_calls: list | None = None) -> types.ModuleType:
    """Build a stand-in ``whisper`` module that returns *transcribe_result*."""
    mod = types.ModuleType("whisper")

    class FakeModel:
        def transcribe(self, _audio_path, **kwargs):
            FakeModel.last_kwargs = kwargs
            return transcribe_result

    FakeModel.last_kwargs = {}

    def load_model(size: str):
        if load_model_calls is not None:
            load_model_calls.append(size)
        return FakeModel()

    mod.load_model = load_model  # pyright: ignore[reportAttributeAccessIssue]
    mod._FakeModel = FakeModel  # pyright: ignore[reportAttributeAccessIssue]  # accessor for tests
    return mod


class TestAudio2TextMediaConverter:
    def test_source_and_target_types(self):
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        assert c.source_type == "audio"
        assert c.target_type == "text"

    def test_name_and_display(self):
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        assert c.name == "audio2text"
        assert "Whisper" in c.display_name or "ASR" in c.display_name

    def test_fields_have_expected_keys(self):
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        keys = {f.key for f in c.fields}
        assert {"model_size", "language"} <= keys

    def test_field_defaults(self):
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        assert c.get_param({}, "model_size") == "base"
        assert c.get_param({}, "language") == ""

    def test_model_size_options_include_standard_sizes(self):
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        size_field = next(f for f in c.fields if f.key == "model_size")
        assert {"tiny", "base", "small", "medium", "large"} <= set(size_field.options or [])

    def test_convert_no_data(self):
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        assert c.convert({"filename": "x.wav"}) == []

    def test_convert_empty_bytes(self):
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        assert c.convert({"filename": "x.wav", "media_bytes": b""}) == []

    def test_convert_no_whisper_returns_empty(self):
        """When ``whisper`` is not importable, convert returns []."""
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        # Setting the module to None in sys.modules makes `import whisper` raise.
        with patch.dict(sys.modules, {"whisper": None}):
            assert c.convert(media) == []

    def test_convert_with_mocked_whisper(self):
        """End-to-end happy path: whisper returns a transcript, we emit one text media."""
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "podcast.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        load_calls: list[str] = []
        fake = _fake_whisper_module(
            {"text": "  Hello world, this is a test transcript.  "},
            load_model_calls=load_calls,
        )

        with patch.dict(sys.modules, {"whisper": fake}):
            results = c.convert(media, {"model_size": "tiny"})

        assert load_calls == ["tiny"]
        assert len(results) == 1
        r = results[0]
        assert r["filename"] == "podcast.txt"
        assert r["media_string"] == "Hello world, this is a test transcript."
        assert r["word_count"] == 7
        assert r["character_count"] == len("Hello world, this is a test transcript.")
        assert r["duration"] == 0

    def test_convert_empty_transcript_returns_empty_list(self):
        """Whisper returning blank text shouldn't produce a zero-content media."""
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "silent.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        fake = _fake_whisper_module({"text": "   "})

        with patch.dict(sys.modules, {"whisper": fake}):
            assert c.convert(media) == []

    def test_convert_passes_language_to_whisper(self):
        """The language param flows into model.transcribe() kwargs."""
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "fr.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        fake = _fake_whisper_module({"text": "Bonjour le monde."})

        with patch.dict(sys.modules, {"whisper": fake}):
            results = c.convert(media, {"language": "fr"})

        assert results
        kwargs = fake._FakeModel.last_kwargs  # pyright: ignore[reportAttributeAccessIssue]
        assert kwargs.get("language") == "fr"
        assert kwargs.get("fp16") is False

    def test_blank_language_omits_language_kwarg(self):
        """Blank language string must NOT be passed (would force Whisper to '')."""
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        fake = _fake_whisper_module({"text": "Hello."})

        with patch.dict(sys.modules, {"whisper": fake}):
            c.convert(media, {"language": "   "})

        assert "language" not in fake._FakeModel.last_kwargs  # pyright: ignore[reportAttributeAccessIssue]

    def test_convert_passes_model_size(self):
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        load_calls: list[str] = []
        fake = _fake_whisper_module({"text": "Hi."}, load_model_calls=load_calls)

        with patch.dict(sys.modules, {"whisper": fake}):
            c.convert(media, {"model_size": "small"})

        assert load_calls == ["small"]

    def test_convert_default_model_size_is_base(self):
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        load_calls: list[str] = []
        fake = _fake_whisper_module({"text": "Hi."}, load_model_calls=load_calls)

        with patch.dict(sys.modules, {"whisper": fake}):
            c.convert(media)

        assert load_calls == ["base"]

    def test_convert_from_path(self, tmp_path):
        """Converter accepts a media_path and does NOT delete a user-owned file."""
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        wav_path = tmp_path / "tone.wav"
        wav_path.write_bytes(_make_silent_wav())
        media = {"filename": "tone.wav", "media_bytes": None, "media_path": str(wav_path)}
        c = Audio2TextMediaConverter()
        fake = _fake_whisper_module({"text": "Path-loaded transcript."})

        with patch.dict(sys.modules, {"whisper": fake}):
            results = c.convert(media)

        assert len(results) == 1
        assert results[0]["media_string"] == "Path-loaded transcript."
        # The caller-supplied file must still exist after conversion.
        assert wav_path.exists()

    def test_convert_handles_load_model_failure(self):
        """A failed model load returns [] cleanly without raising."""
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        mod = types.ModuleType("whisper")

        def load_model(_size):
            raise RuntimeError("boom")

        mod.load_model = load_model  # pyright: ignore[reportAttributeAccessIssue]

        with patch.dict(sys.modules, {"whisper": mod}):
            assert c.convert(media) == []

    def test_convert_handles_transcribe_failure(self):
        """A failed transcribe call returns [] cleanly without raising."""
        from vtsearch.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        mod = types.ModuleType("whisper")

        class FailingModel:
            def transcribe(self, *_args, **_kwargs):
                raise RuntimeError("decoder boom")

        mod.load_model = lambda _size: FailingModel()  # pyright: ignore[reportAttributeAccessIssue]

        with patch.dict(sys.modules, {"whisper": mod}):
            assert c.convert(media) == []


class TestAudio2TextRegistryIntegration:
    def test_audio2text_in_registry(self):
        from vtsearch.converters import get_converter, list_converters

        assert get_converter("audio2text") is not None
        names = [c.name for c in list_converters()]
        assert "audio2text" in names

    def test_audio2text_listed_for_text_target(self):
        from vtsearch.converters import list_converters_for_target

        names = [c.name for c in list_converters_for_target("text")]
        assert "audio2text" in names

    def test_audio2text_listed_for_audio_source(self):
        from vtsearch.converters import list_converters_for_source

        names = [c.name for c in list_converters_for_source("audio")]
        assert "audio2text" in names

    def test_api_lists_audio2text(self, client):
        resp = client.get("/api/converters")
        assert resp.status_code == 200
        names = [c["name"] for c in resp.get_json()["converters"]]
        assert "audio2text" in names
