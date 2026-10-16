"""Tests for the audio→text (Whisper ASR) converter."""

from __future__ import annotations

import io
import sys
import types
import wave
from unittest.mock import patch

import numpy as np


# Fakes go into ``sys.modules`` via ``monkeypatch.setitem``, never
# ``patch.dict(sys.modules, ...)``: ``convert`` resolves the device, which
# imports torch, and ``patch.dict`` restores the whole table on exit. When the
# test was the first to import torch, that evicts it, and the next
# ``import torch`` re-initialises ``torch._C`` and crashes (see tests_lib/detectors/test_face.py).


def _make_silent_wav(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Build a tiny mono 16-bit PCM WAV; content doesn't matter (Whisper is mocked)."""
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
        last_kwargs: dict = {}
        last_device: str | None = None

        def transcribe(self, _audio_path, **kwargs):
            FakeModel.last_kwargs = kwargs
            return transcribe_result

    def load_model(size: str, device=None):
        if load_model_calls is not None:
            load_model_calls.append(size)
        FakeModel.last_device = device
        return FakeModel()

    mod.load_model = load_model  # pyright: ignore[reportAttributeAccessIssue]
    mod._FakeModel = FakeModel  # pyright: ignore[reportAttributeAccessIssue]  # accessor for tests
    return mod


class TestAudio2TextMediaConverter:
    def test_source_and_target_types(self):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        assert c.source_type == "audio"
        assert c.target_type == "text"

    def test_name_and_display(self):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        assert c.name == "audio2text"
        assert "Whisper" in c.display_name or "ASR" in c.display_name

    def test_fields_have_expected_keys(self):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        keys = {f.key for f in c.fields}
        assert {"model_size", "language"} <= keys

    def test_field_defaults(self):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        assert c.get_param({}, "model_size") == "base"
        assert c.get_param({}, "language") == ""

    def test_model_size_options_include_standard_sizes(self):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        size_field = next(f for f in c.fields if f.key == "model_size")
        assert {"tiny", "base", "small", "medium", "large"} <= set(size_field.options or [])

    def test_convert_no_data(self):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        assert c.convert({"filename": "x.wav"}) == []

    def test_convert_empty_bytes(self):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        assert c.convert({"filename": "x.wav", "media_bytes": b""}) == []

    def test_convert_no_whisper_returns_empty(self, monkeypatch):
        """When ``whisper`` is not importable, convert returns []."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        # Setting the module to None in sys.modules makes `import whisper` raise.
        monkeypatch.setitem(sys.modules, "whisper", None)
        assert c.convert(media) == []

    def test_convert_with_mocked_whisper(self, monkeypatch):
        """End-to-end happy path: whisper returns a transcript, we emit one text media."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "podcast.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        load_calls: list[str] = []
        fake = _fake_whisper_module(
            {"text": "  Hello world, this is a test transcript.  "},
            load_model_calls=load_calls,
        )

        monkeypatch.setitem(sys.modules, "whisper", fake)
        results = c.convert(media, {"model_size": "tiny"})

        assert load_calls == ["tiny"]
        assert len(results) == 1
        r = results[0]
        assert r["filename"] == "podcast.txt"
        assert r["media_string"] == "Hello world, this is a test transcript."
        assert r["word_count"] == 7
        assert r["character_count"] == len("Hello world, this is a test transcript.")
        assert r["duration"] == 0

    def test_convert_empty_transcript_returns_empty_list(self, monkeypatch):
        """Whisper returning blank text shouldn't produce a zero-content media."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "silent.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        fake = _fake_whisper_module({"text": "   "})

        monkeypatch.setitem(sys.modules, "whisper", fake)
        assert c.convert(media) == []

    def test_convert_passes_language_to_whisper(self, monkeypatch):
        """The language param flows into model.transcribe() kwargs."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "fr.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        fake = _fake_whisper_module({"text": "Bonjour le monde."})

        monkeypatch.setitem(sys.modules, "whisper", fake)
        results = c.convert(media, {"language": "fr"})

        assert results
        kwargs = fake._FakeModel.last_kwargs  # pyright: ignore[reportAttributeAccessIssue]
        assert kwargs.get("language") == "fr"
        assert kwargs.get("fp16") is False

    def test_blank_language_omits_language_kwarg(self, monkeypatch):
        """Blank language string must NOT be passed (would force Whisper to '')."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        fake = _fake_whisper_module({"text": "Hello."})

        monkeypatch.setitem(sys.modules, "whisper", fake)
        c.convert(media, {"language": "   "})

        assert "language" not in fake._FakeModel.last_kwargs  # pyright: ignore[reportAttributeAccessIssue]

    def test_convert_passes_model_size(self, monkeypatch):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        load_calls: list[str] = []
        fake = _fake_whisper_module({"text": "Hi."}, load_model_calls=load_calls)

        monkeypatch.setitem(sys.modules, "whisper", fake)
        c.convert(media, {"model_size": "small"})

        assert load_calls == ["small"]

    def test_convert_loads_on_resolved_device(self, monkeypatch):
        """Whisper loads on resolve_device()'s choice; on a CPU host that's
        ``"cpu"`` with fp16 disabled (FP16 is enabled only on CUDA)."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        fake = _fake_whisper_module({"text": "Hi."})

        monkeypatch.setitem(sys.modules, "whisper", fake)
        c.convert(media)

        assert fake._FakeModel.last_device == "cpu"  # pyright: ignore[reportAttributeAccessIssue]
        assert fake._FakeModel.last_kwargs.get("fp16") is False  # pyright: ignore[reportAttributeAccessIssue]

    def test_convert_default_model_size_is_base(self, monkeypatch):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        load_calls: list[str] = []
        fake = _fake_whisper_module({"text": "Hi."}, load_model_calls=load_calls)

        monkeypatch.setitem(sys.modules, "whisper", fake)
        c.convert(media)

        assert load_calls == ["base"]

    def test_convert_from_path(self, tmp_path, monkeypatch):
        """Converter accepts a media_path and does NOT delete a user-owned file."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        wav_path = tmp_path / "tone.wav"
        wav_path.write_bytes(_make_silent_wav())
        media = {"filename": "tone.wav", "media_bytes": None, "media_path": str(wav_path)}
        c = Audio2TextMediaConverter()
        fake = _fake_whisper_module({"text": "Path-loaded transcript."})

        monkeypatch.setitem(sys.modules, "whisper", fake)
        results = c.convert(media)

        assert len(results) == 1
        assert results[0]["media_string"] == "Path-loaded transcript."
        # The caller-supplied file must still exist after conversion.
        assert wav_path.exists()

    def test_convert_handles_load_model_failure(self, monkeypatch):
        """A failed model load returns [] cleanly without raising."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        mod = types.ModuleType("whisper")

        def load_model(_size, device=None):
            raise RuntimeError("boom")

        mod.load_model = load_model  # pyright: ignore[reportAttributeAccessIssue]

        monkeypatch.setitem(sys.modules, "whisper", mod)
        assert c.convert(media) == []

    def test_convert_handles_transcribe_failure(self, monkeypatch):
        """A failed transcribe call returns [] cleanly without raising."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        c = Audio2TextMediaConverter()
        mod = types.ModuleType("whisper")

        class FailingModel:
            def transcribe(self, *_args, **_kwargs):
                raise RuntimeError("decoder boom")

        loaded: list[str] = []

        def load_model(size, device=None):
            loaded.append(size)
            return FailingModel()

        mod.load_model = load_model  # pyright: ignore[reportAttributeAccessIssue]

        monkeypatch.setitem(sys.modules, "whisper", mod)
        assert c.convert(media) == []
        assert loaded == ["base"], "the model must load, so the failure is transcribe's"


class TestAudio2TextModelCache:
    """The Whisper model loads once per (size, device), not once per file (#4979)."""

    def test_loads_once_across_files(self, monkeypatch):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        load_calls: list[str] = []
        fake = _fake_whisper_module({"text": "Hi."}, load_model_calls=load_calls)

        monkeypatch.setitem(sys.modules, "whisper", fake)
        for i in range(4):
            assert c.convert({"filename": f"{i}.wav", "media_bytes": _make_silent_wav()})

        assert load_calls == ["base"]

    def test_reloads_when_model_size_changes(self, monkeypatch):
        """A new size replaces the cached model; the slot holds one, so going back reloads."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        load_calls: list[str] = []
        fake = _fake_whisper_module({"text": "Hi."}, load_model_calls=load_calls)

        monkeypatch.setitem(sys.modules, "whisper", fake)
        for size in ["tiny", "tiny", "small", "small", "tiny"]:
            assert c.convert({"filename": "x.wav", "media_bytes": _make_silent_wav()}, {"model_size": size})

        assert load_calls == ["tiny", "small", "tiny"]

    def test_reloads_when_device_changes(self, monkeypatch):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        fake = _fake_whisper_module({"text": "Hi."})
        devices: list[str | None] = []
        real_load = fake.load_model  # pyright: ignore[reportAttributeAccessIssue]

        def load_model(size, device=None):
            devices.append(device)
            return real_load(size, device=device)

        fake.load_model = load_model  # pyright: ignore[reportAttributeAccessIssue]
        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}

        monkeypatch.setitem(sys.modules, "whisper", fake)
        for resolved in ["cpu", "cuda:0", "cuda:0", "cpu"]:
            with patch("vtscore.config.resolve_device", return_value=resolved):
                assert c.convert(media)

        assert devices == ["cpu", "cuda:0", "cpu"]

    def test_language_change_reuses_model(self, monkeypatch):
        """Language is a transcribe argument, not part of the model, so it never reloads."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        load_calls: list[str] = []
        fake = _fake_whisper_module({"text": "Hi."}, load_model_calls=load_calls)

        monkeypatch.setitem(sys.modules, "whisper", fake)
        for language in ["en", "fr", ""]:
            assert c.convert({"filename": "x.wav", "media_bytes": _make_silent_wav()}, {"language": language})

        assert load_calls == ["base"]

    def test_failed_load_is_retried_on_next_call(self, monkeypatch):
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        attempts: list[str] = []
        mod = types.ModuleType("whisper")

        class FakeModel:
            def transcribe(self, _audio_path, **_kwargs):
                return {"text": "Second time lucky."}

        def load_model(size, device=None):
            attempts.append(size)
            if len(attempts) == 1:
                raise RuntimeError("download interrupted")
            return FakeModel()

        mod.load_model = load_model  # pyright: ignore[reportAttributeAccessIssue]
        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}

        monkeypatch.setitem(sys.modules, "whisper", mod)
        assert c.convert(media) == []
        results = c.convert(media)
        assert c.convert(media) == results

        assert results[0]["media_string"] == "Second time lucky."
        assert attempts == ["base", "base"]

    def test_a_new_whisper_module_gets_its_own_model(self, monkeypatch):
        """The loader is part of the key, so the shared ``CONVERTER`` never serves one
        test's fake model to the next."""
        from vtscore.converters.audio2text import CONVERTER

        media = {"filename": "x.wav", "media_bytes": _make_silent_wav()}
        first_calls: list[str] = []
        second_calls: list[str] = []
        first = _fake_whisper_module({"text": "From the first."}, load_model_calls=first_calls)
        second = _fake_whisper_module({"text": "From the second."}, load_model_calls=second_calls)

        monkeypatch.setitem(sys.modules, "whisper", first)
        assert CONVERTER.convert(media)[0]["media_string"] == "From the first."
        monkeypatch.setitem(sys.modules, "whisper", second)
        assert CONVERTER.convert(media)[0]["media_string"] == "From the second."

        assert first_calls == ["base"]
        assert second_calls == ["base"]

    def test_transcribe_runs_under_the_model_lock(self, monkeypatch):
        """Concurrent loads share the singleton, and one Whisper model cannot run two
        transcriptions at once, so transcribe must hold the lock, not only the load."""
        from vtscore.converters.audio2text import Audio2TextMediaConverter

        c = Audio2TextMediaConverter()
        held: list[bool] = []
        mod = types.ModuleType("whisper")

        class FakeModel:
            def transcribe(self, _audio_path, **_kwargs):
                held.append(c._model_slot.lock.locked())
                return {"text": "Hi."}

        mod.load_model = lambda size, device=None: FakeModel()  # pyright: ignore[reportAttributeAccessIssue]

        monkeypatch.setitem(sys.modules, "whisper", mod)
        assert c.convert({"filename": "x.wav", "media_bytes": _make_silent_wav()})

        assert held == [True]
        assert not c._model_slot.lock.locked()


class TestAudio2TextRegistryIntegration:
    def test_audio2text_in_registry(self):
        from vtscore.converters import get_converter, list_converters

        assert get_converter("audio2text") is not None
        names = [c.name for c in list_converters()]
        assert "audio2text" in names

    def test_audio2text_listed_for_text_target(self):
        from vtscore.converters import list_converters_for_target

        names = [c.name for c in list_converters_for_target("text")]
        assert "audio2text" in names

    def test_audio2text_listed_for_audio_source(self):
        from vtscore.converters import list_converters_for_source

        names = [c.name for c in list_converters_for_source("audio")]
        assert "audio2text" in names

    def test_api_hides_audio2text_from_picker(self, client):
        # audio2text carries hidden_from_picker=True: registered (see the
        # list_converters_for_* tests above) but excluded from the
        # picker-facing /api/converters endpoint.
        resp = client.get("/api/converters")
        assert resp.status_code == 200
        names = [c["name"] for c in resp.get_json()["converters"]]
        assert "audio2text" not in names
