"""Transcribe speech in an audio file to text via OpenAI Whisper (ASR)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from vtsearch.converters.base import MediaConverter
from vtsearch.plugins import PluginField


class Audio2TextMediaConverter(MediaConverter):
    """Transcribe an audio file to text using OpenAI Whisper.

    Produces a single text media containing the concatenated transcript
    of all detected speech segments. Lets users feed podcasts, voice
    notes, and the audio track of videos (via the ``video2audio`` →
    ``audio2text`` chain) through text embedders like E5 / BGE without
    leaving VTSearch.

    The Whisper model files are downloaded on first use to the standard
    Whisper cache (``~/.cache/whisper``). The smallest (``tiny``) is
    ~75 MB on disk; ``base`` ~145 MB; ``small`` ~480 MB; ``medium``
    ~1.5 GB; ``large`` ~3 GB. Inference runs on CPU by default and
    automatically uses CUDA if a GPU is available.

    User-configurable parameters
    ----------------------------
    ``model_size``
        Whisper model variant. ``tiny`` and ``base`` are fast and
        English-leaning; ``small`` and above are needed for solid
        multilingual quality. Defaults to ``"base"``.
    ``language``
        Optional ISO 639-1 language code (e.g. ``"en"``, ``"fr"``,
        ``"de"``). Leave blank for Whisper's built-in auto-detect.
    """

    display_name = "Audio → Text (Whisper ASR)"
    converter_description = "Transcribe speech in audio to text via Whisper"
    fields = [
        PluginField(
            key="model_size",
            label="Whisper model size",
            field_type="select",
            description="Larger models are more accurate but slower and use more memory.",
            options=["tiny", "base", "small", "medium", "large"],
            default="base",
            required=False,
        ),
        PluginField(
            key="language",
            label="Language (ISO code, blank = auto-detect)",
            field_type="text",
            description="ISO 639-1 code, e.g. 'en', 'fr', 'de'. Leave blank to auto-detect.",
            default="",
            required=False,
        ),
    ]

    @property
    def source_type(self) -> str:
        return "audio"

    @property
    def target_type(self) -> str:
        return "text"

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        model_size = str(self.get_param(params, "model_size") or "base")
        language = str(self.get_param(params, "language") or "").strip() or None

        media_bytes = media.get("media_bytes")
        media_path = media.get("media_path")
        filename = media.get("filename", "audio.wav")
        stem = Path(filename).stem

        audio_path = self._resolve_audio_path(media_bytes, media_path)
        if audio_path is None:
            return []
        owns_temp = media_path is None or not Path(media_path).exists()

        try:
            import whisper  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        except ImportError:
            print("Audio2TextMediaConverter requires openai-whisper: pip install openai-whisper")
            if owns_temp:
                audio_path.unlink(missing_ok=True)
            return []

        try:
            model = whisper.load_model(model_size)
        except Exception as e:
            print(f"Audio2TextMediaConverter: failed to load Whisper model '{model_size}': {e}")
            if owns_temp:
                audio_path.unlink(missing_ok=True)
            return []

        try:
            kwargs: dict[str, Any] = {"fp16": False}
            if language:
                kwargs["language"] = language
            result = model.transcribe(str(audio_path), **kwargs)
        except Exception as e:
            print(f"Audio2TextMediaConverter: transcription failed on {filename}: {e}")
            return []
        finally:
            if owns_temp:
                audio_path.unlink(missing_ok=True)

        full_text = (result.get("text") or "").strip()
        if not full_text:
            return []

        return [
            {
                "filename": f"{stem}.txt",
                "media_string": full_text,
                "duration": 0,
                "word_count": len(full_text.split()),
                "character_count": len(full_text),
            }
        ]

    @staticmethod
    def _resolve_audio_path(media_bytes: bytes | None, media_path: str | None) -> Path | None:
        """Return a filesystem path to the audio, writing a temp WAV if needed."""
        if media_path:
            p = Path(media_path)
            if p.exists():
                return p
        if not media_bytes:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(media_bytes)
        finally:
            tmp.close()
        return Path(tmp.name)


CONVERTER = Audio2TextMediaConverter()
