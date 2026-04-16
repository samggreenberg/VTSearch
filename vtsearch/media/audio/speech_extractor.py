"""Speech extractor using OpenAI Whisper (tiny model) for transcription."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from vtsearch.media.processors import Extractor


class SpeechExtractor(Extractor):
    """Extracts speech segments from audio clips using Whisper (tiny).

    Running :meth:`extract` on an audio clip returns a list of dicts, one per
    detected speech segment::

        [
            {
                "confidence": 0.85,
                "label": "hello world",
                "start": 0.0,
                "end": 2.5,
            },
            ...
        ]

    The ``label`` field contains the transcribed text for the segment.
    ``start`` and ``end`` are timestamps in seconds.
    """

    def __init__(self, name: str, model_size: str = "tiny", language: str | None = None) -> None:
        self._name = name
        self._model_size = model_size
        self._language = language
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def model_size(self) -> str:
        return self._model_size

    @property
    def language(self) -> str | None:
        return self._language

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if self._model is not None:
            return
        import whisper  # noqa: PLC0415

        self._model = whisper.load_model(self._model_size)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, clip: dict[str, Any]) -> list[dict[str, Any]]:
        """Transcribe speech in *clip* and return segments with timestamps.

        The *clip* dict must contain ``"media_bytes"`` (raw audio bytes, WAV format)
        or ``"media_path"`` (path to audio file on disk).

        Returns a list of dicts, each with ``"confidence"``, ``"label"`` (text),
        ``"start"`` (seconds), and ``"end"`` (seconds).
        """
        self.load_model()
        assert self._model is not None

        # Resolve audio to a file path for whisper
        audio_path = self._resolve_audio_path(clip)
        if audio_path is None:
            return []

        # Track whether we created a temp file so we can clean it up after
        is_temp = clip.get("media_path") is None or not Path(clip.get("media_path", "")).exists()

        try:
            kwargs: dict[str, Any] = {"fp16": False}
            if self._language:
                kwargs["language"] = self._language

            result = self._model.transcribe(str(audio_path), **kwargs)
        except Exception:
            return []
        finally:
            # Clean up temp file if we created one
            if is_temp and audio_path.exists():
                audio_path.unlink(missing_ok=True)

        hits: list[dict[str, Any]] = []
        segments = result.get("segments", [])
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            # Whisper provides avg_logprob; convert to a 0-1 confidence approximation
            avg_logprob = seg.get("avg_logprob", -1.0)
            no_speech_prob = seg.get("no_speech_prob", 0.0)
            # Simple confidence: sigmoid-like mapping of avg_logprob
            import math  # noqa: PLC0415

            confidence = 1.0 / (1.0 + math.exp(-avg_logprob - 0.5))
            # Discount by no_speech_prob
            confidence *= 1.0 - no_speech_prob
            confidence = max(0.0, min(1.0, confidence))

            hits.append(
                {
                    "confidence": round(confidence, 4),
                    "label": text,
                    "start": round(seg.get("start", 0.0), 2),
                    "end": round(seg.get("end", 0.0), 2),
                }
            )

        hits.sort(key=lambda h: h["start"])
        return hits

    def _resolve_audio_path(self, clip: dict[str, Any]) -> Path | None:
        """Get a file path for the audio clip, writing to a temp file if needed."""
        media_path = clip.get("media_path")
        if media_path:
            p = Path(media_path)
            if p.exists():
                return p

        media_bytes = clip.get("media_bytes")
        if media_bytes is None:
            return None

        # Write to a temporary WAV file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(media_bytes)
        tmp.close()
        return Path(tmp.name)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["extractor_type"] = "speech"
        d["config"] = {
            "model_size": self._model_size,
            "language": self._language,
        }
        return d

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "SpeechExtractor":
        """Reconstruct a ``SpeechExtractor`` from a saved config dict."""
        return cls(
            name=name,
            model_size=config.get("model_size", "tiny"),
            language=config.get("language"),
        )
