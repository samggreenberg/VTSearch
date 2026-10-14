"""Audio embedder - EmbeddingGemma 2 (google/embeddinggemma-2), audio + text (#4837).

The audio half of :mod:`vtscore.media._embeddinggemma2`: sounds and typed
queries land in the model's one shared 768-d space, the same space the
``embeddinggemma2`` image embedder writes.  A clip enters the text backbone as
soft tokens from the audio encoder (25 a second) and the vector is the
backbone's pooled output.

Audio is decoded to 16 kHz mono and embedded from its first 30 s, a
deterministic window (a random crop would break the origin -> embedding
rederivation contract); a clip under a second is zero-padded to one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vtscore.config import (
    EMBEDDINGGEMMA2_MAX_SAMPLES,
    EMBEDDINGGEMMA2_MIN_SAMPLES,
    EMBEDDINGGEMMA2_MODEL_ID,
    EMBEDDINGGEMMA2_SAMPLE_RATE,
)
from vtscore.media._embeddinggemma2 import load_embeddinggemma2, media_messages, pooled_forward, query_messages
from vtscore.media.embedder import MediaEmbedder


class AudioEmbeddingGemma2Embedder(MediaEmbedder):
    """Embeds audio files using EmbeddingGemma 2 (google/embeddinggemma-2).

    * Audio files → 768-dimensional vectors: the audio encoder's soft tokens
      run through the text backbone, mean-pooled.
    * Text queries → 768-dimensional vectors in the same space, with the
      model's retrieval-query prefix.

    Loads the text backbone and the audio encoder (570M parameters), not the
    vision encoder.
    """

    _label = "EmbeddingGemma 2 Audio"

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return "embeddinggemma2_audio"

    @property
    def display_name(self) -> str:
        return "EmbeddingGemma 2 (audio)"

    @property
    def model_id(self) -> str:
        return EMBEDDINGGEMMA2_MODEL_ID

    @property
    def media_type_id(self) -> str:
        return "audio"

    @property
    def embedding_dim(self) -> int:
        return 768

    @property
    def description_wrappers(self) -> list[str]:
        """No wrappers: the query already carries the model's own task prefix (see the image half)."""
        return []

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return
        self._model, self._processor = load_embeddinggemma2(self._on_progress, "audio", self._label)

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        # Prefer in-memory bytes (set by clip re-embed) over a disk path so the
        # caller does not need to round-trip through a tempfile.
        audio_bytes = media.get("media_bytes")
        file_path: Optional[Path] = None
        if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
            audio_bytes = None
            path_str = media.get("media_path")
            if not path_str:
                return None
            file_path = Path(path_str)
        source_repr = file_path if file_path is not None else "<bytes>"
        try:
            from vtscore.media.audio.decode import decode_audio  # noqa: PLC0415

            if audio_bytes is not None:
                source: bytes | Path = bytes(audio_bytes)
            else:
                assert file_path is not None  # narrowed by the path_str check above
                source = file_path
            audio_data, _sr = decode_audio(source, sr=EMBEDDINGGEMMA2_SAMPLE_RATE, mono=True)
            audio_data = np.ascontiguousarray(audio_data[:EMBEDDINGGEMMA2_MAX_SAMPLES], dtype=np.float32)
            if audio_data.shape[0] < EMBEDDINGGEMMA2_MIN_SAMPLES:
                audio_data = np.pad(audio_data, (0, EMBEDDINGGEMMA2_MIN_SAMPLES - audio_data.shape[0]))
            return pooled_forward(
                self._model,
                self._processor,
                media_messages("audio", [audio_data]),
                sampling_rate=EMBEDDINGGEMMA2_SAMPLE_RATE,
            )[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", source_repr)
            return None

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            return pooled_forward(self._model, self._processor, query_messages([text]))[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for audio (%s)", self._label)
            return None


EMBEDDER = AudioEmbeddingGemma2Embedder()
