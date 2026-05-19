"""Audio embedder — Whisper encoder (speech-rich datasets).

Uses the encoder half of OpenAI's Whisper model (``openai/whisper-base``)
and time-averages the last hidden states into a fixed-size vector.  The
decoder (text generation) is never invoked — this is purely a speech-
aware audio embedder, not a transcription pipeline.

There is no paired text encoder in the same vector space, so
:attr:`supports_text` is ``False`` and the UI hides text-search
affordances for datasets embedded with this model.  For
speech-keyword-style search, transcribe upstream and embed the
transcript with a text embedder instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vtsearch.config import WHISPER_MODEL_ID, WHISPER_SAMPLE_RATE
from vtsearch.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    load_pretrained_local_first,
    timed_progress,
)


class AudioWhisperEncoderEmbedder(MediaEmbedder):
    """Embeds audio via Whisper's encoder, time-pooled to a single vector.

    * Audio files → 512-dim vector (Whisper-base encoder hidden size).
    * No text branch; :meth:`embed_text` returns ``None``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "whisper_encoder"

    @property
    def display_name(self) -> str:
        return "Whisper encoder (speech)"

    @property
    def media_type_id(self) -> str:
        return "audio"

    @property
    def supports_text(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 3):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers…", 2, 3):
            from transformers import WhisperFeatureExtractor, WhisperModel  # noqa: PLC0415

        with timed_progress(self._on_progress, "loading", "Importing librosa…", 3, 3):
            import librosa  # noqa: F401, PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading Whisper encoder weights…")
        with intercept_tqdm_progress(self._on_progress):
            full_model = load_pretrained_local_first(
                WhisperModel.from_pretrained,
                WHISPER_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
            )
        # Encoder only — drop the decoder so we don't keep ~half the
        # weights in RSS for no benefit (we never invoke it).
        full_model = full_model.to("cpu")
        self._model = full_model.encoder
        self._model.eval()

        self._on_progress("loading", "Loading Whisper feature extractor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                WhisperFeatureExtractor.from_pretrained, WHISPER_MODEL_ID, cache_dir=cache_dir, token=False
            )

        # Warmup: run a dummy forward pass.
        self._on_progress("loading", "Warming up Whisper pipeline: preprocessing…", 1, 2)
        dummy_audio = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)
        inputs = self._processor(dummy_audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
        self._on_progress("loading", "Warming up Whisper pipeline: running encoder…", 2, 2)
        device = next(self._model.parameters()).device
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            self._model(input_features)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        file_path = Path(media["media_path"])
        try:
            import librosa  # noqa: PLC0415
            import torch  # noqa: PLC0415

            audio_data, _sr = librosa.load(file_path, sr=WHISPER_SAMPLE_RATE, mono=True)
            inputs = self._processor(audio_data, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
            device = next(self._model.parameters()).device
            input_features = inputs.input_features.to(device)
            with torch.no_grad():
                # Whisper feature extractor zero-pads short clips to the
                # full 30 s receptive field.  Mean-pooling over the time
                # axis is the standard recipe for turning Whisper-encoder
                # hidden states into a single utterance-level vector.
                hidden = self._model(input_features).last_hidden_state
                pooled = hidden.mean(dim=1).detach().cpu().numpy()
            return pooled[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", file_path)
            return None


EMBEDDER = AudioWhisperEncoderEmbedder()
