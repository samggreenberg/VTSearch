"""Audio embedder — CLAP General 2024 (laion/larger_clap_general)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vtscore.config import CLAP_GENERAL_MODEL_ID, CLAP_SAMPLE_RATE
from vtscore.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    load_pretrained_local_first,
    timed_progress,
)


class AudioClapGeneralEmbedder(MediaEmbedder):
    """Embeds audio files using the larger CLAP general checkpoint (laion/larger_clap_general).

    * Audio files → 512-dimensional vectors via CLAP's audio encoder.
    * Text queries → 512-dimensional vectors via CLAP's text encoder.

    Compared to the original ``laion/clap-htsat-unfused`` baseline, this
    larger general-purpose checkpoint is trained on a broader audio mix
    and tends to give stronger zero-shot transfer for general sounds.
    """

    def __init__(self) -> None:
        super().__init__()
        # Typed ``Any``: transformers stubs miss several ``ClapProcessor.__call__``
        # kwargs we pass at runtime; runtime ``None`` checks guard the calls.
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "clap_general"

    @property
    def display_name(self) -> str:
        return "CLAP (general 2024)"

    @property
    def media_type_id(self) -> str:
        return "audio"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 3):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers…", 2, 3):
            from transformers import ClapModel, ClapProcessor  # noqa: PLC0415

        with timed_progress(self._on_progress, "loading", "Importing librosa…", 3, 3):
            import librosa  # noqa: F401, PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading CLAP General model weights…")
        with intercept_tqdm_progress(self._on_progress):
            self._model = load_pretrained_local_first(
                ClapModel.from_pretrained,
                CLAP_GENERAL_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
            )
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading CLAP General processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                ClapProcessor.from_pretrained, CLAP_GENERAL_MODEL_ID, cache_dir=cache_dir, token=False
            )

        # Warmup: run a dummy forward pass
        self._on_progress("loading", "Warming up CLAP General pipeline: preprocessing…", 1, 2)
        dummy_audio = np.zeros(CLAP_SAMPLE_RATE, dtype=np.float32)
        inputs = self._processor(
            audio=dummy_audio,
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,
            truncation=True,
        )
        self._on_progress("loading", "Warming up CLAP General pipeline: running model…", 2, 2)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.audio_model(**inputs)
            self._model.audio_projection(outputs.pooler_output)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @property
    def description_wrappers(self) -> list[str]:
        return [
            "the sound of {text}",
            "a recording of {text}",
            "{text}",
            "audio of {text}",
            "the noise of {text}",
        ]

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
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
            import io  # noqa: PLC0415
            import librosa  # noqa: PLC0415
            import torch  # noqa: PLC0415

            source = io.BytesIO(bytes(audio_bytes)) if audio_bytes is not None else file_path
            audio_data, _sr = librosa.load(source, sr=CLAP_SAMPLE_RATE, mono=True)
            inputs = self._processor(
                audio=audio_data,
                sampling_rate=CLAP_SAMPLE_RATE,
                return_tensors="pt",
                padding="max_length",
                max_length=480000,
                truncation=True,
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.audio_model(**inputs)
                embedding = self._model.audio_projection(outputs.pooler_output).detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", source_repr)
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._processor(text=[text], return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.text_model(**inputs)
                text_vec = self._model.text_projection(outputs.pooler_output).detach().cpu().numpy()[0]
            return text_vec
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for audio (CLAP General)")
            return None

    # Internal helper used by loader.py bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor


EMBEDDER = AudioClapGeneralEmbedder()
