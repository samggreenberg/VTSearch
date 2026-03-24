"""Audio embedder — CLAP Music & Speech (laion/larger_clap_music_and_speech)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import CLAP_MUSIC_MODEL_ID, CLAP_SAMPLE_RATE, MODELS_CACHE_DIR
from vtsearch.media.base import MediaEmbedder, intercept_tqdm_progress

if TYPE_CHECKING:
    from transformers import ClapModel, ClapProcessor


class AudioClapMusicEmbedder(MediaEmbedder):
    """Embeds audio files using the larger CLAP model (laion/larger_clap_music_and_speech).

    * Audio files → 512-dimensional vectors via CLAP's audio encoder.
    * Text queries → 512-dimensional vectors via CLAP's text encoder.

    This variant is trained on music and speech data, providing better
    performance for music retrieval and genre classification compared to
    the unfused CLAP model.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[ClapModel] = None
        self._processor: Optional[ClapProcessor] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "clap_music"

    @property
    def media_type_id(self) -> str:
        return "audio"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return
        import gc

        from transformers import ClapModel, ClapProcessor  # noqa: PLC0415

        from vtsearch.models.loader import ensure_torch_configured

        ensure_torch_configured()
        gc.collect()
        cache_dir = str(MODELS_CACHE_DIR)
        self._on_progress("loading", "Loading CLAP Music model weights…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._model = ClapModel.from_pretrained(
                CLAP_MUSIC_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False
            )
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading CLAP Music processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = ClapProcessor.from_pretrained(CLAP_MUSIC_MODEL_ID, cache_dir=cache_dir, token=False)

        # Warmup: import librosa and run a dummy forward pass
        self._on_progress("loading", "Warming up CLAP Music pipeline: importing libraries…", 1, 3)
        import librosa  # noqa: F401, PLC0415
        import torch  # noqa: PLC0415

        self._on_progress("loading", "Warming up CLAP Music pipeline: preprocessing…", 2, 3)
        dummy_audio = np.zeros(CLAP_SAMPLE_RATE, dtype=np.float32)
        inputs = self._processor(
            audio=dummy_audio,
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,
            truncation=True,
        )
        self._on_progress("loading", "Warming up CLAP Music pipeline: running model…", 3, 3)
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

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import librosa  # noqa: PLC0415
            import torch  # noqa: PLC0415

            audio_data, _sr = librosa.load(file_path, sr=CLAP_SAMPLE_RATE, mono=True)
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
        except Exception as e:
            print(f"Error embedding {file_path}: {e}")
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
        except Exception as e:
            print(f"Error embedding text query for audio (CLAP Music): {e}")
            return None

    # Internal helper used by loader.py bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor
