"""Audio embedder — CLAP (laion/clap-htsat-unfused)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import CLAP_MODEL_ID, CLAP_SAMPLE_RATE
from vtsearch.media.base import MediaEmbedder, embedder_load_setup, intercept_tqdm_progress, load_pretrained_local_first

if TYPE_CHECKING:
    from transformers import ClapModel, ClapProcessor


class AudioClapEmbedder(MediaEmbedder):
    """Embeds audio files using the CLAP model (laion/clap-htsat-unfused).

    * Audio files → 512-dimensional vectors via CLAP's audio encoder.
    * Text queries → 512-dimensional vectors via CLAP's text encoder.
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
        return "clap"

    @property
    def media_type_id(self) -> str:
        return "audio"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        self._on_progress("loading", "Importing audio libraries…", 0, 0)
        from transformers import ClapModel, ClapProcessor  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading CLAP model weights…")
        with intercept_tqdm_progress(self._on_progress):
            self._model = load_pretrained_local_first(
                ClapModel.from_pretrained, CLAP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False
            )
        # Materialize any tensors left on the ``meta`` device.
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading CLAP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                ClapProcessor.from_pretrained, CLAP_MODEL_ID, cache_dir=cache_dir, token=False
            )

        # Warmup: import librosa (heavy — pulls in numba, scipy, etc.),
        # trigger the numba JIT for audio resampling, and run a single
        # dummy forward pass so that the first real embed_media call runs
        # at the same speed as every subsequent one.
        self._on_progress("loading", "Warming up audio pipeline: importing libraries…", 1, 4)
        import librosa  # noqa: F401, PLC0415
        import torch  # noqa: PLC0415

        # Trigger librosa/soxr resampling JIT by loading a tiny WAV at a
        # different sample rate.  Without this, the first embed_media()
        # call stalls for 10-30 s while numba compiles resampling kernels,
        # making the embedding progress bar appear frozen.
        self._on_progress("loading", "Warming up audio pipeline: resampling JIT…", 2, 4)
        import io  # noqa: PLC0415

        import soundfile as sf  # noqa: PLC0415

        _warmup_sr = 16000  # intentionally different from CLAP_SAMPLE_RATE
        _warmup_buf = io.BytesIO()
        sf.write(_warmup_buf, np.zeros(_warmup_sr, dtype=np.float32), _warmup_sr, format="WAV")
        _warmup_buf.seek(0)
        librosa.load(_warmup_buf, sr=CLAP_SAMPLE_RATE, mono=True)

        self._on_progress("loading", "Warming up audio pipeline: preprocessing…", 3, 4)
        dummy_audio = np.zeros(CLAP_SAMPLE_RATE, dtype=np.float32)
        inputs = self._processor(
            audio=dummy_audio,
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,
            truncation=True,
        )
        self._on_progress("loading", "Warming up audio pipeline: running model…", 4, 4)
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
            print(f"Error embedding text query for audio: {e}")
            return None

    # Internal helper used by loader.py bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor
