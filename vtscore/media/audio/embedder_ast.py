"""Audio embedder — AST (Audio Spectrogram Transformer).

Wraps the AudioSet-finetuned ViT-style spectrogram transformer from MIT
(``MIT/ast-finetuned-audioset-10-10-0.4593``).  AST is audio-only — there
is no paired text encoder — so :attr:`supports_text` is ``False`` and the
UI hides text-search affordances for datasets embedded with this model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vtscore.config import AST_MODEL_ID, AST_SAMPLE_RATE
from vtscore.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    load_pretrained_local_first,
    timed_progress,
)


class AudioASTEmbedder(MediaEmbedder):
    """Embeds audio files using the Audio Spectrogram Transformer.

    * Audio files → 768-dimensional vectors via AST's pooled hidden state.
    * No text encoder; :meth:`embed_text` returns ``None``.
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
        return "ast"

    @property
    def display_name(self) -> str:
        return "AST (audio spectrogram)"

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
            from transformers import ASTFeatureExtractor, ASTModel  # noqa: PLC0415

        with timed_progress(self._on_progress, "loading", "Importing librosa…", 3, 3):
            import librosa  # noqa: F401, PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading AST model weights…")
        with intercept_tqdm_progress(self._on_progress):
            self._model = load_pretrained_local_first(
                ASTModel.from_pretrained,
                AST_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
            )
        self._model = self._model.to("cpu")
        self._model.eval()
        self._on_progress("loading", "Loading AST feature extractor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                ASTFeatureExtractor.from_pretrained, AST_MODEL_ID, cache_dir=cache_dir, token=False
            )

        # Warmup: run a dummy forward pass so the first real embed_media
        # call runs at steady-state speed.
        self._on_progress("loading", "Warming up AST pipeline: preprocessing…", 1, 2)
        dummy_audio = np.zeros(AST_SAMPLE_RATE, dtype=np.float32)
        inputs = self._processor(dummy_audio, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt")
        self._on_progress("loading", "Warming up AST pipeline: running model…", 2, 2)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            self._model(**inputs)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

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

            if audio_bytes is not None:
                source: io.BytesIO | Path = io.BytesIO(bytes(audio_bytes))
            else:
                assert file_path is not None  # narrowed by the path_str check above
                source = file_path
            audio_data, _sr = librosa.load(source, sr=AST_SAMPLE_RATE, mono=True)
            inputs = self._processor(audio_data, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                # AST returns ``BaseModelOutputWithPooling``; ``pooler_output``
                # is the mean of patch tokens (768-dim for the base model).
                embedding = outputs.pooler_output.detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", source_repr)
            return None


EMBEDDER = AudioASTEmbedder()
