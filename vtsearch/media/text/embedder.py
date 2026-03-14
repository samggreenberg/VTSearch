"""Text embedder — E5-base-v2 (sentence-transformers)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import E5_MODEL_ID, MODELS_CACHE_DIR
from vtsearch.media.base import MediaEmbedder, intercept_tqdm_progress, intercept_weight_loading_progress

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class TextE5Embedder(MediaEmbedder):
    """Embeds text using the E5-base-v2 model.

    * Text files → 768-dimensional vectors via ``"passage: "`` prefix (E5's
      asymmetric retrieval design).
    * Text queries → 768-dimensional vectors via ``"query: "`` prefix.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "e5"

    @property
    def media_type_id(self) -> str:
        return "paragraph"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        if self._model is not None:
            return
        import gc

        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        from vtsearch.models.loader import ensure_torch_configured

        ensure_torch_configured()
        gc.collect()
        cache_dir = str(MODELS_CACHE_DIR)
        self._on_progress("loading", "Loading E5 model…", 0, 0)
        with intercept_tqdm_progress(self._on_progress), intercept_weight_loading_progress(
            self._on_progress, "Loading E5 model…"
        ):
            self._model = SentenceTransformer(E5_MODEL_ID, cache_folder=cache_dir, token=False)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @property
    def description_wrappers(self) -> list[str]:
        return [
            "a document about {text}",
            "an article discussing {text}",
            "{text}",
            "a text passage about {text}",
            "writing on the topic of {text}",
        ]

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()
            if not text_content:
                print(f"Warning: empty text file {file_path}")
                return None
            return self._model.encode(f"passage: {text_content}", normalize_embeddings=True)
        except Exception as e:
            print(f"Error embedding {file_path}: {e}")
            return None

    def embed_text_passage(self, text: str) -> Optional[np.ndarray]:
        """Embed *text* as a passage (used when loading demo datasets in-memory)."""
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            return self._model.encode(f"passage: {text}", normalize_embeddings=True)
        except Exception as e:
            print(f"Error embedding passage: {e}")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            return self._model.encode(f"query: {text}", normalize_embeddings=True)
        except Exception as e:
            print(f"Error embedding text query for text: {e}")
            return None

    # Internal helper used by loader.py bridge
    def _get_model(self) -> Optional[SentenceTransformer]:
        if self._model is None:
            self.load_models()
        return self._model
