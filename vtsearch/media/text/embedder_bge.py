"""Text embedder — BGE-base-en-v1.5 (sentence-transformers)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import BGE_MODEL_ID, MODELS_CACHE_DIR
from vtsearch.media.base import MediaEmbedder, intercept_tqdm_progress

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class TextBGEEmbedder(MediaEmbedder):
    """Embeds text using the BGE-base-en-v1.5 model.

    * Text files → 768-dimensional vectors (passage embedding).
    * Text queries → 768-dimensional vectors via ``"Represent this sentence: "``
      prefix (BGE's asymmetric retrieval design).
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "bge"

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
        self._on_progress("loading", "Loading BGE model…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._model = SentenceTransformer(BGE_MODEL_ID, cache_folder=cache_dir, token=False)

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
            return self._model.encode(text_content, normalize_embeddings=True)
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
            return self._model.encode(text, normalize_embeddings=True)
        except Exception as e:
            print(f"Error embedding passage (BGE): {e}")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            return self._model.encode(f"Represent this sentence: {text}", normalize_embeddings=True)
        except Exception as e:
            print(f"Error embedding text query for text (BGE): {e}")
            return None

    # Internal helper used by loader.py bridge
    def _get_model(self) -> Optional[SentenceTransformer]:
        if self._model is None:
            self.load_models()
        return self._model
