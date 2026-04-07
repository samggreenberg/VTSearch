"""Text embedder — E5-base-v2 (sentence-transformers)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import E5_MODEL_ID
from vtsearch.media.base import MediaEmbedder, embedder_load_setup, intercept_tqdm_progress, intercept_weight_loading_progress, load_pretrained_local_first

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
        return "text"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        self._on_progress("loading", "Importing torch…", 1, 3)
        import torch  # noqa: F401, PLC0415

        self._on_progress("loading", "Importing transformers…", 2, 3)
        from transformers import BertModel  # noqa: PLC0415

        self._on_progress("loading", "Importing sentence_transformers…", 3, 3)
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading E5 model…")
        BertModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        with intercept_tqdm_progress(self._on_progress), intercept_weight_loading_progress(
            self._on_progress, "Loading E5 model…"
        ):
            self._model = load_pretrained_local_first(SentenceTransformer, E5_MODEL_ID, cache_folder=cache_dir, token=False)

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
            logging.getLogger(__name__).exception("Error embedding %s", file_path)
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
            logging.getLogger(__name__).exception("Error embedding passage")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            return self._model.encode(f"query: {text}", normalize_embeddings=True)
        except Exception as e:
            logging.getLogger(__name__).exception("Error embedding text query for text")
            return None

    # Internal helper used by loader.py bridge
    def _get_model(self) -> Optional[SentenceTransformer]:
        if self._model is None:
            self.load_models()
        return self._model
