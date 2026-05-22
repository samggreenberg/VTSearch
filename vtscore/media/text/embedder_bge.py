"""Text embedder — BGE-base-en-v1.5 (sentence-transformers)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtscore.config import BGE_MODEL_ID
from vtscore.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _read_text(media: dict) -> Optional[str]:
    """Return the embeddable text for *media*, or ``None`` on failure.

    Prefers ``media_string`` (set directly by clip re-embed and by some
    importers) over ``media_path`` so the bulk surface does not need to
    round-trip the content through a tempfile.
    """
    text = media.get("media_string")
    if isinstance(text, str):
        return text.strip()
    file_path = media.get("media_path")
    if not file_path:
        return None
    try:
        with open(Path(file_path), "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        logging.getLogger(__name__).exception("Error reading %s", file_path)
        return None


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
    def display_name(self) -> str:
        return "BGE (text)"

    @property
    def media_type_id(self) -> str:
        return "text"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 3):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers…", 2, 3):
            from transformers import BertModel  # noqa: PLC0415

        with timed_progress(self._on_progress, "loading", "Importing sentence_transformers…", 3, 3):
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading BGE model…")
        BertModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading BGE model…"),
        ):
            self._model = load_pretrained_local_first(
                SentenceTransformer, BGE_MODEL_ID, cache_folder=cache_dir, token=False
            )

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

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        text_content = _read_text(media)
        if text_content is None:
            return None
        if not text_content:
            print("Warning: empty text content")
            return None
        try:
            return np.asarray(self._model.encode(text_content, normalize_embeddings=True))
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text content")
            return None

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        """Batch-encode every text file through sentence-transformers in one call."""
        if self._model is None:
            self.load_models()
        if self._model is None:
            return [None] * len(medias)

        passages: list[Optional[str]] = []
        for media in medias:
            text = _read_text(media)
            if not text:
                passages.append(None)
                continue
            passages.append(text)

        ready_indices = [i for i, p in enumerate(passages) if p is not None]
        if not ready_indices:
            return [None] * len(medias)

        total = len(medias)
        self._on_progress("embedding", f"Embedding BGE {len(ready_indices)}/{total}...", 0, total)
        with self._embed_lock:
            try:
                vectors = self._model.encode(
                    [passages[i] for i in ready_indices],
                    normalize_embeddings=True,
                    batch_size=self.embed_batch_size,
                )
            except Exception:
                logging.getLogger(__name__).exception("BGE bulk encode failed")
                return [None] * len(medias)

        results: list[Optional[np.ndarray]] = [None] * len(medias)
        for slot, vec in zip(ready_indices, vectors):
            results[slot] = np.asarray(vec)
        self._on_progress("embedding", f"Embedding BGE {total}/{total}...", total, total)
        return results

    def embed_text_passage(self, text: str) -> Optional[np.ndarray]:
        """Embed *text* as a passage (used when loading demo datasets in-memory)."""
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            return np.asarray(self._model.encode(text, normalize_embeddings=True))
        except Exception:
            logging.getLogger(__name__).exception("Error embedding passage (BGE)")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            return np.asarray(self._model.encode(f"Represent this sentence: {text}", normalize_embeddings=True))
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for text (BGE)")
            return None

    # Internal helper used by loader.py bridge
    def _get_model(self) -> Optional[SentenceTransformer]:
        if self._model is None:
            self.load_models()
        return self._model


EMBEDDER = TextBGEEmbedder()
