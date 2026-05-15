"""Text embedder — E5-base-v2 (sentence-transformers)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import E5_MODEL_ID
from vtsearch.media.embedder import (
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

    @property
    def is_default(self) -> bool:
        return True

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

        cache_dir = embedder_load_setup(self._on_progress, "Loading E5 model…")
        BertModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading E5 model…"),
        ):
            self._model = load_pretrained_local_first(
                SentenceTransformer, E5_MODEL_ID, cache_folder=cache_dir, token=False
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
            return self._model.encode(f"passage: {text_content}", normalize_embeddings=True)
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text content")
            return None

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        """Batch-encode every text file through sentence-transformers in one call.

        Sentence-Transformers' ``encode`` natively chunks long input lists
        through the model in ``batch_size`` groups — we feed the whole
        ready set in at once and let it do the GPU batching.
        """
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
            passages.append(f"passage: {text}")

        ready_indices = [i for i, p in enumerate(passages) if p is not None]
        if not ready_indices:
            return [None] * len(medias)

        total = len(medias)
        self._on_progress("embedding", f"Embedding E5 {len(ready_indices)}/{total}...", 0, total)
        with self._embed_lock:
            try:
                vectors = self._model.encode(
                    [passages[i] for i in ready_indices],
                    normalize_embeddings=True,
                    batch_size=self.embed_batch_size,
                )
            except Exception:
                logging.getLogger(__name__).exception("E5 bulk encode failed")
                return [None] * len(medias)

        results: list[Optional[np.ndarray]] = [None] * len(medias)
        for slot, vec in zip(ready_indices, vectors):
            results[slot] = np.asarray(vec)
        self._on_progress("embedding", f"Embedding E5 {total}/{total}...", total, total)
        return results

    def embed_text_passage(self, text: str) -> Optional[np.ndarray]:
        """Embed *text* as a passage (used when loading demo datasets in-memory)."""
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            return self._model.encode(f"passage: {text}", normalize_embeddings=True)
        except Exception:
            logging.getLogger(__name__).exception("Error embedding passage")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            return self._model.encode(f"query: {text}", normalize_embeddings=True)
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for text")
            return None

    # Internal helper used by loader.py bridge
    def _get_model(self) -> Optional[SentenceTransformer]:
        if self._model is None:
            self.load_models()
        return self._model


EMBEDDER = TextE5Embedder()
