"""Text (paragraph) media type — E5-base-v2 embeddings, TXT/MD files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import E5_MODEL_ID, MODELS_CACHE_DIR

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
from vtsearch.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    intercept_tqdm_progress,
)


class TextMediaType(MediaType):
    """Handles plain-text paragraphs using the E5-base-v2 model.

    * Embeds text files with the ``"passage: "`` prefix required by E5's
      asymmetric retrieval design (768-dim, L2-normalised).
    * Embeds text queries with the ``"query: "`` prefix so they land in the
      same space.
    * Serves clips as JSON objects containing the text content and word/
      character statistics (no binary bytes).
    """

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "paragraph"

    @property
    def name(self) -> str:
        return "Text"

    @property
    def icon(self) -> str:
        return "📄"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.txt", "*.md"]

    @property
    def folder_import_name(self) -> str:
        return "paragraphs"

    @property
    def tab_title(self) -> str:
        return "Texts"

    @property
    def dir_key(self) -> str:
        return "text_dir"

    @property
    def legacy_bytes_keys(self) -> list[str]:
        return ["text_content"]

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    # Shared categories for all S/M/L text demo datasets.
    # All three sizes use the same 15 categories; only the underlying
    # articles differ (disjoint slices of each category's texts).
    _DEMO_CATEGORIES = [
        "sports",
        "science",
        "cars",
        "hockey",
        "electronics",
        "religion",
        "world",
        "business",
        "technology",
        "medicine",
        "crypto",
        "atheism",
        "motorcycles",
        "mideast",
        "guns",
    ]

    # Categories for AG News (4 topic categories).
    _AG_NEWS_CATEGORIES = [
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    ]

    # Categories for BBC News (5 topic categories).
    _BBC_NEWS_CATEGORIES = [
        "business",
        "entertainment",
        "politics",
        "sport",
        "tech",
    ]

    # Categories for IMDB Movie Reviews (2 sentiment classes).
    _IMDB_CATEGORIES = [
        "pos",
        "neg",
    ]

    @property
    def demo_datasets(self) -> list:
        cats = self._DEMO_CATEGORIES
        return [
            DemoDataset(
                id="20newsgroups_s",
                label="20 Newsgroups (S)",
                description="Usenet posts from the early 1990s across technical, recreational, and political topics.",
                categories=cats,
                source="ag_news_sample",
                slice_start=0,
                slice_end=25,
                download_size_mb=15,
            ),
            DemoDataset(
                id="20newsgroups_m",
                label="20 Newsgroups (M)",
                description="Usenet posts from the early 1990s across technical, recreational, and political topics.",
                categories=cats,
                source="ag_news_sample",
                slice_start=25,
                slice_end=75,
                download_size_mb=15,
            ),
            DemoDataset(
                id="20newsgroups_l",
                label="20 Newsgroups (L)",
                description="Usenet posts from the early 1990s across technical, recreational, and political topics.",
                categories=cats,
                source="ag_news_sample",
                slice_start=75,
                slice_end=200,
                download_size_mb=15,
            ),
            DemoDataset(
                id="ag_news_a",
                label="AG News (A)",
                description="Short news summaries, well-balanced across world, sports, business, and tech.",
                categories=self._AG_NEWS_CATEGORIES,
                source="ag_news",
                slice_start=0,
                slice_end=30000,
                download_size_mb=15,
            ),
            DemoDataset(
                id="bbc_news_a",
                label="BBC News (A)",
                description="Full BBC news articles — professionally written and cleanly labeled.",
                categories=self._BBC_NEWS_CATEGORIES,
                source="bbc_news",
                slice_start=0,
                slice_end=445,
                download_size_mb=15,
            ),
            DemoDataset(
                id="imdb_a",
                label="IMDB Movie Reviews (A)",
                description="Long-form user-written movie reviews with binary positive/negative sentiment labels.",
                categories=self._IMDB_CATEGORIES,
                source="imdb",
                slice_start=0,
                slice_end=25000,
                download_size_mb=15,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def load_demo_source(self, source, categories, slice_start, slice_end, clips, on_progress=None):
        import hashlib  # noqa: PLC0415

        if on_progress is None:
            from vtsearch.utils import update_progress

            on_progress = update_progress

        if source != "ag_news_sample":
            raise ValueError(f"Unsupported text source: {source!r}")

        from vtsearch.datasets.downloader import download_20newsgroups  # noqa: PLC0415

        texts, labels, category_names = download_20newsgroups(categories, on_progress=on_progress)

        selected_texts = []
        selected_categories = []
        for cat_name in categories:
            if cat_name in category_names:
                cat_idx = category_names.index(cat_name)
                cat_texts = [texts[i] for i, lbl in enumerate(labels) if lbl == cat_idx]
                for text in cat_texts[slice_start : (slice_end or len(cat_texts))]:
                    selected_texts.append(text)
                    selected_categories.append(cat_name)

        if getattr(self, "_model", None) is None:
            on_progress("loading", "Loading text embedding model…", 0, 0)
            self.load_models()

        clip_id = 1
        total = len(selected_texts)
        on_progress("embedding", f"Starting embedding for {total} paragraphs...", 0, total)
        demo_origin: dict = {"importer": "demo", "params": {}}

        for i, (text_content, category) in enumerate(zip(selected_texts, selected_categories)):
            on_progress("embedding", f"Embedding {category}: paragraph {i + 1}/{total}", i + 1, total)
            text_content = text_content[:1000].strip()
            if not text_content:
                continue
            try:
                embedding = self.embed_text_passage(text_content)
            except Exception as e:
                print(f"Error embedding paragraph: {e}")
                continue
            if embedding is None:
                continue
            word_count = len(text_content.split())
            character_count = len(text_content)
            text_bytes = text_content.encode("utf-8")
            fname = f"{category}_{clip_id}.txt"
            clips[clip_id] = {
                "id": clip_id,
                "type": self.type_id,
                "duration": 0,
                "file_size": len(text_bytes),
                "md5": hashlib.md5(text_bytes).hexdigest(),
                "embedding": embedding,
                "clip_bytes": None,
                "clip_string": text_content,
                "filename": fname,
                "category": category,
                "word_count": word_count,
                "character_count": character_count,
                "origin": demo_origin,
                "origin_name": fname,
            }
            clip_id += 1

        return None  # text content is inline

    # ------------------------------------------------------------------
    # Embeddings
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

    def load_models(self) -> None:
        if self._model is not None:
            return
        import gc

        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        gc.collect()
        cache_dir = str(MODELS_CACHE_DIR)
        self._on_progress("loading", "Loading E5 model…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._model = SentenceTransformer(E5_MODEL_ID, cache_folder=cache_dir, token=False)

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

    # internal helper used by loader.py's get_e5_model() bridge
    def _get_model(self) -> Optional[SentenceTransformer]:
        if self._model is None:
            self.load_models()
        return self._model

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_clip_data(self, file_path: Path) -> dict:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()
        except Exception:
            text_content = ""
        return {
            "clip_string": text_content,
            "duration": 0,
            "word_count": len(text_content.split()),
            "character_count": len(text_content),
        }

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def clip_response(self, clip: dict) -> MediaResponse:
        content = self._resolve_clip_string(clip)
        return MediaResponse(
            data={
                "content": content,
                "word_count": clip.get("word_count", 0) or len(content.split()),
                "character_count": clip.get("character_count", 0) or len(content),
            },
            mimetype="application/json",
        )
