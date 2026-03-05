"""Text (paragraph) media type — TXT/MD files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from vtsearch.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
)


class TextMediaType(MediaType):
    """Handles plain-text paragraphs — file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtsearch.media.text.embedder.TextE5Embedder`.
    """

    def __init__(self) -> None:
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

    @property
    def pickle_extra_fields(self) -> list[str]:
        return ["word_count", "character_count"]

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    _DEMO_CATEGORIES = [
        "sports", "science", "cars", "hockey", "electronics",
        "religion", "world", "business", "technology", "medicine",
        "crypto", "atheism", "motorcycles", "mideast", "guns",
    ]

    _AG_NEWS_CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

    _BBC_NEWS_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

    _IMDB_CATEGORIES = ["pos", "neg"]

    @property
    def demo_datasets(self) -> list:
        cats = self._DEMO_CATEGORIES
        return [
            DemoDataset(
                id="20newsgroups_s", label="20 Newsgroups (S)",
                description="Usenet posts from the early 1990s across technical, recreational, and political topics.",
                categories=cats, source="ag_news_sample",
                slice_start=0, slice_end=25, download_size_mb=15,
            ),
            DemoDataset(
                id="20newsgroups_m", label="20 Newsgroups (M)",
                description="Usenet posts from the early 1990s across technical, recreational, and political topics.",
                categories=cats, source="ag_news_sample",
                slice_start=25, slice_end=75, download_size_mb=15,
            ),
            DemoDataset(
                id="20newsgroups_l", label="20 Newsgroups (L)",
                description="Usenet posts from the early 1990s across technical, recreational, and political topics.",
                categories=cats, source="ag_news_sample",
                slice_start=75, slice_end=200, download_size_mb=15,
            ),
            DemoDataset(
                id="ag_news_a", label="AG News (A)",
                description="Short news summaries, well-balanced across world, sports, business, and tech.",
                categories=self._AG_NEWS_CATEGORIES, source="ag_news",
                slice_start=0, slice_end=30000, download_size_mb=15,
            ),
            DemoDataset(
                id="bbc_news_a", label="BBC News (A)",
                description="Full BBC news articles — professionally written and cleanly labeled.",
                categories=self._BBC_NEWS_CATEGORIES, source="bbc_news",
                slice_start=0, slice_end=445, download_size_mb=15,
            ),
            DemoDataset(
                id="imdb_a", label="IMDB Movie Reviews (A)",
                description="Long-form user-written movie reviews with binary positive/negative sentiment labels.",
                categories=self._IMDB_CATEGORIES, source="imdb",
                slice_start=0, slice_end=25000, download_size_mb=15,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def load_demo_source(self, source, categories, slice_start, slice_end, clips, on_progress=None, embedder=None):
        import hashlib  # noqa: PLC0415

        if on_progress is None:
            from vtsearch.utils import update_progress

            on_progress = update_progress

        if embedder is None:
            from vtsearch.media import embedders_for_type

            avail = embedders_for_type(self.type_id)
            if not avail:
                raise ValueError(f"No embedders registered for media type {self.type_id!r}")
            embedder = avail[0]

        selected_texts = []
        selected_categories = []

        if source == "ag_news_sample":
            from vtsearch.datasets.downloader import download_20newsgroups  # noqa: PLC0415

            texts, labels, category_names = download_20newsgroups(categories, on_progress=on_progress)

            for cat_name in categories:
                if cat_name in category_names:
                    cat_idx = category_names.index(cat_name)
                    cat_texts = [texts[i] for i, lbl in enumerate(labels) if lbl == cat_idx]
                    for text in cat_texts[slice_start:(slice_end or len(cat_texts))]:
                        selected_texts.append(text)
                        selected_categories.append(cat_name)

        elif source == "ag_news":
            from vtsearch.datasets.downloader import download_ag_news  # noqa: PLC0415

            categories_articles = download_ag_news(on_progress=on_progress)

            for cat_name in categories:
                articles = categories_articles.get(cat_name, [])
                for article in articles[slice_start:(slice_end or len(articles))]:
                    selected_texts.append(article)
                    selected_categories.append(cat_name)

        elif source == "bbc_news":
            from vtsearch.datasets.downloader import download_bbc_news  # noqa: PLC0415

            categories_articles = download_bbc_news(on_progress=on_progress)

            for cat_name in categories:
                articles = categories_articles.get(cat_name, [])
                for article in articles[slice_start:(slice_end or len(articles))]:
                    selected_texts.append(article)
                    selected_categories.append(cat_name)

        elif source == "imdb":
            from vtsearch.datasets.downloader import download_imdb  # noqa: PLC0415

            categories_reviews = download_imdb(on_progress=on_progress)

            for cat_name in categories:
                reviews = categories_reviews.get(cat_name, [])
                for review in reviews[slice_start:(slice_end or len(reviews))]:
                    selected_texts.append(review)
                    selected_categories.append(cat_name)

        else:
            raise ValueError(f"Unsupported text source: {source!r}")

        if getattr(embedder, "_model", None) is None:
            on_progress("loading", "Loading text embedding model…", 0, 0)
            embedder.load_models()

        clip_id = 1
        total = len(selected_texts)
        on_progress("embedding", f"Starting embedding for {total} paragraphs...", 0, total)
        demo_origin_template: dict = {"importer": "demo", "params": {}}

        for i, (text_content, category) in enumerate(zip(selected_texts, selected_categories)):
            on_progress("embedding", f"Embedding {category}: paragraph {i + 1}/{total}", i + 1, total)
            text_content = text_content[:1000].strip()
            if not text_content:
                continue
            try:
                embedding = embedder.embed_text_passage(text_content)
            except Exception as e:
                print(f"Error embedding paragraph: {e}")
                continue
            if embedding is None:
                continue
            word_count = len(text_content.split())
            character_count = len(text_content)
            text_bytes = text_content.encode("utf-8")
            fname = f"{category}/{category}_{clip_id}.txt"
            clips[clip_id] = {
                "id": clip_id,
                "type": self.type_id,
                "embedder": embedder.name,
                "duration": 0,
                "file_size": len(text_bytes),
                "md5": hashlib.md5(text_bytes).hexdigest(),
                "embedding": embedding,
                "media_bytes": None,
                "media_string": text_content,
                "filename": fname,
                "category": category,
                "word_count": word_count,
                "character_count": character_count,
                "origin": {**demo_origin_template},
                "origin_name": fname,
            }
            clip_id += 1

        return None  # text content is inline

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_media_data(self, file_path: Path) -> dict:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()
        except Exception:
            text_content = ""
        return {
            "media_string": text_content,
            "duration": 0,
            "word_count": len(text_content.split()),
            "character_count": len(text_content),
        }

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def media_response(self, media: dict) -> MediaResponse:
        content = self._resolve_media_string(media)
        return MediaResponse(
            data={
                "content": content,
                "word_count": media.get("word_count", 0) or len(content.split()),
                "character_count": media.get("character_count", 0) or len(content),
            },
            mimetype="application/json",
        )
