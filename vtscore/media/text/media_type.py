"""Text (paragraph) media type - TXT/MD files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast


from vtscore.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    demo_slice,
)


class TextMediaType(MediaType):
    """Handles plain-text paragraphs - file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtscore.media.text.embedder_e5.TextE5Embedder`.
    """

    def __init__(self) -> None:
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "text"

    @property
    def name(self) -> str:
        return "Text"

    @property
    def icon(self) -> str:
        return "file-text"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.txt", "*.md"]

    @property
    def folder_import_name(self) -> str:
        return "text"

    @property
    def dir_key(self) -> str:
        return "text_dir"

    @property
    def pickle_extra_fields(self) -> list[str]:
        return ["word_count", "character_count"]

    # ------------------------------------------------------------------
    # Display metadata
    # ------------------------------------------------------------------

    def display_metadata(self, media: dict) -> dict:
        result: dict = {}
        cat = media.get("category")
        if cat and cat not in ("unknown", "custom"):
            result["Category"] = cat
        wc = media.get("word_count")
        if wc:
            result["Word Count"] = wc
        cc = media.get("character_count")
        if cc:
            result["Characters"] = cc
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        result.update({k: v for k, v in super().display_metadata(media).items() if k not in result})
        return result

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

    _AG_NEWS_CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

    _BBC_NEWS_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

    _IMDB_CATEGORIES = ["pos", "neg"]

    _DBPEDIA_CATEGORIES = [
        "Company",
        "EducationalInstitution",
        "Artist",
        "Athlete",
        "OfficeHolder",
        "MeanOfTransportation",
        "Building",
        "NaturalPlace",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "WrittenWork",
    ]

    _ARXIV_CATEGORIES = [
        "cs.AI",
        "cs.CV",
        "cs.LG",
        "cs.CL",
        "cs.CR",
        "math.AG",
        "math.CO",
        "math.PR",
        "physics.gen-ph",
        "q-bio.GN",
        "astro-ph.CO",
        "stat.ML",
    ]

    _REUTERS_CATEGORIES = [
        "earn",
        "acq",
        "money-fx",
        "grain",
        "crude",
        "trade",
        "interest",
        "ship",
        "wheat",
        "corn",
    ]

    @property
    def demo_datasets(self) -> list:
        cats = self._DEMO_CATEGORIES
        ng_desc = "Early-90s Usenet posts"
        ag_desc = "Short news summaries"
        imdb_desc = "Long-form movie reviews"
        wiki_desc = "Wikipedia abstracts"
        arxiv_desc = "arXiv titles & abstracts"
        reuters_desc = "Financial newswire"
        return [
            DemoDataset(
                id="20newsgroups_s",
                label="20 Newsgroups (S)",
                description=ng_desc,
                categories=cats,
                source="20newsgroups",
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=950,
                download_size_mb=15,
            ),
            DemoDataset(
                id="20newsgroups_m",
                label="20 Newsgroups (M)",
                description=ng_desc,
                categories=cats,
                source="20newsgroups",
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=950,
                download_size_mb=15,
            ),
            DemoDataset(
                id="20newsgroups_l",
                label="20 Newsgroups (L)",
                description=ng_desc,
                categories=cats,
                source="20newsgroups",
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=950,
                download_size_mb=15,
            ),
            DemoDataset(
                id="20newsgroups_a",
                label="20 Newsgroups (A)",
                description=ng_desc,
                categories=cats,
                source="20newsgroups",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=950,
                download_size_mb=15,
            ),
            DemoDataset(
                id="ag_news_s",
                label="AG News (S)",
                description=ag_desc,
                categories=self._AG_NEWS_CATEGORIES,
                source="ag_news",
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=30000,
                download_size_mb=15,
            ),
            DemoDataset(
                id="ag_news_m",
                label="AG News (M)",
                description=ag_desc,
                categories=self._AG_NEWS_CATEGORIES,
                source="ag_news",
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=30000,
                download_size_mb=15,
            ),
            DemoDataset(
                id="ag_news_l",
                label="AG News (L)",
                description=ag_desc,
                categories=self._AG_NEWS_CATEGORIES,
                source="ag_news",
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=30000,
                download_size_mb=15,
            ),
            DemoDataset(
                id="ag_news_a",
                label="AG News (A)",
                description=ag_desc,
                categories=self._AG_NEWS_CATEGORIES,
                source="ag_news",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=30000,
                download_size_mb=15,
            ),
            DemoDataset(
                id="bbc_news_a",
                label="BBC News (A)",
                description="BBC news articles",
                categories=self._BBC_NEWS_CATEGORIES,
                source="bbc_news",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=445,
                download_size_mb=15,
            ),
            DemoDataset(
                id="imdb_s",
                label="IMDB Movie Reviews (S)",
                description=imdb_desc,
                categories=self._IMDB_CATEGORIES,
                source="imdb",
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=25000,
                download_size_mb=15,
            ),
            DemoDataset(
                id="imdb_m",
                label="IMDB Movie Reviews (M)",
                description=imdb_desc,
                categories=self._IMDB_CATEGORIES,
                source="imdb",
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=25000,
                download_size_mb=15,
            ),
            DemoDataset(
                id="imdb_l",
                label="IMDB Movie Reviews (L)",
                description=imdb_desc,
                categories=self._IMDB_CATEGORIES,
                source="imdb",
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=25000,
                download_size_mb=15,
            ),
            DemoDataset(
                id="imdb_a",
                label="IMDB Movie Reviews (A)",
                description=imdb_desc,
                categories=self._IMDB_CATEGORIES,
                source="imdb",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=25000,
                download_size_mb=15,
            ),
            DemoDataset(
                id="wikipedia_topics_s",
                label="Wikipedia Topics (S)",
                description=wiki_desc,
                categories=self._DBPEDIA_CATEGORIES,
                source="dbpedia",
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=40000,
                download_size_mb=70,
            ),
            DemoDataset(
                id="wikipedia_topics_m",
                label="Wikipedia Topics (M)",
                description=wiki_desc,
                categories=self._DBPEDIA_CATEGORIES,
                source="dbpedia",
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=40000,
                download_size_mb=70,
            ),
            DemoDataset(
                id="wikipedia_topics_l",
                label="Wikipedia Topics (L)",
                description=wiki_desc,
                categories=self._DBPEDIA_CATEGORIES,
                source="dbpedia",
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=40000,
                download_size_mb=70,
            ),
            DemoDataset(
                id="wikipedia_topics_a",
                label="Wikipedia Topics (A)",
                description=wiki_desc,
                categories=self._DBPEDIA_CATEGORIES,
                source="dbpedia",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=40000,
                download_size_mb=70,
            ),
            DemoDataset(
                id="arxiv_abstracts_s",
                label="arXiv Abstracts (S)",
                description=arxiv_desc,
                categories=self._ARXIV_CATEGORIES,
                source="arxiv",
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=2000,
                download_size_mb=30,
            ),
            DemoDataset(
                id="arxiv_abstracts_m",
                label="arXiv Abstracts (M)",
                description=arxiv_desc,
                categories=self._ARXIV_CATEGORIES,
                source="arxiv",
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=2000,
                download_size_mb=30,
            ),
            DemoDataset(
                id="arxiv_abstracts_l",
                label="arXiv Abstracts (L)",
                description=arxiv_desc,
                categories=self._ARXIV_CATEGORIES,
                source="arxiv",
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=2000,
                download_size_mb=30,
            ),
            DemoDataset(
                id="arxiv_abstracts_a",
                label="arXiv Abstracts (A)",
                description=arxiv_desc,
                categories=self._ARXIV_CATEGORIES,
                source="arxiv",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=2000,
                download_size_mb=30,
            ),
            DemoDataset(
                id="reuters21578_s",
                label="Reuters-21578 (S)",
                description=reuters_desc,
                categories=self._REUTERS_CATEGORIES,
                source="reuters21578",
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=4000,
                download_size_mb=8,
            ),
            DemoDataset(
                id="reuters21578_m",
                label="Reuters-21578 (M)",
                description=reuters_desc,
                categories=self._REUTERS_CATEGORIES,
                source="reuters21578",
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=4000,
                download_size_mb=8,
            ),
            DemoDataset(
                id="reuters21578_l",
                label="Reuters-21578 (L)",
                description=reuters_desc,
                categories=self._REUTERS_CATEGORIES,
                source="reuters21578",
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=4000,
                download_size_mb=8,
            ),
            DemoDataset(
                id="reuters21578_a",
                label="Reuters-21578 (A)",
                description=reuters_desc,
                categories=self._REUTERS_CATEGORIES,
                source="reuters21578",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=4000,
                download_size_mb=8,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def _collect_demo_texts(
        self,
        source: str,
        categories: list,
        slice_start: int,
        slice_end: int | None,
        slice_frac_start: float | None,
        slice_frac_end: float | None,
        on_progress,
    ) -> tuple[list[str], list[str]]:
        """Resolve a demo text source → (selected_texts, selected_categories)."""

        def _slice_by_dict(
            categories_articles: dict[str, list[str]],
        ) -> tuple[list[str], list[str]]:
            texts: list[str] = []
            cats: list[str] = []
            for cat_name in categories:
                articles = categories_articles.get(cat_name, [])
                for article in demo_slice(
                    articles,
                    slice_start,
                    slice_end or len(articles),
                    slice_frac_start,
                    slice_frac_end,
                ):
                    texts.append(article)
                    cats.append(cat_name)
            return texts, cats

        if source == "20newsgroups":
            from vtscore.datasets.downloader import download_20newsgroups  # noqa: PLC0415

            texts_in, labels, category_names = download_20newsgroups(categories, on_progress=on_progress)
            selected_texts: list[str] = []
            selected_categories: list[str] = []
            for cat_name in categories:
                if cat_name in category_names:
                    cat_idx = category_names.index(cat_name)
                    cat_texts = [texts_in[i] for i, lbl in enumerate(labels) if lbl == cat_idx]
                    for text in demo_slice(
                        cat_texts,
                        slice_start,
                        slice_end or len(cat_texts),
                        slice_frac_start,
                        slice_frac_end,
                    ):
                        selected_texts.append(text)
                        selected_categories.append(cat_name)
            return selected_texts, selected_categories

        from vtscore.datasets import downloader  # noqa: PLC0415

        dict_downloaders = {
            "ag_news": downloader.download_ag_news,
            "bbc_news": downloader.download_bbc_news,
            "imdb": downloader.download_imdb,
            "dbpedia": downloader.download_dbpedia,
            "reuters21578": downloader.download_reuters21578,
        }
        if source in dict_downloaders:
            return _slice_by_dict(dict_downloaders[source](on_progress=on_progress))

        if source == "arxiv":
            return _slice_by_dict(
                downloader.download_arxiv_abstracts(
                    categories=categories,
                    on_progress=on_progress,
                )
            )

        raise ValueError(f"Unsupported text source: {source!r}")

    def load_demo_source(
        self,
        source,
        categories,
        slice_start,
        slice_end,
        clips,
        on_progress=None,
        embedder=None,
        slice_frac_start=None,
        slice_frac_end=None,
        **kwargs,
    ):
        import hashlib  # noqa: PLC0415

        if on_progress is None:
            from vtscore.concurrency.progress import update_progress

            on_progress = update_progress

        if embedder is None:
            from vtscore.media import embedders_for_type

            avail = embedders_for_type(self.type_id)
            if not avail:
                raise ValueError(f"No embedders registered for media type {self.type_id!r}")
            embedder = avail[0]

        selected_texts, selected_categories = self._collect_demo_texts(
            source,
            categories,
            slice_start,
            slice_end,
            slice_frac_start,
            slice_frac_end,
            on_progress,
        )

        if getattr(embedder, "_model", None) is None:
            on_progress("loading", "Loading text embedding model…", 0, 0)
            original_cb = embedder._on_progress
            embedder._on_progress = on_progress
            try:
                embedder.load_models()
            finally:
                embedder._on_progress = original_cb

        clip_id = max(clips.keys(), default=0) + 1
        total = len(selected_texts)
        on_progress("embedding", f"Starting embedding for {total} paragraphs...", 0, total)
        demo_origin_template: dict = {"importer": "demo", "params": {}}

        for i, (text_content, category) in enumerate(zip(selected_texts, selected_categories)):
            on_progress("embedding", f"Embedding {category}", i + 1, total)
            text_content = text_content[:1000].strip()
            if not text_content:
                continue
            try:
                embedding = cast(Any, embedder).embed_text_passage(text_content)
            except Exception:
                logging.getLogger(__name__).exception("Error embedding paragraph")
                continue
            if embedding is None:
                continue
            word_count = len(text_content.split())
            character_count = len(text_content)
            text_bytes = text_content.encode("utf-8")
            fname = f"{category}/{category}_{clip_id}.txt"
            clips[clip_id] = {
                "id": clip_id,
                "media_type": self.type_id,
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

    def load_media_data(self, file_path: Path, media_bytes: bytes | None = None) -> dict:
        try:
            if media_bytes is not None:
                text_content = media_bytes.decode("utf-8", errors="replace").strip()
            else:
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
