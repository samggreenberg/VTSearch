"""Text (paragraph) media type - TXT/MD files."""

from __future__ import annotations

from pathlib import Path


from vtscore.media.base import (
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
)
from vtscore.media.text._demo_sources import build_demo_datasets, load_demo_source


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

    @property
    def demo_datasets(self) -> list:
        return build_demo_datasets()

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
        skip_embedding=False,
        **kwargs,
    ):
        return load_demo_source(
            source,
            categories,
            slice_start,
            slice_end,
            clips,
            on_progress=on_progress,
            embedder=embedder,
            slice_frac_start=slice_frac_start,
            slice_frac_end=slice_frac_end,
            skip_embedding=skip_embedding,
            **kwargs,
        )

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
