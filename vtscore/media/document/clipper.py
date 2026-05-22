"""Document clippers — pass-through document media."""

from __future__ import annotations

from typing import Any

from vtscore.media.clipper import MediaClipper


class DocumentDefaultClipper(MediaClipper):
    """Returns the document media unchanged."""

    @property
    def name(self) -> str:
        return "document_default"

    @property
    def media_type(self) -> str:
        return "document"

    @property
    def description(self) -> str:
        return "Import each document as-is, without splitting."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [media]
