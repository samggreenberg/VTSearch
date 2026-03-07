"""Document clippers — pass-through document media."""

from __future__ import annotations

from typing import Any

from vtsearch.media.base import MediaClipper


class DocumentDefaultClipper(MediaClipper):
    """Returns the document media unchanged."""

    @property
    def name(self) -> str:
        return "document_default"

    @property
    def media_type(self) -> str:
        return "document"

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [media]
