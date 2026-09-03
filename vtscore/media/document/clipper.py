"""Document clippers - pass-through document media."""

from __future__ import annotations


from vtscore.media.clipper import DefaultClipper


class DocumentDefaultClipper(DefaultClipper):
    """Returns the document media unchanged."""

    def __init__(self) -> None:
        super().__init__("document_default", "document", "Import each document as-is, without splitting.")
