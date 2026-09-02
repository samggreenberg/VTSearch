"""Text clippers - split into paragraphs or sentences, or pass-through text media."""

from __future__ import annotations

import re
from typing import Any

from vtscore.media.clipper import DefaultClipper, MediaClipper


def _emit_text_pieces(media: dict[str, Any], pieces: list[str]) -> list[dict[str, Any]]:
    """Turn each string in *pieces* into a clip dict derived from *media*.

    Shared by the paragraph and sentence clippers: each piece becomes the
    new ``media_string`` with ``word_count`` / ``character_count`` recomputed
    and the running ``clip_index`` stamped on.
    """
    results: list[dict[str, Any]] = []
    for idx, piece in enumerate(pieces):
        tile = dict(media)
        tile["media_string"] = piece
        tile["word_count"] = len(piece.split())
        tile["character_count"] = len(piece)
        tile["clip_index"] = idx
        results.append(tile)
    return results


class TextDefaultClipper(DefaultClipper):
    """Returns the text media unchanged."""

    def __init__(self) -> None:
        super().__init__("text_default", "text", "Import each text entry as-is, without splitting.")


class TextParagraphClipper(MediaClipper):
    """Split a text media into one media per paragraph.

    Paragraphs are delimited by one or more blank lines (``\\n\\n``,
    tolerating Windows ``\\r\\n\\r\\n`` and blank lines containing only
    whitespace).  If the text contains no paragraph breaks, it is returned
    unchanged.
    """

    # Match a blank line: a newline, any whitespace (which includes \r and
    # additional newlines), then another newline.  Greedy \s* collapses
    # runs of blank lines into a single split boundary.
    _PARAGRAPH_RE = re.compile(r"\n\s*\n")

    @property
    def name(self) -> str:
        return "text_paragraph"

    @property
    def media_type(self) -> str:
        return "text"

    @property
    def description(self) -> str:
        return "Split each text entry into paragraphs separated by blank lines."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        text = media.get("media_string") or ""
        if not text:
            return [media]

        paragraphs = [p.strip() for p in self._PARAGRAPH_RE.split(text) if p.strip()]
        if len(paragraphs) <= 1:
            return [media]

        return _emit_text_pieces(media, paragraphs)


class TextSentenceClipper(MediaClipper):
    """Split a text media into one media per sentence.

    Sentences are detected by splitting on common sentence-ending
    punctuation (``.``, ``!``, ``?``) followed by whitespace or end-of-string.
    If the text contains no sentence boundaries (i.e. is a single sentence),
    it is returned unchanged.
    """

    # Match sentence-ending punctuation followed by whitespace or end.
    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    @property
    def name(self) -> str:
        return "text_sentence"

    @property
    def media_type(self) -> str:
        return "text"

    @property
    def description(self) -> str:
        return "Split each text entry into individual sentences."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        text = media.get("media_string") or ""
        if not text:
            return [media]

        sentences = [s.strip() for s in self._SENTENCE_RE.split(text) if s.strip()]
        if len(sentences) <= 1:
            return [media]

        return _emit_text_pieces(media, sentences)
