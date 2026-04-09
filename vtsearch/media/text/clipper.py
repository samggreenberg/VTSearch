"""Text clippers — split into sentences or pass-through text media."""

from __future__ import annotations

import re
from typing import Any

from vtsearch.media.base import MediaClipper


class TextDefaultClipper(MediaClipper):
    """Returns the text media unchanged."""

    @property
    def name(self) -> str:
        return "text_default"

    @property
    def media_type(self) -> str:
        return "text"

    @property
    def description(self) -> str:
        return "Import each text entry as-is, without splitting."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [media]


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

        results: list[dict[str, Any]] = []
        for idx, sentence in enumerate(sentences):
            tile = dict(media)
            tile["media_string"] = sentence
            tile["word_count"] = len(sentence.split())
            tile["character_count"] = len(sentence)
            tile["clip_index"] = idx
            results.append(tile)
        return results
