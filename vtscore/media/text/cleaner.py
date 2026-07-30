"""Text cleaners - 1→1 cleanup gates run on each text unit before embedding."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from vtscore.media.cleaner import MediaCleaner


def _retype_text(media: dict[str, Any], text: str) -> dict[str, Any]:
    """Return a copy of *media* carrying *text* and its refreshed counts.

    Returns *media* itself when nothing changed, which is what tells the chain
    runner to skip the ``original_*`` snapshot for this item.
    """
    if text == media.get("media_string"):
        return media
    cleaned = dict(media)
    cleaned["media_string"] = text
    cleaned["word_count"] = len(text.split())
    cleaned["character_count"] = len(text)
    if media.get("file_size") is not None:
        cleaned["file_size"] = len(text.encode("utf-8"))
    return cleaned


class TextWhitespaceCleaner(MediaCleaner):
    """Normalise whitespace and repair words broken across line ends.

    Text extracted from a PDF, a scanned page, or a scraped article arrives full
    of layout artifacts that carry no meaning: a hard line break every 80
    columns, words split by a hyphen at the right margin ("para-\\ngraph"),
    runs of spaces where a column gutter used to be, soft hyphens and
    zero-width joiners left over from typesetting, and stray control bytes.  The
    tokenizer sees every one of them - ``para`` and ``graph`` become two
    unrelated tokens - so the embedding of a paragraph depends partly on the
    page width it was typeset at.

    This gate rewrites the text as flowing prose:

    * Unicode line/paragraph separators and CRLF become plain newlines.
    * Control characters (other than newline and tab) and zero-width formatting
      characters are dropped; non-breaking and other exotic spaces become plain
      spaces.
    * A word hyphenated across a line break is rejoined (``"para-\\ngraph"``
      → ``"paragraph"``).
    * Runs of spaces/tabs collapse to one space, trailing space is stripped from
      each line, and runs of three or more newlines collapse to a blank line.

    Paragraph structure survives on purpose: a single blank line still separates
    paragraphs, so a downstream reader (and a human looking at the Original
    toggle) sees the same document, just without the typesetter's debris.
    """

    #: Characters that are invisible but tokenize: soft hyphen, zero-width
    #: space/non-joiner/joiner, word joiner, BOM.
    _INVISIBLE = str.maketrans({c: None for c in "\u00ad\u200b\u200c\u200d\u2060\ufeff"})

    #: Unicode separators that mean "newline": LINE / PARAGRAPH SEPARATOR.
    _LINE_SEPARATORS = re.compile("[\u2028\u2029]")
    #: Separators that mean "space": NBSP, ogham space, the en/em quad family,
    #: narrow NBSP, medium mathematical space, ideographic space.
    _EXOTIC_SPACES = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")

    #: A word split by a hyphen at a line break: letter, hyphen (any dash the
    #: typesetter may have used), end of line, then the rest of the word.
    _BROKEN_WORD = re.compile(r"(\w)[-\u2010\u2011]\n[ \t]*(\w)")

    _HORIZONTAL_RUN = re.compile(r"[ \t]{2,}")
    _TRAILING_SPACE = re.compile(r"[ \t]+\n")
    _BLANK_RUN = re.compile(r"\n{3,}")

    @property
    def name(self) -> str:
        return "text_whitespace"

    @property
    def media_type(self) -> str:
        return "text"

    @property
    def display_name(self) -> str:
        return "Whitespace & Hyphenation"

    @property
    def description(self) -> str:
        return (
            "Collapse whitespace runs, drop control characters, and rejoin words hyphen-broken "
            "across line endings. Highest value on text extracted from PDFs."
        )

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        text = media.get("media_string")
        if not isinstance(text, str) or not text:
            return media

        out = text.replace("\r\n", "\n").replace("\r", "\n")
        out = self._LINE_SEPARATORS.sub("\n", out)
        out = out.translate(self._INVISIBLE)
        out = self._EXOTIC_SPACES.sub(" ", out)
        out = "".join(c for c in out if c in "\n\t" or unicodedata.category(c) != "Cc")
        # De-hyphenate before collapsing, so the line break is still there to
        # tell a broken word from a genuine hyphenated compound.
        out = self._BROKEN_WORD.sub(r"\1\2", out)
        out = self._HORIZONTAL_RUN.sub(" ", out)
        out = self._TRAILING_SPACE.sub("\n", out)
        out = self._BLANK_RUN.sub("\n\n", out)
        out = out.strip()

        return _retype_text(media, out)


class TextMarkupStripCleaner(MediaCleaner):
    """Strip HTML tags and Markdown syntax so the embedder sees prose.

    Scraped pages and README-style corpora carry their formatting inline.  The
    embedder has no notion of markup, so ``<div class="col-md-6">`` and
    ``**bold**`` are just tokens competing with the sentence they wrap: two
    articles about different subjects can end up neighbours because they share a
    site template.

    What this removes:

    * HTML comments, and the *contents* of ``<script>`` / ``<style>`` blocks
      (code and CSS are never the content the user meant to search).
    * HTML tags, replaced by nothing inside a line and by a space for block-level
      tags so words don't fuse; HTML entities are then unescaped.
    * Markdown fences and inline code ticks, heading markers, blockquote
      markers, list bullets, horizontal rules, and emphasis markers - the text
      inside each is kept.
    * Link and image syntax collapses to its label: ``[docs](url)`` → ``docs``,
      ``![a cat](url)`` → ``a cat``.

    Deliberately conservative: only a well-formed tag-looking token is treated
    as a tag (so ``a < b and b > c`` survives), and ``_`` is only stripped when
    it wraps a span at word boundaries, so ``snake_case`` identifiers are left
    alone.
    """

    _COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
    _SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
    #: An opening/closing tag with a real element name - not a bare "<" in prose.
    _TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*?)?/?>")
    #: Block-level elements become a space (or a newline for the big ones) so
    #: the words on either side don't run together once the tag is gone.
    _BLOCK_TAG = re.compile(
        r"</?(?:p|div|br|hr|li|ul|ol|tr|td|th|table|section|article|header|footer|nav|aside|"
        r"h[1-6]|blockquote|pre|figure|figcaption)\b[^<>]*>",
        re.IGNORECASE,
    )

    _FENCE = re.compile(r"^[ \t]*(?:```|~~~).*$", re.MULTILINE)
    _INLINE_CODE = re.compile(r"`+")
    _IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
    _LINK = re.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")
    _REF_LINK = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")
    _AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
    _HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
    _HORIZONTAL_RULE = re.compile(r"^[ \t]{0,3}(?:[-*_][ \t]*){3,}$", re.MULTILINE)
    _QUOTE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE)
    _BULLET = re.compile(r"^([ \t]*)(?:[-*+]|\d{1,3}[.)])[ \t]+", re.MULTILINE)
    _BOLD_ITALIC = re.compile(r"(\*{1,3})(\S(?:.*?\S)?)\1", re.DOTALL)
    #: ``_emphasis_`` only at word boundaries, so ``snake_case`` is untouched.
    _UNDERSCORE_EMPHASIS = re.compile(r"(?<![\w_])(_{1,3})(\S(?:.*?\S)?)\1(?![\w_])", re.DOTALL)
    _STRIKE = re.compile(r"~~(\S(?:.*?\S)?)~~", re.DOTALL)

    _HORIZONTAL_RUN = re.compile(r"[ \t]{2,}")
    _TRAILING_SPACE = re.compile(r"[ \t]+\n")
    _BLANK_RUN = re.compile(r"\n{3,}")

    @property
    def name(self) -> str:
        return "text_markup_strip"

    @property
    def media_type(self) -> str:
        return "text"

    @property
    def display_name(self) -> str:
        return "Markup Strip"

    @property
    def description(self) -> str:
        return (
            "Remove HTML tags and Markdown syntax, keeping the text inside, so the embedder reads "
            "prose instead of angle brackets and asterisks."
        )

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        import html  # noqa: PLC0415

        text = media.get("media_string")
        if not isinstance(text, str) or not text:
            return media

        out = self._COMMENT.sub("", text)
        out = self._SCRIPT_STYLE.sub(" ", out)
        out = self._AUTOLINK.sub(r"\1", out)
        out = self._BLOCK_TAG.sub("\n", out)
        out = self._TAG.sub("", out)
        out = html.unescape(out)

        out = self._FENCE.sub("", out)
        out = self._IMAGE.sub(r"\1", out)
        out = self._LINK.sub(r"\1", out)
        out = self._REF_LINK.sub(r"\1", out)
        out = self._HORIZONTAL_RULE.sub("", out)
        out = self._HEADING.sub("", out)
        out = self._QUOTE.sub("", out)
        out = self._BULLET.sub(r"\1", out)
        out = self._STRIKE.sub(r"\1", out)
        out = self._BOLD_ITALIC.sub(r"\2", out)
        out = self._UNDERSCORE_EMPHASIS.sub(r"\2", out)
        out = self._INLINE_CODE.sub("", out)

        out = self._HORIZONTAL_RUN.sub(" ", out)
        out = self._TRAILING_SPACE.sub("\n", out)
        out = self._BLANK_RUN.sub("\n\n", out)
        out = out.strip()

        return _retype_text(media, out)
