"""Read a text manifest line by line, so no single line can sink the import (#5561).

A Manifest import's text format is one media path per line.  Manifests arrive
in worse shape than that: two CSV exports concatenated end to end (a header row
turns up in the middle), a quoted column whose value held a line break (its
tail lands on a line of its own), a byte that isn't UTF-8, a value so long it
overflows the filesystem's name limit.  The last two used to raise out of the
reader and fail the whole import; the first two were dropped without a word.

The rules here:

- **Every line is judged on its own.**  A line that names an existing file or
  folder is kept; any other line is skipped with a reason.  Nothing a line
  holds can raise out of :func:`read_text_manifest`.
- **A CSV row is read for its path.**  When a whole line names nothing but
  parses as a CSV row, the row's path column is tried instead: the column a
  header row named (``path``, ``filepath``, ``image`` …), else the first.  The
  format still lists only paths, so the other columns are ignored, and
  :attr:`TextManifest.csv_rows` counts the rows that had some, so the importer
  can say so.
- **Reasons are written for the user.**  Each :class:`SkippedLine` carries the
  line number, an excerpt and a plain-English reason; the importer lists them
  in a warning toast when the import goes ahead, and in its error when
  nothing is left to import.
"""

from __future__ import annotations

import csv
import errno
import re
from dataclasses import dataclass, field
from pathlib import Path

NOT_FOUND = "names no existing file or folder"
NOT_UTF8 = "isn't valid UTF-8 and names no existing file or folder"
CSV_HEADER = "looks like a CSV header row"
QUOTE_FRAGMENT = "has an unclosed quote: part of a quoted value that held a line break"
NUL_BYTE = "holds a NUL byte, which no path can contain"
TOO_LONG = "is too long to be a file path"
NOT_A_PATH = "can't be read as a file path"

_BOM = "﻿"
_EXCERPT_CHARS = 100

#: Header names that mark a CSV column as the one holding media paths.
_PATH_COLUMN_NAMES = frozenset(
    {
        "path",
        "paths",
        "filepath",
        "file_path",
        "full_path",
        "abs_path",
        "abspath",
        "rel_path",
        "relpath",
        "relative_path",
        "media_path",
        "image_path",
        "img_path",
        "audio_path",
        "video_path",
        "clip_path",
        "file",
        "filename",
        "file_name",
        "fname",
        "media",
        "image",
        "img",
        "audio",
        "video",
        "clip",
    }
)
#: A cell that could be a column name: no path separator, no extension dot.
_COLUMN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_ -]{0,63}")


@dataclass(frozen=True)
class SkippedLine:
    """A manifest line that named no media, and why."""

    line_no: int
    excerpt: str
    reason: str

    def describe(self) -> str:
        """One line for a toast's item list or an error message."""
        return f"line {self.line_no}: {self.reason} — {self.excerpt}"


@dataclass
class TextManifest:
    """What :func:`read_text_manifest` made of a manifest."""

    #: Every listed file or folder that exists, in manifest order.
    paths: list[Path] = field(default_factory=list)
    #: Every non-blank, non-comment line that named nothing.
    skipped: list[SkippedLine] = field(default_factory=list)
    #: Lines read as CSV rows whose columns besides the path were ignored.
    csv_rows: int = 0


def read_text_manifest(paths_file: Path) -> TextManifest:
    """Read *paths_file* as one path per line; see the module docstring.

    Relative paths resolve against the manifest's own folder.  Blank lines
    and lines starting with ``#`` are ignored.  Raises only when the manifest
    itself can't be read (missing, unreadable); never for what a line holds.
    """
    reader = _ManifestReader(paths_file.resolve().parent)
    for line_no, raw in enumerate(paths_file.read_bytes().splitlines(), start=1):
        reader.read_line(line_no, raw)
    return reader.manifest


class _ManifestReader:
    """Judges manifest lines in order, remembering the path column a header named."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._path_col = 0
        self.manifest = TextManifest()

    def read_line(self, line_no: int, raw: bytes) -> None:
        text, is_utf8 = _decode(raw)
        # A BOM opens every line that began a file, which is mid-manifest when
        # two exports were concatenated.
        text = text.lstrip(_BOM).strip()
        if not text or text.startswith("#"):
            return
        reason = self._judge(text)
        if reason is None:
            return
        if reason == NOT_FOUND and not is_utf8:
            reason = NOT_UTF8
        self.manifest.skipped.append(SkippedLine(line_no, _excerpt(raw), reason))

    def _judge(self, text: str) -> str | None:
        """Keep the path *text* names and return ``None``, or return why not."""
        if "\x00" in text:
            return NUL_BYTE
        path, reason = _probe(self._base_dir, text)
        if path is None:
            path, reason = self._judge_csv_row(text, reason)
        if path is not None:
            self.manifest.paths.append(path)
            return None
        return reason

    def _judge_csv_row(self, text: str, reason: str) -> tuple[Path | None, str]:
        """Try *text* as a CSV row; *reason* is why the whole line named nothing."""
        fields = _csv_fields(text)
        if not fields:
            return None, reason
        cell = fields[self._path_col].strip() if self._path_col < len(fields) else ""
        path = _probe(self._base_dir, cell)[0] if cell else None
        if path is not None:
            if len(fields) > 1:
                self.manifest.csv_rows += 1
            return path, ""
        # Checked only once the row named nothing, so a file that happens to be
        # called ``image`` is still imported rather than taken for a header.
        header_col = _header_path_column(fields)
        if header_col is not None:
            self._path_col = header_col
            return None, CSV_HEADER
        if text.count('"') % 2:
            return None, QUOTE_FRAGMENT
        return None, reason


def _decode(raw: bytes) -> tuple[str, bool]:
    """Decode a line as UTF-8; ``surrogateescape`` keeps a non-UTF-8 filename usable."""
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="surrogateescape"), False


def _excerpt(raw: bytes) -> str:
    """The line as the user would recognise it, made printable and cut short."""
    text = raw.decode("utf-8", errors="replace").lstrip(_BOM).strip().replace("\x00", "\\x00")
    if len(text) > _EXCERPT_CHARS:
        text = text[: _EXCERPT_CHARS - 1] + "…"
    return text


def _probe(base_dir: Path, text: str) -> tuple[Path | None, str]:
    """Return ``(path, "")`` when *text* names an existing file or folder, else ``(None, reason)``."""
    try:
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        if candidate.is_file() or candidate.is_dir():
            return candidate, ""
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            return None, TOO_LONG
        return None, f"can't be checked ({exc.strerror or exc})"
    except (ValueError, RuntimeError):
        return None, NOT_A_PATH
    return None, NOT_FOUND


def _csv_fields(text: str) -> list[str] | None:
    """Parse *text* as one CSV (or, when it has tabs and no commas, TSV) row."""
    delimiter = "\t" if "\t" in text and "," not in text else ","
    try:
        return next(csv.reader([text], delimiter=delimiter, skipinitialspace=True), None)
    except csv.Error:  # e.g. a value longer than the csv module's field limit
        return None


def _header_path_column(fields: list[str]) -> int | None:
    """Return the index of the path column when *fields* is a header row, else ``None``.

    A header row is one whose every non-empty cell could be a column name and
    at least one of which names a path column.  A name with ``path`` in it
    beats one with ``file``, which beats a bare media word (``image``).
    """
    names = [f.strip().lower().replace(" ", "_").replace("-", "_") for f in fields]
    if not all(_COLUMN_NAME.fullmatch(f.strip()) for f in fields if f.strip()):
        return None
    candidates = [i for i, name in enumerate(names) if name in _PATH_COLUMN_NAMES]
    if not candidates:
        return None

    def rank(i: int) -> tuple[int, int]:
        name = names[i]
        return (0 if "path" in name else 1 if "file" in name or name == "fname" else 2, i)

    return min(candidates, key=rank)
