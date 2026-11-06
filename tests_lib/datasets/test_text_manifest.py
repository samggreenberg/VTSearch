"""A malformed text manifest imports what it can and says what it couldn't (#5561).

A Manifest import lists one media path per line.  The manifest in the report
was two CSV exports concatenated, so a header row sat in the middle, and a
quoted ``landmark`` value that held a line break left its tail on a line of
its own.  One line too long for a filename, or one byte that wasn't UTF-8,
used to raise out of the reader and fail the whole import; header rows and
fragments were dropped without a word.  Now each line is judged on its own, a
CSV row is read for its path column, and every skipped line is reported with
its line number and reason.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests_lib.helpers import make_raw_wav_bytes
from vtscore.concurrency.notifications import Notification, notifications
from vtscore.datasets.importers.server_files import ServerFilesDatasetImporter, _expand_paths
from vtscore.datasets.importers.server_files._text_manifest import (
    CSV_HEADER,
    NOT_FOUND,
    NOT_UTF8,
    NUL_BYTE,
    QUOTE_FRAGMENT,
    TOO_LONG,
    read_text_manifest,
)


@pytest.fixture
def collected():
    """Subscribe a collector to the process-wide notification broker."""
    received: list[Notification] = []
    notifications.subscribe(received.append)
    try:
        yield received
    finally:
        notifications.unsubscribe(received.append)


@pytest.fixture
def wavs(tmp_path) -> list[Path]:
    """Three distinct WAV files, so dedup keeps all three."""
    out = []
    for i in range(3):
        p = tmp_path / f"take_{i}.wav"
        p.write_bytes(make_raw_wav_bytes() + b"\x00\x00" * i)
        out.append(p)
    return out


def _manifest(tmp_path: Path, body: str | bytes) -> Path:
    f = tmp_path / "list.txt"
    f.write_bytes(body if isinstance(body, bytes) else body.encode("utf-8"))
    return f


def _reasons(manifest) -> dict[int, str]:
    return {s.line_no: s.reason for s in manifest.skipped}


class TestReadTextManifest:
    def test_plain_paths_are_kept_in_order(self, tmp_path, wavs):
        read = read_text_manifest(_manifest(tmp_path, "".join(f"{p}\n" for p in wavs)))
        assert read.paths == wavs
        assert read.skipped == []
        assert read.csv_rows == 0

    def test_leaked_header_and_quote_tail_are_skipped_with_reasons(self, tmp_path, wavs):
        a, b, c = wavs
        body = f'{a}\n{b}\npath,landmark\n{c}\nseen from the bridge, at night"\n'
        read = read_text_manifest(_manifest(tmp_path, body))
        assert read.paths == wavs
        assert _reasons(read) == {3: CSV_HEADER, 5: QUOTE_FRAGMENT}

    def test_csv_rows_import_their_path_column(self, tmp_path, wavs):
        a, b, _ = wavs
        body = f'path,landmark\n{a},"Eiffel Tower"\n{b},"Big Ben\nat night"\n'
        read = read_text_manifest(_manifest(tmp_path, body))
        assert read.paths == [a, b]
        assert read.csv_rows == 2
        # The header, and the tail of the value that held a line break.
        assert _reasons(read) == {1: CSV_HEADER, 4: QUOTE_FRAGMENT}

    def test_header_picks_the_path_column(self, tmp_path, wavs):
        a, b, _ = wavs
        body = f"id,filepath,landmark\n1,{a},x\n2,{b},y\n3,gone.wav,z\n"
        read = read_text_manifest(_manifest(tmp_path, body))
        assert read.paths == [a, b]
        assert _reasons(read) == {1: CSV_HEADER, 4: NOT_FOUND}

    def test_tab_separated_rows(self, tmp_path, wavs):
        a, _, _ = wavs
        read = read_text_manifest(_manifest(tmp_path, f"path\tlabel\n{a}\tcat\n"))
        assert read.paths == [a]
        assert _reasons(read) == {1: CSV_HEADER}

    def test_a_file_named_like_a_header_is_still_imported(self, tmp_path):
        media = tmp_path / "image"
        media.write_bytes(b"x")
        read = read_text_manifest(_manifest(tmp_path, "image,label\n"))
        assert read.paths == [media.resolve()]

    def test_a_quoted_single_path_is_not_a_csv_row(self, tmp_path):
        media = tmp_path / "a b.wav"
        media.write_bytes(b"x")
        read = read_text_manifest(_manifest(tmp_path, '"a b.wav"\n'))
        assert read.paths == [media.resolve()]
        assert read.csv_rows == 0

    def test_a_line_too_long_for_a_filename_is_skipped_not_raised(self, tmp_path, wavs):
        a, b, _ = wavs
        long_value = "a very long landmark description " * 12
        read = read_text_manifest(_manifest(tmp_path, f"{a}\n{long_value}\n{b}\n"))
        assert read.paths == [a, b]
        assert _reasons(read) == {2: TOO_LONG}

    def test_a_long_value_after_a_path_still_imports_the_path(self, tmp_path, wavs):
        a, _, _ = wavs
        read = read_text_manifest(_manifest(tmp_path, f'{a},"{"word " * 80}\n'))
        assert read.paths == [a]

    def test_a_line_beyond_the_csv_field_limit_is_skipped(self, tmp_path, wavs):
        a, _, _ = wavs
        read = read_text_manifest(_manifest(tmp_path, f"{a}\n{'x' * 200_000}\n"))
        assert read.paths == [a]
        assert _reasons(read) == {2: TOO_LONG}

    def test_nul_byte_is_skipped_for_relative_and_absolute_lines(self, tmp_path, wavs):
        a, _, _ = wavs
        read = read_text_manifest(_manifest(tmp_path, f"{a}\nrel\x00.wav\n/abs\x00.wav\n"))
        assert read.paths == [a]
        assert _reasons(read) == {2: NUL_BYTE, 3: NUL_BYTE}
        assert "\\x00" in read.skipped[0].excerpt

    def test_non_utf8_line_is_skipped_not_raised(self, tmp_path, wavs):
        a, b, _ = wavs
        body = f"{a}\n".encode() + b"na\xefve.wav\n" + f"{b}\n".encode()
        read = read_text_manifest(_manifest(tmp_path, body))
        assert read.paths == [a, b]
        assert _reasons(read) == {2: NOT_UTF8}

    @pytest.mark.skipif(os.name != "posix", reason="POSIX filenames are bytes")
    def test_non_utf8_filename_that_exists_is_imported(self, tmp_path):
        name = os.fsdecode(b"na\xefve.wav")
        try:
            (tmp_path / name).write_bytes(b"x")
        except (OSError, UnicodeEncodeError):
            pytest.skip("filesystem refuses non-UTF-8 names")
        read = read_text_manifest(_manifest(tmp_path, b"na\xefve.wav\n"))
        assert [p.name for p in read.paths] == [name]

    def test_byte_order_marks_are_stripped_mid_file(self, tmp_path, wavs):
        a, b, _ = wavs
        # Two Excel exports concatenated: each opened with a BOM.
        body = f"﻿{a}\n﻿path,landmark\n{b},x\n"
        read = read_text_manifest(_manifest(tmp_path, body))
        assert read.paths == [a, b]
        assert _reasons(read) == {2: CSV_HEADER}

    def test_crlf_blank_lines_and_comments(self, tmp_path, wavs):
        a, b, _ = wavs
        read = read_text_manifest(_manifest(tmp_path, f"# list\r\n\r\n{a}\r\n   \r\n{b}\r\n"))
        assert read.paths == [a, b]
        assert read.skipped == []

    def test_excerpt_is_cut_short(self, tmp_path):
        read = read_text_manifest(_manifest(tmp_path, "missing-" + "y" * 300 + ".wav\n"))
        (skipped,) = read.skipped
        assert len(skipped.excerpt) <= 100
        assert skipped.excerpt.endswith("…")
        assert skipped.describe().startswith("line 1: ")


class TestImporterReportsSkippedLines:
    def _run(self, listing: Path) -> dict:
        medias: dict = {}
        ServerFilesDatasetImporter().run({"paths_file": str(listing), "media_type": "audio"}, medias)
        return medias

    def test_import_goes_ahead_and_warns_with_every_skipped_line(self, tmp_path, wavs, collected):
        a, b, c = wavs
        body = f'path,landmark\n{a},"Eiffel"\npath,landmark\n{b},"Big Ben\nat night"\n{c}\n'
        medias = self._run(_manifest(tmp_path, body))

        assert {m["origin_name"] for m in medias.values()} == {str(a), str(b), str(c)}
        (toast,) = [n for n in collected if n.source == "Manifest"]
        assert toast.level == "warning"
        assert "Skipped 3 manifest line(s)" in toast.message
        assert "2 line(s) were CSV rows" in (toast.detail or "")
        assert list(toast.items) == [
            "line 1: looks like a CSV header row — path,landmark",
            "line 3: looks like a CSV header row — path,landmark",
            f'line 5: {QUOTE_FRAGMENT} — at night"',
        ]

    def test_clean_csv_says_the_other_columns_were_ignored(self, tmp_path, wavs, collected):
        a, b, _ = wavs
        medias = self._run(_manifest(tmp_path, f"{a},Eiffel\n{b},Big Ben\n"))
        assert len(medias) == 2
        (toast,) = [n for n in collected if n.source == "Manifest"]
        assert toast.level == "info"
        assert "2 manifest line(s) as CSV rows" in toast.message

    def test_clean_manifest_raises_no_toast(self, tmp_path, wavs, collected):
        self._run(_manifest(tmp_path, "".join(f"{p}\n" for p in wavs)))
        assert [n for n in collected if n.source == "Manifest"] == []

    def test_nothing_importable_raises_naming_the_lines(self, tmp_path):
        body = "path,landmark\ngone.wav,x\n" + "".join(f"missing_{i}.wav\n" for i in range(6))
        with pytest.raises(ValueError) as excinfo:
            self._run(_manifest(tmp_path, body))
        message = str(excinfo.value)
        assert "No line of list.txt names an existing file" in message
        assert "Skipped 8 line(s)" in message
        assert "line 1: looks like a CSV header row — path,landmark" in message
        assert "and 3 more" in message

    def test_run_chunked_reports_too(self, tmp_path, wavs, collected):
        a, _, _ = wavs
        imp = ServerFilesDatasetImporter()
        listing = _manifest(tmp_path, f"{a}\nnot-a-file\n")
        chunks = list(imp.run_chunked({"paths_file": str(listing), "media_type": "audio"}, chunk_size=10))
        assert sum(len(c) for c in chunks) == 1
        (toast,) = [n for n in collected if n.source == "Manifest"]
        assert list(toast.items) == ["line 2: names no existing file or folder — not-a-file"]


class TestExpandPathsSurvivesHostileEntries:
    def test_symlink_cycles_are_walked_once(self, tmp_path):
        root = tmp_path / "media"
        root.mkdir()
        (root / "x.wav").write_bytes(b"x")
        # Two links back up the tree made the old walk exponential.
        (root / "up").symlink_to("..")
        (root / "up2").symlink_to("..")
        (root / "self").symlink_to(".")

        files = _expand_paths([root])
        assert [p.name for p in files].count("x.wav") == 1

    def test_name_too_long_is_dropped_not_raised(self, tmp_path):
        good = tmp_path / "a.wav"
        good.write_bytes(b"a")
        assert _expand_paths([tmp_path / ("x" * 300), good]) == [good]
