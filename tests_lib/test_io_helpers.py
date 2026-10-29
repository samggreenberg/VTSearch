"""Tests for ``vtscore.io``: the shared JSON read / atomic-write / CSV-cell helpers."""

from __future__ import annotations

import json
import os
import threading

import pytest

from vtscore.io import (
    atomic_write_json,
    atomic_write_stream,
    atomic_write_text,
    desanitize_csv_cell,
    read_server_json,
    sanitize_csv_cell,
)


class TestReadServerJson:
    def test_reads_dict(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"a": 1, "b": "two"}))
        assert read_server_json(p) == {"a": 1, "b": "two"}

    def test_reads_list(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps([1, 2, 3]))
        assert read_server_json(p) == [1, 2, 3]

    def test_accepts_string_path(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text("{}")
        assert read_server_json(str(p)) == {}

    def test_missing_raises_by_default(self, tmp_path):
        p = tmp_path / "missing.json"
        with pytest.raises(ValueError, match="File not found"):
            read_server_json(p)

    def test_missing_ok_returns_none(self, tmp_path):
        assert read_server_json(tmp_path / "absent.json", missing_ok=True) is None

    def test_directory_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Not a file"):
            read_server_json(tmp_path)

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            read_server_json(p)

    def test_invalid_utf8_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_bytes(b"\xff\xfe not valid utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            read_server_json(p)


class TestAtomicWriteStream:
    def test_basic_write(self, tmp_path):
        p = tmp_path / "out.txt"
        with atomic_write_stream(p) as f:
            f.write("hello ")
            f.write("world")
        assert p.read_text() == "hello world"

    def test_creates_parents(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "out.txt"
        with atomic_write_stream(p) as f:
            f.write("ok")
        assert p.read_text() == "ok"

    def test_no_tmp_left_behind_on_success(self, tmp_path):
        p = tmp_path / "out.txt"
        with atomic_write_stream(p) as f:
            f.write("data")
        assert [entry.name for entry in tmp_path.iterdir()] == ["out.txt"]

    def test_destination_untouched_until_commit(self, tmp_path):
        """A reader mid-stream sees the *previous* content, never a partial write."""
        p = tmp_path / "out.txt"
        p.write_text("old")
        with atomic_write_stream(p) as f:
            f.write("new")
            f.flush()
            assert p.read_text() == "old"
        assert p.read_text() == "new"

    def test_raising_block_leaves_destination_and_removes_tmp(self, tmp_path):
        """The failure mode this helper exists for: a records iterator dying mid-export."""
        p = tmp_path / "out.txt"
        p.write_text("old")
        with pytest.raises(RuntimeError, match="records blew up"):
            with atomic_write_stream(p) as f:
                f.write("partial")
                raise RuntimeError("records blew up")
        assert p.read_text() == "old"
        assert [entry.name for entry in tmp_path.iterdir()] == ["out.txt"]

    def test_tmp_removed_when_destination_did_not_exist(self, tmp_path):
        p = tmp_path / "out.txt"
        with pytest.raises(RuntimeError):
            with atomic_write_stream(p) as f:
                f.write("partial")
                raise RuntimeError("boom")
        assert not p.exists()
        assert list(tmp_path.iterdir()) == []

    def test_tmp_cleaned_up_when_rename_fails(self, tmp_path, monkeypatch):
        from vtscore import io as vtio

        p = tmp_path / "out.txt"

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(vtio.os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            with atomic_write_stream(p) as f:
                f.write("data")
        assert list(tmp_path.iterdir()) == []

    def test_preserves_csv_crlf(self, tmp_path):
        """``newline=""`` keeps a ``csv`` writer's CRLF endings from being doubled."""
        import csv

        p = tmp_path / "out.csv"
        with atomic_write_stream(p) as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b"])
            writer.writerow([1, 2])
        assert p.read_bytes() == b"a,b\r\n1,2\r\n"

    def test_tmp_name_is_per_writer_unique(self, tmp_path):
        """Two writers racing on one destination must not share a tmp path."""
        p = tmp_path / "out.txt"
        seen = []
        with atomic_write_stream(p):
            seen.append(next(e.name for e in tmp_path.iterdir()))
            with atomic_write_stream(p):
                names = sorted(e.name for e in tmp_path.iterdir())
        assert len(names) == 2, names
        assert names[0] != names[1]


class TestAtomicWriteText:
    def test_basic_write(self, tmp_path):
        p = tmp_path / "out.txt"
        atomic_write_text(p, "hello world")
        assert p.read_text() == "hello world"

    def test_creates_parents(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "out.txt"
        atomic_write_text(p, "ok")
        assert p.read_text() == "ok"

    def test_no_tmp_left_behind_on_success(self, tmp_path):
        p = tmp_path / "out.txt"
        atomic_write_text(p, "data")
        # After a successful write, only the final file should remain.
        siblings = [entry.name for entry in tmp_path.iterdir()]
        assert siblings == ["out.txt"]

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "out.txt"
        p.write_text("old")
        atomic_write_text(p, "new")
        assert p.read_text() == "new"

    def test_preserves_csv_crlf(self, tmp_path):
        """``newline=""`` keeps already-formatted CRLF endings intact."""
        p = tmp_path / "out.csv"
        atomic_write_text(p, "a,b\r\n1,2\r\n")
        # No translation: bytes round-trip exactly.
        assert p.read_bytes() == b"a,b\r\n1,2\r\n"

    def test_tmp_cleaned_up_on_error(self, tmp_path, monkeypatch):
        from vtscore import io as vtio

        p = tmp_path / "out.txt"

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(vtio.os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            atomic_write_text(p, "data")
        # ``out.txt.tmp`` must not linger after the failed write.
        siblings = [entry.name for entry in tmp_path.iterdir()]
        assert siblings == []


class TestAtomicWriteJson:
    def test_writes_dict(self, tmp_path):
        p = tmp_path / "out.json"
        atomic_write_json(p, {"k": [1, 2]})
        assert json.loads(p.read_text()) == {"k": [1, 2]}

    def test_trailing_newline(self, tmp_path):
        p = tmp_path / "out.json"
        atomic_write_json(p, {})
        assert p.read_text().endswith("\n")

    def test_indent_default_is_2(self, tmp_path):
        p = tmp_path / "out.json"
        atomic_write_json(p, {"a": 1})
        # 2-space indent surfaces as a newline + two spaces before "a".
        assert '\n  "a"' in p.read_text()


class TestConcurrentAtomicWrites:
    def test_no_partial_writes_under_concurrent_overwrite(self, tmp_path):
        """A reader that opens *path* at any moment sees a fully-formed
        file; never partial bytes left behind by a tmp file."""
        p = tmp_path / "out.json"
        p.write_text("{}")  # initialise

        stop = threading.Event()
        observed_invalid = []

        def writer(payload: dict):
            for _ in range(50):
                if stop.is_set():
                    return
                atomic_write_json(p, payload)

        def reader():
            while not stop.is_set():
                try:
                    data = json.loads(p.read_text())
                    if data not in ({"a": 1}, {"b": 2}, {}):
                        observed_invalid.append(data)
                except json.JSONDecodeError as exc:
                    observed_invalid.append(repr(exc))

        threads = [
            threading.Thread(target=writer, args=({"a": 1},)),
            threading.Thread(target=writer, args=({"b": 2},)),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        # Let the writers churn for a moment.
        threads[0].join(timeout=2)
        threads[1].join(timeout=2)
        stop.set()
        threads[2].join(timeout=2)

        assert observed_invalid == []

    def test_replace_is_atomic_marker(self, tmp_path):
        """Smoke-test that the destination file is never deleted between
        writes; overwrite goes through ``os.replace``, not ``unlink`` +
        ``open``."""
        p = tmp_path / "out.txt"
        atomic_write_text(p, "first")
        first_inode = os.stat(p).st_ino
        atomic_write_text(p, "second")
        second_inode = os.stat(p).st_ino
        # ``os.replace`` swaps inodes on POSIX, proving the rename was
        # atomic rather than an in-place truncate.
        assert first_inode != second_inode


class TestCsvCellSanitization:
    @pytest.mark.parametrize("prefix", ["=", "+", "-", "@", "\t", "\r"])
    def test_formula_prefix_is_escaped(self, prefix):
        assert sanitize_csv_cell(f"{prefix}take2.wav") == f"'{prefix}take2.wav"

    @pytest.mark.parametrize("value", ["take2.wav", "", "a-b", "'quoted", "0.5"])
    def test_safe_value_is_untouched(self, value):
        assert sanitize_csv_cell(value) == value

    @pytest.mark.parametrize("prefix", ["=", "+", "-", "@", "\t", "\r"])
    def test_desanitize_inverts_sanitize(self, prefix):
        original = f"{prefix}handle_post.txt"
        assert desanitize_csv_cell(sanitize_csv_cell(original)) == original

    @pytest.mark.parametrize("value", ["take2.wav", "", "'", "'quoted", "a-b", "-"])
    def test_desanitize_leaves_unescaped_values_alone(self, value):
        assert desanitize_csv_cell(value) == value

    def test_literal_apostrophe_prefix_is_ambiguous(self):
        # ``'-x`` is both the sanitized form of ``-x`` and a literal value
        # the sanitizer would pass through untouched.  The reader resolves
        # it as the escaped form; nothing in the CSV can tell them apart.
        assert sanitize_csv_cell("'-x") == "'-x"
        assert desanitize_csv_cell("'-x") == "-x"

    def test_sanitize_round_trips_the_desanitized_form(self):
        # Every value the reader produces re-encodes to what it was read from.
        for written in ["'-take2.wav", "'@handle.txt", "plain.wav"]:
            assert sanitize_csv_cell(desanitize_csv_cell(written)) == written
