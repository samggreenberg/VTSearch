"""Resumable streamed exports: the write helpers, the ABC contract, and the two file exporters (#5280).

A ``--stream-results --resume`` run writes through
``ResultsExporter.export_cli_streaming_resumable``, committing a position after
each chunk so a later run can cut the output back to it and append.  These
tests pin the in-place writer in :mod:`vtscore.io`, the default refusal and
derived support flag on the base class, and that each file exporter's resumed
output is byte-for-byte the output of one uninterrupted call.
"""

from __future__ import annotations

import json

import pytest

from vtscore.exporters import get_exporter
from vtscore.exporters.base import ResultsExporter, StreamResume
from vtscore.io import commit_write_stream, resumable_write_stream

_HEADER = {
    "media_type": "audio",
    "detectors": [{"detector_name": "dog_bark", "threshold": 0.5, "balance": None}],
    "keep_negatives": True,
}


def _chunks() -> list[list[tuple[str, dict]]]:
    """Three chunks, the middle one scoring to nothing."""
    return [
        [("dog_bark", {"id": 1, "filename": "a.wav", "score": 0.9, "label": "good"})],
        [],
        [
            ("dog_bark", {"id": 3, "filename": "c.wav", "score": 0.7, "label": "good"}),
            ("dog_bark", {"id": 4, "filename": "d.wav", "score": 0.2, "label": "bad"}),
        ],
    ]


class _Recorder:
    """A ``StreamResume.commit`` that keeps every position it is handed."""

    def __init__(self) -> None:
        self.positions: list[int] = []

    def __call__(self, position: int) -> None:
        self.positions.append(position)


# ---------------------------------------------------------------------------
# vtscore.io
# ---------------------------------------------------------------------------


class TestResumableWriteStream:
    def test_a_fresh_stream_truncates_and_creates_parents(self, tmp_path):
        path = tmp_path / "deep" / "out.txt"
        path.parent.mkdir()
        path.write_text("stale")

        with resumable_write_stream(path, None) as f:
            f.write("new")
            assert commit_write_stream(f) == 3

        assert path.read_text() == "new"

    def test_a_resumed_stream_cuts_back_to_the_position_and_appends(self, tmp_path):
        path = tmp_path / "out.txt"
        path.write_text("kept|torn")

        with resumable_write_stream(path, 5) as f:
            f.write("more")

        assert path.read_text() == "kept|more"

    def test_what_was_committed_survives_an_exception(self, tmp_path):
        path = tmp_path / "out.txt"
        with pytest.raises(RuntimeError), resumable_write_stream(path, None) as f:
            f.write("chunk 1\n")
            commit_write_stream(f)
            raise RuntimeError

        assert path.read_text() == "chunk 1\n"

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="is missing"), resumable_write_stream(tmp_path / "gone.txt", 4):
            pass

    def test_a_file_shorter_than_the_position_is_refused(self, tmp_path):
        path = tmp_path / "out.txt"
        path.write_text("ab")
        with pytest.raises(ValueError, match="holds 2 bytes, fewer than the 4"), resumable_write_stream(path, 4):
            pass
        assert path.read_text() == "ab"

    def test_the_committed_position_counts_bytes_not_characters(self, tmp_path):
        with resumable_write_stream(tmp_path / "out.txt", None) as f:
            f.write("é")
            assert commit_write_stream(f) == 2


# ---------------------------------------------------------------------------
# The ABC
# ---------------------------------------------------------------------------


class TestBaseContract:
    def test_default_refuses(self):
        class Plain(ResultsExporter):
            name = "plain"
            display_name = "Plain"
            description = "x"
            fields = []

        exporter = Plain()
        assert exporter.supports_resumable_streaming is False
        assert exporter.resumable_stream_path({}) == ""
        with pytest.raises(NotImplementedError):
            exporter.export_cli_streaming_resumable(_HEADER, iter([]), {}, StreamResume(None, lambda _p: None))

    def test_support_is_derived_from_overriding_the_method(self):
        class Resumable(ResultsExporter):
            name = "resumable"
            display_name = "Resumable"
            description = "x"
            fields = []

            def export_cli_streaming_resumable(self, header, chunks, field_values, resume):
                return {"message": "ok"}

        assert Resumable().supports_resumable_streaming is True

    @pytest.mark.parametrize(
        ("name", "resumable"),
        [("server_json_file", True), ("server_csv_file", True), ("gui", False), ("webhook", False)],
    )
    def test_built_ins(self, name, resumable):
        assert get_exporter(name).supports_resumable_streaming is resumable


# ---------------------------------------------------------------------------
# The file exporters
# ---------------------------------------------------------------------------


def _export(name: str, path, chunks, position=None) -> _Recorder:
    commits = _Recorder()
    get_exporter(name).export_cli_streaming_resumable(
        _HEADER, iter(chunks), {"filepath": str(path)}, StreamResume(position, commits)
    )
    return commits


@pytest.mark.parametrize("name", ["server_json_file", "server_csv_file"])
class TestFileExporters:
    def test_commits_after_the_header_and_after_every_chunk(self, tmp_path, name):
        path = tmp_path / "out"

        commits = _export(name, path, _chunks())

        assert len(commits.positions) == 1 + 3
        assert commits.positions == sorted(commits.positions)
        assert commits.positions[-1] == path.stat().st_size
        # The empty chunk commits too, at the position the chunk before it left.
        assert commits.positions[1] == commits.positions[2]

    def test_matches_the_plain_streamed_export(self, tmp_path, name):
        plain = tmp_path / "plain"
        exporter = get_exporter(name)
        records = [record for chunk in _chunks() for record in chunk]
        exporter.export_cli_streaming(_HEADER, iter(records), {"filepath": str(plain)})

        _export(name, tmp_path / "resumable", _chunks())

        assert (tmp_path / "resumable").read_bytes() == plain.read_bytes()

    def test_a_resumed_export_continues_without_a_second_header(self, tmp_path, name):
        whole = tmp_path / "whole"
        _export(name, whole, _chunks())
        path = tmp_path / "out"
        first = _export(name, path, _chunks()[:1])
        # A kill mid-chunk leaves a torn line past the last commit.
        with path.open("a") as f:
            f.write("torn")

        commits = _export(name, path, _chunks()[1:], position=first.positions[-1])

        assert path.read_bytes() == whole.read_bytes()
        assert len(commits.positions) == 2

    def test_the_resumable_path_is_the_filepath(self, tmp_path, name):
        assert get_exporter(name).resumable_stream_path({"filepath": "/srv/out.ndjson"}) == "/srv/out.ndjson"


def test_the_ndjson_header_is_the_streamed_one(tmp_path):
    path = tmp_path / "out.ndjson"
    _export("server_json_file", path, _chunks())

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["_meta"]["format"] == "vtsearch-hits-ndjson/v1"
    assert [line["id"] for line in lines[1:]] == [1, 3, 4]
