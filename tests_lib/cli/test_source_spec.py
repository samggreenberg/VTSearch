"""Tests for ``vtscore.cli._SourceSpec``.

The four public ``autodetect_*_main`` entry points are a 2x2 matrix
(pickle/importer x whole/chunked) whose cells used to hand-copy three
things apiece: the loader call, the ``--dry-run`` source description, and
the "nothing loaded" error text.  They drifted (the chunked pair grew
``stream_results``/``keep_negatives`` and the whole pair did not).
``_SourceSpec`` owns all three, so these tests pin the mapping from a
spec to each of them - the description in particular, since it is the
part the CLI's ``--dry-run`` output is read from.
"""

from __future__ import annotations

import pytest

from vtscore.cli import _SourceSpec


class TestEmptyError:
    def test_pickle_names_the_dataset(self):
        spec = _SourceSpec(kind="pickle", dataset_path="/data/ds.pkl")
        assert spec.empty_error == "No medias loaded from dataset: /data/ds.pkl"

    def test_importer_names_the_importer(self):
        spec = _SourceSpec(kind="importer", importer_name="server_folder")
        assert spec.empty_error == "No medias loaded by importer 'server_folder'"


class TestDescribe:
    def test_pickle_whole(self):
        spec = _SourceSpec(kind="pickle", dataset_path="/data/ds.pkl")
        assert spec.describe(stream_results=False, keep_negatives=False) == {
            "kind": "pickle",
            "dataset": "/data/ds.pkl",
            "chunk_size": None,
            "stream_results": False,
            "keep_negatives": False,
        }

    def test_pickle_chunked_carries_chunk_size_and_flags(self):
        spec = _SourceSpec(kind="pickle", dataset_path="/data/ds.pkl", chunk_size=250)
        assert spec.describe(stream_results=True, keep_negatives=True) == {
            "kind": "pickle",
            "dataset": "/data/ds.pkl",
            "chunk_size": 250,
            "stream_results": True,
            "keep_negatives": True,
        }

    def test_importer_carries_params(self):
        params = {"path": "/srv/sounds", "media_type": "audio"}
        spec = _SourceSpec(kind="importer", importer_name="server_folder", field_values=params)
        assert spec.describe(stream_results=False, keep_negatives=False) == {
            "kind": "importer",
            "importer": "server_folder",
            "params": params,
            "chunk_size": None,
            "stream_results": False,
            "keep_negatives": False,
        }

    @pytest.mark.parametrize(
        "spec",
        [
            _SourceSpec(kind="pickle", dataset_path="/data/ds.pkl"),
            _SourceSpec(kind="pickle", dataset_path="/data/ds.pkl", chunk_size=10),
            _SourceSpec(kind="importer", importer_name="server_folder"),
            _SourceSpec(kind="importer", importer_name="server_folder", chunk_size=10),
        ],
    )
    def test_every_cell_reports_the_streaming_flags(self, spec):
        """All four cells describe streaming - the drift this class guards against."""
        described = spec.describe(stream_results=True, keep_negatives=True)
        assert described["stream_results"] is True
        assert described["keep_negatives"] is True


class TestLoad:
    """``load()`` picks the loader; it must not consume the source eagerly."""

    def test_missing_pickle_raises_only_on_iteration(self, tmp_path):
        spec = _SourceSpec(kind="pickle", dataset_path=str(tmp_path / "nope.pkl"))
        source = spec.load()  # generator: no I/O yet
        with pytest.raises(FileNotFoundError):
            next(iter(source))

    def test_unknown_importer_raises_only_on_iteration(self):
        spec = _SourceSpec(kind="importer", importer_name="definitely_not_an_importer")
        source = spec.load()
        with pytest.raises(ValueError, match="Unknown importer"):
            next(iter(source))
