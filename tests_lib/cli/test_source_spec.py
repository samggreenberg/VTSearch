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
            "reference_files": False,
        }

    def test_pickle_chunked_carries_chunk_size_and_flags(self):
        spec = _SourceSpec(kind="pickle", dataset_path="/data/ds.pkl", chunk_size=250)
        assert spec.describe(stream_results=True, keep_negatives=True) == {
            "kind": "pickle",
            "dataset": "/data/ds.pkl",
            "chunk_size": 250,
            "stream_results": True,
            "keep_negatives": True,
            "reference_files": False,
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
            "reference_files": False,
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


class TestReferenceFiles:
    """``--reference-files`` decides thin mode; it used to be forced on.

    The CLI hardcoded ``thin=True`` on all four loader paths while the GUI
    passed the user's "Reference files in place" choice, so the same source
    ingested differently in the two - and any media whose bytes could not be
    re-read from outside the source was dropped in the CLI alone (issue #3556).
    """

    def test_defaults_off_so_the_cli_ingests_like_the_gui(self):
        assert _SourceSpec(kind="pickle", dataset_path="/data/ds.pkl").reference_files is False

    def test_describe_reports_the_choice(self):
        spec = _SourceSpec(kind="importer", importer_name="server_folder", reference_files=True)
        assert spec.describe(stream_results=False, keep_negatives=False)["reference_files"] is True

    @pytest.mark.parametrize("reference_files", [False, True])
    @pytest.mark.parametrize("chunk_size", [None, 250])
    def test_pickle_load_forwards_the_choice_as_thin(self, monkeypatch, chunk_size, reference_files):
        seen: dict[str, object] = {}

        def _fake(path, medias, thin=False):
            seen["thin"] = thin
            medias[1] = {"id": 1}

        def _fake_chunked(path, size, thin=False):
            seen["thin"] = thin
            yield {1: {"id": 1}}

        monkeypatch.setattr("vtscore.datasets.loader.load_dataset_from_pickle", _fake, raising=False)
        monkeypatch.setattr("vtscore.cli.load_dataset_from_pickle", _fake, raising=False)
        monkeypatch.setattr("vtscore.datasets.loader.load_dataset_from_pickle_chunked", _fake_chunked, raising=False)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

        spec = _SourceSpec(
            kind="pickle",
            dataset_path="/data/ds.pkl",
            chunk_size=chunk_size,
            reference_files=reference_files,
        )
        list(spec.load())
        assert seen["thin"] is reference_files
