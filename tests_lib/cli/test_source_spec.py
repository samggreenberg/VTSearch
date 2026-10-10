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
from vtscore.datasets.importers.base import OutputSpec


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
            "save_dataset": False,
        }

    def test_pickle_chunked_carries_chunk_size_and_flags(self):
        spec = _SourceSpec(kind="pickle", dataset_path="/data/ds.pkl", chunk_size=250)
        assert spec.describe(stream_results=True, keep_negatives=True) == {
            "kind": "pickle",
            "dataset": "/data/ds.pkl",
            "chunk_size": 250,
            "stream_results": True,
            "keep_negatives": True,
            "save_dataset": False,
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
            "save_dataset": False,
        }

    def test_save_dataset_is_reported(self):
        """``--dry-run`` reads whether the import would be kept from here (#4226)."""
        spec = _SourceSpec(kind="pickle", dataset_path="/data/ds.pkl")
        assert spec.describe(stream_results=False, keep_negatives=False, save_dataset=True)["save_dataset"] is True

    def test_outputs_are_reported_one_dict_per_dataset(self):
        """A multi-dataset source (#4707) lists its datasets, in the shape the web API's ``outputs`` take."""
        spec = _SourceSpec(
            kind="importer",
            importer_name="server_folder",
            field_values={"path": "/srv/kit"},
            outputs=(OutputSpec("audio"), OutputSpec("image", category="document", dataset_name="scans")),
        )
        described = spec.describe(stream_results=False, keep_negatives=False, save_dataset=True)
        assert described["importer"] == "server_folder"
        assert described["params"] == {"path": "/srv/kit"}
        assert [o["media_type"] for o in described["outputs"]] == ["audio", "image"]
        assert described["outputs"][1]["category"] == "document"
        assert described["outputs"][1]["dataset_name"] == "scans"

    def test_a_single_dataset_source_lists_no_outputs(self):
        spec = _SourceSpec(kind="importer", importer_name="server_folder")
        assert "outputs" not in spec.describe(stream_results=False, keep_negatives=False)

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


class TestOutputs:
    """A source with ``outputs`` is several datasets from one importer run (#4707)."""

    def test_only_an_importer_can_make_several_datasets(self):
        with pytest.raises(ValueError, match="Only an importer source"):
            _SourceSpec(kind="pickle", dataset_path="/data/ds.pkl", outputs=(OutputSpec("audio"),))

    def test_load_is_the_wrong_opener(self):
        spec = _SourceSpec(kind="importer", importer_name="server_folder", outputs=(OutputSpec("audio"),))
        with pytest.raises(ValueError, match="load_outputs"):
            spec.load()

    def test_load_outputs_needs_outputs(self):
        spec = _SourceSpec(kind="importer", importer_name="server_folder")
        with pytest.raises(ValueError, match="one dataset"):
            spec.load_outputs()

    def test_unknown_importer_raises_only_on_iteration(self):
        spec = _SourceSpec(kind="importer", importer_name="definitely_not_an_importer", outputs=(OutputSpec("audio"),))
        source = spec.load_outputs()
        with pytest.raises(ValueError, match="Unknown importer"):
            next(iter(source))

    def test_a_single_dataset_importer_is_refused_on_iteration(self):
        spec = _SourceSpec(kind="importer", importer_name="demo", outputs=(OutputSpec("audio"),))
        with pytest.raises(ValueError, match="produces one dataset per run"):
            next(iter(spec.load_outputs()))
