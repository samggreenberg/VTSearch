"""Blur overrides follow their items into a new dataset (#5693): the library half.

A flow that makes a new dataset from an old one renumbers the media ids, so an
override travels *on its item* (``blur_override``) from the file it was read
from to the save that writes the new dataset's sidecar. These tests pin that
carrier: an exported container folds the overrides in one per entry, reading
it puts each back on its item, a source's sidecar wins over its container, and
the save takes them off again. The load pipeline, the export route and promote
are covered in ``tests/datasets/test_blur_carry_flows.py`` and
``tests/datasets/test_promote_dataset.py``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vtscore.datasets.blur_state as blur_state
from vtscore.datasets.blur_state import (
    BLUR_OVERRIDE_FIELD,
    blur_overrides,
    blur_sidecar_path,
    carry_blur_overrides,
    pending_blur_writes,
    set_blur_override,
    stamp_blur_overrides,
    take_blur_overrides,
    write_blur_sidecar,
)
from vtscore.datasets.importers.combine_datasets import CombineDatasetsImporter
from vtscore.datasets.importers.pickle import PickleDatasetImporter
from vtscore.datasets.load_pipeline import consume_chunks_into
from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
from vtscore.datasets.registry import register_dataset
from vtscore.plugins.uploads import BytesIOUploadedFile, CliUploadedFile
from vtscore.state.core import DatasetContext


def _media(cid: int, tag: str = "a", **extra: Any) -> dict[str, Any]:
    rng = np.random.default_rng(cid)
    return {
        "id": cid,
        "media_type": "image",
        "duration": 0,
        "file_size": 64,
        "md5": f"md5-{tag}{cid}",
        "embedder": "siglip",
        "embeddings": {"siglip": rng.standard_normal(16).astype(np.float32)},
        "filename": f"{tag}{cid}.jpg",
        "category": "dog",
        "media_bytes": f"{tag}{cid}".encode() * 8,
        "media_string": None,
        "media_path": None,
        **extra,
    }


def _medias(ids: list[int], tag: str = "a") -> dict[int, dict[str, Any]]:
    return {cid: _media(cid, tag) for cid in ids}


def _export(path: Path, medias: dict[int, dict[str, Any]], **kw: Any) -> Path:
    path.write_bytes(export_dataset_to_file(medias, embedder="siglip", media_type="image", created_at=1.0, **kw))
    return path


def _by_md5(medias: dict[int, dict[str, Any]]) -> dict[str, bool]:
    """Each item's carried override, keyed by what it is rather than by its (renumbered) id."""
    return {m["md5"]: m[BLUR_OVERRIDE_FIELD] for m in medias.values() if BLUR_OVERRIDE_FIELD in m}


def _entries(path: Path) -> dict[int, dict[str, Any]]:
    """The container's raw pickle entries."""
    from vtscore.datasets.container import read_container

    data, _meta = read_container(path)
    return data["medias"]


class TestTheCarrier:
    def test_stamp_then_take_round_trips_and_leaves_no_field(self):
        medias = _medias([1, 2, 3])
        stamp_blur_overrides(medias, {1: True, 3: False})
        assert take_blur_overrides(medias) == {1: True, 3: False}
        assert not any(BLUR_OVERRIDE_FIELD in m for m in medias.values())

    def test_stamp_drops_an_id_the_medias_do_not_hold(self):
        medias = _medias([1])
        stamp_blur_overrides(medias, {1: True, 9: True})
        assert take_blur_overrides(medias) == {1: True}

    def test_stamp_replaces_a_carried_field(self):
        medias = _medias([1])
        medias[1][BLUR_OVERRIDE_FIELD] = True
        stamp_blur_overrides(medias, {1: False})
        assert medias[1][BLUR_OVERRIDE_FIELD] is False

    def test_take_drops_a_field_that_is_not_a_boolean(self):
        medias = _medias([1, 2])
        medias[1][BLUR_OVERRIDE_FIELD] = "yes"
        medias[2][BLUR_OVERRIDE_FIELD] = True
        assert take_blur_overrides(medias) == {2: True}
        assert BLUR_OVERRIDE_FIELD not in medias[1]


class TestCarryOntoTheContext:
    def test_the_carried_overrides_become_the_contexts(self):
        ctx = DatasetContext("carry")
        ctx.medias.update(_medias([1, 2, 3]))
        stamp_blur_overrides(ctx.medias, {1: True, 2: False})
        revision = ctx._blur_overrides_revision

        assert carry_blur_overrides(ctx) == {1: True, 2: False}
        assert blur_overrides(ctx) == {1: True, 2: False}
        assert ctx._blur_overrides_revision > revision, "the blurred-id cache must be invalidated"
        assert not any(BLUR_OVERRIDE_FIELD in m for m in ctx.medias.values())

    def test_an_override_set_during_the_import_wins(self):
        ctx = DatasetContext("carry-newer")
        ctx.medias.update(_medias([1, 2]))
        set_blur_override(ctx, 1, False)  # a click on the in-flight dataset
        stamp_blur_overrides(ctx.medias, {1: True, 2: True})

        assert carry_blur_overrides(ctx) == {1: False, 2: True}
        assert blur_overrides(ctx) == {1: False, 2: True}

    def test_nothing_carried_returns_what_the_context_already_holds(self):
        ctx = DatasetContext("carry-none")
        ctx.medias.update(_medias([1, 2]))
        assert carry_blur_overrides(ctx) == {}
        set_blur_override(ctx, 2, True)
        assert carry_blur_overrides(ctx) == {2: True}


class TestWriteBlurSidecar:
    def test_writes_exactly_the_overrides(self, tmp_path):
        pkl = tmp_path / "ds_new.pkl"
        pkl.write_bytes(b"container")
        write_blur_sidecar(pkl, {3: False, 1: True})
        assert json.loads(blur_sidecar_path(pkl).read_text()) == {"1": True, "3": False}

    def test_no_overrides_leave_no_sidecar(self, tmp_path):
        pkl = tmp_path / "ds_new.pkl"
        pkl.write_bytes(b"container")
        write_blur_sidecar(pkl, {})
        assert not blur_sidecar_path(pkl).exists()
        write_blur_sidecar(pkl, {1: True})
        write_blur_sidecar(pkl, {})
        assert not blur_sidecar_path(pkl).exists()

    def test_replaces_a_queued_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr(blur_state, "WRITE_DELAY_SECONDS", 3600.0)
        saved = tmp_path / "saved"
        saved.mkdir()
        pkl = saved / "ds_queued.pkl"
        pkl.write_bytes(b"container")
        entry = register_dataset(name="queued", media_type="image", num_items=2, pkl_path=str(pkl))
        ctx = DatasetContext(entry["id"])
        ctx.medias.update(_medias([1, 2]))
        set_blur_override(ctx, 1, True)
        assert blur_sidecar_path(pkl) in pending_blur_writes()

        write_blur_sidecar(pkl, {2: True})

        assert blur_sidecar_path(pkl) not in pending_blur_writes()
        assert json.loads(blur_sidecar_path(pkl).read_text()) == {"2": True}


class TestExportThenImport:
    @pytest.mark.parametrize("thin", [False, True])
    def test_every_override_arrives_on_its_item(self, tmp_path, thin):
        pkl = _export(tmp_path / "out.pkl", _medias([1, 2, 3, 4]), blur_overrides={1: True, 3: False})
        loaded: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl, loaded, thin=thin)

        assert loaded[1][BLUR_OVERRIDE_FIELD] is True
        assert loaded[3][BLUR_OVERRIDE_FIELD] is False
        assert BLUR_OVERRIDE_FIELD not in loaded[2]
        assert BLUR_OVERRIDE_FIELD not in loaded[4]

    def test_the_given_overrides_alone_decide(self, tmp_path):
        """A live dataset's overrides are passed in; a stale field on an item is not written."""
        medias = _medias([1, 2])
        medias[1][BLUR_OVERRIDE_FIELD] = True
        pkl = _export(tmp_path / "out.pkl", medias, blur_overrides={2: False})
        entries = _entries(pkl)
        assert BLUR_OVERRIDE_FIELD not in entries[1]
        assert entries[2][BLUR_OVERRIDE_FIELD] is False

    def test_a_carried_field_is_written_when_no_overrides_are_given(self, tmp_path):
        """The staging flow: an imported file staged again for a combine keeps its overrides."""
        first = _export(tmp_path / "first.pkl", _medias([1, 2]), blur_overrides={2: True})
        loaded: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(first, loaded)
        staged = _export(tmp_path / "staged.pkl", loaded)
        assert _entries(staged)[2][BLUR_OVERRIDE_FIELD] is True
        assert BLUR_OVERRIDE_FIELD not in _entries(staged)[1]

    def test_no_overrides_export_exactly_as_today(self, tmp_path):
        plain = _export(tmp_path / "plain.pkl", _medias([1, 2]))
        empty = _export(tmp_path / "empty.pkl", _medias([1, 2]), blur_overrides={})
        with zipfile.ZipFile(plain) as a, zipfile.ZipFile(empty) as b:
            assert a.namelist() == b.namelist()
            for name in a.namelist():
                assert a.read(name) == b.read(name), name
        assert not any(BLUR_OVERRIDE_FIELD in e for e in _entries(plain).values())

        loaded: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(plain, loaded)
        assert take_blur_overrides(loaded) == {}


class TestPickleImporter:
    def test_a_server_path_reads_the_sidecar_beside_it(self, tmp_path):
        """The sidecar is the file's live state, so it wins over the container's own."""
        pkl = _export(tmp_path / "ds_src.pkl", _medias([1, 2, 3]), blur_overrides={1: True, 2: True})
        write_blur_sidecar(pkl, {2: False, 3: True})

        medias: dict[int, dict[str, Any]] = {}
        PickleDatasetImporter().run({"file": CliUploadedFile(pkl)}, medias)

        assert take_blur_overrides(medias) == {1: True, 2: False, 3: True}

    def test_an_upload_brings_the_containers_overrides(self, tmp_path):
        pkl = _export(tmp_path / "upload.pkl", _medias([1, 2]), blur_overrides={2: True})
        medias: dict[int, dict[str, Any]] = {}
        PickleDatasetImporter().run({"file": BytesIOUploadedFile(pkl.read_bytes(), "upload.pkl")}, medias)
        assert take_blur_overrides(medias) == {2: True}


class TestCombine:
    """Combine merges sources with fresh ids; each item keeps its own override."""

    def _sources(self, tmp_path: Path) -> list[str]:
        # Source A: overrides in its container, and a sidecar (its live state)
        # that overrules one of them and adds another.
        a = _export(tmp_path / "ds_a.pkl", _medias([2, 5, 9], "a"), blur_overrides={2: True, 5: True})
        write_blur_sidecar(a, {5: False, 9: True})
        # Source B: overrides in its container only, plus a duplicate of A's
        # item 2 (the merge keeps the first copy, and with it A's override).
        b_medias = _medias([1, 4], "b")
        b_medias[7] = {**_media(7, "b"), "md5": "md5-a2"}
        b = _export(tmp_path / "ds_b.pkl", b_medias, blur_overrides={4: False, 7: False})
        return [str(a), str(b)]

    EXPECTED = {"md5-a2": True, "md5-a5": False, "md5-a9": True, "md5-b4": False}

    def test_run(self, tmp_path):
        medias: dict[int, dict[str, Any]] = {}
        CombineDatasetsImporter().run({"datasets": self._sources(tmp_path)}, medias)
        assert sorted(medias) == [1, 2, 3, 4, 5]
        assert _by_md5(medias) == self.EXPECTED

    def test_chunked_as_the_pipeline_runs_it(self, tmp_path):
        """Each source is its own chunk with ids from 1; the pipeline renumbers them again."""
        medias: dict[int, dict[str, Any]] = {}
        consume_chunks_into(medias, CombineDatasetsImporter().run_chunked({"datasets": self._sources(tmp_path)}, 0))
        assert sorted(medias) == [1, 2, 3, 4, 5]
        assert _by_md5(medias) == self.EXPECTED
        overrides = take_blur_overrides(medias)
        assert {medias[mid]["md5"]: blurred for mid, blurred in overrides.items()} == self.EXPECTED
