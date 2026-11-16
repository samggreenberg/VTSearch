"""Blur overrides survive the flows that make a new dataset from an old one (#5693).

A user's Blur / Unblur overrides live in a sidecar keyed by media id, beside
the dataset's own file. A new dataset made from that file must bring them
along under its own ids, or the corrections vanish silently. These tests drive
the real flows: the export route, and an import of the exported file through
the load pipeline (whose chunk consumption renumbers the ids). Promote is
covered with its other tests in ``test_promote_dataset.py``; the carrier itself
(the per-item ``blur_override`` field, combine, the pickle importer) in
``tests_lib/datasets/test_blur_carry.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from vtscore.datasets.blur_state import (
    BLUR_OVERRIDE_FIELD,
    blur_overrides,
    blur_sidecar_path,
    set_blur_override,
)
from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import load_dataset_from_pickle


def _media(cid: int) -> dict[str, Any]:
    """An already-embedded audio media, so the pipeline needs no model."""
    return {
        "id": cid,
        "media_type": "audio",
        "duration": 1.0,
        "file_size": 100,
        "md5": f"carry-md5-{cid}",
        "embedder": "",
        "embedding": np.full(8, float(cid), dtype=np.float32),
        "filename": f"clip_{cid}.wav",
        "category": "unknown",
        "origin": None,
        "origin_name": f"clip_{cid}.wav",
        "media_bytes": f"bytes-{cid}".encode() * 8,
        "media_string": None,
        "media_path": None,
    }


def _sync_thread_factory():
    """Run the load body inline so the assertions see a finished load."""

    def fake_thread(target, daemon=True, name=None):
        t = mock.MagicMock()
        t.start = lambda: target()
        return t

    return fake_thread


def _import_file(path: Path) -> dict:
    """Import *path* through the real load pipeline, synchronously; return the new registry entry."""
    from vtscore.datasets import get_importer
    from vtscore.datasets.load_pipeline import _run_importer_in_background
    from vtscore.datasets.registry import list_datasets

    before = {e["id"] for e in list_datasets()}
    with mock.patch("vtscore.datasets.load_workers.threading.Thread", side_effect=_sync_thread_factory()):
        _run_importer_in_background(get_importer("pickle"), {"file": str(path)})
    (entry,) = [e for e in list_datasets() if e["id"] not in before]
    return entry


def _sidecar(pkl_path: str) -> dict[str, bool] | None:
    path = blur_sidecar_path(pkl_path)
    return json.loads(path.read_text()) if path.exists() else None


class TestImportThroughThePipeline:
    def test_every_override_arrives_on_the_right_item(self, tmp_path):
        """The file's ids have gaps, so the import renumbers: each override follows its item."""
        source = {cid: _media(cid) for cid in (3, 7, 8, 12)}
        exported = tmp_path / "exported.pkl"
        exported.write_bytes(export_dataset_to_file(source, blur_overrides={7: True, 12: False}))

        entry = _import_file(exported)

        reloaded: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(Path(entry["pkl_path"]), reloaded)
        new_id = {m["md5"]: mid for mid, m in reloaded.items()}
        assert sorted(reloaded) == [1, 2, 3, 4]
        assert _sidecar(entry["pkl_path"]) == {
            str(new_id["carry-md5-7"]): True,
            str(new_id["carry-md5-12"]): False,
        }
        # The saved file holds no copy of them: the sidecar is their one home.
        assert not any(BLUR_OVERRIDE_FIELD in m for m in reloaded.values())

    def test_the_open_dataset_has_them_and_its_items_do_not_carry_them(self, tmp_path):
        from vtscore.state.dataset_registry import get_context

        exported = tmp_path / "exported.pkl"
        exported.write_bytes(export_dataset_to_file({1: _media(1), 2: _media(2)}, blur_overrides={2: True}))

        entry = _import_file(exported)

        ctx = get_context(entry["id"])
        assert ctx is not None
        assert blur_overrides(ctx) == {2: True}
        assert not any(BLUR_OVERRIDE_FIELD in m for m in ctx.medias.values())

    def test_a_dataset_with_no_overrides_imports_as_today(self, tmp_path):
        exported = tmp_path / "exported.pkl"
        exported.write_bytes(export_dataset_to_file({1: _media(1), 2: _media(2)}))

        entry = _import_file(exported)

        assert _sidecar(entry["pkl_path"]) is None
        saved = sorted(p.name for p in Path(entry["pkl_path"]).parent.iterdir())
        assert saved == [Path(entry["pkl_path"]).name]


class TestExportRoute:
    def test_folds_the_datasets_overrides_into_the_file(self, client, tmp_path):
        from vtsearch.state import get_active_context, snapshot_medias

        first, second, third = sorted(snapshot_medias())[:3]
        ctx = get_active_context()
        set_blur_override(ctx, first, True)
        set_blur_override(ctx, third, False)

        resp = client.get("/api/dataset/export")
        assert resp.status_code == 200
        exported = tmp_path / "download.pkl"
        exported.write_bytes(resp.data)

        loaded: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(exported, loaded)
        assert loaded[first][BLUR_OVERRIDE_FIELD] is True
        assert loaded[third][BLUR_OVERRIDE_FIELD] is False
        assert BLUR_OVERRIDE_FIELD not in loaded[second]

    def test_no_overrides_export_no_field(self, client, tmp_path):
        resp = client.get("/api/dataset/export")
        assert resp.status_code == 200
        exported = tmp_path / "download.pkl"
        exported.write_bytes(resp.data)

        loaded: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(exported, loaded)
        assert loaded
        assert not any(BLUR_OVERRIDE_FIELD in m for m in loaded.values())


class TestAFailedSaveLeavesNothing:
    def test_a_registry_failure_removes_the_file_and_its_sidecar(self, monkeypatch):
        from vtscore.datasets.registry import get_saved_datasets_dir
        from vtscore.datasets.stages import registry as registry_stage

        def _refuse(**_kw):
            raise RuntimeError("registry is read-only")

        monkeypatch.setattr(registry_stage, "_reg_register", _refuse)
        medias = {1: {**_media(1), BLUR_OVERRIDE_FIELD: True}}

        assert registry_stage._auto_register_dataset(medias, name="doomed") is None

        saved = get_saved_datasets_dir()
        assert not saved.exists() or list(saved.iterdir()) == []
