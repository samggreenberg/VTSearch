"""``GET /api/medias/blurred`` and ``PUT /api/medias/<id>/blur`` (#5692).

One blur state per dataset, shared by everyone who can read it: an item's
override when someone pressed Blur / Unblur, else whether the server's
``blur_rules`` match its ingest metadata. The override persists in a sidecar
beside the dataset's file. The state model itself is pinned in
``tests_lib/datasets/test_blur_state.py``; these tests cover the routes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import vtsearch.settings as settings_mod
from vtscore.datasets.blur_state import blur_sidecar_path, flush_blur_writes
from vtscore.datasets.registry import register_dataset, unregister_dataset
from vtscore.state.core import DatasetContext, register_context, unregister_context

PORN = {"field": "Porn", "op": ">=", "value": 0.8}
GORE = {"field": "Gore", "op": "truthy"}


def _media(mid: int, **metadata: Any) -> dict[str, Any]:
    return {"id": mid, "media_type": "image", "md5": f"md5-{mid}", "filename": f"{mid}.jpg", "custom_metadata": metadata}


def _medias() -> list[dict[str, Any]]:
    return [_media(1, Porn=0.95), _media(2, Porn=0.1), _media(3, Gore="yes"), _media(4)]


def _dataset(tmp_path: Path, name: str = "blur", **kw: Any) -> tuple[str, Path]:
    """A registered, loaded dataset of :func:`_medias` with a real container file."""
    saved = tmp_path / "saved"
    saved.mkdir(exist_ok=True)
    pkl = saved / f"ds_{name}.pkl"
    pkl.write_bytes(b"container")
    entry = register_dataset(name=name, media_type="image", num_items=4, pkl_path=str(pkl), **kw)
    _open(entry["id"])
    return entry["id"], pkl


def _open(dataset_id: str) -> DatasetContext:
    """Open the dataset as the load route does: a fresh context, filled, then published."""
    ctx = DatasetContext(dataset_id)
    ctx.medias.update({m["id"]: m for m in _medias()})
    register_context(ctx)
    return ctx


def _blurred(client, dataset_id: str):
    return client.get("/api/medias/blurred", headers={"X-Dataset-Id": dataset_id})


def _set(client, dataset_id: str, media_id: int, blurred: Any):
    return client.put(f"/api/medias/{media_id}/blur", json={"blurred": blurred}, headers={"X-Dataset-Id": dataset_id})


@pytest.fixture
def rules():
    settings_mod.set_blur_rules([PORN])


class TestBlurredIds:
    def test_no_rules_and_no_overrides_blur_nothing(self, client, tmp_path):
        dataset_id, _ = _dataset(tmp_path)
        resp = _blurred(client, dataset_id)
        assert resp.status_code == 200
        assert resp.get_json() == {"ids": []}

    def test_the_rules_decide_items_without_an_override(self, client, tmp_path, rules):
        dataset_id, _ = _dataset(tmp_path)
        assert _blurred(client, dataset_id).get_json() == {"ids": [1]}

    def test_overrides_win_in_both_directions(self, client, tmp_path, rules):
        dataset_id, _ = _dataset(tmp_path)
        assert _set(client, dataset_id, 1, False).status_code == 200
        assert _set(client, dataset_id, 4, True).status_code == 200
        assert _blurred(client, dataset_id).get_json() == {"ids": [4]}

    def test_a_rules_change_redecides_only_items_without_an_override(self, client, tmp_path, rules):
        dataset_id, _ = _dataset(tmp_path)
        _set(client, dataset_id, 1, False)
        _set(client, dataset_id, 2, True)
        assert _blurred(client, dataset_id).get_json() == {"ids": [2]}
        settings_mod.set_blur_rules([PORN, GORE])
        # 3 is newly matched by Gore; 1 stays overruled, 2 stays pinned.
        assert _blurred(client, dataset_id).get_json() == {"ids": [2, 3]}
        settings_mod.set_blur_rules([])
        assert _blurred(client, dataset_id).get_json() == {"ids": [2]}

    def test_the_dataset_header_is_required(self, client):
        resp = client.get("/api/medias/blurred", headers={"X-Dataset-Id": ""})
        assert resp.status_code == 400

    def test_an_unloaded_dataset_is_a_409(self, client, tmp_path):
        dataset_id, _ = _dataset(tmp_path)
        unregister_context(dataset_id)
        assert _blurred(client, dataset_id).status_code == 409


class TestSetBlur:
    def test_records_the_override_and_answers_with_it(self, client, tmp_path):
        dataset_id, _ = _dataset(tmp_path)
        resp = _set(client, dataset_id, 2, True)
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True, "id": 2, "blurred": True}
        assert _blurred(client, dataset_id).get_json() == {"ids": [2]}

    def test_is_idempotent(self, client, tmp_path):
        dataset_id, _ = _dataset(tmp_path)
        _set(client, dataset_id, 2, True)
        assert _set(client, dataset_id, 2, True).get_json() == {"ok": True, "id": 2, "blurred": True}
        assert _blurred(client, dataset_id).get_json() == {"ids": [2]}

    def test_persists_in_the_sidecar(self, client, tmp_path):
        dataset_id, pkl = _dataset(tmp_path)
        _set(client, dataset_id, 2, True)
        _set(client, dataset_id, 1, False)
        flush_blur_writes()
        assert json.loads(blur_sidecar_path(pkl).read_text()) == {"1": False, "2": True}

    def test_an_override_survives_closing_and_reopening_the_dataset(self, client, tmp_path, rules):
        dataset_id, _ = _dataset(tmp_path)
        _set(client, dataset_id, 1, False)
        _set(client, dataset_id, 4, True)
        flush_blur_writes()
        unregister_context(dataset_id)
        _open(dataset_id)
        assert _blurred(client, dataset_id).get_json() == {"ids": [4]}

    def test_an_unknown_media_is_a_404(self, client, tmp_path):
        dataset_id, _ = _dataset(tmp_path)
        assert _set(client, dataset_id, 99, True).status_code == 404

    @pytest.mark.parametrize("body", [{}, {"blurred": None}, {"blurred": "maybe"}])
    def test_a_missing_or_malformed_state_is_a_422(self, client, tmp_path, body):
        dataset_id, _ = _dataset(tmp_path)
        resp = client.put("/api/medias/2/blur", json=body, headers={"X-Dataset-Id": dataset_id})
        assert resp.status_code == 422

    def test_the_dataset_header_is_required(self, client):
        resp = client.put("/api/medias/1/blur", json={"blurred": True}, headers={"X-Dataset-Id": ""})
        assert resp.status_code == 400

    def test_one_state_per_dataset(self, client, tmp_path):
        first, _ = _dataset(tmp_path, "first")
        second, _ = _dataset(tmp_path, "second")
        _set(client, first, 2, True)
        assert _blurred(client, first).get_json() == {"ids": [2]}
        assert _blurred(client, second).get_json() == {"ids": []}


class TestAccess:
    """A user who may not read the dataset can neither read nor write its blur state."""

    def test_a_non_reader_cannot_read_it(self, client, tmp_path):
        dataset_id, _ = _dataset(tmp_path, created_by="alice")
        assert _blurred(client, dataset_id).status_code == 403

    def test_a_non_reader_cannot_write_it(self, client, tmp_path):
        dataset_id, pkl = _dataset(tmp_path, created_by="alice")
        assert _set(client, dataset_id, 2, True).status_code == 403
        flush_blur_writes()
        assert not blur_sidecar_path(pkl).exists()

    def test_a_non_reader_learns_nothing_about_an_unloaded_dataset(self, client, tmp_path):
        dataset_id, _ = _dataset(tmp_path, created_by="alice")
        unregister_context(dataset_id)
        assert _blurred(client, dataset_id).status_code == 403

    def test_a_listed_reader_shares_the_one_state(self, client, tmp_path):
        dataset_id, _ = _dataset(tmp_path, created_by="alice", readers=["default"])
        assert _set(client, dataset_id, 2, True).status_code == 200
        assert _blurred(client, dataset_id).get_json() == {"ids": [2]}

    def test_a_public_dataset_is_readable_by_anyone(self, client, tmp_path):
        dataset_id, _ = _dataset(tmp_path, created_by="alice", readers=["*"])
        assert _blurred(client, dataset_id).status_code == 200


class TestRemovedWithTheDataset:
    def test_the_delete_route_removes_the_sidecar(self, client, tmp_path):
        dataset_id, pkl = _dataset(tmp_path)
        _set(client, dataset_id, 2, True)
        flush_blur_writes()
        assert blur_sidecar_path(pkl).is_file()
        from vtscore.datasets.registry import add_loaded_id

        add_loaded_id(dataset_id)
        assert client.delete(f"/api/datasets/registry/{dataset_id}").status_code == 200
        assert list(pkl.parent.iterdir()) == []

    def test_a_click_queued_before_the_delete_does_not_come_back(self, client, tmp_path, monkeypatch):
        import vtscore.datasets.blur_state as blur_state

        monkeypatch.setattr(blur_state, "WRITE_DELAY_SECONDS", 3600.0)
        dataset_id, pkl = _dataset(tmp_path)
        _set(client, dataset_id, 2, True)
        assert unregister_dataset(dataset_id) is True
        flush_blur_writes()
        assert list(pkl.parent.iterdir()) == []
