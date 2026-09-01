"""Regression tests: every detector-JSON read→merge→write runs under the sync lock.

Audit follow-up (ninth pass): ``sync_labels_to_loaded_detector`` serialises its
RMW of ``detectors/<slug>.json`` under ``label_sync_write_lock``, but four
route-layer writers (``save_detector_labels``, ``vote_detector_label``,
``import_labels_into_detector``, ``find_corrections_to_detector``) did the
same file's RMW unlocked — a concurrent locked sync and an unlocked route
write could interleave and drop each other's entries (lost update).  These
tests pin that each route's ``_write_detector`` call happens while the lock
is held.

Also pins the sentinel guard on ``save_detector_labels``: a header-less
request resolves the request-missing detector/dataset sentinels, whose
``validated_vote_snapshot`` is ``safe=True`` over frozen-empty votes/medias,
so before the ``require_*_header`` decorators it would overwrite the named
detector's labelset with an empty one.
"""

from __future__ import annotations

from vtscore.detectors.labelset_ops import label_sync_write_lock
from vtscore.detectors.store import _detector_path, _read_detector
from vtsearch.state import medias


def _record_locked_writes(monkeypatch, module, attr="_write_detector"):
    """Wrap *module*'s detector writer to record whether the lock was held."""
    real_write = getattr(module, attr)
    locked_at_write: list[bool] = []

    def recording_write(path, data):
        locked_at_write.append(label_sync_write_lock.locked())
        return real_write(path, data)

    monkeypatch.setattr(module, attr, recording_write)
    return locked_at_write


def _create_detector(client, name: str) -> None:
    resp = client.post(
        "/api/detectors",
        json={"name": name, "media_type": "audio", "text_query": "test"},
    )
    assert resp.status_code in (200, 201), resp.get_json()


class TestDetectorJsonWritersHoldSyncLock:
    def test_save_detector_labels_holds_lock(self, client, monkeypatch):
        import vtsearch.routes.detectors.labels as labels_mod

        _create_detector(client, "lock-save")
        first_id = next(iter(medias))
        assert client.post(f"/api/medias/{first_id}/vote", json={"target": "good"}).status_code == 200

        locked = _record_locked_writes(monkeypatch, labels_mod)
        resp = client.post("/api/detectors/lock-save/labels")
        assert resp.status_code == 200, resp.get_json()
        assert locked and all(locked)

    def test_vote_detector_label_holds_lock(self, client, monkeypatch):
        import vtsearch.routes.detectors.labels as labels_mod

        _create_detector(client, "lock-vote")
        first_id = next(iter(medias))
        assert client.post(f"/api/medias/{first_id}/vote", json={"target": "good"}).status_code == 200
        assert client.post("/api/detectors/lock-vote/labels").status_code == 200

        detail = client.get("/api/detectors/lock-vote/labels-detail").get_json()
        elements = detail["good"] + detail["bad"]
        assert elements, "expected at least one saved label element"
        element_id = elements[0]["id"]

        locked = _record_locked_writes(monkeypatch, labels_mod)
        resp = client.post(
            f"/api/detectors/lock-vote/labels/{element_id}/vote",
            json={"target": "bad"},
        )
        assert resp.status_code == 200, resp.get_json()
        assert locked and all(locked)

    def test_import_labels_into_detector_holds_lock(self, client, monkeypatch):
        import vtsearch.routes.detectors.labels as labels_mod
        from vtscore.config import DATA_DIR

        _create_detector(client, "lock-import")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        labels_file = DATA_DIR / "lock_import_labels.json"
        labels_file.write_text('{"labels": [{"md5": "aa11", "label": "good", "origin_name": "x.wav"}]}')
        try:
            locked = _record_locked_writes(monkeypatch, labels_mod)
            resp = client.post(
                "/api/detectors/lock-import/import-labels/server_json_file",
                json={"filepath": str(labels_file)},
            )
            assert resp.status_code == 200, resp.get_json()
            assert locked and all(locked)
        finally:
            labels_file.unlink(missing_ok=True)


class TestSaveLabelsHeaderGuard:
    def test_headerless_save_is_rejected_and_labelset_untouched(self, client):
        """No X-Detector-Id → 400, and the on-disk labelset is not wiped."""
        _create_detector(client, "guard-save")
        first_id = next(iter(medias))
        assert client.post(f"/api/medias/{first_id}/vote", json={"target": "good"}).status_code == 200
        assert client.post("/api/detectors/guard-save/labels").status_code == 200
        before_data = _read_detector(_detector_path("guard-save"))
        assert before_data is not None
        before = before_data["labelset"]
        assert before["labels"], "setup should have persisted one label"

        resp = client.post(
            "/api/detectors/guard-save/labels",
            headers={"X-Detector-Id": ""},
        )
        assert resp.status_code == 400

        after_data = _read_detector(_detector_path("guard-save"))
        assert after_data is not None
        assert after_data["labelset"] == before

    def test_headerless_dataset_save_is_rejected(self, client):
        _create_detector(client, "guard-save-ds")
        resp = client.post(
            "/api/detectors/guard-save-ds/labels",
            headers={"X-Dataset-Id": ""},
        )
        assert resp.status_code == 400
