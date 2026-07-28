"""The labelset media ingest runs as a background task, not inside the request.

Both import paths pull the media of labels the active dataset doesn't have in
from their origins, which is one fetch + embed per label.  Issue #2703: that
used to run inline, so the request hung for the duration (and a long enough
ingest hit a gateway timeout with the import already half-applied).  Both now
hand the work to ``detector_loading_tasks`` and answer immediately with a
task id.

Covers, for ``POST /api/detectors/registry/from-labelset/<importer>`` and
``POST /api/label-importers/import/<importer>``:

- the request returns while the ingest is still running,
- the task is visible on the detector-task tracker (the SSE feed's source),
- cancelling the task surfaces as ``error="Cancelled"``,
- the task publishes its terminal counts as ``ingest_result``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from unittest.mock import patch

import pytest

from vtscore.concurrency.progress import detector_loading_tasks
from vtscore.datasets import ingest as ingest_module

from tests import wait_for_detector_task


@pytest.fixture
def foreign_labels(tmp_path):
    """A labels JSON file whose two entries resolve to no loaded media.

    Returns ``(labels_path, clips_dir)``.  The clips are real files on disk so
    a non-stubbed ingest can actually resolve them.
    """
    from helpers import make_wav_file

    clips_dir = tmp_path / "ingest_task_clips"
    clips_dir.mkdir()
    origin = {
        "importer": "server_folder",
        "params": {"path": str(clips_dir), "media_type": "audio"},
    }
    entries = []
    for i, (freq, label) in enumerate(((311.0, "good"), (523.0, "bad")), start=1):
        path = make_wav_file(clips_dir, f"task_{i}.wav", frequency=freq)
        entries.append(
            {
                "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                "label": label,
                "origin": origin,
                "origin_name": path.name,
                "filename": path.name,
            }
        )
    labels_path = tmp_path / "task_labels.json"
    labels_path.write_text(json.dumps({"labels": entries}))
    return labels_path, clips_dir


class _BlockingIngest:
    """Stand-in for ``ingest_missing_medias`` that parks until released."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.entry_count = 0

    def __call__(self, entries, medias, on_progress=None):
        self.entry_count = len(entries)
        self.started.set()
        if on_progress is not None:
            on_progress("ingesting", "Re-ingesting from server_folder…", 0, len(entries))
        # Deliberately unbounded except for the release: a bounded loop can
        # outrun a slow cancel on a loaded machine (see CLAUDE.md).
        self.release.wait(timeout=30)
        if on_progress is not None:
            # The real ingest reports per item, so a cancel raised between
            # items is what actually stops it; report once more so a test that
            # cancelled mid-park sees the same path.
            on_progress("ingesting", "Re-ingesting from server_folder…", 1, len(entries))
        return 0


def _active_task(task_id: str) -> dict:
    """The tracker snapshot for *task_id*, asserted to still be running."""
    matches = [t for t in detector_loading_tasks.list_tasks() if t["task_id"] == task_id]
    assert matches, f"task {task_id!r} is not on the detector task tracker"
    assert matches[0]["status"] != "idle", f"task {task_id!r} already finished"
    return matches[0]


class TestFromLabelsetIngestIsBackgrounded:
    def test_request_returns_while_ingest_runs(self, client, foreign_labels):
        labels_path, _ = foreign_labels
        blocker = _BlockingIngest()

        with patch.object(ingest_module, "ingest_missing_medias", blocker):
            res = client.post(
                "/api/detectors/registry/from-labelset/server_json_file",
                json={"name": "BackgroundIngest", "filepath": str(labels_path)},
            )
            assert res.status_code == 201, res.get_json()
            task_id = res.get_json()["ingest_task_id"]
            assert task_id, "the route must hand back the ingest task id"
            assert blocker.started.wait(timeout=10)
            # The response landed while the ingest is still parked, which is
            # the whole point: the modal is no longer held open by the fetch.
            assert _active_task(task_id)["detector_id"] == res.get_json()["detector"]["id"]
            assert blocker.entry_count == 2
            blocker.release.set()

        wait_for_detector_task(task_id)

    def test_cancel_stops_the_ingest(self, client, foreign_labels):
        labels_path, _ = foreign_labels
        blocker = _BlockingIngest()

        with patch.object(ingest_module, "ingest_missing_medias", blocker):
            res = client.post(
                "/api/detectors/registry/from-labelset/server_json_file",
                json={"name": "CancelledIngest", "filepath": str(labels_path)},
            )
            task_id = res.get_json()["ingest_task_id"]
            assert blocker.started.wait(timeout=10)

            cancel = client.post(f"/api/detectors/cancel/{task_id}")
            assert cancel.status_code == 200
            blocker.release.set()

        snapshot = wait_for_detector_task(task_id)
        # The stub reports progress once *after* the cancel flag is set, and
        # the task's progress callback is what raises CancelledError.
        assert snapshot["error"] == "Cancelled"

    def test_ingest_result_reports_what_landed(self, client, foreign_labels):
        labels_path, _ = foreign_labels
        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"name": "ResultIngest", "filepath": str(labels_path)},
        )
        assert res.status_code == 201, res.get_json()
        snapshot = wait_for_detector_task(res.get_json()["ingest_task_id"])
        assert not snapshot["error"]
        assert snapshot["ingest_result"] == {"ingested": 2}


class TestLabelImportIngestIsBackgrounded:
    def test_request_returns_while_auto_resolve_runs(self, client, foreign_labels):
        labels_path, _ = foreign_labels
        blocker = _BlockingIngest()

        with patch.object(ingest_module, "ingest_missing_medias", blocker):
            res = client.post(
                "/api/label-importers/import/server_json_file",
                json={"filepath": str(labels_path)},
            )
            assert res.status_code == 200, res.get_json()
            body = res.get_json()
            task_id = body["ingest_task_id"]
            assert task_id
            assert body["ingest_pending_count"] == 2
            assert "in the background" in body["message"]
            assert blocker.started.wait(timeout=10)
            _active_task(task_id)
            blocker.release.set()

        wait_for_detector_task(task_id)

    def test_auto_resolve_applies_labels_and_publishes_counts(self, client, foreign_labels):
        """The backgrounded pass still lands the votes the inline one used to."""
        import app as app_module

        labels_path, _ = foreign_labels
        saved = dict(app_module.medias)
        try:
            res = client.post(
                "/api/label-importers/import/server_json_file",
                json={"filepath": str(labels_path)},
            )
            assert res.status_code == 200, res.get_json()
            snapshot = wait_for_detector_task(res.get_json()["ingest_task_id"])
            assert snapshot["ingest_result"] == {
                "ingested": 2,
                "applied": 2,
                "unresolved": 0,
                "failed": 0,
            }
            names = {m.get("origin_name") for m in app_module.medias.values()}
            assert {"task_1.wav", "task_2.wav"} <= names
            voted = set(app_module.good_votes) | set(app_module.bad_votes)
            ingested_ids = {
                cid
                for cid, m in app_module.medias.items()
                if m.get("origin_name") in ("task_1.wav", "task_2.wav")
            }
            assert ingested_ids <= voted, "auto-resolved media must carry their labels"
        finally:
            app_module.medias.clear()
            app_module.medias.update(saved)
