"""Regression tests: an *unexpected* exception still parks ``find_progress``.

Issue #2949.  The scoring routes park the shared ``find_progress`` singleton at
``"idle"`` on every anticipated exit (success, ``abort()``, cancel), but an
unhandled exception used to skip all of them: the request 500'd through the
global handler while the tracker stayed at ``"running"`` on whatever step it
died on — broadcast to every SSE client until the next Find reset it, with the
``auto_finish`` timing recorder still subscribed to the singleton.

``vtsearch.routes._shared.find_idle_on_crash`` now guards the post-resolve body
of ``/api/find``, ``/api/find-label``, and ``/api/auto-detect``, mirroring
``sort_clips``' exception branch.  ``/api/auto-detect`` additionally never
parked the tracker on its *success* path (a cold detector's train writes
"running" from inside the workers), which is covered here too.
"""

from __future__ import annotations

import pytest
from helpers import setup_trainable_model_in_registry
from vtscore.concurrency.progress import find_progress, get_find_progress
from vtsearch.state import snapshot_medias

import app as app_module  # noqa: F401  (ensures routes are registered)


@pytest.fixture(autouse=True)
def _clear_cancel_flag():
    """Never leak a set cancel flag into (or out of) a test."""
    find_progress.reset_cancel()
    yield
    find_progress.reset_cancel()


class _Boom(RuntimeError):
    """Stands in for e.g. an embedding-dimension mismatch raised mid-scoring."""


class TestGuardUnit:
    """``find_idle_on_crash`` itself."""

    def test_parks_idle_and_reraises(self):
        from vtsearch.routes._shared import find_idle_on_crash

        find_progress.update("running", "Scoring…", step=3, total_steps=4)
        with pytest.raises(_Boom), find_idle_on_crash():
            raise _Boom("dimension mismatch")

        snap = get_find_progress()
        assert snap["status"] == "idle"
        # The whole-job step frame is cleared too, so the next job's bar starts
        # from scratch instead of inheriting a half-finished one.
        assert snap["step"] is None
        assert snap["total_steps"] is None

    def test_closes_the_recorder_as_a_failed_run(self):
        """The idle update's ``auto_finish`` hook would otherwise bank a crashed
        run's partial phase timings as a good cost sample."""
        from vtsearch.routes._shared import find_idle_on_crash

        calls: list[bool] = []

        class _Recorder:
            def finish(self, n=None, size_mb=None, ok=True):
                calls.append(ok)

        with pytest.raises(_Boom), find_idle_on_crash(_Recorder()):
            raise _Boom("dimension mismatch")

        assert calls == [False]
        assert get_find_progress()["status"] == "idle"

    def test_abort_passes_through_untouched(self):
        """``abort()`` already parked the tracker; the guard must not re-push an
        idle frame or re-render flask-smorest's envelope."""
        from werkzeug.exceptions import HTTPException

        from vtsearch.routes._shared import find_idle_on_crash

        calls: list[bool] = []

        class _Recorder:
            def finish(self, n=None, size_mb=None, ok=True):
                calls.append(ok)

        with app_module.app.test_request_context("/api/find-label"):
            with pytest.raises(HTTPException) as excinfo:
                with find_idle_on_crash(_Recorder()):
                    from flask_smorest import abort

                    abort(409, message="Find cancelled")
        assert excinfo.value.code == 409
        assert calls == []


class TestFindLabelCrash:
    def test_scoring_failure_returns_500_and_resets_progress(self, client, monkeypatch):
        import vtscore.detectors.training as training_mod

        detector_id = setup_trainable_model_in_registry(
            "crash-find-label",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )

        def boom(*args, **kwargs):
            raise _Boom("mat1 and mat2 shapes cannot be multiplied")

        # Patched on the defining module: find_label imports it inside the body.
        monkeypatch.setattr(training_mod, "score_media_with_model", boom)

        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 500, resp.get_json()
        assert get_find_progress()["status"] == "idle"


class TestMultiFindCrash:
    def test_dataset_scoring_failure_returns_500_and_resets_progress(self, client, monkeypatch):
        import vtsearch.routes.detectors.find as find_mod

        detector_id = setup_trainable_model_in_registry(
            "crash-multi-find",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )

        monkeypatch.setattr(
            find_mod,
            "_resolve_find_datasets",
            lambda dataset_ids: [{"name": "phantom", "pkl_path": "/nonexistent.pkl", "num_items": 1}],
        )

        def boom(*args, **kwargs):
            raise _Boom("mat1 and mat2 shapes cannot be multiplied")

        monkeypatch.setattr(find_mod, "_score_dataset", boom)

        resp = client.post(
            "/api/find",
            json={"dataset_ids": ["anything"], "detector_ids": [detector_id]},
        )
        assert resp.status_code == 500, resp.get_json()
        assert get_find_progress()["status"] == "idle"

    def test_detector_prep_failure_resets_progress(self, client, monkeypatch):
        """The guard starts *before* detector resolution, which reports step 1."""
        import vtsearch.routes.detectors.find as find_mod

        detector_id = setup_trainable_model_in_registry(
            "crash-multi-find-prep",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )

        monkeypatch.setattr(
            find_mod,
            "_resolve_find_datasets",
            lambda dataset_ids: [{"name": "phantom", "pkl_path": "/nonexistent.pkl", "num_items": 1}],
        )

        def boom(_detectors):
            raise _Boom("unreadable detector JSON")

        monkeypatch.setattr(find_mod, "_build_detector_configs", boom)

        resp = client.post(
            "/api/find",
            json={"dataset_ids": ["anything"], "detector_ids": [detector_id]},
        )
        assert resp.status_code == 500, resp.get_json()
        assert get_find_progress()["status"] == "idle"


class TestAutoDetectParksTracker:
    def test_success_parks_tracker_at_idle(self, client):
        """A cold detector's train leaves the tracker at "running"; the route
        owns parking it again once the workers are drained."""
        from vtsearch.settings import set_autofind_detectors

        name = "autodetect-idle"
        setup_trainable_model_in_registry(
            name,
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        set_autofind_detectors([name])

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_json()
        assert get_find_progress()["status"] == "idle"
