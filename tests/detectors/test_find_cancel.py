"""Regression tests: ``POST /api/find/cancel`` actually stops the scoring routes.

Audit follow-up (ninth pass): the cancel route set ``find_progress``'s cancel
flag, and every scoring route cleared it on entry, but **no scoring loop ever
read it** — cancelling an in-flight Find was a silent no-op that ran to
completion.  The routes now poll the flag at their stage boundaries
(``multi_find``'s dataset loop, ``_score_dataset``'s detector loop,
``find_label``'s train/score/apply boundaries, and the start of each
``auto-detect`` worker) and unwind with a 409.
"""

from __future__ import annotations

import pytest
from tests.helpers import setup_trainable_model_in_registry
from vtscore.concurrency.progress import CancelledError, find_progress, get_find_progress
from vtsearch.state import snapshot_medias

import app as app_module


@pytest.fixture(autouse=True)
def _clear_cancel_flag():
    """Never leak a set cancel flag into (or out of) a test."""
    find_progress.reset_cancel()
    yield
    find_progress.reset_cancel()


class TestCancelPrimitives:
    """The building blocks the routes poll."""

    def test_abort_if_find_cancelled_aborts_409_and_resets_progress(self):
        from werkzeug.exceptions import HTTPException

        from vtsearch.routes.detectors.scoring import _abort_if_find_cancelled

        find_progress.cancel()
        with app_module.app.test_request_context("/api/find-label"):
            with pytest.raises(HTTPException) as excinfo:
                _abort_if_find_cancelled()
        assert excinfo.value.code == 409
        assert get_find_progress()["status"] == "idle"

    def test_abort_if_find_cancelled_noop_without_cancel(self):
        from vtsearch.routes.detectors.scoring import _abort_if_find_cancelled

        with app_module.app.test_request_context("/api/find-label"):
            _abort_if_find_cancelled()  # must not raise

    def test_auto_detect_worker_raises_before_scoring(self):
        """The worker polls before its catch-all try: a cancel must propagate
        to the route as CancelledError, not be swallowed as a failed detector."""
        from vtsearch.routes.detectors.scoring import _score_detector_for_auto_detect

        find_progress.cancel()
        with pytest.raises(CancelledError):
            _score_detector_for_auto_detect("x", {}, None, "audio", {}, [], None)


class TestFindLabelCancel:
    def test_cancel_mid_run_returns_409_and_applies_no_labels(self, client, monkeypatch):
        """A cancel landing during training aborts before any label is applied."""
        import vtsearch.routes.detectors.scoring as scoring_mod

        detector_id = setup_trainable_model_in_registry(
            "cancel-find-label",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )

        real_resolve = scoring_mod.resolve_or_train_detector

        def cancelling_resolve(*args, **kwargs):
            # Simulate POST /api/find/cancel arriving while training runs.
            find_progress.cancel()
            return real_resolve(*args, **kwargs)

        monkeypatch.setattr(scoring_mod, "resolve_or_train_detector", cancelling_resolve)

        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 409, resp.get_json()
        assert "cancel" in resp.get_json()["message"].lower()
        # The tracker is reset so the frontend's progress bar doesn't hang.
        assert get_find_progress()["status"] == "idle"
        # The apply stage never ran: no votes were written.
        votes = client.get("/api/votes").get_json()
        assert votes["good"] == [] and votes["bad"] == []


class TestMultiFindCancel:
    def test_cancel_before_dataset_loop_returns_409(self, client, monkeypatch):
        """The dataset loop polls the flag; a cancel set after the route's
        reset_cancel() aborts the run with 409 instead of scoring."""
        import vtsearch.routes.detectors.find as find_mod

        detector_id = setup_trainable_model_in_registry(
            "cancel-multi-find",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )

        def cancelling_resolve_datasets(dataset_ids):
            find_progress.cancel()
            # Never reached by the scorer: the loop's poll fires first.
            return [{"name": "phantom", "pkl_path": "/nonexistent.pkl"}]

        monkeypatch.setattr(find_mod, "_resolve_find_datasets", cancelling_resolve_datasets)

        resp = client.post(
            "/api/find",
            json={"dataset_ids": ["anything"], "detector_ids": [detector_id]},
        )
        assert resp.status_code == 409, resp.get_json()
        assert "cancel" in resp.get_json()["message"].lower()
        assert get_find_progress()["status"] == "idle"
