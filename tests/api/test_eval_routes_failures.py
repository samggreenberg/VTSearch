"""Failure-path tests for the eval / labeling-progress routes.

The happy paths for these routes live in ``tests/sorting/test_sorting.py``
(``TestEvalTrainAndScoreAsync``), ``tests/api/test_api_contracts.py``, and
``tests/core/test_votes.py``.  This module covers the error branches those
suites skip: precondition rejects (missing votes / history), internal
computation failures surfacing as 500, schema rejects (422), and the
job-lifecycle branches of the poll/cancel endpoints (missing job → 404,
errored job → 500, cancelled job → ``"cancelled"``).
"""

from __future__ import annotations

import unittest.mock

from vtscore.concurrency.progress import CancelledError
from vtsearch.state import bad_votes, good_votes, label_history


def _seed_votes_and_history():
    """A handful of good/bad votes plus matching label history — enough to
    satisfy the labeling-progress preconditions and drive a real eval job."""
    for cid in (1, 2, 3):
        good_votes[cid] = None
        label_history.append((cid, "good", 0.0))
    for cid in (4, 5, 6):
        bad_votes[cid] = None
        label_history.append((cid, "bad", 0.0))


class TestLabelingProgressFailures:
    """POST /api/labeling-progress precondition and computation failures."""

    def test_no_votes_returns_400(self, client):
        resp = client.post("/api/labeling-progress")
        assert resp.status_code == 400

    def test_votes_but_no_history_returns_400(self, client):
        # Votes present but no label history recorded (the second precondition
        # branch, distinct from the no-votes reject).  Set votes directly so
        # no history is appended by the vote endpoint.
        good_votes[1] = None
        bad_votes[2] = None
        assert not label_history
        resp = client.post("/api/labeling-progress")
        assert resp.status_code == 400
        assert "no label history" in resp.get_json()["message"].lower()

    def test_computation_error_returns_500(self, client):
        _seed_votes_and_history()
        with unittest.mock.patch(
            "vtsearch.routes.eval.analyze_labeling_progress",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.post("/api/labeling-progress")
        assert resp.status_code == 500
        assert "computation failed" in resp.get_json()["message"].lower()


class TestLabelingStatusFailures:
    """GET /api/labeling-status computation failure surfaces as 500."""

    def test_computation_error_returns_500(self, client):
        with unittest.mock.patch(
            "vtsearch.routes.eval.compute_labeling_status",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.get("/api/labeling-status")
        assert resp.status_code == 500
        assert "computation failed" in resp.get_json()["message"].lower()


class TestIndicatorScoreHistoryFailures:
    """GET /api/indicator-score-history schema and computation failures."""

    def test_missing_metric_returns_422(self, client):
        resp = client.get("/api/indicator-score-history")
        assert resp.status_code == 422
        assert "metric" in resp.get_json()["errors"]["query"]

    def test_invalid_metric_returns_422(self, client):
        resp = client.get("/api/indicator-score-history?metric=bogus")
        assert resp.status_code == 422

    def test_computation_error_returns_500(self, client):
        with unittest.mock.patch(
            "vtsearch.routes.eval.calculate_error_cost_over_time",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.get("/api/indicator-score-history?metric=smart")
        assert resp.status_code == 500
        assert "score history" in resp.get_json()["message"].lower()


class TestEvalTrainAndScoreStartFailures:
    """POST /api/eval/train-and-score schema and (wait=true) error branches."""

    def test_invalid_metric_returns_422(self, client):
        resp = client.post("/api/eval/train-and-score", json={"metric": "bogus", "wait": True})
        assert resp.status_code == 422

    def test_wait_true_job_error_returns_500(self, client):
        _seed_votes_and_history()
        with unittest.mock.patch(
            "vtsearch.routes.eval.calculate_prediction_stability_over_time",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.post("/api/eval/train-and-score", json={"metric": "stable", "wait": True})
        assert resp.status_code == 500
        assert resp.get_json()["message"]


class TestEvalTrainAndScoreResultFailures:
    """GET /api/eval/train-and-score/result job-lifecycle branches."""

    def test_missing_job_id_returns_422(self, client):
        resp = client.get("/api/eval/train-and-score/result")
        assert resp.status_code == 422
        assert "job_id" in resp.get_json()["errors"]["query"]

    def test_unknown_job_returns_404(self, client):
        # The missing-job branch is signalled by the HTTP status code; the
        # body is the standard ``{"error", "request_id"}`` Not-Found envelope
        # (the abort's extra kwargs don't surface in it).
        resp = client.get("/api/eval/train-and-score/result?job_id=does-not-exist")
        assert resp.status_code == 404

    def test_errored_job_polls_to_500(self, client):
        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import eval_jobs

        _seed_votes_and_history()
        with unittest.mock.patch(
            "vtsearch.routes.eval.calculate_prediction_stability_over_time",
            side_effect=RuntimeError("boom"),
        ):
            envelope = client.post("/api/eval/train-and-score", json={"metric": "stable"}).get_json()
            job_id = envelope["job_id"]
            _wait_for_job(eval_jobs)
            resp = client.get(f"/api/eval/train-and-score/result?job_id={job_id}")
        assert resp.status_code == 500

    def test_cancelled_job_polls_to_cancelled(self, client):
        """A running job that unwinds via ``CancelledError`` (cooperative user
        cancel) is a terminal *non-error* state: the poll reports
        ``"cancelled"`` with a 200, not a 500."""
        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import eval_jobs

        _seed_votes_and_history()
        with unittest.mock.patch(
            "vtsearch.routes.eval.calculate_prediction_stability_over_time",
            side_effect=CancelledError("cancelled by user"),
        ):
            envelope = client.post("/api/eval/train-and-score", json={"metric": "stable"}).get_json()
            job_id = envelope["job_id"]
            _wait_for_job(eval_jobs)
            resp = client.get(f"/api/eval/train-and-score/result?job_id={job_id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "cancelled"
        assert data["job_id"] == job_id


class TestEvalTrainAndScoreCancelFailures:
    """POST /api/eval/train-and-score/cancel/<job_id> branches."""

    def test_unknown_job_returns_404(self, client):
        resp = client.post("/api/eval/train-and-score/cancel/does-not-exist")
        assert resp.status_code == 404

    def test_cancel_existing_job_returns_ok(self, client):
        """Cancel returns 200/ok for a real job id, even when the job has
        already finished (the flag-set is idempotent and never 404s a job it
        can see)."""
        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import eval_jobs

        _seed_votes_and_history()
        envelope = client.post("/api/eval/train-and-score", json={"metric": "stable"}).get_json()
        job_id = envelope["job_id"]

        resp = client.post(f"/api/eval/train-and-score/cancel/{job_id}")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

        # Drain so the background thread doesn't leak into the next test.
        _wait_for_job(eval_jobs)
