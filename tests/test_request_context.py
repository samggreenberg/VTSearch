"""Tests for request-scoped dataset/model context resolution.

Phase 1 of the "active" state simplification: the frontend can send
``X-Dataset-Id`` and ``X-Model-Id`` headers to select which loaded
dataset/model a request operates on, without mutating global "active" state.
"""

import numpy as np
import pytest

from vtsearch.utils.state_core import (
    DatasetContext,
    DetectorContext,
    get_active_context,
    get_active_detector_context,
    get_context,
    get_thread_dataset_context,
    get_thread_detector_context,
    register_context,
    register_detector_context,
    set_thread_dataset_context,
    set_thread_detector_context,
    medias,
    good_votes,
    bad_votes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(ds_id, media_ids):
    """Create and register a DatasetContext with fake media items."""
    ctx = DatasetContext(ds_id)
    rng = np.random.default_rng(hash(ds_id) % 2**32)
    for mid in media_ids:
        ctx.medias[mid] = {
            "id": mid,
            "type": "audio",
            "embedding": rng.standard_normal(4).astype(np.float32),
        }
    register_context(ctx)
    return ctx


def _make_detector(det_id):
    """Create and register a DetectorContext."""
    ctx = DetectorContext(det_id)
    register_detector_context(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Tests: X-Dataset-Id header overrides active dataset per-request
# ---------------------------------------------------------------------------


class TestRequestScopedDataset:
    """X-Dataset-Id header makes proxies resolve to a specific dataset."""

    def test_header_overrides_active_dataset(self, client):
        """Sending X-Dataset-Id routes proxy reads to that dataset."""
        _make_dataset("req_ds_a", [100, 101])
        _make_dataset("req_ds_b", [200, 201])
        set_thread_dataset_context(get_context("req_ds_a"))

        from app import app

        # Without header: proxy resolves to thread-local active (A)
        with app.test_request_context():
            app.preprocess_request()
            assert get_active_context().dataset_id == "req_ds_a"
            assert 100 in medias

        # With X-Dataset-Id header: proxy resolves to B
        with app.test_request_context(headers={"X-Dataset-Id": "req_ds_b"}):
            app.preprocess_request()
            ctx = get_active_context()
            assert ctx.dataset_id == "req_ds_b"
            assert 200 in medias
            assert 100 not in medias

        # Thread-local active pointer was NOT mutated
        assert get_thread_dataset_context().dataset_id == "req_ds_a"

    def test_header_with_unloaded_id_falls_back_to_active(self, client):
        """If X-Dataset-Id refers to a dataset not in memory, fall back to global active."""
        _make_dataset("req_fallback", [300])
        set_thread_dataset_context(get_context("req_fallback"))

        from app import app

        with app.test_request_context(headers={"X-Dataset-Id": "nonexistent"}):
            app.preprocess_request()
            # Falls back to global active
            ctx = get_active_context()
            assert ctx.dataset_id == "req_fallback"
            assert 300 in medias

    def test_no_header_uses_global_active(self, client):
        """Without the header, behaviour is identical to before (global active)."""
        _make_dataset("req_global", [400])
        set_thread_dataset_context(get_context("req_global"))

        from app import app

        with app.test_request_context():
            app.preprocess_request()
            ctx = get_active_context()
            assert ctx.dataset_id == "req_global"
            assert 400 in medias


# ---------------------------------------------------------------------------
# Tests: X-Model-Id header overrides active detector per-request
# ---------------------------------------------------------------------------


class TestRequestScopedModel:
    """X-Model-Id header makes vote proxies resolve to a specific detector."""

    def test_header_overrides_active_detector(self, client):
        """Sending X-Model-Id routes vote reads to that detector."""
        det_a = _make_detector("req_det_a")
        det_b = _make_detector("req_det_b")

        det_a.good_votes[1] = None
        det_b.good_votes[2] = None

        from vtsearch.utils.state_core import get_detector_context
        set_thread_detector_context(get_detector_context("req_det_a"))

        # Without header: votes come from A
        resp = client.get("/api/votes")
        assert resp.status_code == 200
        data = resp.get_json()
        assert 1 in set(data.get("good", []))

        # With header: votes come from B
        resp = client.get("/api/votes", headers={"X-Model-Id": "req_det_b"})
        assert resp.status_code == 200
        data = resp.get_json()
        good_ids = set(data.get("good", []))
        assert 2 in good_ids
        assert 1 not in good_ids

        # Thread-local active pointer was NOT mutated
        assert get_thread_detector_context().detector_id == "req_det_a"

    def test_model_header_with_unloaded_id_falls_back(self, client):
        """If X-Model-Id refers to a detector not in memory, fall back to global."""
        det = _make_detector("req_det_fb")
        det.good_votes[5] = None
        from vtsearch.utils.state_core import get_detector_context
        set_thread_detector_context(get_detector_context("req_det_fb"))

        resp = client.get("/api/votes", headers={"X-Model-Id": "nonexistent"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert 5 in set(data.get("good", []))

    def test_header_resolves_correct_detector_in_request(self, client):
        """Verify the proxy objects resolve correctly inside a request context."""
        det_a = _make_detector("req_det_ctx_a")
        det_b = _make_detector("req_det_ctx_b")
        det_a.good_votes[10] = None
        det_b.bad_votes[20] = None
        from vtsearch.utils.state_core import get_detector_context
        set_thread_detector_context(get_detector_context("req_det_ctx_a"))

        from app import app

        with app.test_request_context(headers={"X-Model-Id": "req_det_ctx_b"}):
            app.preprocess_request()
            ctx = get_active_detector_context()
            assert ctx.detector_id == "req_det_ctx_b"
            assert 20 in bad_votes
            assert 10 not in good_votes


# ---------------------------------------------------------------------------
# Tests: Both headers together
# ---------------------------------------------------------------------------


class TestRequestScopedBoth:
    """Both X-Dataset-Id and X-Model-Id can be sent together."""

    def test_both_headers_resolve_independently(self, client):
        """Dataset and model headers resolve to their respective contexts."""
        _make_dataset("req_both_ds", [500, 501])
        det = _make_detector("req_both_det")
        det.good_votes[500] = None

        set_thread_dataset_context(get_context("req_both_ds"))
        from vtsearch.utils.state_core import get_detector_context
        set_thread_detector_context(get_detector_context("req_both_det"))

        from app import app

        with app.test_request_context(
            headers={"X-Dataset-Id": "req_both_ds", "X-Model-Id": "req_both_det"}
        ):
            app.preprocess_request()
            assert get_active_context().dataset_id == "req_both_ds"
            assert get_active_detector_context().detector_id == "req_both_det"
            assert 500 in medias
            assert 500 in good_votes


# ---------------------------------------------------------------------------
# Tests: Request isolation — sequential requests don't interfere
# ---------------------------------------------------------------------------


class TestRequestIsolation:
    """Request-scoped context does not leak between requests."""

    def test_sequential_requests_with_different_headers(self, client):
        """Two sequential requests with different headers see different data."""
        _make_dataset("req_iso_a", [600])
        _make_dataset("req_iso_b", [700])
        set_thread_dataset_context(get_context("req_iso_a"))

        from app import app

        # Request targeting B
        with app.test_request_context(headers={"X-Dataset-Id": "req_iso_b"}):
            app.preprocess_request()
            assert 700 in medias
            assert 600 not in medias

        # Next request without header falls back to A
        with app.test_request_context():
            app.preprocess_request()
            assert 600 in medias
            assert 700 not in medias

    def test_header_does_not_mutate_global_state(self, client):
        """Using X-Dataset-Id never changes the global _active_dataset_id."""
        _make_dataset("req_nomut_a", [800])
        _make_dataset("req_nomut_b", [900])
        set_thread_dataset_context(get_context("req_nomut_a"))

        from app import app

        for _ in range(3):
            with app.test_request_context(headers={"X-Dataset-Id": "req_nomut_b"}):
                app.preprocess_request()
                assert 900 in medias

        # Thread-local pointer untouched
        assert get_thread_dataset_context().dataset_id == "req_nomut_a"
        assert 800 in medias
