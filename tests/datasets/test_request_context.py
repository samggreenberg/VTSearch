"""Tests for request-scoped dataset/model context resolution.

Phase 1 of the "active" state simplification: the frontend can send
``X-Dataset-Id`` and ``X-Detector-Id`` headers to select which loaded
dataset/model a request operates on, without mutating global "active" state.
"""

import numpy as np

from vtsearch.state import bad_votes, good_votes, medias
from vtscore.state.core import (
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
        ctx_a = get_thread_dataset_context()
        assert ctx_a is not None and ctx_a.dataset_id == "req_ds_a"

    def test_header_with_unloaded_id_raises_at_proxy_access(self, client):
        """If X-Dataset-Id names an unloaded dataset, proxy access raises (H16).

        Previously this fell back silently to the thread-local / empty
        context, returning stale data with HTTP 200. Now the resolver
        raises ``DatasetNotLoadedError`` (mapped to 409) the moment a
        handler touches the dataset proxies.
        """
        import pytest

        from vtscore.state.core import DatasetNotLoadedError

        _make_dataset("req_fallback", [300])
        set_thread_dataset_context(get_context("req_fallback"))

        from app import app

        with app.test_request_context(headers={"X-Dataset-Id": "nonexistent"}):
            app.preprocess_request()
            with pytest.raises(DatasetNotLoadedError) as excinfo:
                get_active_context()
            assert excinfo.value.dataset_id == "nonexistent"
            # Proxy access through the module-level name also raises.
            with pytest.raises(DatasetNotLoadedError):
                _ = 300 in medias

    def test_header_with_unloaded_id_returns_409(self, client):
        """End-to-end: a request with an unloaded X-Dataset-Id gets 409."""
        _make_dataset("req_e2e_loaded", [400])

        # A route that touches the medias proxy via snapshot_medias().
        resp = client.get("/api/medias/ids", headers={"X-Dataset-Id": "nope"})
        assert resp.status_code == 409
        body = resp.get_json()
        assert body["code"] == "dataset_not_loaded"
        assert body["dataset_id"] == "nope"

    def test_unloaded_header_does_not_block_routes_that_skip_proxies(self, client):
        """Routes that don't touch the dataset proxies still respond 200.

        The registry-listing endpoint operates on the global context store,
        not on the active ``medias``, so an unloaded ``X-Dataset-Id``
        header must not break it — clients need this endpoint precisely
        to discover which datasets *are* loaded.
        """
        resp = client.get("/api/datasets/registry", headers={"X-Dataset-Id": "nope"})
        assert resp.status_code == 200

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
# Tests: X-Detector-Id header overrides active detector per-request
# ---------------------------------------------------------------------------


class TestRequestScopedModel:
    """X-Detector-Id header makes vote proxies resolve to a specific detector."""

    def test_header_overrides_active_detector(self, client):
        """Sending X-Detector-Id routes vote reads to that detector."""
        det_a = _make_detector("req_det_a")
        det_b = _make_detector("req_det_b")

        det_a.good_votes[1] = None
        det_b.good_votes[2] = None

        from vtscore.state.core import get_detector_context

        set_thread_detector_context(get_detector_context("req_det_a"))

        # Without header: votes come from A
        resp = client.get("/api/votes")
        assert resp.status_code == 200
        data = resp.get_json()
        assert 1 in set(data.get("good", []))

        # With header: votes come from B
        resp = client.get("/api/votes", headers={"X-Detector-Id": "req_det_b"})
        assert resp.status_code == 200
        data = resp.get_json()
        good_ids = set(data.get("good", []))
        assert 2 in good_ids
        assert 1 not in good_ids

        # Thread-local active pointer was NOT mutated
        det_a = get_thread_detector_context()
        assert det_a is not None and det_a.detector_id == "req_det_a"

    def test_model_header_with_unloaded_id_returns_409(self, client):
        """If X-Detector-Id names an unloaded detector, the route returns 409 (H34).

        Previously this fell back to the thread-local detector and the
        client saw votes from a different (or stale) detector under HTTP
        200. Now the resolver raises ``DetectorNotLoadedError`` (mapped to
        409) at proxy access.
        """
        det = _make_detector("req_det_fb")
        det.good_votes[5] = None
        from vtscore.state.core import get_detector_context

        set_thread_detector_context(get_detector_context("req_det_fb"))

        resp = client.get("/api/votes", headers={"X-Detector-Id": "nonexistent"})
        assert resp.status_code == 409
        body = resp.get_json()
        assert body["code"] == "detector_not_loaded"
        assert body["detector_id"] == "nonexistent"

    def test_header_resolves_correct_detector_in_request(self, client):
        """Verify the proxy objects resolve correctly inside a request context."""
        det_a = _make_detector("req_det_ctx_a")
        det_b = _make_detector("req_det_ctx_b")
        det_a.good_votes[10] = None
        det_b.bad_votes[20] = None
        from vtscore.state.core import get_detector_context

        set_thread_detector_context(get_detector_context("req_det_ctx_a"))

        from app import app

        with app.test_request_context(headers={"X-Detector-Id": "req_det_ctx_b"}):
            app.preprocess_request()
            ctx = get_active_detector_context()
            assert ctx.detector_id == "req_det_ctx_b"
            assert 20 in bad_votes
            assert 10 not in good_votes


# ---------------------------------------------------------------------------
# Tests: Both headers together
# ---------------------------------------------------------------------------


class TestRequestScopedBoth:
    """Both X-Dataset-Id and X-Detector-Id can be sent together."""

    def test_both_headers_resolve_independently(self, client):
        """Dataset and model headers resolve to their respective contexts."""
        _make_dataset("req_both_ds", [500, 501])
        det = _make_detector("req_both_det")
        det.good_votes[500] = None

        set_thread_dataset_context(get_context("req_both_ds"))
        from vtscore.state.core import get_detector_context

        set_thread_detector_context(get_detector_context("req_both_det"))

        from app import app

        with app.test_request_context(headers={"X-Dataset-Id": "req_both_ds", "X-Detector-Id": "req_both_det"}):
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
        ctx_nomut = get_thread_dataset_context()
        assert ctx_nomut is not None and ctx_nomut.dataset_id == "req_nomut_a"
        assert 800 in medias


# ---------------------------------------------------------------------------
# Tests: Query-param context resolution (for browser-native requests)
# ---------------------------------------------------------------------------


class TestQueryParamContext:
    """Query params ``dataset_id`` / ``detector_id`` work as fallback for
    browser-native requests (``<img src>``, ``<audio src>``, etc.) that
    bypass Angular's HttpClient interceptor and therefore cannot send
    custom headers.
    """

    def test_dataset_id_query_param(self, client):
        """?dataset_id= resolves the correct dataset context."""
        _make_dataset("qp_ds_a", [1000])
        _make_dataset("qp_ds_b", [2000])
        set_thread_dataset_context(get_context("qp_ds_a"))

        from app import app

        with app.test_request_context("/?dataset_id=qp_ds_b"):
            app.preprocess_request()
            assert get_active_context().dataset_id == "qp_ds_b"
            assert 2000 in medias
            assert 1000 not in medias

    def test_model_id_query_param(self, client):
        """?detector_id= resolves the correct detector context."""
        det_a = _make_detector("qp_det_a")
        det_b = _make_detector("qp_det_b")
        det_a.good_votes[10] = None
        det_b.good_votes[20] = None
        from vtscore.state.core import get_detector_context

        set_thread_detector_context(get_detector_context("qp_det_a"))

        from app import app

        with app.test_request_context("/?detector_id=qp_det_b"):
            app.preprocess_request()
            assert get_active_detector_context().detector_id == "qp_det_b"
            assert 20 in good_votes
            assert 10 not in good_votes

    def test_header_takes_precedence_over_query_param(self, client):
        """X-Dataset-Id header wins over ?dataset_id= query param."""
        _make_dataset("qp_hdr", [3000])
        _make_dataset("qp_qp", [4000])

        from app import app

        with app.test_request_context(
            "/?dataset_id=qp_qp",
            headers={"X-Dataset-Id": "qp_hdr"},
        ):
            app.preprocess_request()
            assert get_active_context().dataset_id == "qp_hdr"
            assert 3000 in medias
