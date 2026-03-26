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
    get_active_dataset_id,
    get_active_detector_id,
    get_context,
    register_context,
    register_detector_context,
    set_active_dataset_id,
    set_active_detector_id,
    medias,
    good_votes,
    bad_votes,
    label_history,
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
        # Set up two datasets
        ctx_a = _make_dataset("req_ds_a", [100, 101])
        ctx_b = _make_dataset("req_ds_b", [200, 201])

        # Activate A globally
        set_active_dataset_id("req_ds_a")

        # Without header: /api/medias returns A's medias
        resp = client.get("/api/medias")
        assert resp.status_code == 200
        ids_default = {m["id"] for m in resp.get_json()["medias"]}
        assert 100 in ids_default

        # With header: /api/medias returns B's medias
        resp = client.get("/api/medias", headers={"X-Dataset-Id": "req_ds_b"})
        assert resp.status_code == 200
        ids_header = {m["id"] for m in resp.get_json()["medias"]}
        assert 200 in ids_header
        assert 100 not in ids_header

        # Global active pointer was NOT mutated
        assert get_active_dataset_id() == "req_ds_a"

    def test_header_with_unloaded_id_falls_back_to_active(self, client):
        """If X-Dataset-Id refers to a dataset not in memory, fall back to global active."""
        ctx_a = _make_dataset("req_fallback", [300])
        set_active_dataset_id("req_fallback")

        resp = client.get("/api/medias", headers={"X-Dataset-Id": "nonexistent"})
        assert resp.status_code == 200
        ids = {m["id"] for m in resp.get_json()["medias"]}
        assert 300 in ids

    def test_no_header_uses_global_active(self, client):
        """Without the header, behaviour is identical to before (global active)."""
        ctx = _make_dataset("req_global", [400])
        set_active_dataset_id("req_global")

        resp = client.get("/api/medias")
        assert resp.status_code == 200
        ids = {m["id"] for m in resp.get_json()["medias"]}
        assert 400 in ids


# ---------------------------------------------------------------------------
# Tests: X-Model-Id header overrides active detector per-request
# ---------------------------------------------------------------------------


class TestRequestScopedModel:
    """X-Model-Id header makes vote proxies resolve to a specific detector."""

    def test_header_overrides_active_detector(self, client):
        """Sending X-Model-Id routes vote reads to that detector."""
        det_a = _make_detector("req_det_a")
        det_b = _make_detector("req_det_b")

        # Vote on A
        det_a.good_votes[1] = None
        # Vote on B
        det_b.good_votes[2] = None

        # Activate A globally
        set_active_detector_id("req_det_a")

        # Without header: GET /api/votes returns A's votes
        resp = client.get("/api/votes")
        assert resp.status_code == 200
        data = resp.get_json()
        good_ids_default = set(data.get("good", []))
        assert 1 in good_ids_default

        # With header: GET /api/votes returns B's votes
        resp = client.get("/api/votes", headers={"X-Model-Id": "req_det_b"})
        assert resp.status_code == 200
        data = resp.get_json()
        good_ids_header = set(data.get("good", []))
        assert 2 in good_ids_header
        assert 1 not in good_ids_header

        # Global active pointer was NOT mutated
        assert get_active_detector_id() == "req_det_a"

    def test_model_header_with_unloaded_id_falls_back(self, client):
        """If X-Model-Id refers to a detector not in memory, fall back to global."""
        det = _make_detector("req_det_fb")
        det.good_votes[5] = None
        set_active_detector_id("req_det_fb")

        resp = client.get("/api/votes", headers={"X-Model-Id": "nonexistent"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert 5 in set(data.get("good", []))


# ---------------------------------------------------------------------------
# Tests: Both headers together
# ---------------------------------------------------------------------------


class TestRequestScopedBoth:
    """Both X-Dataset-Id and X-Model-Id can be sent together."""

    def test_both_headers_resolve_independently(self, client):
        """Dataset and model headers resolve to their respective contexts."""
        ctx_ds = _make_dataset("req_both_ds", [500, 501])
        det = _make_detector("req_both_det")
        det.good_votes[500] = None

        set_active_dataset_id("req_both_ds")
        set_active_detector_id("req_both_det")

        resp = client.get(
            "/api/votes",
            headers={
                "X-Dataset-Id": "req_both_ds",
                "X-Model-Id": "req_both_det",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert 500 in set(data.get("good", []))


# ---------------------------------------------------------------------------
# Tests: Request isolation — concurrent requests don't interfere
# ---------------------------------------------------------------------------


class TestRequestIsolation:
    """Request-scoped context does not leak between requests."""

    def test_sequential_requests_with_different_headers(self, client):
        """Two sequential requests with different headers see different data."""
        _make_dataset("req_iso_a", [600])
        _make_dataset("req_iso_b", [700])
        set_active_dataset_id("req_iso_a")

        # Request targeting B
        resp_b = client.get("/api/medias", headers={"X-Dataset-Id": "req_iso_b"})
        ids_b = {m["id"] for m in resp_b.get_json()["medias"]}
        assert 700 in ids_b

        # Next request without header falls back to A
        resp_a = client.get("/api/medias")
        ids_a = {m["id"] for m in resp_a.get_json()["medias"]}
        assert 600 in ids_a
        assert 700 not in ids_a
