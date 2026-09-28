"""``GET|POST /api/min-precision`` and the precision floor as a setting (#4245).

The counterpart of ``/api/inclusion``: a pure cutoff knob whose response carries
the line it draws in the same round trip.  Two owner rulings are pinned here
(2026-09-28): a user who has set no floor gets one at 50%, and a set floor wins
over Inclusion - clearing it (``null``) hands the line back to the knob.
"""

from __future__ import annotations

import pytest

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import get_active_detector_context
from vtscore.training.thresholds import DEFAULT_MIN_PRECISION, resolve_min_precision
from vtsearch.state import snapshot_medias

_STATES = {"promised", "unreachable", "insufficient_evidence"}


def _load_trained_detector(client) -> None:
    detector_id = setup_trainable_model_in_registry(
        "floor-route", good_ids=[1, 2, 3, 4, 5, 6], bad_ids=[7, 8, 9, 10, 11, 12], snap=snapshot_medias()
    )
    load_detector_and_wait(client, detector_id)
    client.post("/api/find-label", json={"detector_id": detector_id})


class TestTheSetting:
    def test_an_unset_floor_is_the_owners_default(self, client):
        data = client.get("/api/min-precision").get_json()
        assert data["min_precision"] == DEFAULT_MIN_PRECISION == 0.5

    def test_the_settings_default_is_the_one_the_eval_default_arm_resolves(self):
        """What ``thresholds.min_precision_default`` in the eval/app sync gate leans on."""
        from vtsearch.settings_models import UserSettings

        assert UserSettings().min_precision == resolve_min_precision(None)

    def test_it_is_a_state_tier_setting_like_inclusion(self):
        from vtsearch.routes.settings import api

        assert "min_precision" in api._STATE_TIER_SETTERS
        assert "min_precision" not in api._CUSTOM_SETTERS

    def test_put_settings_and_the_route_agree(self, client):
        assert client.put("/api/settings", json={"min_precision": 0.75}).get_json()["min_precision"] == 0.75
        assert client.get("/api/min-precision").get_json()["min_precision"] == 0.75
        assert client.put("/api/settings", json={"min_precision": None}).get_json()["min_precision"] is None
        assert client.get("/api/min-precision").get_json()["min_precision"] is None


class TestTheRoute:
    def test_no_trained_detector_no_evidence(self, client):
        data = client.get("/api/min-precision").get_json()
        assert data["status"] == "insufficient_evidence"
        # ``threshold`` is whatever ``/api/inclusion`` reports for the context.
        assert data["threshold"] == client.get("/api/inclusion").get_json()["threshold"]
        assert data["n_returned"] is None
        assert data["calibration_positives"] == 0

    def test_set_persists_and_clear_reports_no_status(self, client):
        data = client.post("/api/min-precision", json={"min_precision": 0.8}).get_json()
        assert data["min_precision"] == 0.8 and data["status"] in _STATES
        assert client.get("/api/min-precision").get_json()["min_precision"] == 0.8

        cleared = client.post("/api/min-precision", json={"min_precision": None}).get_json()
        assert cleared["min_precision"] is None and cleared["status"] is None

    @pytest.mark.parametrize(("sent", "stored"), [(0, 0.01), (-3, 0.01), (5, 1.0), (0.9, 0.9)])
    def test_out_of_range_is_clamped(self, client, sent, stored):
        assert client.post("/api/min-precision", json={"min_precision": sent}).get_json()["min_precision"] == stored

    @pytest.mark.parametrize("body", [{"min_precision": "half"}, {"min_precision": True}, {"wrong": 0.5}])
    def test_a_malformed_body_is_rejected(self, client, body):
        assert client.post("/api/min-precision", json=body).status_code == 422


class TestTheLine:
    def test_an_unmet_floor_draws_the_inclusion_zero_line(self, client):
        """The test labels carry no learned-sort provenance, so no promise can be made."""
        _load_trained_detector(client)
        ctx = get_active_detector_context()
        assert ctx.anchored_cut_cache is not None

        data = client.post("/api/min-precision", json={"min_precision": 0.5}).get_json()
        assert data["status"] == "insufficient_evidence"
        assert data["calibration_positives"] == 0
        assert data["threshold"] == ctx.threshold == ctx.anchored_cut_cache.threshold_at(0)
        assert data["n_returned"] == ctx.precision_floor_cache.count_at(ctx.threshold)

    def test_a_set_floor_wins_over_inclusion_and_null_hands_it_back(self, client):
        _load_trained_detector(client)
        ctx = get_active_detector_context()
        client.post("/api/min-precision", json={"min_precision": 0.5})
        floored = ctx.threshold

        moved = client.post("/api/inclusion", json={"inclusion": -10}).get_json()
        assert moved["threshold"] == floored, "a set floor draws the line whatever Inclusion says"

        released = client.post("/api/min-precision", json={"min_precision": None}).get_json()
        assert released["threshold"] == ctx.anchored_cut_cache.threshold_at(-10)
        assert released["threshold"] != floored
