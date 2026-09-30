"""``GET|POST /api/min-precision`` and the precision floor as a setting (#4245).

A pure cutoff knob whose response carries the line it draws in the same round
trip.  Two owner rulings are pinned here: a user who has set no floor gets one
at 50% (2026-09-28), and every detector has a floor - ``null`` is refused, now
that Inclusion is retired as a user preference (#4269).
"""

from __future__ import annotations

import pytest

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import get_active_detector_context, human_voted_ids
from vtscore.training.thresholds import DEFAULT_MIN_PRECISION, FLOOR_STATES, resolve_min_precision
from vtsearch.state import snapshot_medias

_STATES = set(FLOOR_STATES)


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

    def test_it_is_a_state_tier_setting(self):
        from vtsearch.routes.settings import api

        assert "min_precision" in api._STATE_TIER_SETTERS
        assert "min_precision" not in api._CUSTOM_SETTERS

    def test_put_settings_and_the_route_agree(self, client):
        assert client.put("/api/settings", json={"min_precision": 0.75}).get_json()["min_precision"] == 0.75
        assert client.get("/api/min-precision").get_json()["min_precision"] == 0.75
        assert client.put("/api/settings", json={"min_precision": None}).status_code == 422
        assert client.get("/api/min-precision").get_json()["min_precision"] == 0.75


class TestTheRoute:
    def test_no_trained_detector_is_unchecked_with_no_line(self, client):
        data = client.get("/api/min-precision").get_json()
        assert data["status"] == "unchecked" and data["range"] is None
        assert data["n_returned"] is None
        assert data["count"] == 32 and data["schedule"] == {"candidate": 32, "rounds": 3, "picks": 5}

    def test_set_persists_and_null_is_refused(self, client):
        data = client.post("/api/min-precision", json={"min_precision": 0.8}).get_json()
        assert data["min_precision"] == 0.8 and data["status"] in _STATES
        assert client.get("/api/min-precision").get_json()["min_precision"] == 0.8

        assert client.post("/api/min-precision", json={"min_precision": None}).status_code == 422
        assert client.get("/api/min-precision").get_json()["min_precision"] == 0.8

    @pytest.mark.parametrize(("sent", "stored"), [(0, 0.01), (-3, 0.01), (5, 1.0), (0.9, 0.9)])
    def test_out_of_range_is_clamped(self, client, sent, stored):
        assert client.post("/api/min-precision", json={"min_precision": sent}).get_json()["min_precision"] == stored

    @pytest.mark.parametrize("body", [{"min_precision": "half"}, {"min_precision": True}, {"wrong": 0.5}])
    def test_a_malformed_body_is_rejected(self, client, body):
        assert client.post("/api/min-precision", json=body).status_code == 422


class TestTheLine:
    def test_an_unchecked_floor_keeps_the_starting_candidate(self, client):
        """Before any check the line sits at the last of the top K unvoted items (#4272)."""
        _load_trained_detector(client)
        ctx = get_active_detector_context()
        assert ctx.line_ranking is not None

        data = client.post("/api/min-precision", json={"min_precision": 0.5}).get_json()
        assert data["status"] == "unchecked" and data["range"] is None
        # 20 media, 12 voted: the 8 unvoted are the whole candidate.
        assert data["count"] == 8
        assert data["threshold"] == ctx.threshold == ctx.line_ranking.threshold_for(32, human_voted_ids(ctx))
        assert data["n_returned"] == ctx.line_ranking.above(ctx.threshold)


class TestTheClampHasOneOwner:
    """The floor's ``[0.01, 1]`` bound is declared exactly once.

    It lives on ``UserSettings.min_precision`` in :mod:`vtsearch.settings_models`,
    and both write paths reach it through the accessor generated from that
    field (``settings.validate_min_precision``): ``POST /api/min-precision``
    calls it directly, and ``PUT /api/settings`` reaches it via
    ``settings.validate_setting`` because ``min_precision`` is dispatched
    through ``_STATE_TIER_SETTERS``.  A bespoke clamp on either route would be
    a second copy of the range (issue #3416, first found on Inclusion).
    """

    VALUES = [-3, 0, 0.005, 0.01, 0.25, 0.9, 1, 1.5, 100]

    def test_both_write_paths_agree_with_the_pydantic_field(self, client):
        from vtsearch import settings

        for value in self.VALUES:
            expected = settings.validate_min_precision(float(value))

            post = client.post("/api/min-precision", json={"min_precision": value})
            assert post.status_code == 200, f"POST /api/min-precision rejected {value!r}"
            assert post.get_json()["min_precision"] == expected, value

            put = client.put("/api/settings", json={"min_precision": value})
            assert put.status_code == 200, f"PUT /api/settings rejected {value!r}"
            assert put.get_json()["min_precision"] == expected, value
