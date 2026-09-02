class TestInclusionEndpoints:
    def test_get_default_inclusion(self, client):
        resp = client.get("/api/inclusion")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "inclusion" in data
        assert isinstance(data["inclusion"], int)

    def test_set_inclusion_valid_value(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": 5})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] == 5

        # Verify it persists
        resp = client.get("/api/inclusion")
        data = resp.get_json()
        assert data["inclusion"] == 5

    def test_set_inclusion_negative_value(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": -5})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] == -5

    def test_set_inclusion_clamped_to_max(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": 100})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] == 10  # Clamped to max

    def test_set_inclusion_clamped_to_min(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": -100})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] == -10  # Clamped to min

    def test_set_inclusion_float_converted_to_int(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": 3.7})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] == 3  # Converted to int

    def test_set_inclusion_invalid_type(self, client):
        # Marshmallow-validated route: non-numeric values surface as 422
        # with the standard ``errors`` envelope.
        resp = client.post("/api/inclusion", json={"inclusion": "not a number"})
        assert resp.status_code == 422

    def test_set_inclusion_missing_field(self, client):
        # Marshmallow-validated route: missing required ``inclusion`` →
        # 422 with the standard ``errors`` envelope.
        resp = client.post("/api/inclusion", json={"wrong": 5})
        assert resp.status_code == 422


class TestInclusionClampHasOneOwner:
    """``inclusion``'s ``[-10, 10]`` bound is declared exactly once.

    It lives on ``UserSettings.inclusion`` in :mod:`vtsearch.settings_models`,
    and both write paths reach it through the accessor generated from that
    field (``settings.validate_inclusion``): ``POST /api/inclusion`` calls it
    directly, and ``PUT /api/settings`` reaches it via
    ``settings.validate_setting`` because ``inclusion`` is dispatched through
    ``_STATE_TIER_SETTERS`` like its ``calibrate_count`` /
    ``calibration_fraction`` siblings.

    Before issue #3416 the range was spelled out three times -- once on the
    pydantic field and once as a hand-rolled ``max/min`` in each route -- so
    widening it meant three edits and any missed one silently clamped harder
    than the others. These tests fail if a bespoke clamp is reintroduced on
    either route.
    """

    # Straddles both bounds, and includes the fractional value the
    # ``fields.Raw`` schema on ``POST /api/inclusion`` admits.
    VALUES = [-100, -11, -10, -3, 0, 3, 3.7, 10, 11, 100]

    def test_both_write_paths_agree_with_the_pydantic_field(self, client):
        from vtsearch import settings

        for value in self.VALUES:
            expected = settings.validate_inclusion(int(value))

            post = client.post("/api/inclusion", json={"inclusion": value})
            assert post.status_code == 200, f"POST /api/inclusion rejected {value!r}"
            assert post.get_json()["inclusion"] == expected, (
                f"POST /api/inclusion clamped {value!r} to "
                f"{post.get_json()['inclusion']!r}, not {expected!r}"
            )

            put = client.put("/api/settings", json={"inclusion": value})
            assert put.status_code == 200, f"PUT /api/settings rejected {value!r}"
            assert put.get_json()["inclusion"] == expected, (
                f"PUT /api/settings clamped {value!r} to "
                f"{put.get_json()['inclusion']!r}, not {expected!r}"
            )

    def test_inclusion_is_dispatched_like_its_training_siblings(self):
        """``inclusion`` routes through the state tier, not a custom setter.

        A ``_CUSTOM_SETTERS`` entry would bypass ``validate_setting`` (see
        ``_plan_one_key``) and so would need its own validator -- which is
        exactly the second copy of the range this issue removed.
        """
        from vtsearch.routes.settings import api

        assert "inclusion" in api._STATE_TIER_SETTERS
        assert "inclusion" not in api._CUSTOM_SETTERS
