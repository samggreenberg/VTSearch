"""Tests for the server-tier ``blur_rules`` setting (#5691).

An instance names the ingest metadata fields that mark an image as worth
blurring: an ordered list of ``{"field", "op", "value"}`` rules the operator
writes into the server settings file. It lives on
:class:`~vtsearch.settings.models.ServerSettings`, is read-only over the API,
and is normalized so only well-formed rules reach the frontend; a malformed
one is dropped and named at startup, and the good rules beside it survive.
The rules' semantics are pinned in ``tests_lib/datasets/test_blur_rules.py``.
"""

from __future__ import annotations

import json

import app as app_module
import vtsearch.settings as settings_mod
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema
from vtsearch.settings.models import ServerSettings, UserSettings
from vtsearch.settings import schema as settings_schema
from vtsearch.settings import tiers as settings_tiers

PORN = {"field": "Porn", "op": ">=", "value": 0.8}
GORE = {"field": "Gore", "op": "truthy"}
RATING = {"field": "Rating", "op": "in", "value": ["R", "X"]}
BAD = {"field": "Porn", "op": "~=", "value": 0.8}


def _hand_edit_server_file(blur_rules) -> None:
    """Write *blur_rules* into the server settings file, as an operator would."""
    path = settings_tiers.SETTINGS_PATH
    data = json.loads(path.read_text()) if path.exists() else {}
    data["blur_rules"] = blur_rules
    path.write_text(json.dumps(data) + "\n")
    settings_mod.reset()  # re-read from disk


class TestTier:
    def test_lives_on_the_server_model(self):
        assert "blur_rules" in ServerSettings.model_fields
        assert "blur_rules" not in UserSettings.model_fields
        assert "blur_rules" in settings_schema._SERVER_KEYS

    def test_defaults_to_no_rules(self, isolated_settings):
        assert settings_mod.get_blur_rules() == []
        assert settings_mod.get_all()["blur_rules"] == []

    def test_round_trips_in_order(self, isolated_settings):
        settings_mod.set_blur_rules([RATING, PORN, GORE])
        assert settings_mod.get_blur_rules() == [RATING, PORN, GORE]
        assert settings_mod.get_all()["blur_rules"] == [RATING, PORN, GORE]


class TestNormalization:
    def test_setter_stores_only_well_formed_rules(self, isolated_settings):
        settings_mod.set_blur_rules([PORN, BAD, {"field": " Gore ", "op": "TRUTHY", "value": "ignored"}])
        assert settings_mod.get_blur_rules() == [PORN, GORE]

    def test_non_list_value_reads_as_no_rules(self, isolated_settings):
        _hand_edit_server_file(PORN)
        assert settings_mod.get_blur_rules() == []
        assert settings_mod.get_rejected_blur_rules() == []


class TestHandEditedFile:
    def test_accessor_and_get_all_drop_bad_rules(self, isolated_settings):
        _hand_edit_server_file([PORN, BAD, GORE])
        assert settings_mod.get_blur_rules() == [PORN, GORE]
        assert settings_mod.get_all()["blur_rules"] == [PORN, GORE]

    def test_rejected_rules_are_reported_with_why(self, isolated_settings):
        _hand_edit_server_file([PORN, BAD, "Gore"])
        rejected = settings_mod.get_rejected_blur_rules()
        assert [entry for entry, _ in rejected] == [BAD, "Gore"]
        assert "op must be one of" in rejected[0][1]
        assert "must be an object" in rejected[1][1]

    def test_get_settings_serves_only_well_formed_rules(self, client, isolated_settings):
        _hand_edit_server_file([BAD, PORN, RATING])
        assert client.get("/api/settings").get_json()["blur_rules"] == [PORN, RATING]


class TestApi:
    def test_put_cannot_change_it(self, client, isolated_settings):
        """A PUT body carrying blur_rules is silently ignored: the rules are the operator's."""
        settings_mod.set_blur_rules([PORN])
        resp = client.put("/api/settings", json={"blur_rules": []})
        assert resp.status_code == 200
        assert settings_mod.get_blur_rules() == [PORN]
        assert resp.get_json()["blur_rules"] == [PORN]

    def test_schema_marks_it_read_only(self):
        assert AppSettingsSchema().fields["blur_rules"].dump_only
        assert "blur_rules" not in SettingsUpdateSchema().fields


class TestStartupReport:
    def test_silent_by_default(self, isolated_settings, capsys):
        app_module._report_blur_rules()
        assert capsys.readouterr().out == ""

    def test_names_the_rules_and_each_dropped_one_with_why(self, isolated_settings, capsys):
        _hand_edit_server_file([PORN, BAD, GORE, RATING])
        app_module._report_blur_rules()
        out = capsys.readouterr().out
        assert "Blur rules: Porn >= 0.8; Gore is truthy; Rating in R, X" in out
        assert "Ignoring blur_rules entry" in out
        assert "'~='" in out
        assert "op must be one of truthy, ==, !=, >, >=, <, <=, in" in out
