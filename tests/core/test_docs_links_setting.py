"""Tests for the server-tier ``docs_links`` setting (#4310).

A deployment that adds its own plugins or extensions lists the docs for them
in the Help modal, beside the built-in user guide: an ordered list of
``{"label", "url"}`` objects the operator writes into the server settings
file. It lives on :class:`~vtsearch.settings_models.ServerSettings`, is
read-only over the API, and is normalized so only usable links (a label, and
an ``http(s)`` URL or a root-relative path) reach the frontend.
"""

from __future__ import annotations

import json

import app as app_module
import vtsearch.settings as settings_mod
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema
from vtsearch.settings_models import ServerSettings, UserSettings, partition_docs_links

ACME = {"label": "Acme plugin guide", "url": "https://acme.example/docs"}
WIKI = {"label": "Lab wiki", "url": "/wiki/vtsearch"}


def _hand_edit_server_file(docs_links) -> None:
    """Write *docs_links* into the server settings file, as an operator would."""
    path = settings_mod.SETTINGS_PATH
    data = json.loads(path.read_text()) if path.exists() else {}
    data["docs_links"] = docs_links
    path.write_text(json.dumps(data) + "\n")
    settings_mod.reset()  # re-read from disk


class TestTier:
    def test_lives_on_the_server_model(self):
        assert "docs_links" in ServerSettings.model_fields
        assert "docs_links" not in UserSettings.model_fields
        assert "docs_links" in settings_mod._SERVER_KEYS

    def test_defaults_to_no_links(self, isolated_settings):
        assert settings_mod.get_docs_links() == []
        assert settings_mod.get_all()["docs_links"] == []

    def test_round_trips_in_order(self, isolated_settings):
        settings_mod.set_docs_links([WIKI, ACME])
        assert settings_mod.get_docs_links() == [WIKI, ACME]
        assert settings_mod.get_all()["docs_links"] == [WIKI, ACME]


class TestNormalization:
    def test_trims_and_drops_extra_keys(self):
        kept, rejected = partition_docs_links(
            [{"label": "  Acme plugin guide ", "url": " https://acme.example/docs ", "x": 1}]
        )
        assert kept == [ACME]
        assert rejected == []

    def test_accepts_http_https_and_root_relative(self):
        entries = [
            {"label": "a", "url": "https://a.example/"},
            {"label": "b", "url": "http://b.example/x?y=1,2#z"},
            {"label": "c", "url": "HTTPS://C.example"},
            {"label": "d", "url": "/docs/plugin.html"},
        ]
        kept, rejected = partition_docs_links(entries)
        assert [link["label"] for link in kept] == ["a", "b", "c", "d"]
        assert rejected == []

    def test_rejects_unusable_entries(self):
        bad = [
            {"label": "", "url": "https://a.example"},
            {"label": "   ", "url": "https://a.example"},
            {"label": "No url"},
            {"url": "https://a.example"},
            {"label": "Script", "url": "javascript:alert(1)"},
            {"label": "Data", "url": "data:text/html,<b>x</b>"},
            {"label": "Protocol-relative", "url": "//evil.example/x"},
            {"label": "Bare relative", "url": "docs/x.html"},
            {"label": "No host", "url": "https:///x"},
            {"label": "Ftp", "url": "ftp://files.example/x"},
            {"label": 3, "url": "https://a.example"},
            "https://a.example",
            ["Acme", "https://a.example"],
            None,
        ]
        kept, rejected = partition_docs_links([ACME, *bad, WIKI])
        assert kept == [ACME, WIKI]
        assert rejected == bad

    def test_setter_stores_only_usable_links(self, isolated_settings):
        settings_mod.set_docs_links([ACME, {"label": "Script", "url": "javascript:alert(1)"}, WIKI])
        assert settings_mod.get_docs_links() == [ACME, WIKI]

    def test_non_list_value_reads_as_no_links(self, isolated_settings):
        _hand_edit_server_file({"label": "Acme", "url": "https://acme.example"})
        assert settings_mod.get_docs_links() == []
        assert settings_mod.get_rejected_docs_links() == []


class TestHandEditedFile:
    BAD = {"label": "Script", "url": "javascript:alert(1)"}

    def test_accessor_and_get_all_drop_bad_entries(self, isolated_settings):
        _hand_edit_server_file([ACME, self.BAD, {"label": " Lab wiki ", "url": "/wiki/vtsearch"}])
        assert settings_mod.get_docs_links() == [ACME, WIKI]
        assert settings_mod.get_all()["docs_links"] == [ACME, WIKI]

    def test_rejected_entries_are_reported(self, isolated_settings):
        _hand_edit_server_file([ACME, self.BAD])
        assert settings_mod.get_rejected_docs_links() == [self.BAD]

    def test_get_settings_serves_only_usable_links(self, client, isolated_settings):
        _hand_edit_server_file([self.BAD, ACME])
        assert client.get("/api/settings").get_json()["docs_links"] == [ACME]


class TestApi:
    def test_put_cannot_change_it(self, client, isolated_settings):
        """A PUT body carrying docs_links is silently ignored: the list is the operator's."""
        settings_mod.set_docs_links([ACME])
        resp = client.put("/api/settings", json={"docs_links": [{"label": "Phish", "url": "https://evil.example"}]})
        assert resp.status_code == 200
        assert settings_mod.get_docs_links() == [ACME]
        assert resp.get_json()["docs_links"] == [ACME]

    def test_schema_marks_it_read_only(self):
        assert AppSettingsSchema().fields["docs_links"].dump_only
        assert "docs_links" not in SettingsUpdateSchema().fields


class TestStartupReport:
    def test_silent_by_default(self, isolated_settings, capsys):
        app_module._report_docs_links()
        assert capsys.readouterr().out == ""

    def test_names_the_links_and_each_dropped_entry(self, isolated_settings, capsys):
        _hand_edit_server_file([ACME, {"label": "Script", "url": "javascript:alert(1)"}, WIKI])
        app_module._report_docs_links()
        out = capsys.readouterr().out
        assert "Help-modal docs: Acme plugin guide, Lab wiki" in out
        assert "Ignoring docs_links entry" in out
        assert "javascript:alert(1)" in out
