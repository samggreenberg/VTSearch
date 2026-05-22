"""Tests for the dataset-importer picker-tab registry."""

from __future__ import annotations

import pytest

from vtscore.datasets.importers.tabs import (
    _PICKER_TABS,
    list_picker_tabs,
    register_picker_tab,
)


@pytest.fixture
def restore_tabs():
    """Snapshot/restore the picker-tab registry around tests that mutate it."""
    snapshot = [dict(t) for t in _PICKER_TABS]
    yield
    _PICKER_TABS.clear()
    _PICKER_TABS.extend(snapshot)


class TestBuiltinTabs:
    def test_built_in_tabs_present(self):
        ids = {t["id"] for t in list_picker_tabs()}
        assert {"services", "server", "local", "demo"} <= ids

    def test_built_in_tabs_have_labels_and_icons(self):
        for tab in list_picker_tabs():
            assert tab.get("label")
            assert tab.get("icon")

    def test_tabs_are_sorted_by_order(self):
        tabs = list_picker_tabs()
        orders = [t.get("order", 100) for t in tabs]
        assert orders == sorted(orders)


@pytest.mark.usefixtures("restore_tabs")
class TestRegisterPickerTab:
    def test_register_appends_new_tab(self):
        register_picker_tab({"id": "cloud", "label": "Cloud", "icon": "cloud", "order": 25})
        ids = [t["id"] for t in list_picker_tabs()]
        assert "cloud" in ids

    def test_register_replaces_existing_tab(self):
        register_picker_tab({"id": "server", "label": "Servers", "icon": "server", "order": 5})
        tabs = list_picker_tabs()
        server = next(t for t in tabs if t["id"] == "server")
        assert server["label"] == "Servers"
        assert server["order"] == 5

    def test_register_requires_id(self):
        with pytest.raises(ValueError):
            register_picker_tab({"label": "Bad"})

    def test_register_requires_label(self):
        with pytest.raises(ValueError):
            register_picker_tab({"id": "bad"})


class TestApiResponse:
    def test_all_importers_includes_tabs(self, client):
        resp = client.get("/api/dataset/all-importers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "tabs" in data
        assert isinstance(data["tabs"], list)
        ids = {t["id"] for t in data["tabs"]}
        assert {"services", "server", "local", "demo"} <= ids

    def test_tab_entries_have_id_and_label(self, client):
        resp = client.get("/api/dataset/all-importers")
        for tab in resp.get_json()["tabs"]:
            assert tab.get("id")
            assert tab.get("label")
