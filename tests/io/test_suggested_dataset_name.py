"""Tests for ``POST /api/dataset/import/<name>/suggested-name``.

The route surfaces an importer's ``default_display_name`` live, so the
Add-Dataset form can prefill its Dataset Name box with the same name the
import would fall back to.  Its reason for existing is the case no
client-side derivation can handle: an importer that maps an opaque
internal selection (an id, a saved-query key) onto a human-readable label.

Covers:

- The route returns the importer's suggestion for the supplied form values.
- A plugin that resolves an id to a label is reflected verbatim.
- The user's own ``dataset_name`` is excluded from what the plugin sees.
- Unknown importer → 404; a raising plugin → 502 carrying its message.
- Importers that override nothing fall back to the generic derivation.
"""

from __future__ import annotations

import pytest

from vtscore.datasets.importers.base import DatasetImporter
from vtscore.plugins import PluginField

_QUERY_LABELS = {"q-8f31": "Q1 Field Survey", "q-2b07": "Q2 Coastal Sweep"}


class _LabelResolvingImporter(DatasetImporter):
    """The issue's shape: the form holds an id, only the plugin knows its name."""

    name = "_stub_suggested_name"
    display_name = "Stub Suggested Name"
    description = "Test importer that names datasets after a selected query."
    fields = [
        PluginField("query_id", "Query", "select", options=sorted(_QUERY_LABELS)),
    ]

    def __init__(self, fail_with: Exception | None = None) -> None:
        super().__init__()
        self._fail_with = fail_with
        self.seen_values: dict | None = None

    def default_display_name(self, field_values):
        if self._fail_with is not None:
            raise self._fail_with
        self.seen_values = dict(field_values)
        return _QUERY_LABELS.get(field_values.get("query_id", ""), self.display_name)

    def run(self, field_values, medias, thin=False):  # pragma: no cover; unused here
        return None


class _DerivingImporter(DatasetImporter):
    """Overrides no naming code; relies on the base derivation."""

    name = "_stub_derived_name"
    display_name = "Stub Derived Name"
    description = "Test importer with a URL field and no naming code."
    fields = [PluginField("url", "URL", "url")]

    def run(self, field_values, medias, thin=False):  # pragma: no cover; unused here
        return None


@pytest.fixture
def register_importer():
    """Register importers in the live registry for the duration of a test."""
    from vtscore.datasets.importers import get_importer

    registry = get_importer.__self__
    registry._ensure_discovered()
    registered: list[str] = []

    def _register(importer):
        registry._items[importer.name] = importer
        registered.append(importer.name)
        return importer

    try:
        yield _register
    finally:
        for name in registered:
            registry._items.pop(name, None)


def _post(client, importer_name, values):
    return client.post(f"/api/dataset/import/{importer_name}/suggested-name", json={"values": values})


class TestSuggestedNameRoute:
    def test_resolves_an_opaque_id_to_a_label(self, client, register_importer):
        stub = register_importer(_LabelResolvingImporter())
        resp = _post(client, stub.name, {"query_id": "q-8f31"})
        assert resp.status_code == 200
        assert resp.get_json()["dataset_name"] == "Q1 Field Survey"

    def test_tracks_the_current_selection(self, client, register_importer):
        stub = register_importer(_LabelResolvingImporter())
        assert _post(client, stub.name, {"query_id": "q-2b07"}).get_json()["dataset_name"] == "Q2 Coastal Sweep"

    def test_unrecognised_selection_falls_back_to_the_importer_name(self, client, register_importer):
        stub = register_importer(_LabelResolvingImporter())
        assert _post(client, stub.name, {"query_id": "nope"}).get_json()["dataset_name"] == stub.display_name

    def test_user_typed_name_is_hidden_from_the_plugin(self, client, register_importer):
        """The plugin is asked what it *would* pick, not handed the answer."""
        stub = register_importer(_LabelResolvingImporter())
        resp = _post(client, stub.name, {"query_id": "q-8f31", "dataset_name": "My Corpus"})
        assert resp.get_json()["dataset_name"] == "Q1 Field Survey"
        assert stub.seen_values is not None
        assert "dataset_name" not in stub.seen_values

    def test_missing_values_key_is_allowed(self, client, register_importer):
        stub = register_importer(_LabelResolvingImporter())
        resp = client.post(f"/api/dataset/import/{stub.name}/suggested-name", json={})
        assert resp.status_code == 200
        assert resp.get_json()["dataset_name"] == stub.display_name

    def test_suggestion_is_whitespace_stripped(self, client, register_importer):
        class _Padded(_LabelResolvingImporter):
            name = "_stub_padded_name"

            def default_display_name(self, field_values):
                return "  Padded  "

        stub = register_importer(_Padded())
        assert _post(client, stub.name, {}).get_json()["dataset_name"] == "Padded"

    def test_unknown_importer_is_404(self, client):
        assert _post(client, "_no_such_importer", {}).status_code == 404

    def test_plugin_error_is_502_with_its_message(self, client, register_importer):
        stub = register_importer(_LabelResolvingImporter(fail_with=RuntimeError("query service down")))
        resp = _post(client, stub.name, {"query_id": "q-8f31"})
        assert resp.status_code == 502
        assert "query service down" in resp.get_data(as_text=True)

    def test_falls_back_to_the_generic_derivation(self, client, register_importer):
        """An importer that overrides nothing still suggests something useful."""
        stub = register_importer(_DerivingImporter())
        resp = _post(client, stub.name, {"url": "https://example.org/sets/genres.tar.gz"})
        assert resp.get_json()["dataset_name"] == "genres"

    def test_suggestion_matches_the_name_the_import_would_use(self, client, register_importer):
        """The whole point: what the box shows is what a blank box would produce."""
        stub = register_importer(_LabelResolvingImporter())
        values = {"query_id": "q-8f31"}
        suggested = _post(client, stub.name, values).get_json()["dataset_name"]
        assert suggested == stub.resolve_display_name({**values, "dataset_name": ""})


class TestBuiltinImporterSuggestions:
    """Spot-check that the route works against real registered importers."""

    def test_server_folder_suggests_the_leaf_directory(self, client):
        resp = _post(client, "server_folder", {"path": "/data/sounds/sirens"})
        assert resp.status_code == 200
        assert resp.get_json()["dataset_name"] == "sirens"

    def test_http_archive_strips_the_archive_extension(self, client):
        resp = _post(client, "http_archive", {"url": "https://example.org/data/genres.tar.gz"})
        assert resp.status_code == 200
        assert resp.get_json()["dataset_name"] == "genres"
