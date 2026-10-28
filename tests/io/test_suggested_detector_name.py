"""Tests for ``POST /api/label-importers/suggested-name/<name>`` (issue #5342).

The route surfaces a label importer's ``default_detector_name`` live, so the
New Detector › Trained form can prefill its Detector name box.  Its reason
for existing is the case no client-side derivation can handle: an importer
that maps an opaque internal selection (an id, a saved-query key) onto a
human-readable name.

Covers:

- The route returns the importer's suggestion for the supplied form values.
- Importers that override nothing fall back to the base derivation.
- Unknown importer → 404; a raising plugin → 502 carrying its message.
"""

from __future__ import annotations

import pytest

from vtscore.labels.importers.base import LabelImporter, PluginField

_QUERY_LABELS = {"q-8f31": "Q1 Field Survey", "q-2b07": "Q2 Coastal Sweep"}


class _LabelResolvingImporter(LabelImporter):
    """The issue's shape: the form holds an id, only the plugin knows its name."""

    name = "_stub_suggested_detector_name"
    display_name = "Stub Saved Query"
    description = "Test label importer that names detectors after a selected query."
    fields = [
        PluginField("query_id", "Query", "select", options=sorted(_QUERY_LABELS)),
    ]

    def __init__(self, fail_with: Exception | None = None) -> None:
        super().__init__()
        self._fail_with = fail_with
        self.seen_values: dict | None = None

    def default_detector_name(self, field_values):
        if self._fail_with is not None:
            raise self._fail_with
        self.seen_values = dict(field_values)
        return _QUERY_LABELS.get(field_values.get("query_id", ""), "") or super().default_detector_name(field_values)

    def run(self, field_values):  # pragma: no cover; unused here
        return []


class _UploadImporter(LabelImporter):
    """Overrides no naming code; relies on the base derivation."""

    name = "_stub_upload_detector_name"
    display_name = "Stub Upload"
    description = "Test label importer with a file field and no naming code."
    fields = [PluginField("file", "Labels file", "file")]

    def run(self, field_values):  # pragma: no cover; unused here
        return []


@pytest.fixture
def register_label_importer():
    """Register label importers in the live registry for the duration of a test."""
    from vtscore.labels.importers import get_label_importer

    registry = get_label_importer.__self__
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
    return client.post(f"/api/label-importers/suggested-name/{importer_name}", json={"values": values})


class TestSuggestedDetectorNameRoute:
    def test_resolves_an_opaque_id_to_a_name(self, client, register_label_importer):
        stub = register_label_importer(_LabelResolvingImporter())
        resp = _post(client, stub.name, {"query_id": "q-8f31"})
        assert resp.status_code == 200
        assert resp.get_json()["detector_name"] == "Q1 Field Survey"
        assert stub.seen_values == {"query_id": "q-8f31"}

    def test_tracks_the_current_selection(self, client, register_label_importer):
        stub = register_label_importer(_LabelResolvingImporter())
        assert _post(client, stub.name, {"query_id": "q-2b07"}).get_json()["detector_name"] == "Q2 Coastal Sweep"

    def test_missing_values_key_is_allowed(self, client, register_label_importer):
        stub = register_label_importer(_LabelResolvingImporter())
        resp = client.post(f"/api/label-importers/suggested-name/{stub.name}", json={})
        assert resp.status_code == 200
        assert resp.get_json()["detector_name"] == "Stub Saved Query detector"

    def test_suggestion_is_whitespace_stripped(self, client, register_label_importer):
        class _Padded(_LabelResolvingImporter):
            name = "_stub_padded_detector_name"

            def default_detector_name(self, field_values):
                return "  Padded  "

        stub = register_label_importer(_Padded())
        assert _post(client, stub.name, {}).get_json()["detector_name"] == "Padded"

    def test_falls_back_to_the_base_derivation(self, client, register_label_importer):
        """An importer that overrides nothing names the detector after its file."""
        stub = register_label_importer(_UploadImporter())
        assert _post(client, stub.name, {"file": "barking dogs.json"}).get_json()["detector_name"] == "barking dogs"
        assert _post(client, stub.name, {}).get_json()["detector_name"] == "Stub Upload detector"

    def test_stock_importer_suggests_its_own_name(self, client):
        resp = _post(client, "server_json_file", {"filepath": "/data/labels/dogs.json"})
        assert resp.status_code == 200
        assert resp.get_json()["detector_name"] == "Server JSON File detector"

    def test_unknown_importer_is_404(self, client):
        assert _post(client, "_no_such_label_importer", {}).status_code == 404

    def test_plugin_error_is_502_with_its_message(self, client, register_label_importer):
        stub = register_label_importer(_LabelResolvingImporter(fail_with=RuntimeError("query service down")))
        resp = _post(client, stub.name, {"query_id": "q-8f31"})
        assert resp.status_code == 502
        assert "query service down" in resp.get_data(as_text=True)
