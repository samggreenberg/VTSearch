"""Tests for the usage-metrics sink, ``POST /api/metrics`` (issue #5475).

Covers the JSON-lines file the route appends to, the flattening of event
attributes, the request validation, and the server-tier ``enable_metrics``
switch: read-only at ``GET /api/settings``, ignored by ``PUT``, and when off
the route records nothing and says so.
"""

from __future__ import annotations

import json
import math

import pytest

from vtsearch import metrics
from vtsearch import settings as settings_mod
from vtsearch.schemas.metrics import MAX_EVENTS_PER_BATCH


@pytest.fixture
def metrics_dir(tmp_path, monkeypatch):
    path = tmp_path / "metrics"
    monkeypatch.setattr(metrics, "METRICS_DIR", path)
    return path


def _lines(metrics_dir):
    files = sorted(metrics_dir.glob("events-*.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text(encoding="utf-8").splitlines()]


class TestRecording:
    def test_appends_one_line_per_event(self, client, isolated_settings, metrics_dir):
        resp = client.post(
            "/api/metrics",
            json={
                "session": "abc123",
                "events": [
                    {"group": "view", "attributes": {"view": "browse"}, "client_ts": 1793385084301},
                    {"group": "resultsExported", "attributes": {"exporter": "csv"}},
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.get_json() == {"recorded": 2, "enabled": True}

        files = list(metrics_dir.glob("events-*.jsonl"))
        assert len(files) == 1
        assert files[0].name.startswith("events-") and len(files[0].stem) == len("events-YYYY-MM")
        first, second = _lines(metrics_dir)
        assert first["user"] == "default"
        assert first["session"] == "abc123"
        assert first["group"] == "view"
        assert first["attributes"] == {"view": "browse"}
        assert first["client_ts"] == 1793385084301
        assert first["ts"].endswith("Z")
        assert second["group"] == "resultsExported"
        assert "client_ts" not in second

    def test_batches_append_to_the_same_file(self, client, isolated_settings, metrics_dir):
        for view in ("dashboard", "label"):
            client.post("/api/metrics", json={"events": [{"group": "view", "attributes": {"view": view}}]})
        assert [line["attributes"]["view"] for line in _lines(metrics_dir)] == ["dashboard", "label"]

    def test_attributes_default_to_empty(self, client, isolated_settings, metrics_dir):
        client.post("/api/metrics", json={"events": [{"group": "click"}]})
        (line,) = _lines(metrics_dir)
        assert line["attributes"] == {}
        assert line["session"] is None

    def test_empty_batch_writes_nothing(self, client, isolated_settings, metrics_dir):
        resp = client.post("/api/metrics", json={"events": []})
        assert resp.get_json() == {"recorded": 0, "enabled": True}
        assert not metrics_dir.exists()

    def test_write_failure_is_dropped_not_raised(self, client, isolated_settings, tmp_path, monkeypatch):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        monkeypatch.setattr(metrics, "METRICS_DIR", blocker / "metrics")
        resp = client.post("/api/metrics", json={"events": [{"group": "view"}]})
        assert resp.status_code == 200
        assert resp.get_json() == {"recorded": 0, "enabled": True}


class TestAttributeCleaning:
    def test_scalars_survive(self):
        raw = {"s": "x", "i": 3, "f": 1.5, "b": False, "n": None}
        assert metrics.clean_attributes(raw) == raw

    def test_nested_and_non_finite_values_are_dropped(self):
        raw = {"keep": 1, "list": [1, 2], "dict": {"a": 1}, "nan": math.nan, "inf": math.inf}
        assert metrics.clean_attributes(raw) == {"keep": 1}

    def test_long_strings_and_keys_are_cut(self):
        cleaned = metrics.clean_attributes({"k" * 100: "v" * 1000})
        ((key, value),) = cleaned.items()
        assert len(key) == metrics.MAX_KEY_LENGTH
        assert len(value) == metrics.MAX_STRING_LENGTH

    def test_attribute_count_is_capped(self):
        raw = {f"k{i}": i for i in range(metrics.MAX_ATTRIBUTES + 5)}
        assert len(metrics.clean_attributes(raw)) == metrics.MAX_ATTRIBUTES

    def test_route_flattens_attributes(self, client, isolated_settings, metrics_dir):
        client.post(
            "/api/metrics",
            json={"events": [{"group": "settingChanged", "attributes": {"key": "theme", "value": {"x": 1}}}]},
        )
        (line,) = _lines(metrics_dir)
        assert line["attributes"] == {"key": "theme"}


class TestValidation:
    @pytest.mark.parametrize("group", ["", "1view", "has space", "x" * 65])
    def test_bad_group_is_rejected(self, client, isolated_settings, metrics_dir, group):
        resp = client.post("/api/metrics", json={"events": [{"group": group}]})
        assert resp.status_code == 422
        assert not metrics_dir.exists()

    def test_events_are_required(self, client, isolated_settings, metrics_dir):
        assert client.post("/api/metrics", json={}).status_code == 422

    def test_oversized_batch_is_rejected(self, client, isolated_settings, metrics_dir):
        events = [{"group": "click"}] * (MAX_EVENTS_PER_BATCH + 1)
        assert client.post("/api/metrics", json={"events": events}).status_code == 422
        assert not metrics_dir.exists()


class TestEnableMetricsSetting:
    def test_default_is_on(self, isolated_settings):
        assert settings_mod.get_enable_metrics() is True
        assert metrics.is_enabled() is True

    def test_disabled_records_nothing(self, client, isolated_settings, metrics_dir):
        settings_mod.set_enable_metrics(False)
        resp = client.post("/api/metrics", json={"events": [{"group": "view"}]})
        assert resp.status_code == 200
        assert resp.get_json() == {"recorded": 0, "enabled": False}
        assert not metrics_dir.exists()

    def test_record_events_honours_the_switch(self, isolated_settings, metrics_dir):
        settings_mod.set_enable_metrics(False)
        assert metrics.record_events([{"group": "view"}], user="alice") == 0
        assert not metrics_dir.exists()

    def test_get_reports_it(self, client, isolated_settings):
        assert client.get("/api/settings").get_json()["enable_metrics"] is True
        settings_mod.set_enable_metrics(False)
        assert client.get("/api/settings").get_json()["enable_metrics"] is False

    def test_not_settable_via_put(self, client, isolated_settings):
        resp = client.put("/api/settings", json={"enable_metrics": False})
        assert resp.status_code == 200
        assert settings_mod.get_enable_metrics() is True
