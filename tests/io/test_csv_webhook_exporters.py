"""Tests for CSV and Webhook exporters."""

import argparse
import csv
import json
from unittest import mock

import pytest

import app as app_module  # noqa: F401
from vtscore.cli import _run_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample_results():
    """Create a sample results dict for testing exporters directly."""
    return {
        "media_type": "audio",
        "detectors_run": 1,
        "results": {
            "test_detector": {
                "detector_name": "test_detector",
                "threshold": 0.5,
                "total_hits": 3,
                "hits": [
                    {"id": 1, "filename": "clip_1.wav", "category": "birds", "score": 0.95},
                    {"id": 2, "filename": "clip_2.wav", "category": "rain", "score": 0.82},
                    {"id": 3, "filename": "clip_3.wav", "category": "birds", "score": 0.71},
                ],
            }
        },
    }


# ---------------------------------------------------------------------------
# CSV Exporter: metadata
# ---------------------------------------------------------------------------


class TestCsvExporterMetadata:
    """CSV exporter metadata and registration."""

    def test_csv_exporter_registered(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_csv_file")
        assert exp is not None

    def test_csv_exporter_display_name(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_csv_file")
        assert exp.display_name == "Server CSV File"

    def test_csv_exporter_icon(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_csv_file")
        assert exp.icon == "\U0001f5a5"

    def test_csv_exporter_has_filepath_field(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_csv_file")
        keys = [f.key for f in exp.fields]
        assert "filepath" in keys

    def test_csv_exporter_to_dict(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_csv_file")
        d = exp.to_dict()
        assert d["name"] == "server_csv_file"
        assert "fields" in d
        assert len(d["fields"]) >= 1


# ---------------------------------------------------------------------------
# CSV Exporter: CLI arguments
# ---------------------------------------------------------------------------


class TestCsvExporterCLI:
    """CLI argument parsing for the CSV exporter."""

    def test_adds_filepath_arg(self):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        args = parser.parse_args(["--filepath", "/tmp/results.csv"])
        assert args.filepath == "/tmp/results.csv"

    def test_filepath_default(self):
        from vtscore.config import DATA_DIR
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        args = parser.parse_args([])
        # Default includes a timestamp template so consecutive runs do not
        # silently overwrite one another.
        assert args.filepath == f"{DATA_DIR}/autodetect_results_{{YYYYMMDD-HHMMSS}}.csv"

    def test_validate_passes(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        exp.validate_cli_field_values({"filepath": str(tmp_path / "out.csv")})

    def test_validate_missing_filepath(self):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        with pytest.raises(ValueError, match="Missing required argument: --filepath"):
            exp.validate_cli_field_values({})


# ---------------------------------------------------------------------------
# CSV Exporter: export functionality
# ---------------------------------------------------------------------------


class TestCsvExporterExport:
    """Tests for the CSV export() method."""

    def test_creates_csv_file(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = _make_sample_results()
        filepath = tmp_path / "output.csv"

        result = exp.export(results, {"filepath": str(filepath)})
        assert filepath.exists()
        assert "message" in result
        assert "Saved" in result["message"]

    def test_csv_has_correct_header(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = _make_sample_results()
        filepath = tmp_path / "output.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["detector", "threshold", "filename", "category", "score", "origin", "origin_name"]

    def test_csv_has_correct_row_count(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = _make_sample_results()
        filepath = tmp_path / "output.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # 1 header + 3 data rows
        assert len(rows) == 4

    def test_csv_row_values(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = _make_sample_results()
        filepath = tmp_path / "output.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            first_row = next(reader)
        assert first_row[0] == "test_detector"
        assert first_row[2] == "clip_1.wav"
        assert first_row[3] == "birds"
        assert first_row[4] == "0.95"

    def test_csv_empty_results(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = {"media_type": "audio", "detectors_run": 0, "results": {}}
        filepath = tmp_path / "empty.csv"

        result = exp.export(results, {"filepath": str(filepath)})
        assert filepath.exists()
        assert "0 hit(s)" in result["message"]

    def test_csv_creates_parent_dirs(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = _make_sample_results()
        filepath = tmp_path / "sub" / "dir" / "output.csv"

        exp.export(results, {"filepath": str(filepath)})
        assert filepath.exists()

    def test_csv_multiple_detectors(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = {
            "media_type": "audio",
            "detectors_run": 2,
            "results": {
                "det_a": {
                    "detector_name": "det_a",
                    "threshold": 0.4,
                    "total_hits": 1,
                    "hits": [{"id": 1, "filename": "a.wav", "category": "x", "score": 0.9}],
                },
                "det_b": {
                    "detector_name": "det_b",
                    "threshold": 0.6,
                    "total_hits": 2,
                    "hits": [
                        {"id": 2, "filename": "b.wav", "category": "y", "score": 0.8},
                        {"id": 3, "filename": "c.wav", "category": "z", "score": 0.7},
                    ],
                },
            },
        }
        filepath = tmp_path / "multi.csv"

        result = exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # 1 header + 3 data rows
        assert len(rows) == 4
        assert "3 hit(s)" in result["message"]
        assert "2 detector(s)" in result["message"]


# ---------------------------------------------------------------------------
# CSV Exporter: labels format with selected columns
# ---------------------------------------------------------------------------


SAMPLE_LABELS = [
    {
        "label": "good",
        "md5": "abc123",
        "origin_name": "dataset_a",
        "filename": "clip1.wav",
        "category": "birds",
        "custom_metadata": {"source": "field", "quality": "high"},
    },
    {
        "label": "bad",
        "md5": "def456",
        "origin_name": "dataset_a",
        "filename": "clip2.wav",
        "category": "rain",
        "custom_metadata": {"source": "studio", "quality": "low"},
    },
    {
        "label": "good",
        "md5": "ghi789",
        "origin_name": "dataset_b",
        "filename": "clip3.wav",
        "category": "birds",
        "custom_metadata": {"source": "field"},
    },
]


class TestCsvExporterLabelsFormat:
    """Tests for CSV export of labels with selected columns."""

    def test_labels_format_uses_selected_columns(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = {
            "labels": SAMPLE_LABELS,
            "selected_columns": ["label", "filename", "category"],
        }
        filepath = tmp_path / "labels.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        # origin is always appended as last column for re-import
        assert header == ["label", "filename", "category", "origin"]

    def test_labels_format_correct_row_count(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = {
            "labels": SAMPLE_LABELS,
            "selected_columns": ["label", "filename"],
        }
        filepath = tmp_path / "labels.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 4  # 1 header + 3 labels

    def test_labels_format_row_values(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = {
            "labels": SAMPLE_LABELS,
            "selected_columns": ["label", "filename", "category"],
        }
        filepath = tmp_path / "labels.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            first_row = next(reader)
        # origin column is always appended (empty when not present on entry)
        assert first_row == ["good", "clip1.wav", "birds", ""]

    def test_labels_format_includes_metadata_columns(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = {
            "labels": SAMPLE_LABELS,
            "selected_columns": ["filename", "source", "quality"],
        }
        filepath = tmp_path / "labels.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            first_row = next(reader)
        assert header == ["filename", "source", "quality", "origin"]
        assert first_row == ["clip1.wav", "field", "high", ""]

    def test_labels_format_missing_metadata_gives_empty(self, tmp_path):
        """When a metadata column is missing from an entry, output empty string."""
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = {
            "labels": SAMPLE_LABELS,
            "selected_columns": ["filename", "quality"],
        }
        filepath = tmp_path / "labels.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # Third entry (clip3.wav) has no "quality" in metadata; origin also empty
        assert rows[3] == ["clip3.wav", "", ""]

    def test_labels_format_changed_columns_reflected(self, tmp_path):
        """Changing selected_columns between exports produces different output."""
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()

        # First export with all base columns
        results1 = {
            "labels": SAMPLE_LABELS,
            "selected_columns": ["label", "md5", "origin_name", "filename", "category"],
        }
        filepath1 = tmp_path / "export1.csv"
        exp.export(results1, {"filepath": str(filepath1)})

        # Second export with only a metadata column
        results2 = {
            "labels": SAMPLE_LABELS,
            "selected_columns": ["source"],
        }
        filepath2 = tmp_path / "export2.csv"
        exp.export(results2, {"filepath": str(filepath2)})

        with open(filepath1, newline="", encoding="utf-8") as f:
            header1 = next(csv.reader(f))
        with open(filepath2, newline="", encoding="utf-8") as f:
            header2 = next(csv.reader(f))

        # origin is always the last column
        assert header1 == ["label", "md5", "origin_name", "filename", "category", "origin"]
        assert header2 == ["source", "origin"]

    def test_labels_format_no_selected_columns_uses_defaults(self, tmp_path):
        """When selected_columns is absent, fall back to base columns."""
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = {"labels": SAMPLE_LABELS}
        filepath = tmp_path / "labels.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == ["label", "md5", "origin_name", "filename", "category", "origin"]

    def test_labels_format_message(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = {
            "labels": SAMPLE_LABELS,
            "selected_columns": ["label", "filename"],
        }
        filepath = tmp_path / "labels.csv"

        result = exp.export(results, {"filepath": str(filepath)})
        assert "3 label(s)" in result["message"]

    def test_labels_format_empty_labels(self, tmp_path):
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        results = {"labels": [], "selected_columns": ["label", "filename"]}
        filepath = tmp_path / "empty.csv"

        result = exp.export(results, {"filepath": str(filepath)})
        assert "0 label(s)" in result["message"]
        with open(filepath, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # header only

    def test_labels_format_origin_dict_serialised_as_json(self, tmp_path):
        """Origin dicts are JSON-serialised in CSV so they survive round-trip."""
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter

        exp = ServerCsvLabelsetExporter()
        origin = {"importer": "demo", "params": {"name": "flowers102"}}
        labels = [
            {
                "label": "good",
                "md5": "abc",
                "origin_name": "rose.jpg",
                "filename": "rose.jpg",
                "category": "",
                "origin": origin,
            }
        ]
        results = {"labels": labels}
        filepath = tmp_path / "with_origin.csv"

        exp.export(results, {"filepath": str(filepath)})

        # Read back and verify origin column is valid JSON
        from vtscore.labels.importers.server_csv_file import _parse_csv_bytes

        parsed = _parse_csv_bytes(filepath.read_bytes())
        assert len(parsed) == 1
        assert parsed[0]["origin"] == origin


# ---------------------------------------------------------------------------
# JSON Exporter: labels format with selected columns
# ---------------------------------------------------------------------------


class TestJsonExporterLabelsFormat:
    """Tests for JSON export of labels with selected columns."""

    def test_labels_format_filters_to_selected_columns(self, tmp_path):
        from vtscore.exporters.server_json_file import ServerJsonLabelsetExporter

        exp = ServerJsonLabelsetExporter()
        results = {
            "labels": SAMPLE_LABELS,
            "selected_columns": ["label", "filename"],
        }
        filepath = tmp_path / "labels.json"

        exp.export(results, {"filepath": str(filepath)})
        written = json.loads(filepath.read_text())
        assert written["selected_columns"] == ["label", "filename"]
        for entry in written["labels"]:
            assert set(entry.keys()) == {"label", "filename"}

    def test_labels_format_includes_metadata(self, tmp_path):
        from vtscore.exporters.server_json_file import ServerJsonLabelsetExporter

        exp = ServerJsonLabelsetExporter()
        results = {
            "labels": SAMPLE_LABELS,
            "selected_columns": ["filename", "source"],
        }
        filepath = tmp_path / "labels.json"

        exp.export(results, {"filepath": str(filepath)})
        written = json.loads(filepath.read_text())
        assert written["labels"][0] == {"filename": "clip1.wav", "source": "field"}

    def test_labels_format_no_selected_columns_keeps_all(self, tmp_path):
        from vtscore.exporters.server_json_file import ServerJsonLabelsetExporter

        exp = ServerJsonLabelsetExporter()
        results = {"labels": SAMPLE_LABELS}
        filepath = tmp_path / "labels.json"

        exp.export(results, {"filepath": str(filepath)})
        written = json.loads(filepath.read_text())
        assert written["labels"] == SAMPLE_LABELS

    def test_labels_format_changed_columns(self, tmp_path):
        """Changing selected_columns between exports produces different JSON."""
        from vtscore.exporters.server_json_file import ServerJsonLabelsetExporter

        exp = ServerJsonLabelsetExporter()

        results1 = {"labels": SAMPLE_LABELS, "selected_columns": ["label", "md5"]}
        filepath1 = tmp_path / "export1.json"
        exp.export(results1, {"filepath": str(filepath1)})

        results2 = {"labels": SAMPLE_LABELS, "selected_columns": ["source"]}
        filepath2 = tmp_path / "export2.json"
        exp.export(results2, {"filepath": str(filepath2)})

        written1 = json.loads(filepath1.read_text())
        written2 = json.loads(filepath2.read_text())

        assert set(written1["labels"][0].keys()) == {"label", "md5"}
        assert set(written2["labels"][0].keys()) == {"source"}

    def test_labels_format_message(self, tmp_path):
        from vtscore.exporters.server_json_file import ServerJsonLabelsetExporter

        exp = ServerJsonLabelsetExporter()
        results = {"labels": SAMPLE_LABELS, "selected_columns": ["label"]}
        filepath = tmp_path / "labels.json"

        result = exp.export(results, {"filepath": str(filepath)})
        assert "3 label(s)" in result["message"]


class TestCsvExporterIntegration:
    """Integration: CSV exporter via _run_exporter."""

    def test_csv_via_run_exporter(self, client, tmp_path):
        results = _make_sample_results()
        output_file = tmp_path / "integrated.csv"
        _run_exporter("server_csv_file", {"filepath": str(output_file)}, results)

        assert output_file.exists()
        with open(output_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header[0] == "detector"


# ---------------------------------------------------------------------------
# Webhook Exporter: metadata
# ---------------------------------------------------------------------------


class TestWebhookExporterMetadata:
    """Webhook exporter metadata and registration."""

    def test_webhook_exporter_registered(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("webhook")
        assert exp is not None

    def test_webhook_exporter_display_name(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("webhook")
        assert exp.display_name == "Webhook (HTTP POST)"

    def test_webhook_exporter_icon(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("webhook")
        assert exp.icon == "\U0001f310"

    def test_webhook_exporter_has_url_field(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("webhook")
        keys = [f.key for f in exp.fields]
        assert "url" in keys

    def test_webhook_url_field_is_url_type(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("webhook")
        url_field = next(f for f in exp.fields if f.key == "url")
        assert url_field.field_type == "url"

    def test_webhook_exporter_has_auth_header_field(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("webhook")
        keys = [f.key for f in exp.fields]
        assert "auth_header" in keys

    def test_webhook_auth_header_not_required(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("webhook")
        auth_field = next(f for f in exp.fields if f.key == "auth_header")
        assert auth_field.required is False

    def test_webhook_exporter_to_dict(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("webhook")
        d = exp.to_dict()
        assert d["name"] == "webhook"
        assert "fields" in d


# ---------------------------------------------------------------------------
# Webhook Exporter: CLI arguments
# ---------------------------------------------------------------------------


class TestWebhookExporterCLI:
    """CLI argument parsing for the Webhook exporter."""

    def test_adds_url_arg(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        args = parser.parse_args(["--url", "https://example.com/hook"])
        assert args.url == "https://example.com/hook"

    def test_validate_passes_with_url(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        # auth_header is optional, so only url is needed
        exp.validate_cli_field_values({"url": "https://example.com/hook"})

    def test_validate_missing_url(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        with pytest.raises(ValueError, match="Missing required argument: --url"):
            exp.validate_cli_field_values({})

    def test_validate_passes_without_auth_header(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        # Should not raise
        exp.validate_cli_field_values({"url": "https://example.com/hook"})


# ---------------------------------------------------------------------------
# Webhook Exporter: export functionality
# ---------------------------------------------------------------------------


class TestWebhookExporterExport:
    """Tests for the Webhook export() method using mocked HTTP.

    Phase B moved ``validate_url`` out of the plugin body and into the
    framework's ``normalize_field_values`` pass that fires before
    ``.export()`` is invoked.  These tests call ``.export()`` directly
    with already-normalized field values, so no URL validator runs and
    no patch is needed.
    """

    def test_posts_json_to_url(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with (
            mock.patch("vtscore.exporters.webhook.requests.post", return_value=mock_resp) as mock_post,
        ):
            result = exp.export(results, {"url": "https://example.com/hook"})

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"] == results
        assert "message" in result
        assert "200" in result["message"]

    def test_sends_auth_header_when_provided(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with (
            mock.patch("vtscore.exporters.webhook.requests.post", return_value=mock_resp) as mock_post,
        ):
            exp.export(results, {"url": "https://example.com/hook", "auth_header": "Bearer my-token"})

        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer my-token"

    def test_no_auth_header_when_empty(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with (
            mock.patch("vtscore.exporters.webhook.requests.post", return_value=mock_resp) as mock_post,
        ):
            exp.export(results, {"url": "https://example.com/hook", "auth_header": ""})

        call_kwargs = mock_post.call_args
        assert "Authorization" not in call_kwargs.kwargs["headers"]

    def test_http_error_propagates(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")

        with mock.patch("vtscore.exporters.webhook.requests.post", return_value=mock_resp):
            with pytest.raises(Exception, match="500 Server Error"):
                exp.export(results, {"url": "https://example.com/hook"})

    def test_message_contains_hit_count(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with mock.patch("vtscore.exporters.webhook.requests.post", return_value=mock_resp):
            result = exp.export(results, {"url": "https://example.com/hook"})

        assert "3 hit(s)" in result["message"]
        assert "1 detector(s)" in result["message"]

    def test_result_includes_status_code_and_url(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 201
        mock_resp.raise_for_status.return_value = None

        with mock.patch("vtscore.exporters.webhook.requests.post", return_value=mock_resp):
            result = exp.export(results, {"url": "https://example.com/hook"})

        assert result["status_code"] == 201
        assert result["url"] == "https://example.com/hook"


class TestWebhookExporterIntegration:
    """Integration: Webhook exporter via _run_exporter."""

    def test_webhook_via_run_exporter(self, client, tmp_path):
        results = _make_sample_results()
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with (
            mock.patch("vtscore.security.url_validation.validate_url"),
            mock.patch("vtscore.exporters.webhook.requests.post", return_value=mock_resp),
        ):
            _run_exporter("webhook", {"url": "https://example.com/hook"}, results)

    def test_unknown_exporter_still_raises(self):
        with pytest.raises(ValueError, match="Unknown exporter"):
            _run_exporter("nonexistent_exporter", {}, {})


# ---------------------------------------------------------------------------
# Filepath template expansion (shared by CSV + JSON exporters)
# ---------------------------------------------------------------------------


class TestFilepathTemplateExpansion:
    """Default filepaths embed a timestamp; templates expand at export time."""

    def test_csv_default_contains_timestamp_template(self):
        from vtscore.config import DATA_DIR
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_csv_file")
        fp = next(f for f in exp.fields if f.key == "filepath")
        expected = f"{DATA_DIR}/autodetect_results_{{YYYYMMDD-HHMMSS}}.csv"
        assert fp.default == expected
        assert fp.placeholder == expected

    def test_json_default_contains_timestamp_template(self):
        from vtscore.config import DATA_DIR
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_json_file")
        fp = next(f for f in exp.fields if f.key == "filepath")
        expected = f"{DATA_DIR}/autodetect_results_{{YYYYMMDD-HHMMSS}}.json"
        assert fp.default == expected
        assert fp.placeholder == expected

    def test_consecutive_csv_exports_do_not_overwrite(self, tmp_path):
        """Two exports a second apart should land in distinct files."""
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter
        from vtscore.plugins.normalize import normalize_field_values

        exp = ServerCsvLabelsetExporter()
        results = _make_sample_results()
        template = str(tmp_path / "out_{YYYYMMDD-HHMMSS}.csv")

        with mock.patch("vtscore.plugins.normalize.datetime") as mock_dt:
            from datetime import datetime as real_dt
            from datetime import timezone

            mock_dt.now.return_value = real_dt(2026, 5, 16, 14, 30, 22, tzinfo=timezone.utc)
            r1 = exp.export(results, normalize_field_values(exp, {"filepath": template}))
            mock_dt.now.return_value = real_dt(2026, 5, 16, 14, 30, 23, tzinfo=timezone.utc)
            r2 = exp.export(results, normalize_field_values(exp, {"filepath": template}))

        # Both files exist with distinct, timestamp-stamped names.
        assert r1["filepath"] != r2["filepath"]
        assert (tmp_path / "out_20260516-143022.csv").exists()
        assert (tmp_path / "out_20260516-143023.csv").exists()
        assert "{YYYYMMDD-HHMMSS}" not in r1["filepath"]

    def test_json_template_expands_timestamp(self, tmp_path):
        from datetime import datetime as real_dt
        from datetime import timezone

        from vtscore.exporters.server_json_file import ServerJsonLabelsetExporter
        from vtscore.plugins.normalize import normalize_field_values

        exp = ServerJsonLabelsetExporter()
        template = str(tmp_path / "j_{YYYYMMDD-HHMMSS}.json")

        with mock.patch("vtscore.plugins.normalize.datetime") as mock_dt:
            mock_dt.now.return_value = real_dt(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
            result = exp.export(_make_sample_results(), normalize_field_values(exp, {"filepath": template}))

        expected = tmp_path / "j_20260102-030405.json"
        assert expected.exists()
        assert result["filepath"] == str(expected.resolve())

    def test_date_only_templates_expand(self, tmp_path):
        """{YYYYMMDD} and the {YYYY}/{MM}/{DD} parts expand to today's UTC
        date, so a daily scheduled Auto-Find can write date-named files."""
        from datetime import datetime as real_dt
        from datetime import timezone

        from vtscore.exporters.server_json_file import ServerJsonLabelsetExporter
        from vtscore.plugins.normalize import normalize_field_values

        exp = ServerJsonLabelsetExporter()
        template = str(tmp_path / "d_{YYYYMMDD}" / "{YYYY}.{MM}.{DD}.json")

        with mock.patch("vtscore.plugins.normalize.datetime") as mock_dt:
            mock_dt.now.return_value = real_dt(2026, 4, 1, 23, 59, 59, tzinfo=timezone.utc)
            result = exp.export(_make_sample_results(), normalize_field_values(exp, {"filepath": template}))

        expected = tmp_path / "d_20260401" / "2026.04.01.json"
        assert expected.exists()
        assert result["filepath"] == str(expected.resolve())

    def test_username_template_expands(self, tmp_path):
        """The {username} template substitutes get_current_user(), sanitised."""
        from vtscore.exporters.server_json_file import ServerJsonLabelsetExporter
        from vtscore.plugins.normalize import normalize_field_values

        exp = ServerJsonLabelsetExporter()
        template = str(tmp_path / "{username}.json")

        with mock.patch("vtsearch.auth.get_current_user", return_value="alice"):
            exp.export(_make_sample_results(), normalize_field_values(exp, {"filepath": template}))

        assert (tmp_path / "alice.json").exists()

    def test_username_template_sanitises_path_separators(self, tmp_path):
        """A malicious username with ``/`` cannot escape the parent directory."""
        from vtscore.exporters.server_json_file import ServerJsonLabelsetExporter
        from vtscore.plugins.normalize import normalize_field_values

        exp = ServerJsonLabelsetExporter()
        template = str(tmp_path / "{username}.json")

        with mock.patch("vtsearch.auth.get_current_user", return_value="../evil"):
            exp.export(_make_sample_results(), normalize_field_values(exp, {"filepath": template}))

        # ``/`` and ``..`` are replaced with ``_``, so the result stays inside tmp_path.
        assert (tmp_path / ".._evil.json").exists()

    def test_detector_name_template_expands(self, tmp_path):
        """The {detector_name} template pulls from the active detector context."""
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter
        from vtscore.plugins.normalize import normalize_field_values
        from vtscore.state.core import DetectorContext, set_thread_detector_context

        exp = ServerCsvLabelsetExporter()
        template = str(tmp_path / "{detector_name}.csv")

        ctx = DetectorContext("det-id-1", name="dog_bark")
        set_thread_detector_context(ctx)
        try:
            exp.export(_make_sample_results(), normalize_field_values(exp, {"filepath": template}))
        finally:
            set_thread_detector_context(None)

        assert (tmp_path / "dog_bark.csv").exists()

    def test_detector_name_template_with_no_active_context_uses_placeholder(self, tmp_path):
        """An empty detector context (fallback) sanitises to ``_``."""
        from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter
        from vtscore.plugins.normalize import normalize_field_values

        exp = ServerCsvLabelsetExporter()
        template = str(tmp_path / "{detector_name}.csv")

        exp.export(_make_sample_results(), normalize_field_values(exp, {"filepath": template}))

        # sanitize_template_value("") -> "_"
        assert (tmp_path / "_.csv").exists()
