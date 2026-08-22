"""Tests for the ``open_url`` labelset exporter.

Covers URL formatting from both payload shapes (LabelSet and auto-detect
results), the identifier/separator/max-items knobs, and the two ways a
template is rejected: an unusable scheme and a URL past the length limit.
"""

from __future__ import annotations

import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from vtscore import cli_progress
from vtscore.cli import _run_exporter
from vtscore.exporters import get_exporter
from vtscore.exporters.open_url import MAX_URL_LENGTH, OpenUrlResultsExporter

LABELSET = {
    "labels": [
        {"md5": "aaa1", "filename": "one.wav", "label": "good", "category": "bark"},
        {"md5": "bbb2", "filename": "two.wav", "label": "good", "category": "meow"},
        {"md5": "ccc3", "filename": "three.wav", "label": "bad", "category": "bark"},
    ]
}

AUTODETECT_RESULTS = {
    "media_type": "audio",
    "detectors_run": 1,
    "results": {
        "dog_bark": {
            "detector_name": "dog_bark",
            "total_hits": 2,
            "hits": [
                {"md5": "aaa1", "filename": "one.wav", "score": 0.9},
                {"md5": "bbb2", "filename": "two.wav", "score": 0.7},
            ],
        }
    },
}


def _ids_param(url: str, key: str = "ids") -> str:
    """Return the decoded value of *key* from *url*'s query string."""
    return parse_qs(urlparse(url).query)[key][0]


def _stub_gui_cli_export(outcome: dict):
    """Make the ``gui`` exporter's CLI export return *outcome*.

    Patches the **class**, not the registry's singleton instance: an
    instance-level patch leaves a shadowing attribute behind on undo, which
    then swallows the class-level patches other exporter tests use.
    """
    return patch.object(type(get_exporter("gui")), "export_cli", return_value=outcome)


@pytest.fixture
def exporter() -> OpenUrlResultsExporter:
    return OpenUrlResultsExporter()


class TestOpenUrlRegistration:
    def test_discovered_by_registry(self):
        assert get_exporter("open_url") is not None

    def test_declares_opens_url(self, exporter):
        assert exporter.opens_url is True

    def test_opens_url_reaches_to_dict(self, exporter):
        assert exporter.to_dict()["opens_url"] is True

    def test_other_exporters_do_not_open_urls(self):
        assert get_exporter("server_json_file").to_dict()["opens_url"] is False


class TestUrlFormatting:
    def test_substitutes_ids_from_labelset(self, exporter):
        out = exporter.export_labelset(LABELSET, {"url_template": "https://example.com/r?ids={ids}"})
        assert _ids_param(out["open_url"]) == "aaa1,bbb2,ccc3"

    def test_substitutes_ids_from_autodetect_results(self, exporter):
        out = exporter.export_find_results(AUTODETECT_RESULTS, {"url_template": "https://example.com/r?ids={ids}"})
        assert _ids_param(out["open_url"]) == "aaa1,bbb2"

    def test_substitutes_count(self, exporter):
        out = exporter.export_labelset(LABELSET, {"url_template": "https://example.com/r?n={count}"})
        assert _ids_param(out["open_url"], "n") == "3"

    def test_id_field_selects_the_identifier(self, exporter):
        out = exporter.export_labelset(
            LABELSET,
            {"url_template": "https://example.com/r?ids={ids}", "id_field": "filename"},
        )
        assert _ids_param(out["open_url"]) == "one.wav,two.wav,three.wav"

    def test_falls_back_to_custom_metadata(self, exporter):
        labelset = {"labels": [{"md5": "aaa1", "custom_metadata": {"asset_id": "XY-7"}}]}
        out = exporter.export_labelset(
            labelset,
            {"url_template": "https://example.com/r?ids={ids}", "id_field": "asset_id"},
        )
        assert _ids_param(out["open_url"]) == "XY-7"

    def test_separator_is_url_encoded_not_literal(self, exporter):
        out = exporter.export_labelset(
            LABELSET,
            {"url_template": "https://example.com/r?ids={ids}", "separator": "/"},
        )
        # The slash must survive as data, not become a path segment.
        assert "/r?ids=aaa1%2Fbbb2%2Fccc3" in out["open_url"]
        assert _ids_param(out["open_url"]) == "aaa1/bbb2/ccc3"

    def test_items_without_the_identifier_are_skipped(self, exporter):
        labelset = {"labels": [{"md5": "aaa1"}, {"filename": "no-md5.wav"}]}
        out = exporter.export_labelset(labelset, {"url_template": "https://example.com/r?ids={ids}"})
        assert _ids_param(out["open_url"]) == "aaa1"

    def test_template_without_placeholders_opens_the_site(self, exporter):
        out = exporter.export_labelset(LABELSET, {"url_template": "https://example.com/review"})
        assert out["open_url"] == "https://example.com/review"

    def test_empty_labelset_yields_an_empty_id_list(self, exporter):
        out = exporter.export_labelset({"labels": []}, {"url_template": "https://example.com/r?ids={ids}"})
        assert out["open_url"] == "https://example.com/r?ids="
        assert out["total_count"] == 0

    def test_spaces_from_template_substitution_are_encoded(self, exporter):
        # ``{detector_name}`` is substituted by the framework *before* export()
        # sees the value, and a detector name may contain spaces.
        out = exporter.export_labelset(LABELSET, {"url_template": "https://example.com/r?d=Bird Calls"})
        assert out["open_url"] == "https://example.com/r?d=Bird%20Calls"

    def test_rejects_control_characters(self, exporter):
        with pytest.raises(ValueError, match="control characters"):
            exporter.export_labelset(LABELSET, {"url_template": "https://example.com/r?d=a\nb"})


class TestTruncation:
    def test_truncates_to_max_items(self, exporter):
        out = exporter.export_labelset(
            LABELSET,
            {"url_template": "https://example.com/r?ids={ids}", "max_items": "2"},
        )
        assert _ids_param(out["open_url"]) == "aaa1,bbb2"
        assert out["included_count"] == 2
        assert out["total_count"] == 3

    def test_truncation_is_reported_in_the_message(self, exporter):
        out = exporter.export_labelset(
            LABELSET,
            {"url_template": "https://example.com/r?ids={ids}", "max_items": "2"},
        )
        assert "first 2 of 3" in out["message"]

    def test_untruncated_message_omits_the_first_n_phrasing(self, exporter):
        out = exporter.export_labelset(LABELSET, {"url_template": "https://example.com/r?ids={ids}"})
        assert "first" not in out["message"]
        assert out["included_count"] == out["total_count"] == 3

    @pytest.mark.parametrize("value", ["", None, "0", "-5", "not-a-number"])
    def test_unusable_max_items_falls_back_to_the_default(self, exporter, value):
        # A cap of zero would emit a URL covering nothing at all.
        assert exporter._resolve_max_items(value) > 0


class TestRejection:
    @pytest.mark.parametrize(
        "template",
        [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "file:///etc/passwd",
            "ftp://example.com/x",
        ],
    )
    def test_rejects_non_http_schemes(self, exporter, template):
        with pytest.raises(ValueError, match="http or https"):
            exporter.export_labelset(LABELSET, {"url_template": template})

    def test_rejects_relative_url(self, exporter):
        with pytest.raises(ValueError, match="http or https"):
            exporter.export_labelset(LABELSET, {"url_template": "/review?ids={ids}"})

    def test_rejects_empty_template(self, exporter):
        with pytest.raises(ValueError, match="required"):
            exporter.export_labelset(LABELSET, {"url_template": "   "})

    def test_rejects_url_over_the_length_limit(self, exporter):
        labelset = {"labels": [{"md5": "x" * 100} for _ in range(100)]}
        with pytest.raises(ValueError, match="over the"):
            exporter.export_labelset(labelset, {"url_template": "https://example.com/r?ids={ids}"})

    def test_length_error_names_the_knob_to_turn(self, exporter):
        labelset = {"labels": [{"md5": "x" * 100} for _ in range(100)]}
        with pytest.raises(ValueError, match="Max items"):
            exporter.export_labelset(labelset, {"url_template": "https://example.com/r?ids={ids}"})

    def test_lowering_max_items_brings_it_under_the_limit(self, exporter):
        labelset = {"labels": [{"md5": "x" * 100} for _ in range(100)]}
        out = exporter.export_labelset(
            labelset,
            {"url_template": "https://example.com/r?ids={ids}", "max_items": "10"},
        )
        assert len(out["open_url"]) <= MAX_URL_LENGTH


class TestCliExport:
    """The CLI variant returns the URL; ``vtscore.cli`` is what surfaces it.

    Writing to stdout here would both duplicate that line and corrupt the
    NDJSON stream under ``--progress-format json``, so these assert silence.
    """

    def test_returns_the_url(self, exporter):
        out = exporter.export_cli(AUTODETECT_RESULTS, {"url_template": "https://example.com/r?ids={ids}"})
        assert _ids_param(out["open_url"]) == "aaa1,bbb2"

    def test_reports_truncation_in_the_message(self, exporter):
        out = exporter.export_cli(
            AUTODETECT_RESULTS,
            {"url_template": "https://example.com/r?ids={ids}", "max_items": "1"},
        )
        assert "first 1 of 2" in out["message"]

    def test_writes_nothing_to_stdout(self, exporter, capsys):
        exporter.export_cli(AUTODETECT_RESULTS, {"url_template": "https://example.com/r?ids={ids}"})
        assert capsys.readouterr().out == ""


class TestCliSurfacesOpenUrl:
    """``_run_exporter`` reports an ``open_url`` instead of dropping it.

    There is no browser on the command line, so the URL has to reach the
    operator (text mode) or the wrapping script (JSON mode) some other way.
    This is what makes the capability usable from *any* exporter, not just the
    built-in one (issue #2898).
    """

    TEMPLATE = {"url_template": "https://example.com/r?ids={ids}"}

    def test_prints_the_url_under_the_message(self, capsys):
        _run_exporter("open_url", dict(self.TEMPLATE), AUTODETECT_RESULTS)
        out = capsys.readouterr().out
        assert "Formatted a URL covering 2 item(s)." in out
        assert "https://example.com/r?ids=aaa1%2Cbbb2" in out

    def test_carries_the_url_on_the_json_event(self, capsys):
        cli_progress.set_format("json")
        _run_exporter("open_url", dict(self.TEMPLATE), AUTODETECT_RESULTS)
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        assert len(lines) == 1, "the URL must not be printed as prose alongside the NDJSON event"
        event = json.loads(lines[0])
        assert event["event"] == "export_complete"
        assert _ids_param(event["open_url"]) == "aaa1,bbb2"

    def test_exporters_without_a_url_are_unchanged(self, capsys):
        with _stub_gui_cli_export({"message": "Wrote it."}):
            _run_exporter("gui", {}, AUTODETECT_RESULTS)
        assert capsys.readouterr().out == "Wrote it.\n"

    def test_unusable_url_is_dropped_rather_than_shown(self, capsys):
        """A plugin must not be able to put a ``javascript:`` URL in front of
        the user — but the export itself already ran, so it isn't failed."""
        with _stub_gui_cli_export({"message": "Wrote it.", "open_url": "javascript:alert(1)"}):
            _run_exporter("gui", {}, AUTODETECT_RESULTS)
        out = capsys.readouterr().out
        assert "javascript:" not in out
        assert "Wrote it." in out
