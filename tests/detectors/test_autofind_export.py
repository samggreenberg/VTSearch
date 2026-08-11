"""Tests for the Auto-Find server-side results auto-export.

Covers :func:`vtsearch.routes.detectors.scoring._run_autofind_export`, which
hands an autodetect run's results to the exporter configured under
``autofind_exporter``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from vtsearch import settings
from vtsearch.routes.detectors.scoring import _run_autofind_export

_SAMPLE_RESULTS = {
    "media_type": "audio",
    "detectors_run": 1,
    "results": {
        "det-a": {
            "detector_name": "det-a",
            "threshold": 0.5,
            "total_hits": 1,
            "hits": [{"id": 1, "score": 0.9, "filename": "a.wav", "md5": "abc"}],
            "negative_hits": [],
        }
    },
}


class TestAutofindExport:
    def test_no_exporter_returns_none(self, isolated_settings):
        settings.set_autofind_exporter("")
        assert _run_autofind_export(dict(_SAMPLE_RESULTS)) is None

    def test_unknown_exporter_reports_error(self, isolated_settings):
        # The raw setter skips the route-layer registry check, so a stale /
        # bogus name can reach the function; it must report rather than raise.
        settings.set_autofind_exporter("no_such_exporter")
        status = _run_autofind_export(dict(_SAMPLE_RESULTS))
        assert status is not None
        assert status["exporter"] == "no_such_exporter"
        assert status["success"] is False
        assert "error" in status

    def test_success_writes_json_file(self, isolated_settings, tmp_path):
        out = tmp_path / "autofind_results.json"
        settings.set_autofind_exporter("server_json_file")
        settings.set_autofind_exporter_field_values({"server_json_file": {"filepath": str(out)}})
        status = _run_autofind_export(dict(_SAMPLE_RESULTS))
        assert status is not None
        assert status["exporter"] == "server_json_file"
        assert status["success"] is True
        assert out.exists()
        written = json.loads(out.read_text())
        assert "results" in written or "det-a" in json.dumps(written)


class TestAutofindExportOpenUrl:
    """An exporter can send the user to a web page instead of delivering the
    results anywhere; the status block carries the URL to the modal (#2898)."""

    def test_open_url_reaches_the_status_block(self, isolated_settings):
        settings.set_autofind_exporter("open_url")
        settings.set_autofind_exporter_field_values({"open_url": {"url_template": "https://example.com/r?ids={ids}"}})
        status = _run_autofind_export(dict(_SAMPLE_RESULTS))
        assert status is not None
        assert status["success"] is True
        assert status["open_url"] == "https://example.com/r?ids=abc"

    def test_unusable_url_is_dropped_not_forwarded(self, isolated_settings):
        """The same scheme allowlist the export route applies — a plugin must
        not be able to push a ``javascript:`` URL at the browser from here."""
        from vtscore.exporters import get_exporter

        # Patch the *class*: the registry hands out a singleton instance, and
        # an instance-level patch would leave a shadowing attribute behind that
        # later class-level patches (tests/io/test_exporters.py) can't reach.
        settings.set_autofind_exporter("gui")
        with patch.object(
            type(get_exporter("gui")),
            "export",
            return_value={"message": "Shown.", "open_url": "javascript:alert(1)"},
        ):
            status = _run_autofind_export(dict(_SAMPLE_RESULTS))
        assert status is not None
        # The export itself ran, so it stays a success — only the URL is lost.
        assert status["success"] is True
        assert "open_url" not in status
        assert "unusable URL" in status["message"]
