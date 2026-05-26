"""Tests for SSRF URL validation.

Verifies that :func:`vtscore.security.url_validation.validate_url` blocks
requests to private/internal network addresses while allowing public URLs.
Also verifies that the HTTP archive importer and webhook exporter both
call the validator.
"""

from __future__ import annotations

from unittest import mock

import pytest

from vtscore.security.url_validation import validate_url


# ---------------------------------------------------------------------------
# validate_url – scheme enforcement
# ---------------------------------------------------------------------------


class TestValidateUrlScheme:
    def test_rejects_ftp_scheme(self):
        with pytest.raises(ValueError, match="http or https"):
            validate_url("ftp://example.com/file.zip")

    def test_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="http or https"):
            validate_url("file:///etc/passwd")

    def test_rejects_no_scheme(self):
        with pytest.raises(ValueError, match="http or https"):
            validate_url("example.com/file.zip")

    def test_rejects_javascript_scheme(self):
        with pytest.raises(ValueError, match="http or https"):
            validate_url("javascript:alert(1)")

    def test_accepts_http(self):
        with mock.patch("vtscore.security.url_validation.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
            assert validate_url("http://example.com") == "http://example.com"

    def test_accepts_https(self):
        with mock.patch("vtscore.security.url_validation.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
            assert validate_url("https://example.com") == "https://example.com"


# ---------------------------------------------------------------------------
# validate_url – hostname enforcement
# ---------------------------------------------------------------------------


class TestValidateUrlHostname:
    def test_rejects_empty_hostname(self):
        with pytest.raises(ValueError, match="hostname"):
            validate_url("http://")

    def test_rejects_missing_hostname(self):
        with pytest.raises(ValueError, match="hostname"):
            validate_url("http:///path")


# ---------------------------------------------------------------------------
# validate_url – private IP blocking
# ---------------------------------------------------------------------------


class TestValidateUrlPrivateIPs:
    def _mock_resolve(self, ip):
        return mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", (ip, 0))],
        )

    def test_blocks_localhost_127_0_0_1(self):
        with self._mock_resolve("127.0.0.1"):
            with pytest.raises(ValueError, match="private/internal"):
                validate_url("http://localhost/admin")

    def test_blocks_127_x(self):
        with self._mock_resolve("127.0.0.2"):
            with pytest.raises(ValueError, match="private/internal"):
                validate_url("http://some-host.example/")

    def test_blocks_10_x(self):
        with self._mock_resolve("10.0.0.1"):
            with pytest.raises(ValueError, match="private/internal"):
                validate_url("http://internal.corp/")

    def test_blocks_172_16_x(self):
        with self._mock_resolve("172.16.0.1"):
            with pytest.raises(ValueError, match="private/internal"):
                validate_url("http://internal.corp/")

    def test_blocks_192_168_x(self):
        with self._mock_resolve("192.168.1.1"):
            with pytest.raises(ValueError, match="private/internal"):
                validate_url("http://router.local/")

    def test_blocks_link_local(self):
        with self._mock_resolve("169.254.169.254"):
            with pytest.raises(ValueError, match="private/internal"):
                validate_url("http://metadata.internal/")

    def test_blocks_ipv6_loopback(self):
        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(10, 1, 0, "", ("::1", 0, 0, 0))],
        ):
            with pytest.raises(ValueError, match="private/internal"):
                validate_url("http://ip6-localhost/")

    def test_allows_public_ip(self):
        with self._mock_resolve("93.184.216.34"):
            result = validate_url("https://example.com/archive.zip")
            assert result == "https://example.com/archive.zip"

    def test_blocks_unresolvable_hostname(self):
        import socket

        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            side_effect=socket.gaierror("Name or service not known"),
        ):
            with pytest.raises(ValueError, match="Could not resolve"):
                validate_url("http://this-does-not-exist.invalid/")

    def test_blocks_zero_address(self):
        with self._mock_resolve("0.0.0.0"):
            with pytest.raises(ValueError, match="private/internal"):
                validate_url("http://zero.example/")

    def test_checks_all_resolved_addresses(self):
        """If a hostname has multiple A records, block if ANY is private."""
        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[
                (2, 1, 0, "", ("93.184.216.34", 0)),
                (2, 1, 0, "", ("127.0.0.1", 0)),
            ],
        ):
            with pytest.raises(ValueError, match="private/internal"):
                validate_url("http://dual-homed.example/")


# ---------------------------------------------------------------------------
# HTTP archive importer – SSRF protection integration
# ---------------------------------------------------------------------------


class TestHttpArchiveImporterSSRF:
    """Verify that the HTTP archive importer validates URLs before downloading."""

    def test_run_rejects_private_url(self):
        # Phase B: ``validate_url`` runs at the framework boundary, not
        # inside ``imp.run()``.  Verify the framework's normalize pass
        # fires on the importer's declared ``url`` field.
        from vtscore.datasets.importers.http_archive import HttpArchiveDatasetImporter
        from vtscore.plugins.normalize import normalize_field_values

        imp = HttpArchiveDatasetImporter()
        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", ("127.0.0.1", 0))],
        ):
            with pytest.raises(ValueError, match="private/internal"):
                normalize_field_values(imp, {"url": "http://localhost:8080/secret.zip", "media_type": "audio"})

    def test_run_rejects_non_http_scheme(self):
        from vtscore.datasets.importers.http_archive import HttpArchiveDatasetImporter
        from vtscore.plugins.normalize import normalize_field_values

        imp = HttpArchiveDatasetImporter()
        with pytest.raises(ValueError, match="http or https"):
            normalize_field_values(imp, {"url": "file:///etc/passwd", "media_type": "audio"})


# ---------------------------------------------------------------------------
# Webhook exporter – SSRF protection integration
# ---------------------------------------------------------------------------


class TestWebhookExporterSSRF:
    """Verify that the webhook exporter validates URLs before POSTing."""

    def test_export_rejects_private_url(self):
        # Phase B: ``validate_url`` runs at the framework boundary, not
        # inside ``exp.export()``.  Verify the framework's normalize
        # pass fires on the exporter's declared ``url`` field.
        from vtscore.exporters.webhook import WebhookLabelsetExporter
        from vtscore.plugins.normalize import normalize_field_values

        exp = WebhookLabelsetExporter()
        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", ("192.168.1.1", 0))],
        ):
            with pytest.raises(ValueError, match="private/internal"):
                normalize_field_values(exp, {"url": "http://192.168.1.1:9090/hook"})

    def test_export_rejects_non_http_scheme(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter
        from vtscore.plugins.normalize import normalize_field_values

        exp = WebhookLabelsetExporter()
        with pytest.raises(ValueError, match="http or https"):
            normalize_field_values(exp, {"url": "ftp://evil.example/hook"})

    def test_export_allows_public_url(self):
        from vtscore.exporters.webhook import WebhookLabelsetExporter

        exp = WebhookLabelsetExporter()
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", ("93.184.216.34", 0))],
        ):
            with mock.patch("vtscore.exporters.webhook.requests.post", return_value=mock_resp):
                result = exp.export(
                    {"detectors_run": 0, "results": {}},
                    {"url": "https://example.com/hook"},
                )
        assert result["status_code"] == 200
