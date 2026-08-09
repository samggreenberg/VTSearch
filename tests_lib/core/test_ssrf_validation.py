"""Tests for SSRF URL validation.

Verifies that :func:`vtscore.security.url_validation.validate_url` blocks
requests to private/internal network addresses while allowing public URLs,
that :func:`~vtscore.security.url_validation.guarded_session` re-checks the
peer address each socket actually reaches (so a rebinding DNS answer cannot
slip past the name check), that
:func:`~vtscore.security.url_validation.open_validated_stream` re-checks every
redirect hop, and that the HTTP archive importer, the webhook exporter, and the
``media_url`` media fetch all go through the guard.
"""

from __future__ import annotations

import socket as socket_mod
from typing import Any, cast
from unittest import mock

import pytest
import requests

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
# guarded_session – peer-address recheck (DNS rebinding)
# ---------------------------------------------------------------------------


@pytest.fixture
def loopback_listener():
    """A real listening socket on ``127.0.0.1``; yields its port.

    Nothing ever accepts on it - a TCP connect still completes through the
    listen backlog, which is all the peer check needs to observe an address.
    """
    srv = socket_mod.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        yield srv.getsockname()[1]
    finally:
        srv.close()


class TestGuardedSessionPeerCheck:
    """``validate_url`` vets a *name*; the socket connects to an *address*.

    Between the two lookups an attacker who runs DNS for their own hostname can
    swap the answer (DNS rebinding), so the name check alone proves nothing.
    These tests drive the real socket path: the address is never mocked, only
    the DNS answer ``validate_url`` sees.
    """

    def _public_dns_answer(self):
        """Make ``validate_url`` believe any hostname is publicly routable."""
        return mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", ("93.184.216.34", 0))],
        )

    def _session(self):
        from vtscore.security.url_validation import guarded_session

        session = guarded_session()
        # Ignore any ambient HTTP(S)_PROXY: a proxied request is peer-checked at
        # the proxy, which is exactly the case this test must not exercise.
        session.trust_env = False
        return session

    def test_rebinding_to_loopback_is_blocked_at_connect(self, loopback_listener):
        from vtscore.security.url_validation import BlockedAddressError

        url = f"http://localhost:{loopback_listener}/admin"
        with self._public_dns_answer():
            # The name check is fooled by the attacker's first DNS answer...
            assert validate_url(url) == url
        # ...but the socket lands on 127.0.0.1, and that is what gets checked.
        with self._session() as session:
            with pytest.raises(BlockedAddressError, match="private/internal"):
                session.get(url, timeout=5)

    def test_https_is_blocked_before_the_tls_handshake(self, loopback_listener):
        """The hook is on the bare socket, so no SNI or request byte is sent."""
        from vtscore.security.url_validation import BlockedAddressError

        with self._session() as session:
            with pytest.raises(BlockedAddressError, match="private/internal"):
                session.get(f"https://localhost:{loopback_listener}/admin", timeout=5)

    def test_a_plain_session_is_not_guarded(self, loopback_listener):
        """Pins why the guard has to be mounted: a bare session has no check.

        The connect succeeds and the request goes out (there is no server, so it
        stalls on the read) - the point is that nothing rejects the address.
        """
        from vtscore.security.url_validation import BlockedAddressError

        session = requests.Session()
        session.trust_env = False
        with session:
            with pytest.raises(requests.RequestException) as excinfo:
                session.get(f"http://localhost:{loopback_listener}/admin", timeout=(5, 0.5))
        assert not isinstance(excinfo.value, BlockedAddressError)

    def test_blocked_error_is_a_value_error(self):
        """Callers already treat a failed guard as ``ValueError``; keep it so."""
        from vtscore.security.url_validation import BlockedAddressError

        assert issubclass(BlockedAddressError, ValueError)


class _FakeSocket:
    """Just enough socket for the peer check: an address and a close flag."""

    def __init__(self, peer):
        self._peer = peer
        self.closed = False

    def getpeername(self):
        if self._peer is None:
            raise OSError("not connected")
        return self._peer

    def close(self):
        self.closed = True


class _FakeConn:
    def __init__(self, proxy=None):
        self.proxy = proxy


def _check_peer(peer, proxy=None) -> tuple[_FakeSocket, object]:
    """Run the peer check over a stubbed socket; return the socket and result."""
    from vtscore.security.url_validation import _reject_internal_peer

    sock = _FakeSocket(peer)
    returned = _reject_internal_peer(cast(Any, _FakeConn(proxy)), cast(Any, sock))
    return sock, returned


class TestRejectInternalPeer:
    """Unit coverage for the peer check itself, socket layer stubbed out."""

    def test_public_peer_passes_through(self):
        sock, returned = _check_peer(("93.184.216.34", 443))
        assert returned is sock
        assert not sock.closed

    @pytest.mark.parametrize(
        "ip",
        ["127.0.0.1", "169.254.169.254", "10.0.0.5", "192.168.1.1", "172.16.0.1", "0.0.0.0", "::1"],
    )
    def test_internal_peer_is_rejected(self, ip):
        from vtscore.security.url_validation import BlockedAddressError

        with pytest.raises(BlockedAddressError, match="private/internal"):
            _check_peer((ip, 80))

    def test_rejected_socket_is_closed(self):
        """Nothing is left half-open for the caller to accidentally write to."""
        from vtscore.security.url_validation import BlockedAddressError, _reject_internal_peer

        sock = _FakeSocket(("127.0.0.1", 80))
        with pytest.raises(BlockedAddressError):
            _reject_internal_peer(cast(Any, _FakeConn()), cast(Any, sock))
        assert sock.closed

    def test_unreadable_peer_fails_closed(self):
        """A socket we cannot read the peer of is a socket we cannot vouch for."""
        from vtscore.security.url_validation import BlockedAddressError

        with pytest.raises(BlockedAddressError, match="unknown"):
            _check_peer(None)

    def test_proxied_connection_is_exempt(self):
        """Through a proxy the peer *is* the proxy, often legitimately private."""
        sock, returned = _check_peer(("127.0.0.1", 3128), proxy="http://127.0.0.1:3128")
        assert returned is sock
        assert not sock.closed


class TestGuardedSessionWiring:
    """The guard is only worth anything if it is actually mounted."""

    def test_both_schemes_get_the_guarded_adapter(self):
        from vtscore.security.url_validation import _GuardedHTTPAdapter, guarded_session

        with guarded_session() as session:
            for prefix in ("http://x/", "https://x/"):
                assert isinstance(session.get_adapter(prefix), _GuardedHTTPAdapter)

    def test_pool_manager_hands_out_guarded_connections(self):
        from vtscore.security.url_validation import (
            _GuardedHTTPConnection,
            _GuardedHTTPSConnection,
            guarded_session,
        )

        with guarded_session() as session:
            manager = session.get_adapter("https://x/").poolmanager  # type: ignore[attr-defined]
            assert manager.pool_classes_by_scheme["http"].ConnectionCls is _GuardedHTTPConnection
            assert manager.pool_classes_by_scheme["https"].ConnectionCls is _GuardedHTTPSConnection

    def test_the_downloader_fetches_on_a_guarded_session(self):
        """A validated URL handed to a bare session is the rebinding hole."""
        from vtscore.security.url_validation import _GuardedHTTPAdapter

        captured: list[requests.Session] = []

        def fake_stream(session, url, headers=None):
            captured.append(session)
            raise requests.ConnectionError("stop here")

        with mock.patch("vtscore.datasets.downloader.core._open_validated_stream", fake_stream):
            from vtscore.datasets.downloader.core import fetch_remote_signature

            with mock.patch(
                "vtscore.security.url_validation.socket.getaddrinfo",
                return_value=[(2, 1, 0, "", ("93.184.216.34", 0))],
            ):
                assert fetch_remote_signature("https://public.example/a.zip") is None
        assert captured, "the downloader never opened a stream"
        assert isinstance(captured[0].get_adapter("https://x/"), _GuardedHTTPAdapter)


# ---------------------------------------------------------------------------
# open_validated_stream – per-hop redirect revalidation
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for a streamed ``requests.Response``."""

    def __init__(self, status_code: int, location: str | None = None, body: bytes = b""):
        self.status_code = status_code
        self.headers = {"Location": location} if location else {}
        self.content = body
        self.closed = False

    @property
    def is_redirect(self) -> bool:
        return 300 <= self.status_code < 400 and "Location" in self.headers

    @property
    def is_permanent_redirect(self) -> bool:
        return self.status_code in (301, 308) and "Location" in self.headers

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class _FakeSession:
    """Session that replays a scripted chain of responses and records hops."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.hops: list[tuple[str, dict]] = []

    def get(self, url, **kwargs):
        self.hops.append((url, kwargs.get("headers") or {}))
        if self._responses:
            return self._responses.pop(0)
        return _FakeResponse(200, body=b"end")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def as_session(self) -> requests.Session:
        """Typed view for the duck-typed ``session`` parameter.

        ``open_validated_stream`` only ever calls ``.get()``, but its signature
        names the real class; this keeps the fake usable without widening the
        production annotation to a bespoke protocol.
        """
        return cast(requests.Session, self)


class TestOpenValidatedStream:
    """A one-time up-front check is not enough: each hop is re-resolved."""

    def _resolve(self, ip):
        return mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", (ip, 0))],
        )

    def test_redirect_to_internal_host_is_rejected(self):
        from vtscore.security.url_validation import open_validated_stream

        session = _FakeSession([_FakeResponse(302, location="http://169.254.169.254/latest/meta-data/")])
        with self._resolve("169.254.169.254"):
            with pytest.raises(ValueError, match="private/internal"):
                open_validated_stream(session.as_session(), "https://public.example/media.wav")
        # Only the first hop was ever issued; the internal target was not fetched.
        assert [hop for hop, _ in session.hops] == ["https://public.example/media.wav"]

    def test_redirect_to_non_http_scheme_is_rejected(self):
        from vtscore.security.url_validation import open_validated_stream

        session = _FakeSession([_FakeResponse(302, location="file:///etc/passwd")])
        with pytest.raises(ValueError, match="http or https"):
            open_validated_stream(session.as_session(), "https://public.example/media.wav")

    def test_follows_a_public_redirect_chain(self):
        from vtscore.security.url_validation import open_validated_stream

        session = _FakeSession(
            [
                _FakeResponse(302, location="https://cdn.example/final.wav"),
                _FakeResponse(200, body=b"payload"),
            ]
        )
        with self._resolve("93.184.216.34"):
            response = open_validated_stream(session.as_session(), "https://public.example/media.wav")
        assert response.content == b"payload"
        assert [hop for hop, _ in session.hops] == [
            "https://public.example/media.wav",
            "https://cdn.example/final.wav",
        ]

    def test_relative_location_is_resolved_against_the_current_hop(self):
        from vtscore.security.url_validation import open_validated_stream

        session = _FakeSession([_FakeResponse(302, location="/elsewhere.wav"), _FakeResponse(200, body=b"ok")])
        with self._resolve("93.184.216.34"):
            open_validated_stream(session.as_session(), "https://public.example/a/media.wav")
        assert session.hops[1][0] == "https://public.example/elsewhere.wav"

    def test_headers_are_recomputed_per_hop(self):
        """Credentials scoped to one host must not be replayed to the next."""
        from vtscore.security.url_validation import open_validated_stream

        session = _FakeSession([_FakeResponse(302, location="https://cdn.example/final.wav"), _FakeResponse(200)])
        with self._resolve("93.184.216.34"):
            open_validated_stream(
                session.as_session(),
                "https://public.example/media.wav",
                headers_for_url=lambda u: {"Authorization": "secret"} if "public.example" in u else {},
            )
        assert session.hops[0][1] == {"Authorization": "secret"}
        assert session.hops[1][1] == {}

    def test_gives_up_past_the_redirect_cap(self):
        from vtscore.security.url_validation import MAX_REDIRECTS, open_validated_stream

        session = _FakeSession(
            [_FakeResponse(302, location=f"https://h{i}.example/x") for i in range(MAX_REDIRECTS + 2)]
        )
        with self._resolve("93.184.216.34"):
            with pytest.raises(requests.TooManyRedirects):
                open_validated_stream(session.as_session(), "https://public.example/media.wav")

    def test_does_not_validate_the_first_url(self):
        """The caller owns the up-front check; this covers only the hops after it."""
        from vtscore.security.url_validation import open_validated_stream

        session = _FakeSession([_FakeResponse(200, body=b"ok")])
        with mock.patch("vtscore.security.url_validation.socket.getaddrinfo") as mock_gai:
            open_validated_stream(session.as_session(), "https://public.example/media.wav")
        mock_gai.assert_not_called()


class TestFetchValidatedUrl:
    def test_rejects_file_scheme_before_any_request(self):
        from vtscore.security.url_validation import fetch_validated_url

        with mock.patch("vtscore.security.url_validation.guarded_session") as mock_session:
            with pytest.raises(ValueError, match="http or https"):
                fetch_validated_url("file:///etc/passwd")
        mock_session.assert_not_called()

    def test_rejects_internal_host_before_any_request(self):
        from vtscore.security.url_validation import fetch_validated_url

        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", ("127.0.0.1", 0))],
        ):
            with mock.patch("vtscore.security.url_validation.guarded_session") as mock_session:
                with pytest.raises(ValueError, match="private/internal"):
                    fetch_validated_url("http://localhost:5000/api/settings")
        mock_session.assert_not_called()

    def test_returns_body_for_a_public_url(self):
        from vtscore.security.url_validation import fetch_validated_url

        session = _FakeSession([_FakeResponse(200, body=b"media-bytes")])
        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", ("93.184.216.34", 0))],
        ):
            with mock.patch("vtscore.security.url_validation.guarded_session", return_value=session):
                assert fetch_validated_url("https://public.example/m.wav") == b"media-bytes"

    def test_raises_on_error_status(self):
        from vtscore.security.url_validation import fetch_validated_url

        session = _FakeSession([_FakeResponse(404)])
        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", ("93.184.216.34", 0))],
        ):
            with mock.patch("vtscore.security.url_validation.guarded_session", return_value=session):
                with pytest.raises(requests.HTTPError):
                    fetch_validated_url("https://public.example/missing.wav")


# ---------------------------------------------------------------------------
# media_url – SSRF protection integration
# ---------------------------------------------------------------------------


class TestMediaUrlSSRF:
    """``media_url`` rides along on a media dict that can come from a loaded
    pickle, and whatever it fetches is served straight back to the requester.
    It must not be able to name a local file or an internal service.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "file://localhost/etc/shadow",
            "ftp://evil.example/creds",
            "/etc/passwd",
        ],
    )
    def test_non_http_media_url_fetches_nothing(self, url):
        from vtscore.media.base import _fetch_media_url

        with mock.patch("vtscore.security.url_validation.guarded_session") as mock_session:
            assert _fetch_media_url(url) is None
        mock_session.assert_not_called()

    @pytest.mark.parametrize("ip", ["127.0.0.1", "169.254.169.254", "10.0.0.5", "192.168.1.1"])
    def test_internal_media_url_fetches_nothing(self, ip):
        from vtscore.media.base import _fetch_media_url

        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", (ip, 0))],
        ):
            with mock.patch("vtscore.security.url_validation.guarded_session") as mock_session:
                assert _fetch_media_url(f"http://{ip}/latest/meta-data/") is None
        mock_session.assert_not_called()

    def test_public_media_url_still_fetches(self):
        from vtscore.media.base import _fetch_media_url

        session = _FakeSession([_FakeResponse(200, body=b"remote-media")])
        with mock.patch(
            "vtscore.security.url_validation.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", ("93.184.216.34", 0))],
        ):
            with mock.patch("vtscore.security.url_validation.guarded_session", return_value=session):
                assert _fetch_media_url("https://pullwrest.example/media/123") == b"remote-media"

    def test_resolve_media_bytes_refuses_a_file_url(self, tmp_path):
        """End-to-end: a pickled ``file://`` media_url must not read the disk."""
        from vtscore.media.audio.media_type import AudioMediaType

        secret = tmp_path / "secret.txt"
        secret.write_bytes(b"top-secret")

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": None, "media_url": secret.as_uri()}
        assert mt._resolve_media_bytes(media) is None

    def test_resolve_media_string_refuses_a_file_url(self, tmp_path):
        from vtscore.media.text.media_type import TextMediaType

        secret = tmp_path / "secret.txt"
        secret.write_text("top-secret")

        mt = TextMediaType()
        media = {"media_string": None, "media_path": None, "media_url": secret.as_uri()}
        assert mt._resolve_media_string(media) == ""


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
            with mock.patch("requests.Session.post", return_value=mock_resp):
                result = exp.export(
                    {"detectors_run": 0, "results": {}},
                    {"url": "https://example.com/hook"},
                )
        assert result["status_code"] == 200


# ---------------------------------------------------------------------------
# validate_browser_url – the scheme guard for URLs the *browser* opens
# ---------------------------------------------------------------------------


class TestValidateBrowserUrl:
    """The browser-URL guard is a scheme allowlist, not an SSRF guard.

    ``open_url`` is fetched by the user's browser, never by the server, so
    private hosts are legitimate targets and resolving them would buy nothing.
    What it must stop is a scheme that executes rather than navigates.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "javascript:alert(1)",
            "JavaScript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "file:///etc/passwd",
            "ftp://example.com/x",
            "example.com/no-scheme",
        ],
    )
    def test_rejects_non_http_schemes(self, url):
        from vtscore.security.url_validation import validate_browser_url

        with pytest.raises(ValueError, match="http or https"):
            validate_browser_url(url)

    def test_rejects_missing_hostname(self):
        from vtscore.security.url_validation import validate_browser_url

        with pytest.raises(ValueError, match="hostname"):
            validate_browser_url("https:///path")

    def test_rejects_empty(self):
        from vtscore.security.url_validation import validate_browser_url

        with pytest.raises(ValueError, match="empty"):
            validate_browser_url("   ")

    @pytest.mark.parametrize("url", ["https://ex ample.com/a", "https://example.com/a\nb", "https://example.com/a\tb"])
    def test_rejects_whitespace_and_control_characters(self, url):
        from vtscore.security.url_validation import validate_browser_url

        with pytest.raises(ValueError, match="whitespace or control"):
            validate_browser_url(url)

    def test_accepts_public_https(self):
        from vtscore.security.url_validation import validate_browser_url

        assert validate_browser_url("https://example.com/r?ids=a,b") == "https://example.com/r?ids=a,b"

    def test_strips_surrounding_whitespace(self):
        from vtscore.security.url_validation import validate_browser_url

        assert validate_browser_url("  https://example.com/r  ") == "https://example.com/r"

    def test_allows_localhost_unlike_the_ssrf_guard(self):
        """A local companion viewer is a legitimate target for the browser."""
        from vtscore.security.url_validation import validate_browser_url

        assert validate_browser_url("http://localhost:9000/viewer") == "http://localhost:9000/viewer"

    def test_allows_private_lan_host(self):
        from vtscore.security.url_validation import validate_browser_url

        assert validate_browser_url("http://192.168.1.20/review") == "http://192.168.1.20/review"

    def test_makes_no_dns_query(self):
        """No name resolution happens — nothing here talks to the network."""
        from vtscore.security.url_validation import validate_browser_url

        with mock.patch("vtscore.security.url_validation.socket.getaddrinfo") as mock_gai:
            validate_browser_url("https://example.com/r")
        mock_gai.assert_not_called()
