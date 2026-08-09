"""URL validation utilities.

Two guards live here, and they defend against different things:

* :func:`validate_url` — the SSRF guard for URLs **the server fetches**.  It
  resolves the hostname and rejects private/internal network addresses.
* :func:`validate_browser_url` — the scheme guard for URLs **the user's
  browser opens**.  No name resolution, no private-IP rejection; the request
  never leaves the user's machine, so a LAN or ``localhost`` viewer is a
  legitimate target.  What it does reject is a non-``http(s)`` scheme
  (``javascript:``, ``data:``, ``file:``), which is the actual attack surface
  when a URL is handed to the frontend to open.

Pick by *who makes the request*, not by where the string came from.

Alongside them sits :func:`open_validated_stream`, the one fetch primitive
every server-side URL fetch is meant to go through.  A single up-front
:func:`validate_url` is not enough on its own: a public URL can 302 to an
internal host, so the redirect chain has to be walked by hand with each hop
re-checked.  Keeping that loop here rather than in one caller is what stops
the next fetch site from quietly re-introducing a bypass.

Name-based checks have a second, subtler hole: :func:`validate_url` vets the
addresses a *hostname* resolved to, but the fetch that follows resolves that
name **again** inside urllib3.  An attacker who runs the authoritative DNS for
their own hostname can answer the validation lookup with a public IP and the
connect-time lookup with ``127.0.0.1`` / ``169.254.169.254`` (classic DNS
rebinding), so passing the name check proves nothing about where the socket
lands.  :func:`guarded_session` closes that window by re-checking the *peer
address* of every freshly connected socket, before TLS or any request bytes.
Server-side fetches must issue their requests on such a session — validating
the URL and then handing it to a bare :class:`requests.Session` is the hole.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any, Callable, Optional
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import DEFAULT_POOLBLOCK, HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.poolmanager import PoolManager

#: Redirect hops :func:`open_validated_stream` will follow before giving up.
MAX_REDIRECTS = 10


class BlockedAddressError(ValueError):
    """A socket connected to a private/internal peer address and was dropped.

    Subclasses :class:`ValueError` so every caller that already treats a
    rejected :func:`validate_url` as a ``ValueError`` handles a rebinding block
    the same way, without a new except clause.
    """


def _is_private_ip(ip_str: str) -> bool:
    """Return True if *ip_str* belongs to a private, loopback, or reserved range."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable → treat as unsafe
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_reserved
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_unspecified
    )


def validate_url(url: str) -> str:
    """Validate *url* for safe external use and return the cleaned URL.

    Raises :class:`ValueError` if the URL uses a non-HTTP(S) scheme, has no
    hostname, or resolves to a private/internal IP address.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must use http or https scheme, got: {parsed.scheme!r}")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must contain a hostname.")

    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        raise ValueError(f"Could not resolve hostname: {hostname}")

    for family, _, _, _, sockaddr in infos:
        ip_str = str(sockaddr[0])
        if _is_private_ip(ip_str):
            raise ValueError(
                f"URL points to a private/internal network address ({ip_str}). Only publicly routable URLs are allowed."
            )

    return url


def _reject_internal_peer(conn: HTTPConnection, sock: socket.socket) -> socket.socket:
    """Return *sock* unless its peer is an internal address, in which case close
    it and raise :class:`BlockedAddressError`.

    This is the anti-rebinding half of the SSRF guard.  ``validate_url`` vets a
    *name*; this vets the *address* the kernel actually connected to, which is
    the only thing an attacker's second DNS answer cannot lie about.

    Failing closed matters here: a socket whose peer we cannot read is a socket
    we cannot vouch for, so an unreadable ``getpeername`` is treated as a block
    rather than waved through.
    """
    if getattr(conn, "proxy", None) is not None:
        # Through a proxy the peer *is* the proxy — routinely a private or
        # loopback address, legitimately so — and the origin hostname is
        # resolved at the far end where we cannot see it.  Blocking on the
        # proxy's address would break every proxied deployment while buying
        # nothing, so the peer check does not apply.
        return sock
    try:
        peer = sock.getpeername()
        ip_str = str(peer[0]) if peer else ""
    except OSError:
        ip_str = ""
    if not ip_str or _is_private_ip(ip_str):
        sock.close()
        raise BlockedAddressError(
            f"URL points to a private/internal network address ({ip_str or 'unknown'}). "
            "Only publicly routable URLs are allowed."
        )
    return sock


class _GuardedHTTPConnection(HTTPConnection):
    """An ``http://`` connection that vets its peer address once connected.

    The hook is ``_new_conn`` rather than ``connect`` so the check sees the bare
    TCP socket: for HTTPS that is *before* the handshake leaks the hostname via
    SNI, and for both schemes before a single request byte is written.
    """

    def _new_conn(self) -> socket.socket:
        return _reject_internal_peer(self, super()._new_conn())


class _GuardedHTTPSConnection(HTTPSConnection):
    """The ``https://`` counterpart of :class:`_GuardedHTTPConnection`."""

    def _new_conn(self) -> socket.socket:
        return _reject_internal_peer(self, super()._new_conn())


# ``ConnectionCls`` is annotated as urllib3's ``BaseHTTP[S]Connection``
# *protocol*, which its own concrete ``HTTP[S]Connection`` does not structurally
# satisfy (mutable ``host``/``assert_hostname`` are invariant), so assigning any
# subclass of the concrete class trips reportAssignmentType. urllib3 makes the
# identical assignment internally; the subclasses below only override
# ``_new_conn``.
class _GuardedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _GuardedHTTPConnection  # pyright: ignore[reportAssignmentType]


class _GuardedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _GuardedHTTPSConnection  # pyright: ignore[reportAssignmentType]


class _GuardedPoolManager(PoolManager):
    """A pool manager that hands out peer-checking connection pools."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pool_classes_by_scheme = {
            "http": _GuardedHTTPConnectionPool,
            "https": _GuardedHTTPSConnectionPool,
        }


class _GuardedHTTPAdapter(HTTPAdapter):
    """A ``requests`` transport adapter whose sockets re-check their peer IP."""

    def init_poolmanager(
        self, connections: int, maxsize: int, block: bool = DEFAULT_POOLBLOCK, **pool_kwargs: Any
    ) -> None:
        # Mirrors HTTPAdapter.init_poolmanager, swapping in the guarded manager.
        # The three attributes it sets are what makes an adapter picklable.
        self._pool_connections = connections
        self._pool_maxsize = maxsize
        self._pool_block = block
        self.poolmanager = _GuardedPoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )


def guarded_session() -> requests.Session:
    """Return a :class:`requests.Session` whose every connection is peer-checked.

    The session behaves exactly like a plain one except that each freshly
    opened socket has its peer address checked against the private/internal
    blocklist and is dropped — with :class:`BlockedAddressError` — if it landed
    somewhere internal.  That is what makes the up-front :func:`validate_url`
    binding: without it, the hostname is resolved a second time at connect
    time and a rebinding DNS server gets to pick the address that lookup
    returns.

    Every server-side fetch of a URL the user had any hand in should run on one
    of these.  Requests routed through an HTTP proxy are exempt (see
    :func:`_reject_internal_peer`).
    """
    session = requests.Session()
    adapter = _GuardedHTTPAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def open_validated_stream(
    session: requests.Session,
    url: str,
    *,
    headers_for_url: Optional[Callable[[str], dict]] = None,
    timeout: tuple[float, float] = (10, 60),
) -> requests.Response:
    """GET *url* as a stream, following redirects manually so every hop is
    re-checked by :func:`validate_url`.

    We follow redirects by hand (``allow_redirects=False``) so a public URL
    cannot redirect to an internal host (SSRF), bypassing the up-front check.
    The ``(connect, read)`` timeout fails fast on an unresponsive host and
    aborts if the server stalls mid-stream.

    The **caller** validates the first URL — pass a *url* that has already been
    through :func:`validate_url`.  This function owns only the hops after it,
    which the caller has no chance to see.  (Keeping the initial check outside
    means a caller that already validated, like the resuming downloader, does
    not pay a fresh DNS resolve on every retry, and a transient resolver
    failure surfaces as a retryable connection error rather than a hard
    ``ValueError``.)

    Args:
        session: The :class:`requests.Session` to issue each hop on.  Pass a
            :func:`guarded_session`; a bare session re-resolves each hostname
            at connect time, which is the DNS-rebinding hole the per-hop
            :func:`validate_url` calls here cannot see.
        url: An already-validated HTTP(S) URL.
        headers_for_url: Optional callable returning the headers to send for a
            given hop's URL.  Recomputed per hop so credentials scoped to one
            host are not replayed to a redirect target on another.
        timeout: ``(connect, read)`` timeout passed to each request.

    Returns:
        The final, non-redirect response; the caller owns closing it.

    Raises:
        ValueError: If a redirect hop fails :func:`validate_url`, or (as
            :class:`BlockedAddressError`) if a hop's socket lands on an
            internal peer address.
        requests.TooManyRedirects: If the chain exceeds :data:`MAX_REDIRECTS`.
    """

    def _headers(target: str) -> dict:
        return headers_for_url(target) if headers_for_url is not None else {}

    current_url = url
    response = session.get(
        current_url,
        stream=True,
        timeout=timeout,
        allow_redirects=False,
        headers=_headers(current_url),
    )
    redirects = 0
    while response.is_redirect or response.is_permanent_redirect:
        if redirects >= MAX_REDIRECTS:
            response.close()
            raise requests.TooManyRedirects(f"Exceeded {MAX_REDIRECTS} redirects following {url}")
        location = response.headers.get("Location")
        if not location:
            break
        next_url = urljoin(current_url, location)
        validate_url(next_url)
        response.close()
        current_url = next_url
        response = session.get(
            current_url,
            stream=True,
            timeout=timeout,
            allow_redirects=False,
            headers=_headers(current_url),
        )
        redirects += 1
    return response


def fetch_validated_url(url: str, *, timeout: tuple[float, float] = (10, 30)) -> bytes:
    """Fetch *url* into memory through the full SSRF guard and return its body.

    The whole-body counterpart to :func:`open_validated_stream`, for the fetch
    sites that want bytes rather than a stream to spool to disk (see
    :func:`vtscore.media.base._fetch_media_url`).  Validates *url* up front,
    re-validates every redirect hop, fetches on a :func:`guarded_session` so no
    hop can rebind onto an internal address, and raises for a non-2xx status.

    Raises:
        ValueError: If *url* — or any redirect hop — is not a publicly
            routable ``http(s)`` URL, or lands on an internal peer address.
        requests.RequestException: On any transport or HTTP error.
    """
    validate_url(url)
    with guarded_session() as session:
        response = open_validated_stream(session, url, timeout=timeout)
        with response:
            response.raise_for_status()
            return response.content


def validate_browser_url(url: str) -> str:
    """Validate a URL that the **user's browser** will open, and return it stripped.

    Used for the ``open_url`` an exporter can return so the frontend opens a
    third-party page in a new tab (see
    :meth:`vtscore.exporters.base.LabelsetExporter.export`).  The fetch is made
    by the browser, not by us, so this is deliberately *not*
    :func:`validate_url`: resolving the host and refusing private IPs would
    block a perfectly reasonable ``http://localhost:9000/viewer`` companion app
    while buying no protection, because no server-side request is ever made.

    What it does enforce is the part that is actually dangerous when a string
    reaches ``window.open`` — the scheme.  Only ``http`` and ``https`` pass;
    ``javascript:``, ``data:``, ``file:`` and friends are rejected.  Embedded
    whitespace and control characters are rejected too, since they let a URL
    render as one target while resolving to another.

    Raises:
        ValueError: If *url* is empty, contains whitespace/control characters,
            uses a non-HTTP(S) scheme, or has no hostname.
    """
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("URL is empty.")

    if any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned):
        raise ValueError("URL must not contain whitespace or control characters.")

    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must use http or https scheme, got: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("URL must contain a hostname.")

    return cleaned
