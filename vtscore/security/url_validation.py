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
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import requests

#: Redirect hops :func:`open_validated_stream` will follow before giving up.
MAX_REDIRECTS = 10


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
        session: The :class:`requests.Session` to issue each hop on.
        url: An already-validated HTTP(S) URL.
        headers_for_url: Optional callable returning the headers to send for a
            given hop's URL.  Recomputed per hop so credentials scoped to one
            host are not replayed to a redirect target on another.
        timeout: ``(connect, read)`` timeout passed to each request.

    Returns:
        The final, non-redirect response; the caller owns closing it.

    Raises:
        ValueError: If a redirect hop fails :func:`validate_url`.
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
    re-validates every redirect hop, and raises for a non-2xx status.

    Raises:
        ValueError: If *url* — or any redirect hop — is not a publicly
            routable ``http(s)`` URL.
        requests.RequestException: On any transport or HTTP error.
    """
    validate_url(url)
    with requests.Session() as session:
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
