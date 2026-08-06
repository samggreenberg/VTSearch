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
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


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
