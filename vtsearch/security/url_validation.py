"""URL validation utilities to prevent SSRF (Server-Side Request Forgery).

Provides :func:`validate_url` which resolves the hostname and rejects URLs
that point to private/internal network addresses.
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
        ip_str = sockaddr[0]
        if _is_private_ip(ip_str):
            raise ValueError(
                f"URL points to a private/internal network address ({ip_str}). Only publicly routable URLs are allowed."
            )

    return url
