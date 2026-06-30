"""In-memory HuggingFace credential store and gated-resource error type.

VTSearch can authenticate its *outbound* requests to the HuggingFace Hub so
that gated demo datasets (and gated model weights, e.g. DINOv3) download
successfully.  The OAuth *flow* that obtains the token lives in the app tier
(:mod:`vtsearch.routes.auth_huggingface`); this module is the library-tier
holder that download and embedder code reads from, so neither needs to import
anything app-specific.

The token is held **in memory only** (process-scoped), never written to disk.
It is a short-lived secret that the HuggingFace OAuth flow re-mints on demand,
so persisting it would add a plaintext-secret-at-rest risk for no real gain;
a restart simply means signing in again.  This mirrors the repo's broader
"re-derive on demand, don't persist" stance for in-memory artifacts.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

# Hosts to which the stored HuggingFace bearer token may be attached.  We send
# it only to the Hub itself; a ``resolve/`` request 302s to a presigned CDN /
# Xet URL whose authorization is carried in the URL signature, so the token is
# never leaked to (and is not needed by) those downstream hosts.
_HF_AUTH_HOSTS = ("huggingface.co", "hf.co")


@dataclass(frozen=True)
class HFCredential:
    """A stored HuggingFace OAuth credential."""

    access_token: str
    username: str = ""
    #: Epoch seconds when the token expires, or ``None`` if unknown/non-expiring.
    expires_at: float | None = None
    #: Space-separated OAuth scopes granted, for display only.
    scopes: str = ""

    def is_expired(self, *, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at


_lock = threading.Lock()
_credential: HFCredential | None = None


def set_credential(
    access_token: str,
    *,
    username: str = "",
    expires_at: float | None = None,
    scopes: str = "",
) -> None:
    """Store the active HuggingFace credential (replacing any previous one)."""
    global _credential
    token = (access_token or "").strip()
    if not token:
        raise ValueError("access_token must be a non-empty string")
    with _lock:
        _credential = HFCredential(
            access_token=token,
            username=username or "",
            expires_at=expires_at,
            scopes=scopes or "",
        )


def clear_credential() -> None:
    """Forget the stored HuggingFace credential (sign out)."""
    global _credential
    with _lock:
        _credential = None


def get_token() -> str | None:
    """Return the stored access token, or ``None`` if unset or expired."""
    with _lock:
        cred = _credential
    if cred is None or cred.is_expired():
        return None
    return cred.access_token


def is_authenticated() -> bool:
    """Return ``True`` if a non-expired HuggingFace credential is stored."""
    return get_token() is not None


def get_status() -> dict[str, object]:
    """Return a JSON-serialisable snapshot of the sign-in state."""
    with _lock:
        cred = _credential
    if cred is None or cred.is_expired():
        return {"authenticated": False, "username": "", "scopes": ""}
    return {"authenticated": True, "username": cred.username, "scopes": cred.scopes}


def _host_is_hf(host: str | None) -> bool:
    if not host:
        return False
    host = host.lower()
    return any(host == h or host.endswith("." + h) for h in _HF_AUTH_HOSTS)


def auth_header_for_url(url: str) -> dict[str, str]:
    """Return ``{"Authorization": "Bearer <token>"}`` iff *url* targets the Hub.

    Returns an empty dict when no token is stored or *url* is not a HuggingFace
    host, so callers can unconditionally merge the result into their headers
    without leaking the token to third-party CDNs or redirect targets.
    """
    token = get_token()
    if not token:
        return {}
    try:
        host = urlparse(url).hostname
    except ValueError:
        return {}
    if not _host_is_hf(host):
        return {}
    return {"Authorization": f"Bearer {token}"}


class GatedResourceError(RuntimeError):
    """A download failed because the resource is gated/requires authentication.

    Raised with a short, user-facing message (the frontend keys off it to offer
    a "Sign in with HuggingFace" affordance).  Carries the originating *url* and
    HTTP *status* for logging/diagnostics.
    """

    def __init__(self, message: str, *, url: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status = status
