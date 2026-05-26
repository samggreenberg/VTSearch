"""Authentication and user management for VTSearch.

Provides a pluggable login-provider abstraction so VTSearch can run in
single-user mode (DefaultLoginProvider) or multi-user mode (e.g.
UserPassLoginProvider, PKILoginProvider) without changing route code.

Routes call :func:`get_current_user` to learn who is making the request.
The active provider is set once at startup via :func:`set_login_provider`.
"""

from __future__ import annotations

import logging
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Thread-local fallback used when no Flask request context is available.
# Background threads spawned by request handlers should call
# :func:`set_thread_user` to propagate the user that triggered them; tests
# can use the same hook to pin a user without a request context.
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# LoginProvider ABC
# ---------------------------------------------------------------------------


class LoginProvider(ABC):
    """Abstract base for authentication providers.

    Subclasses must implement :meth:`get_user` and :meth:`is_authenticated`.
    The default implementations of the other methods are suitable for
    single-user deployments; multi-user providers should override them.
    """

    #: Short identifier for this provider (e.g. ``"default"``, ``"userpass"``).
    name: str = ""

    @abstractmethod
    def get_user(self, request: Any) -> str:
        """Return the username for the current request.

        Args:
            request: The Flask request object.

        Returns:
            A non-empty string identifying the user.
        """

    @abstractmethod
    def is_authenticated(self, request: Any) -> bool:
        """Return ``True`` if the request carries valid credentials.

        For providers that don't require authentication (e.g.
        :class:`DefaultLoginProvider`) this should always return ``True``.
        """

    def login_required(self) -> bool:
        """Whether the frontend should show a login screen at startup."""
        return False

    def get_user_data_dir(self, username: str, base_data_dir: Path) -> Path:
        """Return the data directory for *username*.

        The default implementation returns *base_data_dir* unchanged, which
        is correct for single-user deployments.  Multi-user providers should
        return ``base_data_dir / username`` (or similar) so each user's
        datasets, detectors, and settings are stored separately.
        """
        return base_data_dir

    def status_dict(self, request: Any) -> dict[str, Any]:
        """Return a JSON-serialisable dict describing the auth state.

        Used by the ``/api/auth/status`` endpoint.
        """
        return {
            "provider": self.name,
            "user": self.get_user(request),
            "authenticated": self.is_authenticated(request),
            "login_required": self.login_required(),
        }


# ---------------------------------------------------------------------------
# DefaultLoginProvider: single-user, no authentication
# ---------------------------------------------------------------------------


class DefaultLoginProvider(LoginProvider):
    """No-op provider for single-user deployments.

    Every request is treated as coming from the ``"default"`` user.
    No login screen is shown, and the data directory is used as-is
    (no per-user subdirectory).
    """

    name = "default"

    def get_user(self, request: Any) -> str:  # noqa: ARG002
        return "default"

    def is_authenticated(self, request: Any) -> bool:  # noqa: ARG002
        return True

    def login_required(self) -> bool:
        return False

    def get_user_data_dir(self, username: str, base_data_dir: Path) -> Path:  # noqa: ARG002
        # Single-user mode: no subdirectory, use data/ directly.
        return base_data_dir


# ---------------------------------------------------------------------------
# TrivialLoginProvider: cookie-based, no password
# ---------------------------------------------------------------------------


class TrivialLoginProvider(LoginProvider):
    """Cookie-based login with no password.

    The frontend prompts for a username and sends it via
    ``POST /api/auth/login``.  The server stores the name in a
    signed Flask session cookie.  Completely insecure; useful only
    for testing multi-user features locally.
    """

    name = "trivial"

    _COOKIE_KEY = "vtsearch_user"

    def get_user(self, request: Any) -> str:
        try:
            from flask import session

            return session.get(self._COOKIE_KEY, "anonymous")
        except RuntimeError:
            return "anonymous"

    def is_authenticated(self, request: Any) -> bool:
        try:
            from flask import session

            return self._COOKIE_KEY in session
        except RuntimeError:
            return False

    def login_required(self) -> bool:
        return True

    def get_user_data_dir(self, username: str, base_data_dir: Path) -> Path:
        return base_data_dir / username


# ---------------------------------------------------------------------------
# ApiKeyLoginProvider: bearer-token, for headless integrations
# ---------------------------------------------------------------------------


# Usernames are used as data-directory names (data/<user>/...) so they must
# not contain path separators or traversal segments.
_VALID_USERNAME = re.compile(r"^[A-Za-z0-9._-]+$")


class ApiKeyLoginProvider(LoginProvider):
    """Bearer-token authentication for headless API clients.

    Reads ``Authorization: Bearer <key>`` from the request, hashes the
    presented key with SHA-256, and looks the hash up in a JSON file
    (default ``data/api_keys.json``) of the form::

        {"<sha256_hex_of_key>": "alice", "<sha256_hex_of_key>": "ci-bot"}

    The file is reloaded automatically when its mtime changes, so keys
    can be rotated without restarting the server.

    Generate a key and its hash with::

        python -c "import secrets, hashlib; \\
            k = secrets.token_urlsafe(32); \\
            print('key:', k); \\
            print('hash:', hashlib.sha256(k.encode()).hexdigest())"

    Requests without a valid bearer token resolve to the username
    ``"anonymous"`` and ``is_authenticated`` returns ``False``.  This
    provider does not show a login UI (``login_required`` is ``False``).
    It is meant for headless callers that send the header directly.
    """

    name = "api_key"

    def __init__(self, keys_file: Path | None = None) -> None:
        if keys_file is None:
            from vtscore.config import DATA_DIR

            keys_file = DATA_DIR / "api_keys.json"
        self._keys_file: Path = Path(keys_file)
        self._keys: dict[str, str] = {}
        self._mtime: float | None = None
        self._lock = threading.Lock()
        self._load_keys_if_changed()
        if not self._keys:
            logger.warning(
                "ApiKeyLoginProvider: no keys loaded from %s; every request will be anonymous. "
                "Add entries with the snippet in the class docstring.",
                self._keys_file,
            )

    def _load_keys_if_changed(self) -> None:
        try:
            mtime = self._keys_file.stat().st_mtime
        except FileNotFoundError:
            with self._lock:
                if self._mtime is not None or self._keys:
                    self._keys = {}
                    self._mtime = None
            return
        except OSError:
            logger.exception("ApiKeyLoginProvider: failed to stat %s", self._keys_file)
            return
        if mtime == self._mtime:
            return
        try:
            import json

            with open(self._keys_file) as f:
                data = json.load(f)
        except Exception:
            logger.exception(
                "ApiKeyLoginProvider: failed to load %s; keeping previously loaded keys",
                self._keys_file,
            )
            return
        if not isinstance(data, dict):
            logger.error(
                "ApiKeyLoginProvider: %s must contain a JSON object, got %s",
                self._keys_file,
                type(data).__name__,
            )
            return
        new_keys: dict[str, str] = {}
        for key_hash, username in data.items():
            if not isinstance(key_hash, str) or not isinstance(username, str):
                logger.error(
                    "ApiKeyLoginProvider: skipping non-string entry in %s: %r -> %r",
                    self._keys_file,
                    key_hash,
                    username,
                )
                continue
            if not _VALID_USERNAME.match(username):
                logger.error(
                    "ApiKeyLoginProvider: skipping key for invalid username %r in %s (must match [A-Za-z0-9._-]+)",
                    username,
                    self._keys_file,
                )
                continue
            new_keys[key_hash] = username
        with self._lock:
            self._keys = new_keys
            self._mtime = mtime
        logger.info(
            "ApiKeyLoginProvider: loaded %d key(s) from %s",
            len(new_keys),
            self._keys_file,
        )

    def _lookup_user(self, request: Any) -> str | None:
        if request is None:
            return None
        try:
            header = request.headers.get("Authorization", "") or ""
        except Exception:
            return None
        if not header.startswith("Bearer "):
            return None
        token = header[len("Bearer ") :].strip()
        if not token:
            return None
        import hashlib

        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        self._load_keys_if_changed()
        with self._lock:
            return self._keys.get(token_hash)

    def get_user(self, request: Any) -> str:
        return self._lookup_user(request) or "anonymous"

    def is_authenticated(self, request: Any) -> bool:
        return self._lookup_user(request) is not None

    def login_required(self) -> bool:
        return False

    def get_user_data_dir(self, username: str, base_data_dir: Path) -> Path:
        return base_data_dir / username


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_login_provider: LoginProvider = DefaultLoginProvider()


def set_login_provider(provider: LoginProvider) -> None:
    """Set the active login provider (called once at app startup)."""
    global _login_provider
    _login_provider = provider
    logger.info("Login provider set to %r", provider.name)


def get_login_provider() -> LoginProvider:
    """Return the active login provider."""
    return _login_provider


def get_current_user() -> str:
    """Return the username for the current request.

    Resolution order:

    1. ``g.user`` (set by the ``before_request`` middleware in ``app.py``).
    2. Thread-local fallback (set by :func:`set_thread_user`; background
       threads spawned from a request handler use this).
    3. ``"default"`` (CLI, tests, threads with no explicit user).
    """
    try:
        from flask import g

        return g.user  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        # No Flask request context (CLI mode, background thread, etc.)
        pass
    tl_user = getattr(_thread_local, "user", None)
    if tl_user:
        return tl_user
    return "default"


def set_thread_user(username: str | None) -> None:
    """Set the thread-local user for the current thread.

    Background threads that need :func:`get_current_user` to resolve to a
    specific user (rather than ``"default"``) should snapshot the request
    user before ``Thread.start()`` and call this from the thread's target.
    Pass ``None`` to clear.
    """
    _thread_local.user = username


def get_thread_user() -> str | None:
    """Return the thread-local user, or ``None`` if unset."""
    return getattr(_thread_local, "user", None)


def get_user_data_dir(username: str | None = None) -> Path:
    """Return the data directory for a user.

    If *username* is ``None``, the current request user is used.  For
    :class:`DefaultLoginProvider` this always returns ``DATA_DIR``
    unchanged.
    """
    from vtscore.config import DATA_DIR

    if username is None:
        username = get_current_user()
    return _login_provider.get_user_data_dir(username, DATA_DIR)
