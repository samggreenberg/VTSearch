"""Authentication and user management for VTSearch.

Provides a pluggable login-provider abstraction so VTSearch can run in
single-user mode (DefaultLoginProvider) or multi-user mode (e.g.
UserPassLoginProvider, PKILoginProvider) without changing route code.

Routes call :func:`get_current_user` to learn who is making the request.
The active provider is set once at startup via :func:`set_login_provider`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
# DefaultLoginProvider — single-user, no authentication
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
        # Single-user: no subdirectory — use data/ directly.
        return base_data_dir


# ---------------------------------------------------------------------------
# TrivialLoginProvider — cookie-based, no password
# ---------------------------------------------------------------------------


class TrivialLoginProvider(LoginProvider):
    """Cookie-based login with no password.

    The frontend prompts for a username and sends it via
    ``POST /api/auth/login``.  The server stores the name in a
    signed Flask session cookie.  Completely insecure — useful only
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

    Inside a Flask request context this reads ``g.user`` (set by the
    ``before_request`` middleware in ``app.py``).  Outside a request
    context (e.g. CLI, tests) it falls back to ``"default"``.
    """
    try:
        from flask import g

        return g.user  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        # No Flask request context (CLI mode, background thread, etc.)
        return "default"


def get_user_data_dir(username: str | None = None) -> Path:
    """Return the data directory for a user.

    If *username* is ``None``, the current request user is used.  For
    :class:`DefaultLoginProvider` this always returns ``DATA_DIR``
    unchanged.
    """
    from vtsearch.config import DATA_DIR

    if username is None:
        username = get_current_user()
    return _login_provider.get_user_data_dir(username, DATA_DIR)
