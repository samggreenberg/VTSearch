"""Who may act, and which subtree they are confined to - the Flask-free half of auth.

:mod:`vtscore.state.current_user` answers *"which username is this work
being done for?"*.  This module answers the two questions that follow
from it and that the security boundary actually depends on: **how** an
identity is established (the :class:`LoginProvider` abstraction) and
**where** that identity's data lives
(:func:`get_user_data_dir`, the confinement root handed to
:func:`~vtscore.security.path_validation.validate_server_filepath`).

Both halves have to be library-tier, because the code that consumes them
is: every server-file importer and exporter resolves its path through
:mod:`vtscore.security.path_validation`, which asks
:func:`get_login_provider` whether a per-user boundary exists at all.
Keeping the ABC and the process-wide registry here means a library-only
embedder can opt into confinement - register a provider whose
:meth:`LoginProvider.get_user_data_dir` returns ``base_data_dir /
username`` and every path check in the library starts enforcing it -
without a Flask app anywhere in the process.

What stays app-tier is only what genuinely needs a web framework: the
providers that read ``flask.session`` or an HTTP header
(:class:`~vtsearch.auth.TrivialLoginProvider`,
:class:`~vtsearch.auth.ApiKeyLoginProvider`) and the resolver that reads
``g.user``.  :mod:`vtsearch.auth` re-exports every name defined here, so
there is exactly one active provider per process however you reach it.
"""

from __future__ import annotations

import logging
import re
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
            request: The request object, or ``None`` outside a request
                (CLI, library embedding).  Typed loosely on purpose: the
                library never introspects it, it only hands it back to
                the provider that asked for it.

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

    def enforce_auth(self) -> bool:
        """Whether the server rejects unauthenticated ``/api/*`` requests.

        When ``True``, the ``_enforce_auth`` before_request hook (see
        ``vtsearch.hooks``) aborts with 401 whenever
        :meth:`is_authenticated` is ``False``, except for the small
        allowlist of auth endpoints the SPA needs to reach the login
        screen. Defaults to ``True`` so a custom provider that implements
        real credentials is gated by construction — forgetting to override
        this must fail closed, not silently serve every request as
        ``"anonymous"``. Providers for which anonymous access is a
        legitimate mode (``TrivialLoginProvider``) override this to
        ``False``; :class:`DefaultLoginProvider` is unaffected either way
        since it authenticates every request.

        Like :meth:`get_user`'s *request* argument, this is a policy the
        library only stores: it means nothing without a serving layer to
        enforce it.
        """
        return True

    def www_authenticate(self) -> str | None:
        """Challenge value for the ``WWW-Authenticate`` header on 401s.

        Return the auth scheme clients should use (e.g. ``"Bearer"``), or
        ``None`` to omit the header (cookie/session providers).
        """
        return None

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
# DefaultLoginProvider - single-user, no authentication
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
        # Single-user: no subdirectory, use data/ directly.
        return base_data_dir


# ---------------------------------------------------------------------------
# Username validation
# ---------------------------------------------------------------------------


# Usernames are used as data-directory names (data/<user>/...) so they must
# not contain path separators or traversal segments.  Note that the character
# class alone is not sufficient: it admits "." and "..", which *are* traversal
# segments, so :func:`is_safe_username` rejects those explicitly.
_VALID_USERNAME = re.compile(r"^[A-Za-z0-9._-]+$")


def is_safe_username(username: Any) -> bool:
    """Return ``True`` if *username* is safe to use as a path component.

    A username reaches the filesystem in two places, so a bad one is not
    merely a mislabelled request:

    * ``get_user_data_dir()`` builds ``DATA_DIR / username`` for per-user
      settings, which are written with ``mkdir(parents=True)``.
    * The same directory is the confinement root handed to
      :func:`~vtscore.security.path_validation.validate_server_filepath`,
      which compares against ``base_dir.resolve()`` — so ``..`` segments in
      the root are *collapsed* rather than rejected, silently widening the
      sandbox for every server-file importer and exporter.

    Providers must therefore validate any username they did not construct
    themselves, whatever their authentication strength.
    """
    return isinstance(username, str) and bool(_VALID_USERNAME.match(username)) and username.strip(".") != ""


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


def get_user_data_dir(username: str | None = None) -> Path:
    """Return the data directory for a user.

    If *username* is ``None``, the current request user is used.  For
    :class:`DefaultLoginProvider` this always returns ``DATA_DIR``
    unchanged.
    """
    # Imported lazily: :mod:`vtscore.state` pulls in the embedding stack
    # (numpy, torch) at import time, and this module is on the import path
    # of every path check in the library.
    from vtscore.config import DATA_DIR
    from vtscore.state.current_user import get_current_user

    if username is None:
        username = get_current_user()
    return _login_provider.get_user_data_dir(username, DATA_DIR)
