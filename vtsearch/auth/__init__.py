"""Authentication and user management for VTSearch - the Flask half.

VTSearch runs in single-user mode (:class:`DefaultLoginProvider`) or
multi-user mode (:class:`TrivialLoginProvider`,
:class:`ApiKeyLoginProvider`, or your own subclass) without changing
route code.  Routes call :func:`get_current_user` to learn who is making
the request; the active provider is set once at startup via
:func:`set_login_provider`.

The abstraction itself is **library-tier**: the :class:`LoginProvider`
ABC, the single-user default, the process-wide registry,
:func:`is_safe_username` and :func:`get_user_data_dir` all live in
:mod:`vtscore.security.login`, because
:mod:`vtscore.security.path_validation` consults them on every server-file
path check and must work in a process with no Flask in it (see
``../../vtscore/docs/architecture.md``).  This module re-exports them
verbatim - so ``from vtsearch.auth import LoginProvider`` keeps working
and there is exactly one active provider per process - and adds only the
parts that genuinely need the web framework:

* :class:`TrivialLoginProvider` / :class:`ApiKeyLoginProvider`, which read
  ``flask.session`` and the request headers;
* the request-user resolver that reads ``g.user``.

The same split already backs the current-user thread-local, whose
mechanics live in :mod:`vtscore.state.current_user`.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from vtscore.security.login import (
    DefaultLoginProvider as DefaultLoginProvider,
    LoginProvider as LoginProvider,
    get_login_provider as get_login_provider,
    get_user_data_dir as get_user_data_dir,
    is_safe_username as is_safe_username,
    set_login_provider as set_login_provider,
)
from vtscore.state.current_user import (
    get_current_user as get_current_user,
    get_thread_user as get_thread_user,
    register_request_user_resolver,
    set_thread_user as set_thread_user,
    thread_user as thread_user,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TrivialLoginProvider - cookie-based, no password
# ---------------------------------------------------------------------------


class TrivialLoginProvider(LoginProvider):
    """Cookie-based login with no password.

    The frontend prompts for a username and sends it via
    ``POST /api/auth/login``.  The server stores the name in a
    signed Flask session cookie.  Completely insecure; useful only
    for testing multi-user features locally.

    "Insecure" here means **no isolation between users**: anyone may claim
    any username, and impersonating another user is a feature, not a bug.
    It does *not* mean the username escapes ``DATA_DIR``.  The cookie is
    only integrity-protected by ``app.secret_key``, so a client that knows
    the key can put an arbitrary string in it — hence :meth:`get_user`
    re-validates on the way out, even though the login route already
    validates on the way in.  See :func:`is_safe_username`.
    """

    name = "trivial"

    _COOKIE_KEY = "vtsearch_user"

    def get_user(self, request: Any) -> str:
        try:
            from flask import session

            username = session.get(self._COOKIE_KEY, "anonymous")
        except RuntimeError:
            return "anonymous"
        if not is_safe_username(username):
            logger.warning(
                "TrivialLoginProvider: ignoring unsafe username %r in session cookie; treating request as anonymous",
                username,
            )
            return "anonymous"
        return username

    def is_authenticated(self, request: Any) -> bool:
        try:
            from flask import session

            if self._COOKIE_KEY not in session:
                return False
        except RuntimeError:
            return False
        # A cookie carrying an unsafe name resolves to "anonymous" above, so
        # report it as unauthenticated rather than letting ``status_dict``
        # claim an authenticated session for a user we refused to honour.
        return is_safe_username(session.get(self._COOKIE_KEY))

    def login_required(self) -> bool:
        return True

    def enforce_auth(self) -> bool:
        # Deliberately not enforced server-side: this provider has no
        # password, so a 401 gate adds no security (any caller could just
        # POST /api/auth/login first) — it would only break the SPA's
        # pre-login boot requests and the documented anonymous fallback.
        # The login screen is a UX affordance for switching identities,
        # not an access control.
        return False

    def get_user_data_dir(self, username: str, base_data_dir: Path) -> Path:
        return base_data_dir / username


# ---------------------------------------------------------------------------
# ApiKeyLoginProvider - bearer-token, for headless integrations
# ---------------------------------------------------------------------------


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

    Requests without a valid bearer token are rejected with 401 by the
    ``_enforce_auth`` hook (``enforce_auth()`` is inherited ``True``),
    except for the auth-status/login allowlist.  This provider does not
    show a login UI (``login_required`` is ``False``); it is meant for
    headless callers that send the header directly.
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
            if not is_safe_username(username):
                logger.error(
                    "ApiKeyLoginProvider: skipping key for invalid username %r in %s "
                    "(must match [A-Za-z0-9._-]+ and not be a '.'/'..' path segment)",
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

    def www_authenticate(self) -> str | None:
        return "Bearer"

    def get_user_data_dir(self, username: str, base_data_dir: Path) -> Path:
        return base_data_dir / username


# ---------------------------------------------------------------------------
# Current-user resolution
# ---------------------------------------------------------------------------
# The mechanics (thread-local storage, the ``"default"`` fallback) live in
# ``vtscore.state.current_user`` so library code can resolve a user without
# importing Flask - see ``../../vtscore/docs/architecture.md`` Phase 2.  All
# this module adds is the Flask half: read ``g.user`` when a request is in
# flight.  Registering the resolver at *import* time (rather than from the
# shim's startup wiring) means any process that has the app tier loaded at
# all resolves request users correctly, including CLI and test paths that
# never call ``register_flask_context_resolvers()``.


def _flask_request_user() -> str | None:
    """Return ``g.user`` when inside a Flask request, else ``None``.

    ``ImportError`` is part of the contract, not defensive padding: the
    ``vtscore-clean`` gate makes ``flask`` unimportable to prove the
    library tier runs without it, and this resolver is reachable from
    library code through :func:`vtscore.state.current_user.get_current_user`.
    """
    try:
        from flask import g

        return g.user  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, ImportError):
        # No Flask request context (CLI mode, background thread, etc.),
        # or no Flask at all (library-tier test run).
        return None


register_request_user_resolver(_flask_request_user)
