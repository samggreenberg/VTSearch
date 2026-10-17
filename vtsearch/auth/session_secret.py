"""Keep the shipped Flask ``secret_key`` out of play wherever session cookies are trusted.

``app.secret_key`` signs Flask's session cookie. ``app.py`` reads it from
``VTSEARCH_SECRET_KEY`` and otherwise falls back to :data:`DEFAULT_SECRET_KEY`,
a constant anyone can read in the source, so with the fallback in force
anyone can mint a session cookie carrying whatever they like.

That only matters where the server believes what a session cookie says:

* :class:`~vtsearch.auth.TrivialLoginProvider` (``--login trivial``) keeps the
  username in it;
* the HuggingFace OAuth flow (:mod:`vtsearch.routes.auth_huggingface`), live
  whenever ``HF_OAUTH_CLIENT_ID`` is set under any login provider, keeps the
  handshake's ``state``, PKCE verifier and redirect URI there, and accepts a
  callback only when its ``state`` matches.

:func:`secure_session_secret` runs at server start. When one of those is
active and the key is still the shipped constant, it swaps in a random
per-process key and logs one warning. Single-user mode without the
HuggingFace flow trusts no cookie and is left exactly as it was.
"""

from __future__ import annotations

import logging
import secrets

from flask import Flask

logger = logging.getLogger(__name__)

#: The fallback ``app.py`` uses when ``VTSEARCH_SECRET_KEY`` is unset. Public
#: by construction: it is in the source.
DEFAULT_SECRET_KEY = "vtsearch-dev-key-change-in-production"


def _cookie_trusting_features() -> list[str]:
    """Name each active feature that trusts the contents of a session cookie."""
    from vtsearch.auth import TrivialLoginProvider, get_login_provider
    from vtsearch.routes.auth_huggingface import _is_configured as hf_oauth_configured

    features = []
    if isinstance(get_login_provider(), TrivialLoginProvider):
        features.append("--login trivial")
    if hf_oauth_configured():
        features.append("Sign in with HuggingFace (HF_OAUTH_CLIENT_ID)")
    return features


def secure_session_secret(app: Flask) -> bool:
    """Replace the shipped secret key with a random one if a feature trusts session cookies.

    Returns ``True`` when it replaced the key. An operator-set
    ``VTSEARCH_SECRET_KEY`` is never touched, and a second call is a no-op
    because the key no longer matches :data:`DEFAULT_SECRET_KEY`.

    A per-process key is safe in the shipped configuration because
    ``gunicorn.conf.py`` runs a single worker; each worker of a multi-worker
    deployment would hold its own key and reject the others' cookies, which is
    why the warning tells that deployment to set the variable.
    """
    if app.secret_key != DEFAULT_SECRET_KEY:
        return False
    features = _cookie_trusting_features()
    if not features:
        return False
    app.secret_key = secrets.token_hex(32)
    logger.warning(
        "VTSEARCH_SECRET_KEY is not set, and %s trusts signed session cookies. The built-in default key "
        "is public, so this process signs them with a random key instead: sessions (trivial-login "
        "usernames, HuggingFace sign-ins in progress) will not survive a restart. That is safe with "
        "gunicorn.conf.py's single worker; a deployment running more than one worker must set "
        "VTSEARCH_SECRET_KEY, or each worker will reject the others' cookies.",
        " and ".join(features),
    )
    return True
