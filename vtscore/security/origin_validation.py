"""Confinement validation for origin dicts that arrive from outside the server.

An origin dict (``{"importer": ..., "params": {...}}``) is normally stamped
by the server at import time and trusted thereafter.  Some flows, however,
accept a full origin from a request body or a detector JSON file (the
``POST /api/example-sort-origin`` route, a detector's saved media examples),
and resolving such an origin re-runs filesystem or network access from
originally user-supplied parameters.  Before an externally-supplied origin
is used, its path-like params must pass the same per-user confinement the
ingress applies.

URL-valued params are deliberately *not* path-checked here: they are
re-validated with :func:`~vtscore.security.url_validation.validate_url` at
fetch time by the URL-backed sources (and the downloader re-checks every
redirect hop), and running them through the path validator would spuriously
reject them in multi-user mode.
"""

from __future__ import annotations

from typing import Any


def check_origin_param_confinement(origin: Any) -> None:
    """Raise :class:`ValueError` if a path-like origin param escapes the user's allowed dir.

    Recurses into ``dupe_set``-style ``members`` so a nested member origin
    cannot smuggle an unvalidated path.  A no-op in single-user mode, where
    the base dir is unrestricted (see
    :func:`~vtscore.security.path_validation.get_file_access_base_dir`).
    """
    from vtscore.security.path_validation import get_file_access_base_dir

    _check(origin, get_file_access_base_dir())


def _check(origin: Any, base_dir: Any) -> None:
    if not isinstance(origin, dict):
        return

    from vtscore.security.path_validation import validate_server_filepath

    params = origin.get("params")
    if isinstance(params, dict):
        for val in params.values():
            if not isinstance(val, str) or ("/" not in val and "\\" not in val):
                continue
            if val.startswith(("http://", "https://")):
                continue  # re-validated by validate_url at fetch time
            validate_server_filepath(val, base_dir=base_dir)

    members = origin.get("members")
    if isinstance(members, list):
        for member in members:
            if isinstance(member, dict):
                _check(member.get("origin"), base_dir)
