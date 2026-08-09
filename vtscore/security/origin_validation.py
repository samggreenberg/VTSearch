"""Confinement validation for origin dicts that arrive from outside the server.

An origin dict (``{"importer": ..., "params": {...}}``) is normally stamped
by the server at import time and trusted thereafter.  Some flows, however,
accept a full origin from a request body or a detector JSON file (the
``POST /api/example-sort-origin`` route, a detector's saved media examples),
and resolving such an origin re-runs filesystem or network access from
originally user-supplied parameters.  Before an externally-supplied origin
is used, its path-like params must pass the same per-user confinement the
ingress applies.

The check hands back a *confined copy* of the origin rather than a bare
pass/fail, because under confinement the validator anchors a relative path
at the user's data dir while the media source would anchor it at the process
CWD — so consuming the raw params would read a different path than the one
approved.  See
:func:`~vtscore.security.path_validation.confine_server_filepath`.

Origin params are untyped, so path-like values are recognised by carrying a
path separator.  A separator-free param (``{"filename": "abc.wav"}``) is left
alone — it cannot be told apart from an ordinary scalar like a demo-dataset
name or a media type, and confining it would rewrite non-paths into absolute
paths.  Such a value can still be resolved against the CWD by its consumer,
but it addresses a file in the process's working directory, not another
user's subtree; closing that last gap wants typed origin params.

URL-valued params are deliberately *not* path-checked here: they are
re-validated with :func:`~vtscore.security.url_validation.validate_url` at
fetch time by the URL-backed sources (and the downloader re-checks every
redirect hop), and running them through the path validator would spuriously
reject them in multi-user mode.
"""

from __future__ import annotations

from typing import Any


def confine_origin_params(origin: Any) -> Any:
    """Return *origin* with its path-like params replaced by approved paths.

    Raises :class:`ValueError` if a path-like origin param escapes the user's
    allowed dir.  Recurses into ``dupe_set``-style ``members`` so a nested
    member origin cannot smuggle an unvalidated path.  Non-dict input is
    returned unchanged.

    The input is never mutated — a saved detector example keeps the origin it
    was written with, while the caller gets the canonical form to resolve.
    In single-user mode the base dir is unrestricted (see
    :func:`~vtscore.security.path_validation.get_file_access_base_dir`), so
    every param comes back verbatim.
    """
    from vtscore.security.path_validation import get_file_access_base_dir

    return _confined(origin, get_file_access_base_dir())


def _confined(origin: Any, base_dir: Any) -> Any:
    if not isinstance(origin, dict):
        return origin

    from vtscore.security.path_validation import confine_server_filepath

    confined = dict(origin)

    params = origin.get("params")
    if isinstance(params, dict):
        new_params = dict(params)
        for key, val in params.items():
            if not isinstance(val, str) or ("/" not in val and "\\" not in val):
                continue
            if val.startswith(("http://", "https://")):
                continue  # re-validated by validate_url at fetch time
            new_params[key] = confine_server_filepath(val, base_dir)
        confined["params"] = new_params

    members = origin.get("members")
    if isinstance(members, list):
        confined["members"] = [
            {**m, "origin": _confined(m["origin"], base_dir)} if isinstance(m, dict) and "origin" in m else m
            for m in members
        ]

    return confined
