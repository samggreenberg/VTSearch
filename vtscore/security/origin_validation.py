"""Confinement validation for origin dicts that arrive from outside the server.

An origin dict (``{"importer": ..., "params": {...}}``) is normally stamped
by the server at import time and trusted thereafter.  Some flows, however,
accept a full origin from a request body or a detector JSON file (the
``POST /api/example-sort-origin`` route, a detector's saved media examples),
and resolving such an origin re-runs filesystem or network access from
originally user-supplied parameters.  Before an externally-supplied origin
is used, its params must pass the same per-user confinement the ingress
applies.

Every string param is checked, not just the ones that *look* like paths:
an origin's params are untyped, so there is no reliable way to tell a
filesystem path from an opaque key, and the tokens that matter most
(``..``, ``.``, ``~``) carry no path separator to key off.  A value that is
not a path at all is harmless to check — a plain relative name resolves
inside the user's own directory and passes — so the check errs towards
validating too much rather than too little.

The check hands back a *confined copy* of the origin rather than a bare
pass/fail, because under confinement the validator anchors a relative path
at the user's data dir while the consuming source would anchor it at the
process CWD — so resolving the raw params would open a different path than
the one approved.  Rewriting, unlike checking, cannot err towards doing too
much: turning an opaque key into an absolute path would corrupt it.  So only
the params the source factories actually resolve as filesystem paths are
replaced; see :data:`_PATH_PARAM_KEYS` and
:func:`~vtscore.security.path_validation.confine_server_filepath`.

URL-valued params are deliberately *not* path-checked here: they are
re-validated with :func:`~vtscore.security.url_validation.validate_url` at
fetch time by the URL-backed sources (and the downloader re-checks every
redirect hop), and running them through the path validator would spuriously
reject them in multi-user mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Path components that make a *relative* value unconfinable.  ``.`` and
#: ``..`` are resolved against the process CWD by the sources that consume
#: an origin (``LocalFolderSource(params["path"])`` and friends), not
#: against the user's data dir, so they escape the per-user boundary even
#: when they look inert against it (``base_dir/.`` *is* ``base_dir``).
#: ``~`` is the shell's home-directory token and never names a real
#: directory inside a user's data dir.
_UNCONFINABLE_COMPONENTS = frozenset({".", "..", "~"})

#: Param keys the source factories open as filesystem paths: ``path``
#: (``LocalFolderSource`` / ``LocalArchiveSource``), ``manifest``
#: (``LocalArchiveMemberSource``) and ``paths_file`` (``ServerFilesSource``).
#: Only these are rewritten to the path the check approved.  Every other
#: param is checked and handed back verbatim, because an opaque key (an
#: archive ``member``, a ``media_type``, an embedder name) is not a path and
#: would be destroyed by being resolved into one.
_PATH_PARAM_KEYS = frozenset({"path", "manifest", "paths_file"})


def confine_origin_params(origin: Any) -> Any:
    """Return *origin* with its path params replaced by the approved paths.

    Raises :class:`ValueError` if a param escapes the user's allowed dir.
    Recurses into ``dupe_set``-style ``members`` so a nested member origin
    cannot smuggle an unvalidated path.  Non-dict input is returned
    unchanged.

    The input is never mutated — a saved detector example keeps the origin it
    was written with, while the caller gets the canonical form to resolve.
    In single-user mode the base dir is unrestricted (see
    :func:`~vtscore.security.path_validation.get_file_access_base_dir`), so
    the origin comes back exactly as it went in.
    """
    from vtscore.security.path_validation import get_file_access_base_dir

    base_dir = get_file_access_base_dir()
    if base_dir is None:
        return origin  # single-user / no-auth: every server-readable path is allowed
    return _confined(origin, base_dir)


def _confined(origin: Any, base_dir: Path) -> Any:
    if not isinstance(origin, dict):
        return origin

    confined = dict(origin)

    params = origin.get("params")
    if isinstance(params, dict):
        confined["params"] = {key: _confined_param(str(key), val, base_dir) for key, val in params.items()}

    members = origin.get("members")
    if isinstance(members, list):
        confined["members"] = [
            {**m, "origin": _confined(m["origin"], base_dir)} if isinstance(m, dict) and "origin" in m else m
            for m in members
        ]

    return confined


def _confined_param(key: str, val: Any, base_dir: Path) -> Any:
    """Validate one param value, recursing into nested containers.

    Returns the value the caller should consume: the approved path for a
    path-valued key, the input unchanged for everything else.
    """
    if isinstance(val, str):
        return _confined_value(key, val, base_dir)
    if isinstance(val, dict):
        return {sub_key: _confined_param(f"{key}.{sub_key}", sub_val, base_dir) for sub_key, sub_val in val.items()}
    if isinstance(val, (list, tuple)):
        items = [_confined_param(key, sub_val, base_dir) for sub_val in val]
        return tuple(items) if isinstance(val, tuple) else items
    return val


def _confined_value(key: str, val: str, base_dir: Path) -> str:
    """Validate a single string param against the user's allowed dir."""
    from vtscore.security.path_validation import confine_server_filepath, validate_server_filepath

    if val.startswith(("http://", "https://")):
        return val  # re-validated by validate_url at fetch time

    if not Path(val).is_absolute():
        # A relative value is only safe if it is a plain forward path: the
        # consuming source joins it verbatim, so a ``.`` / ``..`` / ``~``
        # component is resolved somewhere other than the user's data dir.
        tokens = val.replace("\\", "/").split("/")
        if any(token in _UNCONFINABLE_COMPONENTS for token in tokens):
            raise ValueError(
                f"The origin param '{key}' resolves outside the allowed directory: "
                f"a relative path must not contain '.', '..' or '~' components. "
                f"Paths must be within '{base_dir.resolve()}'."
            )

    if key in _PATH_PARAM_KEYS:
        return confine_server_filepath(val, base_dir)
    validate_server_filepath(val, base_dir=base_dir)
    return val
