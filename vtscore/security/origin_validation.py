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


def check_origin_param_confinement(origin: Any) -> None:
    """Raise :class:`ValueError` if an origin param escapes the user's allowed dir.

    Recurses into ``dupe_set``-style ``members`` so a nested member origin
    cannot smuggle an unvalidated path.  A no-op in single-user mode, where
    the base dir is unrestricted (see
    :func:`~vtscore.security.path_validation.get_file_access_base_dir`).
    """
    from vtscore.security.path_validation import get_file_access_base_dir

    base_dir = get_file_access_base_dir()
    if base_dir is None:
        return  # single-user / no-auth: every server-readable path is allowed
    _check(origin, base_dir)


def _check(origin: Any, base_dir: Path) -> None:
    if not isinstance(origin, dict):
        return

    params = origin.get("params")
    if isinstance(params, dict):
        for key, val in params.items():
            _check_param(str(key), val, base_dir)

    members = origin.get("members")
    if isinstance(members, list):
        for member in members:
            if isinstance(member, dict):
                _check(member.get("origin"), base_dir)


def _check_param(key: str, val: Any, base_dir: Path) -> None:
    """Validate one param value, recursing into nested containers."""
    if isinstance(val, str):
        _check_value(key, val, base_dir)
    elif isinstance(val, dict):
        for sub_key, sub_val in val.items():
            _check_param(f"{key}.{sub_key}", sub_val, base_dir)
    elif isinstance(val, (list, tuple)):
        for sub_val in val:
            _check_param(key, sub_val, base_dir)


def _check_value(key: str, val: str, base_dir: Path) -> None:
    """Validate a single string param against the user's allowed dir."""
    from vtscore.security.path_validation import validate_server_filepath

    if val.startswith(("http://", "https://")):
        return  # re-validated by validate_url at fetch time

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

    validate_server_filepath(val, base_dir=base_dir)
