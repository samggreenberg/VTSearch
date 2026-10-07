"""Functions a server admin runs when a user's dataset import finishes.

An admin names them at startup, as ``module:function``::

    python app.py --on-dataset-imported mailer:notify
    VTSEARCH_ON_DATASET_IMPORTED=mailer:notify,audit:record   # gunicorn / Docker

Each one is called with a single
:class:`~vtscore.datasets.import_event.DatasetImported` whenever an import a
user started from the web app ends: when it saved a dataset, and when it
failed (never when the user cancelled it).  That is the extension point for
sending an email, posting to a chat channel, or anything else a deployment
wants to do then, without that code living in this repository: the module only
has to be importable on the server (installed, or on ``PYTHONPATH``).

**Which imports count.** The ones a user starts and waits for: the Add Dataset
dialog's importers, local folder and file uploads, demo datasets, and an
uploaded dataset file -- the same set that offers to run the user's AutoRun
detectors.  Reloading a saved dataset, combining datasets, and the
CLI's ``--autodetect`` do not fire the hooks.

**When and where they run.** On the import's worker thread, after the progress
bar has already shown the import as finished, and after AutoRun (if any) has
started.  So a hook that blocks on a slow mail server holds up nothing the user
can see.  Hooks run in the order they were named.  One that raises is logged
and the next still runs; the dataset is saved either way.

**Why this is not an** :mod:`~vtsearch.admin_overrides` **entry.** Every knob in
that registry overrides a persisted setting and is published at
``/api/settings``.  This one must never be persisted: a settings file that
could name code to run would let anyone who can import a settings file run code
on the server.  So it has a flag and an env var and nothing else, and both
spellings go through :func:`configure` here.
"""

from __future__ import annotations

import logging
import os
import pkgutil
from collections.abc import Callable, Iterable
from typing import Any

from vtscore.datasets.import_event import DatasetImported

logger = logging.getLogger(__name__)

FLAG = "--on-dataset-imported"
ENV = "VTSEARCH_ON_DATASET_IMPORTED"

#: ``(spec, function)`` for every hook in force, in the order they run.
_hooks: list[tuple[str, Callable[[DatasetImported], Any]]] = []

#: The flag or env var that set :data:`_hooks`, for the startup banner.
_source: str | None = None


class HookSpecError(ValueError):
    """A ``module:function`` spec did not name a callable.

    Carries a message written for an operator.  :mod:`vtsearch.cli_main` turns
    it into ``parser.error``; the env path prints it as a warning instead,
    because a bad variable should not stop a container from booting.
    """


def resolve_hook(spec: str, *, source: str = FLAG) -> Callable[[DatasetImported], Any]:
    """Import and return the function *spec* names, or raise :class:`HookSpecError`.

    Importing happens here, at startup, so a typo or a broken module fails
    while an admin is watching rather than at the first import.  *source* is
    the flag or env var the spec came from, so the message names what the
    admin actually set.
    """
    spec = spec.strip()
    if not spec:
        raise HookSpecError(f"{source} expects module:function, got an empty value")
    try:
        target = pkgutil.resolve_name(spec)
    except Exception as exc:  # noqa: BLE001 - importing the admin's module can raise anything
        raise HookSpecError(f"{source} {spec!r} could not be loaded: {exc}") from exc
    if not callable(target):
        raise HookSpecError(f"{source} {spec!r} names a {type(target).__name__}, not a function")
    return target


def configure(specs: Iterable[str], *, source: str) -> None:
    """Resolve every spec in *specs* and make them the hooks in force.

    All or nothing: if any spec fails, :class:`HookSpecError` is raised and
    the hooks already in force are left as they were.
    """
    global _source
    resolved = [(spec.strip(), resolve_hook(spec, source=source)) for spec in specs]
    _hooks[:] = resolved
    _source = source if resolved else None


def add_cli_argument(parser: Any) -> None:
    """Register :data:`FLAG` on the ``python app.py`` argument parser."""
    parser.add_argument(
        FLAG,
        action="append",
        default=[],
        dest="on_dataset_imported",
        metavar="MODULE:FUNCTION",
        help=(
            "Call FUNCTION (imported from MODULE, which must be importable on this "
            "server) with a DatasetImported event whenever a user's dataset import "
            "from the web app succeeds or fails, e.g. to email them. Repeatable; "
            "hooks run in the order given. Applies to every user for the lifetime "
            f"of the process. Also settable as a comma-separated {ENV}."
        ),
    )


def configure_from_args(args: Any) -> None:
    """Apply what argparse parsed for :data:`FLAG`; raises :class:`HookSpecError`."""
    specs = getattr(args, "on_dataset_imported", None) or []
    if specs:
        configure(specs, source=FLAG)


def configure_from_env(*, warn: Callable[[str], None] | None = None) -> None:
    """Apply :data:`ENV` unless :data:`FLAG` already set the hooks.

    This is what makes the hooks reachable under gunicorn, which never parses
    ``argv``.  An explicit flag wins.  A spec that fails is reported through
    *warn* (default: ``print``) and no hook from the variable is installed.
    """
    if _source is not None:
        return
    raw = os.environ.get(ENV, "")
    specs = [token for token in raw.split(",") if token.strip()]
    if not specs:
        return
    try:
        configure(specs, source=ENV)
    except HookSpecError as exc:
        emit = warn if warn is not None else (lambda msg: print(msg, flush=True))
        emit(f"⚠️  Ignoring {ENV}={raw!r}: {exc}")


def describe() -> str | None:
    """One line naming the hooks in force and what set them, or ``None``."""
    if not _hooks:
        return None
    return f"{', '.join(spec for spec, _ in _hooks)} (from {_source})"


def fire_dataset_imported(event: DatasetImported) -> None:
    """Call every hook in force with *event*; the load pipeline's ``on_finished``."""
    for spec, hook in list(_hooks):
        try:
            hook(event)
        except Exception:
            logger.exception("Dataset-import hook %s failed for %s import %r", spec, event.outcome, event.name)


def clear_import_hooks() -> None:
    """Remove every hook (used by tests between cases)."""
    global _source
    _hooks.clear()
    _source = None
