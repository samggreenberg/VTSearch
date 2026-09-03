"""Snapshot and restore the **host seams**: the callbacks an app installs into the library.

``vtscore`` runs without ``vtsearch`` and without Flask, so wherever the
library needs something only a host application can answer - "is this a
Flask request?", "which dataset did this request name?", "persist this
user preference" - it holds a module-level slot with a library default
and exposes a ``register_*`` function for the host to fill it.
``vtsearch/shim/`` fills all of them at app startup (see ``app.py``).

Each seam is deliberately declared next to the code that *calls* it, so
this module does not own them; it owns the one thing no single seam can
know on its own: **the complete list**.  That list is what tests need.
Every slot is a process global shared by thousands of tests in a handful
of long-lived xdist workers, so a test that installs a seam leaks it into
every test that follows.  Before this module, four test files each
hand-rolled their own save/restore against the private globals
(``achievements_hooks._recorders``, ``state.core._request_context_predicate``,
…) and a fifth forgot to, leaking a settings persister for the rest of
the worker's life.

:func:`capture_host_seams` / :func:`restore_host_seams` replace that
boilerplate.  Note they are **snapshot/restore, not reset-to-default**:
the app tier runs with the real ``vtsearch`` wiring installed at import
time, so clearing the slots would strip the very seams those tests
exercise.  The suites capture once at startup - after their conftest has
finished bootstrapping - and restore that snapshot before each test.

The seams
---------

======================================  ============================================
Seam                                    Declared in
======================================  ============================================
``request_context_predicate``           :mod:`vtscore.state.core`
``dataset_context_resolver``            :mod:`vtscore.state.core`
``detector_context_resolver``           :mod:`vtscore.state.core`
``request_user_resolver``               :mod:`vtscore.state.current_user`
``core_config_builder``                 :mod:`vtscore.config.core_config`
``last_embedder_persistence_hook``      :mod:`vtscore.datasets.load_pipeline`
``setting_persisters`` (keyed)          :mod:`vtscore.state`
``achievement_recorders`` (keyed)       :mod:`vtscore.achievements_hooks`
======================================  ============================================

``register_plugin_family`` is **not** a host seam and is deliberately
absent.  It looks like one - the shim calls it at startup - but the
library registers its own families too (``vtscore/plugins/inventory.py``
installs the built-ins at import time), so it is a plugin extension point
with the app as one registrant among several.  Snapshotting it here would
be harmless; *restoring* it would be a live hazard, since a restore
running before the built-ins finish registering would drop them.

There is no ``install(**seams)`` counterpart on purpose.  The eight
``register_*`` functions are public, documented ``vtscore`` surface that
out-of-tree code may call, so they stay the way a host installs a seam;
an alternative spelling of the same thing would be one more name to keep
working forever, for no behaviour nobody has.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HostSeams:
    """An immutable snapshot of every installed host seam.

    The two keyed seams are copied into plain ``dict``s at capture time, so
    a later mutation of the live registry cannot reach back into a snapshot
    already taken.
    """

    request_context_predicate: Callable[[], bool]
    dataset_context_resolver: Callable[[], Any]
    detector_context_resolver: Callable[[], Any]
    request_user_resolver: Callable[[], str | None]
    core_config_builder: Callable[..., Any] | None
    last_embedder_persistence_hook: Callable[[str, str], None] | None
    setting_persisters: dict[str, Callable[[Any], None]]
    achievement_recorders: dict[str, Callable[..., Any]]


def capture_host_seams() -> HostSeams:
    """Snapshot the currently installed host seams.

    Call this once the host has finished installing its seams (for the app
    tier, after ``app.py`` has run its ``register_app_*`` calls); the result
    is what :func:`restore_host_seams` puts back.
    """
    import vtscore.achievements_hooks as achievements_hooks  # noqa: PLC0415
    import vtscore.config.core_config as core_config  # noqa: PLC0415
    import vtscore.datasets.load_pipeline as load_pipeline  # noqa: PLC0415
    import vtscore.state as state  # noqa: PLC0415
    import vtscore.state.core as state_core  # noqa: PLC0415
    import vtscore.state.current_user as current_user  # noqa: PLC0415

    return HostSeams(
        request_context_predicate=state_core._request_context_predicate,
        dataset_context_resolver=state_core._dataset_context_resolver,
        detector_context_resolver=state_core._detector_context_resolver,
        request_user_resolver=current_user._request_user_resolver,
        core_config_builder=core_config._core_config_builder,
        last_embedder_persistence_hook=load_pipeline._last_embedder_persistence_hook,
        setting_persisters=dict(state._setting_persisters),
        achievement_recorders=dict(achievements_hooks._recorders),
    )


def restore_host_seams(seams: HostSeams) -> None:
    """Put every host seam back to the state *seams* recorded.

    The keyed registries are restored in place (cleared and refilled) rather
    than rebound, because :mod:`vtscore.state` and
    :mod:`vtscore.achievements_hooks` read their module globals directly and
    a test may hold a reference to the live dict.
    """
    import vtscore.achievements_hooks as achievements_hooks  # noqa: PLC0415
    import vtscore.config.core_config as core_config  # noqa: PLC0415
    import vtscore.datasets.load_pipeline as load_pipeline  # noqa: PLC0415
    import vtscore.state as state  # noqa: PLC0415
    import vtscore.state.core as state_core  # noqa: PLC0415
    import vtscore.state.current_user as current_user  # noqa: PLC0415

    state_core._request_context_predicate = seams.request_context_predicate
    state_core._dataset_context_resolver = seams.dataset_context_resolver
    state_core._detector_context_resolver = seams.detector_context_resolver
    current_user._request_user_resolver = seams.request_user_resolver
    core_config._core_config_builder = seams.core_config_builder
    load_pipeline._last_embedder_persistence_hook = seams.last_embedder_persistence_hook

    state._setting_persisters.clear()
    state._setting_persisters.update(seams.setting_persisters)
    achievements_hooks._recorders.clear()
    achievements_hooks._recorders.update(seams.achievement_recorders)


__all__ = ["HostSeams", "capture_host_seams", "restore_host_seams"]
