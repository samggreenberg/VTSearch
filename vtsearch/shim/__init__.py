"""Flask-aware glue between the (library-candidate) ``vtsearch.state`` core
and the Flask request lifecycle.

The library is being split into a Flask-free ``vtscore`` and a Flask-
wrapping ``vtsearch`` app - see ``../../vtscore/docs/architecture.md``.  Any
piece of code that wants to read or override the per-request
``DatasetContext`` / ``DetectorContext`` via ``flask.g`` belongs here
so the core package can stay import-clean.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vtscore.config import DATA_DIR


def _flask_dataset_context_resolver() -> Any:
    """Read the request-scoped dataset context off ``flask.g``.

    Returns ``None`` outside a request context (background threads, CLI,
    library callers) so :func:`vtscore.state.core.get_active_context`
    can fall through to its thread-local / empty-context fallback.

    Also returns ``None`` when Flask's test client is still preserving
    the *previous* request's context for inspection (``with
    app.test_client() as c:`` blocks pop-and-keep the request context
    so the caller can read ``session`` / ``g`` after the response).
    Inside that window we are no longer in the active handler - falling
    back to the thread-local matches what the next real request would
    see (and what production code, which never has a preserved context,
    always saw). The ``before_request`` hook sets
    ``g._vts_in_request_handler = True``; ``teardown_request`` clears
    it back to ``False`` before the test client's preserved window
    opens.

    Raises :class:`vtscore.state.core.DatasetNotLoadedError` if the
    request explicitly named a dataset (``X-Dataset-Id`` header or
    ``?dataset_id=`` query param) that is not loaded. Without this, the
    proxy silently resolved to the empty context and the client saw
    stale data - logical-bug-audit H16.
    """
    from flask import g, has_request_context

    if not has_request_context():
        return None
    if not getattr(g, "_vts_in_request_handler", False):
        return None
    ctx = getattr(g, "_dataset_context", None)
    if ctx is None:
        unloaded = getattr(g, "_unloaded_dataset_id", None)
        if unloaded:
            from vtscore.state.core import DatasetNotLoadedError

            raise DatasetNotLoadedError(unloaded)
    return ctx


def _flask_detector_context_resolver() -> Any:
    """Counterpart of :func:`_flask_dataset_context_resolver` for detectors."""
    from flask import g, has_request_context

    if not has_request_context():
        return None
    if not getattr(g, "_vts_in_request_handler", False):
        return None
    ctx = getattr(g, "_detector_context", None)
    if ctx is None:
        unloaded = getattr(g, "_unloaded_detector_id", None)
        if unloaded:
            from vtscore.state.core import DetectorNotLoadedError

            raise DetectorNotLoadedError(unloaded)
    return ctx


def _flask_in_request_handler_predicate() -> bool:
    """Return True while a Flask route handler is actively running.

    ``has_request_context()`` alone is too coarse: Flask's test client
    preserves the popped request context inside ``with app.test_client()
    as c:`` blocks so the caller can inspect ``session`` / ``g`` after
    the response. Inside that preserved window we are no longer "in a
    request" - code that mutates state (e.g. a test's
    ``app_module.medias.update(saved)`` restore in ``finally``) should
    write to the real context, not refuse via the request-missing
    sentinel.

    The ``before_request`` hook in ``app.py`` sets
    ``g._vts_in_request_handler = True``; ``teardown_request`` clears it
    back to ``False`` before the test client's preserved window opens.
    """
    from flask import g, has_request_context

    if not has_request_context():
        return False
    return bool(getattr(g, "_vts_in_request_handler", False))


def register_flask_context_resolvers() -> None:
    """Install Flask-aware request-context resolvers on ``vtscore.state.core``.

    Called once during Flask app startup.  After this, the ``medias`` /
    ``good_votes`` proxies and the ``get_active_*_context()`` helpers
    pick up whatever the ``before_request`` hook stashes on ``g`` for
    the duration of each request.

    Also installs :func:`flask.has_request_context` as the request-context
    predicate so :func:`vtscore.state.core.get_active_context` returns the
    frozen request-missing sentinel (rather than the global empty
    context) when a request arrives without an ``X-Dataset-Id`` /
    ``X-Detector-Id`` header - see logical-bug-audit H13 / H16.
    """
    from vtscore.state.core import (
        register_dataset_context_resolver,
        register_detector_context_resolver,
        register_request_context_predicate,
    )

    register_dataset_context_resolver(_flask_dataset_context_resolver)
    register_detector_context_resolver(_flask_detector_context_resolver)
    register_request_context_predicate(_flask_in_request_handler_predicate)


def register_app_persistence_hooks() -> None:
    """Wire library-side persistence hooks to the app's ``vtsearch.settings``.

    The library exposes a few "let the app persist this" hook points so it
    doesn't import settings directly (the Settings seam in
    ``../../vtscore/docs/architecture.md``).  This function installs the Flask
    app's settings as the backing store for each hook.
    """
    from vtsearch import settings
    from vtscore.datasets.load_pipeline import register_last_embedder_persistence_hook
    from vtsearch.state import register_setting_persister

    register_last_embedder_persistence_hook(settings.set_last_embedder_for_media_type)

    register_setting_persister("beta", settings.set_beta)
    register_setting_persister("calibrate_count", settings.set_calibrate_count)
    register_setting_persister("calibration_fraction", settings.set_calibration_fraction)


def register_app_achievement_recorders() -> None:
    """Wire the library's achievement events to :mod:`vtsearch.achievements`.

    The achievement counters live in the current user's ``vtsearch.settings``
    file, so the recording functions are app-tier - but the events they
    count (a vote landing, a dataset load completing, a detector import, a
    find run) are raised from inside :mod:`vtscore`.  The library dispatches
    them through :func:`vtscore.host.achievements.record_achievement`; this
    installs the app-side recorders.  Without them the events are no-ops,
    which is what a library-only consumer wants.

    Each recorder imports ``vtsearch.achievements`` lazily so app startup
    doesn't pull the settings stack in through this wiring.
    """
    from vtscore.host.achievements import register_achievement_recorder

    def _record_vote(*args: Any, **kwargs: Any) -> None:
        from vtsearch.achievements import record_vote

        record_vote(*args, **kwargs)

    def _record_dataset_load(*args: Any, **kwargs: Any) -> None:
        from vtsearch.achievements import record_dataset_load

        record_dataset_load(*args, **kwargs)

    def _record_detector_import(*args: Any, **kwargs: Any) -> None:
        from vtsearch.achievements import record_detector_import

        record_detector_import(*args, **kwargs)

    def _record_find(*args: Any, **kwargs: Any) -> None:
        from vtsearch.achievements import record_find

        record_find(*args, **kwargs)

    register_achievement_recorder("vote", _record_vote)
    register_achievement_recorder("dataset_load", _record_dataset_load)
    register_achievement_recorder("detector_import", _record_detector_import)
    register_achievement_recorder("find", _record_find)


def register_app_setting_getters() -> None:
    """Answer every :func:`vtscore.config.get_setting` knob from ``vtsearch.settings``.

    Called once at Flask app startup.  Each getter reads its one setting live,
    for the current user, so library code pays for the knob it asks about and
    no other.  Also installs ``settings.set_settings_path`` as the library's
    settings-path redirect, which the CLI's ``--settings`` goes through.

    Every getter looks the settings function up when it is called, not here,
    so a test that patches one on ``vtsearch.settings`` is seen.
    """
    from vtsearch import settings
    from vtscore.config import register_setting_getter, register_settings_path_redirect

    getters: dict[str, Callable[[], Any]] = {
        "saved_datasets_dir": lambda: settings.get_saved_datasets_dir(),
        "detectors_dir": lambda: settings.get_detectors_dir(),
        "max_concurrent_dataset_downloads": lambda: settings.get_max_concurrent_dataset_downloads(),
        "max_concurrent_dataset_embeddings": lambda: settings.get_max_concurrent_dataset_embeddings(),
        "autofind_detectors": lambda: tuple(settings.get_autofind_detectors()),
        "dataset_max_age_days": lambda: settings.get_effective_dataset_max_age_days(),
        "calibrate_count": lambda: settings.get_calibrate_count(),
        "calibration_fraction": lambda: settings.get_calibration_fraction(),
        "enrich_descriptions": lambda: settings.get_enrich_descriptions(),
        "autopilot_goal_diversity": lambda: settings.get_autopilot_goal_diversity(),
        "beta": lambda: settings.get_beta(),
        "data_dir": lambda: DATA_DIR,
        "autofind_exporter": lambda: settings.get_autofind_exporter(),
        "autofind_exporter_field_values": lambda: {
            name: dict(vals) for name, vals in settings.get_autofind_exporter_field_values().items()
        },
        "autofind_cli_delete_dataset": lambda: settings.get_autofind_cli_delete_dataset(),
        "projection_n_neighbors": lambda: settings.get_projection_n_neighbors(),
        "projection_min_dist": lambda: settings.get_projection_min_dist(),
        "signpost_captioner": lambda: dict(settings.get_browse_signpost_captioner()),
        "signpost_vocab": lambda: {mt: list(terms) for mt, terms in settings.get_browse_signpost_vocab().items()},
        "default_labelset_source": lambda: settings.get_default_labelset_source(),
        "labelset_source_push_interval": lambda: settings.get_labelset_source_push_interval_seconds(),
    }
    for name, getter in getters.items():
        register_setting_getter(name, getter)
    register_settings_path_redirect(lambda path: settings.set_settings_path(path))


def register_app_plugin_families() -> None:
    """Register app-only plugin families with :mod:`vtscore.plugins.inventory`.

    The settings importers / exporters / sources live in
    :mod:`vtsearch.settings_io`, which is app-tier (it round-trips
    ``vtsearch.settings`` to disk).  Registering them here keeps
    ``vtscore.plugins.inventory`` free of any ``vtsearch.settings_io``
    import - see the Plugin discovery seam in
    ``../../vtscore/docs/architecture.md``.

    Called once at Flask app startup, before the argparse parser is built
    so ``--list-settings-importers`` and friends get their shortcut flags.
    """
    from vtscore.plugins.inventory import FamilyProvider, register_plugin_family

    def _settings_importers():
        from vtsearch.settings_io.importers import list_settings_importers

        return list_settings_importers()

    def _settings_exporters():
        from vtsearch.settings_io.exporters import list_settings_exporters

        return list_settings_exporters()

    def _settings_sources():
        from vtsearch.settings_io.sources import list_settings_sources

        return list_settings_sources()

    register_plugin_family(FamilyProvider("settings_importers", "Settings importers", _settings_importers))
    register_plugin_family(FamilyProvider("settings_exporters", "Settings exporters", _settings_exporters))
    register_plugin_family(FamilyProvider("settings_sources", "Settings sources", _settings_sources))


__all__ = [
    "register_app_achievement_recorders",
    "register_app_persistence_hooks",
    "register_app_plugin_families",
    "register_app_setting_getters",
    "register_flask_context_resolvers",
]
