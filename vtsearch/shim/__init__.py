"""Flask-aware glue between the (library-candidate) ``vtsearch.state`` core
and the Flask request lifecycle.

The library is being split into a Flask-free ``vtscore`` and a Flask-
wrapping ``vtsearch`` app - see ``../../vtscore/docs/architecture.md``.  Any
piece of code that wants to read or override the per-request
``DatasetContext`` / ``DetectorContext`` via ``flask.g`` belongs here
so the core package can stay import-clean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtscore.config import DATA_DIR, CoreConfig


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
    ``X-Detector-Id`` header - see ``docs/plans/logical-bug-audit.md``
    H13 / H16.
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
    doesn't import settings directly (see Phase 2 of
    ``../../vtscore/docs/architecture.md``).  This function installs the Flask
    app's settings as the backing store for each hook.
    """
    from vtsearch import settings
    from vtscore.datasets.load_pipeline import register_last_embedder_persistence_hook
    from vtsearch.state import register_setting_persister

    register_last_embedder_persistence_hook(settings.set_last_embedder_for_media_type)

    register_setting_persister("inclusion", settings.set_inclusion)
    register_setting_persister("calibrate_count", settings.set_calibrate_count)
    register_setting_persister("calibration_fraction", settings.set_calibration_fraction)
    register_setting_persister("safe_thresholds", settings.set_safe_thresholds)


def build_core_config(settings_path: str | Path | None = None) -> CoreConfig:
    """Snapshot ``vtsearch.settings`` into a :class:`CoreConfig`.

    The app-side implementation of :meth:`CoreConfig.from_settings`.  Lives
    here so the library file ``vtsearch/config.py`` never imports
    ``vtsearch.settings`` - see Phase 8 of
    ``../../vtscore/docs/architecture.md``.

    When *settings_path* is given, the server-tier settings file path is
    redirected to that location before reading.  The CLI uses this to
    point at a run-specific settings JSON.
    """
    from vtsearch import settings as _settings

    if settings_path is not None:
        _settings.set_settings_path(settings_path)

    return CoreConfig(
        saved_datasets_dir=_settings.get_saved_datasets_dir(),
        detectors_dir=_settings.get_detectors_dir(),
        max_concurrent_dataset_downloads=_settings.get_max_concurrent_dataset_downloads(),
        max_concurrent_dataset_embeddings=_settings.get_max_concurrent_dataset_embeddings(),
        autorun_detectors=tuple(_settings.get_autorun_detectors()),
        dataset_max_age_days=_settings.get_dataset_max_age_days(),
        safe_thresholds=_settings.get_safe_thresholds(),
        calibrate_count=_settings.get_calibrate_count(),
        calibration_fraction=_settings.get_calibration_fraction(),
        enrich_descriptions=_settings.get_enrich_descriptions(),
        autopilot_goal_diversity=_settings.get_autopilot_goal_diversity(),
        inclusion=_settings.get_inclusion(),
        data_dir=DATA_DIR,
    )


def register_app_config_builder() -> None:
    """Install :func:`build_core_config` as the backing for
    :meth:`CoreConfig.from_settings`.

    Called once at Flask app startup so library-candidate callers can keep
    using ``CoreConfig.from_settings()`` while the bridge to
    ``vtsearch.settings`` lives entirely in this shim package.
    """
    from vtscore.config import register_core_config_builder

    register_core_config_builder(build_core_config)


def register_app_plugin_families() -> None:
    """Register app-only plugin families with :mod:`vtscore.plugins.inventory`.

    The settings importers / exporters / sources live in
    :mod:`vtsearch.settings_io`, which is app-tier (it round-trips
    ``vtsearch.settings`` to disk).  Registering them here keeps
    ``vtscore.plugins.inventory`` free of any ``vtsearch.settings_io``
    import - see ``../../vtscore/docs/architecture.md`` Phase 5.

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
    "build_core_config",
    "register_app_config_builder",
    "register_app_persistence_hooks",
    "register_app_plugin_families",
    "register_flask_context_resolvers",
]
