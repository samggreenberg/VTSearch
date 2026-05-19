"""Flask-aware glue between the (library-candidate) ``vtsearch.state`` core
and the Flask request lifecycle.

The library is being split into a Flask-free ``vtscore`` and a Flask-
wrapping ``vtsearch`` app — see ``docs/plans/extract-library.md``.  Any
piece of code that wants to read or override the per-request
``DatasetContext`` / ``DetectorContext`` via ``flask.g`` belongs here
so the core package can stay import-clean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtsearch.config import DATA_DIR, CoreConfig


def _flask_dataset_context_resolver() -> Any:
    """Read the request-scoped dataset context off ``flask.g``.

    Returns ``None`` outside a request context (background threads, CLI,
    library callers) so :func:`vtsearch.state.core.get_active_context`
    can fall through to its thread-local / empty-context fallback.
    """
    from flask import g, has_request_context

    if has_request_context():
        return getattr(g, "_dataset_context", None)
    return None


def _flask_detector_context_resolver() -> Any:
    """Counterpart of :func:`_flask_dataset_context_resolver` for detectors."""
    from flask import g, has_request_context

    if has_request_context():
        return getattr(g, "_detector_context", None)
    return None


def register_flask_context_resolvers() -> None:
    """Install Flask-aware request-context resolvers on ``vtsearch.state.core``.

    Called once during Flask app startup.  After this, the ``medias`` /
    ``good_votes`` proxies and the ``get_active_*_context()`` helpers
    pick up whatever the ``before_request`` hook stashes on ``g`` for
    the duration of each request.
    """
    from vtsearch.state.core import (
        register_dataset_context_resolver,
        register_detector_context_resolver,
    )

    register_dataset_context_resolver(_flask_dataset_context_resolver)
    register_detector_context_resolver(_flask_detector_context_resolver)


def register_app_persistence_hooks() -> None:
    """Wire library-side persistence hooks to the app's ``vtsearch.settings``.

    The library exposes a few "let the app persist this" hook points so it
    doesn't import settings directly (see Phase 2 of
    ``docs/plans/extract-library.md``).  This function installs the Flask
    app's settings as the backing store for each hook.
    """
    from vtsearch import settings
    from vtsearch.datasets.load_pipeline import register_last_embedder_persistence_hook
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
    ``vtsearch.settings`` — see Phase 8 of
    ``docs/plans/extract-library.md``.

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
    from vtsearch.config import register_core_config_builder

    register_core_config_builder(build_core_config)


__all__ = [
    "build_core_config",
    "register_app_config_builder",
    "register_app_persistence_hooks",
    "register_flask_context_resolvers",
]
