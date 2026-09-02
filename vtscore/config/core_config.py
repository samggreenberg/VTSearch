"""The :class:`CoreConfig` value object and the app-side builder hook.

This is the seam that lets library code run with or without the Flask app: a
frozen bundle of every knob ``vtscore`` reads, constructed directly by
library-only consumers and snapshotted from ``vtsearch.settings`` by a builder
the app installs at startup.  :mod:`vtscore.config` never imports the app.

**Patch target.** :attr:`_core_config_builder` is module state, so a test that
stubs it must patch ``vtscore.config.core_config._core_config_builder`` - the
package re-exports functions and constants, not this mutable global.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from vtscore.config.runtime import PROJECTION_MIN_DIST, PROJECTION_N_NEIGHBORS

# ---------------------------------------------------------------------------
# CoreConfig: runtime config bundle the (future) ``vtscore`` library consumes
# ---------------------------------------------------------------------------
#
# Today every library-candidate package reaches into ``vtsearch.settings``
# directly for tunables like ``saved_datasets_dir``, ``detectors_dir``,
# ``calibrate_count``, etc.  That couples the library to the app's settings
# layer and makes it impossible to vendor the library as ``vtscore`` (see
# ``docs/architecture.md``, Phase 2).
#
# ``CoreConfig`` is the seam: a frozen value object that bundles every knob
# library code reads.  Follow-up PRs convert each call site to accept (or
# look up) a ``CoreConfig`` instead of importing ``vtsearch.settings``.
# Until those land this class is unused at runtime; the scaffold just
# defines the type so the conversions can happen one file at a time.
#
# The app side will build a fresh ``CoreConfig`` at each request boundary
# via :meth:`CoreConfig.from_settings`; library callers can construct one
# directly with whatever values they want.
#
# The implementation of :meth:`from_settings` is installed by the app via
# :func:`register_core_config_builder`; see ``vtsearch/shim`` for the
# concrete builder that snapshots ``vtsearch.settings``.  This keeps the
# library import-clean: ``vtscore.config`` itself never imports
# ``vtsearch.settings`` (Phase 8 of ``docs/architecture.md``).
# Library-only consumers without an app skip ``from_settings()`` entirely
# and construct ``CoreConfig`` directly.


_core_config_builder: Callable[[str | Path | None], CoreConfig] | None = None


def register_core_config_builder(fn: Callable[[str | Path | None], CoreConfig]) -> None:
    """Install the app-side builder that reads ``vtsearch.settings``.

    The Flask app wires this at startup via
    :func:`vtsearch.shim.register_app_config_builder`.  Once registered,
    :meth:`CoreConfig.from_settings` delegates to *fn*; the library file
    itself stays settings-import-free.
    """
    global _core_config_builder
    _core_config_builder = fn


@dataclass(frozen=True)
class CoreConfig:
    """Runtime configuration bundle the ``vtscore`` library consumes.

    Field set is intentionally narrow; only knobs that library code (loaders,
    detectors, training, embedding) reads.  User-pref concerns like theme or
    grid-icon size are app-tier and stay in ``vtsearch.settings``.
    """

    # Server-tier settings (shared across users, stored in data/settings.json)
    saved_datasets_dir: Path
    detectors_dir: Path
    max_concurrent_dataset_downloads: int
    max_concurrent_dataset_embeddings: int
    autofind_detectors: tuple[str, ...]

    dataset_max_age_days: int | None

    # Per-user settings (stored under each user's data dir)
    calibrate_count: int
    # The user's *explicit* Train/Calibrate split, or ``None`` when unset.
    # ``None`` is resolved at training time to the per-space production
    # default for the detector's embedder (0.3 single-vector / 0.5 patch;
    # issue #3287) - see
    # :func:`vtscore.detectors.training.resolve_calibration_fraction`.
    calibration_fraction: float | None
    enrich_descriptions: bool
    autopilot_goal_diversity: int
    inclusion: int

    # Filesystem root for caches, embeddings, model downloads.  Phase 4 will
    # route every hardcoded ``data/`` path through this field.
    data_dir: Path

    # Auto-Find results exporter (server-tier). When an autodetect run has no
    # explicit ``--exporter``, the CLI falls back to this exporter +
    # field-value map. ``""`` means "no configured exporter" (CLI defaults to
    # ``gui``). Defaulted here so library-only ``CoreConfig(...)`` constructions
    # without the app shim keep working unchanged.
    autofind_exporter: str = ""
    autofind_exporter_field_values: dict[str, dict[str, str]] = field(default_factory=dict)

    # Operator overrides for the Browse projection's UMAP knobs (server-tier
    # ``projection_n_neighbors`` / ``projection_min_dist``).  Mirrored onto the
    # library tier because *both* fit paths — the on-demand route and the
    # ingest-time pre-build, which cannot import ``vtsearch.settings`` — resolve
    # their params through :func:`vtscore.projection.params.resolve_projection_params`.
    # A value equal to the global constant above means "no override", which is
    # what lets ``PROJECTION_DEFAULTS_BY_EMBEDDER`` apply.  Defaulted here so
    # library-only ``CoreConfig(...)`` constructions without the app shim keep
    # working unchanged.
    projection_n_neighbors: int = PROJECTION_N_NEIGHBORS
    projection_min_dist: float = PROJECTION_MIN_DIST

    # Per-media-type opt-in to the generative signpost captioner (image VLM /
    # audio captioner) instead of the default zero-shot tag texts.  ``{}`` (the
    # default) means tags for every type.  Read by
    # :func:`vtscore.projection.signpost_texts.provider_for`.  Defaulted here so
    # library-only ``CoreConfig(...)`` constructions without the app shim keep
    # working unchanged.
    signpost_captioner: dict[str, bool] = field(default_factory=dict)

    # Per-media-type operator-supplied zero-shot tag vocabulary for signpost
    # region names, replacing the built-in AudioSet-527 / OpenImages-600 lists
    # for the whole deployment (the app populates it from the server-tier
    # ``browse_signpost_vocab`` setting, not from a per-user one).  ``{}``
    # (the default) means the shipped vocabulary for every type.  Read by
    # :func:`vtscore.projection.signpost_texts.provider_for`.  Defaulted here so
    # library-only ``CoreConfig(...)`` constructions without the app shim keep
    # working unchanged.
    signpost_vocab: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings_path: str | Path | None = None) -> CoreConfig:
        """Snapshot the current user's ``vtsearch.settings`` into a CoreConfig.

        Called by the Flask app at the request boundary (after auth resolves
        the current user) and by the CLI before kicking off autodetect.  The
        result is a frozen immutable value safe to hand to background
        threads; settings changes during a request will not retroactively
        rewrite a config already in flight.

        When *settings_path* is given, the server-tier settings file path is
        redirected to that location first.  The CLI uses this to point at a
        run-specific settings JSON without each call site importing
        :mod:`vtsearch.settings` directly.

        Implementation note: the body of this classmethod lives in
        ``vtsearch/shim/`` and is installed at app startup via
        :func:`register_core_config_builder`.  Library-only consumers
        without the shim should construct :class:`CoreConfig` directly.
        """
        if _core_config_builder is None:
            raise RuntimeError(
                "CoreConfig.from_settings() requires the app-side builder to be "
                "registered.  The Flask app installs it during startup via "
                "vtsearch.shim.register_app_config_builder().  Library-only "
                "callers should construct CoreConfig(...) directly instead."
            )
        return _core_config_builder(settings_path)
