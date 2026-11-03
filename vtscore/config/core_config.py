"""The :class:`CoreConfig` value object and the setting getters library code reads.

This is the seam that lets library code run with or without the Flask app.
Library code reads one knob at a time with :func:`get_setting`, which answers
from, in order:

1. the getter a host app registered for that knob with
   :func:`register_setting_getter` - the VTSearch app registers one per knob at
   startup, each reading ``vtsearch.settings`` live;
2. the :class:`CoreConfig` a library-only consumer's builder returns
   (:func:`register_core_config_builder`);
3. the knob's :class:`CoreConfig` field default, or :class:`RuntimeError` for a
   required field, which has none.

:mod:`vtscore.config` never imports the app.

**Patch targets.** :attr:`_core_config_builder`, :attr:`_setting_getters` and
:attr:`_settings_path_redirect` are module state, so a test that stubs one must
patch ``vtscore.config.core_config`` - the package re-exports functions and
constants, not these mutable globals.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

from vtscore.config.runtime import (
    DEFAULT_BETA,
    PROJECTION_MIN_DIST,
    PROJECTION_N_NEIGHBORS,
)

#: Default shortest gap, in seconds, between two pushes of one detector's
#: labels to its labelset source (#4913).  The app's server setting
#: ``labelset_source_push_interval_seconds`` overrides it.
DEFAULT_LABELSET_SOURCE_PUSH_INTERVAL: float = 60.0


def _retired_inclusion(where: str, value: float | None, *, stacklevel: int = 3) -> None:
    """Refuse a non-zero Inclusion passed to a retired name, and warn about a zero one.

    Inclusion is no longer a user preference (#4269): the balance is the
    operating point (#4413), and a detector with no balance draws its line at
    the Inclusion 0 cut.  The names that used to set it (*where*) stay importable
    so an out-of-tree caller keeps working, but only at 0 - the one value that
    still means what it did.  Any other value is refused rather than ignored,
    because a knob that silently stops moving the line is worse than an error.
    ``None`` is "not given" and passes silently.  *stacklevel* points the
    warning at the retired name's caller: the default fits a function that
    calls this directly.
    """
    if value is None:
        return
    if value != 0:
        raise ValueError(
            f"{where}: Inclusion is retired as a user preference and is fixed at 0; got {value!r}. "
            "Set a balance instead (beta= / vtscore.state.set_beta)."
        )
    warnings.warn(
        f"{where} is deprecated: Inclusion is retired as a user preference and is always 0. "
        "Set a balance instead (beta= / vtscore.state.set_beta).",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


_core_config_builder: Callable[[str | Path | None], CoreConfig] | None = None

#: The host's per-knob getters, keyed by :class:`CoreConfig` field name.
_setting_getters: dict[str, Callable[[], Any]] = {}

#: The host's hook that points its server-tier settings at another file.
_settings_path_redirect: Callable[[str | Path], None] | None = None


def register_core_config_builder(fn: Callable[[str | Path | None], CoreConfig]) -> None:
    """Install a builder returning the :class:`CoreConfig` the library reads its knobs from.

    The way a library-only consumer configures ``vtscore``: once registered,
    :func:`get_setting` answers every knob no host getter covers from *fn*'s
    result, and :meth:`CoreConfig.from_settings` returns it.  *fn* is called
    with one positional argument, a settings path or ``None``.  The VTSearch
    app registers per-knob getters instead (:func:`register_setting_getter`).
    """
    global _core_config_builder
    _core_config_builder = fn


@dataclass(frozen=True)
class CoreConfig:
    """Every knob the ``vtscore`` library reads, as one frozen value.

    A library-only consumer constructs one and registers a builder returning it
    (:func:`register_core_config_builder`); library code reads a single knob
    at a time with :func:`get_setting`, so its field names are the setting
    names.  Field set is intentionally narrow; only knobs that library code
    (loaders, detectors, training, embedding) reads.  User-pref concerns like
    theme or grid-icon size are app-tier and stay in ``vtsearch.settings``.
    """

    # Server-tier settings (shared across users, stored in data/settings.json)
    saved_datasets_dir: Path
    detectors_dir: Path
    max_concurrent_dataset_downloads: int
    max_concurrent_dataset_embeddings: int
    # Per-user (the current user's AutoFind detector names), despite sitting
    # in the server-tier block: moving it would reorder the positional fields.
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

    # Filesystem root for caches, embeddings, model downloads.  The app
    # snapshots :data:`vtscore.config.DATA_DIR` into it (the Filesystem seam
    # in ``vtscore/docs/architecture.md``).
    data_dir: Path

    # AutoFind results exporter (per-user). When an autodetect run has no
    # explicit ``--exporter``, the CLI falls back to this exporter +
    # field-value map. ``""`` means "no configured exporter" (CLI defaults to
    # ``gui``). Defaulted here so library-only ``CoreConfig(...)`` constructions
    # without the app shim keep working unchanged.
    autofind_exporter: str = ""
    autofind_exporter_field_values: dict[str, dict[str, str]] = field(default_factory=dict)

    # Whether a saving CLI autodetect run deletes the dataset it imported once
    # its detectors have run (#4674; per-user ``autofind_cli_delete_dataset``).
    # A run that detected nothing keeps its dataset either way.  Defaulted off
    # so library-only ``CoreConfig(...)`` constructions keep what they save.
    autofind_cli_delete_dataset: bool = False

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

    # Deployment-wide labelset source every detector syncs its labels to unless
    # it has a binding of its own (#4913; the app populates it from the
    # server-tier ``default_labelset_source``).  ``{"source_name": ...,
    # "field_values": {...}}``; each non-empty value is fixed by the admin.
    # Read by :func:`vtscore.labels.binding.resolve_labelset_source`.  ``None``
    # (the default) means no deployment-wide source.
    default_labelset_source: dict[str, Any] | None = None

    # Shortest gap, in seconds, between two pushes of one detector's labels to
    # its labelset source (server-tier ``labelset_source_push_interval_seconds``).
    # Read by :mod:`vtscore.labels.sync`.  Defaulted here so library-only
    # ``CoreConfig(...)`` constructions keep working unchanged.
    labelset_source_push_interval: float = DEFAULT_LABELSET_SOURCE_PUSH_INTERVAL

    # Deprecated and ignored (#4667).  It was the operator switch that hid the
    # remaining-time estimate on ingest progress bars (#4233); ingest bars now
    # never publish one, for every deployment.  Kept only so an out-of-tree
    # ``CoreConfig(..., hide_ingest_eta=...)`` construction is not a
    # ``TypeError``; nothing reads it.
    hide_ingest_eta: bool = False

    # The balance (#4413): F-beta's beta, the preference the line is drawn at,
    # or ``None`` for no balance - the line is then the Inclusion 0 cut.  Seeds
    # each detector's own balance on first read (``vtscore.state.get_beta``).
    # The app always sets one; ``None`` survives for library callers.
    # Defaulted here so library-only ``CoreConfig(...)`` constructions keep
    # working.
    beta: float | None = DEFAULT_BETA

    # Deprecated (#4269): Inclusion is no longer a user preference, so nothing
    # reads this.  ``None`` is "not given"; ``0`` is accepted with a
    # ``DeprecationWarning`` and any other value is refused (see
    # :func:`_retired_inclusion`).  Kept so an existing ``CoreConfig(inclusion=0)``
    # still constructs.
    inclusion: int | None = None

    def __post_init__(self) -> None:
        _retired_inclusion("CoreConfig(inclusion=...)", self.inclusion, stacklevel=4)

    @classmethod
    def from_settings(cls, settings_path: str | Path | None = None) -> CoreConfig:
        """Every knob's current value, bundled into one :class:`CoreConfig`.

        Library code does not call this: it reads the one knob it needs with
        :func:`get_setting`.  This stays for a caller that wants the whole
        bundle at once, and delegates.  A builder that is all there is answers
        it outright, with *settings_path* passed through.  Otherwise
        *settings_path*, when given, is applied with
        :func:`redirect_settings_path`, and each field is read with
        :func:`get_setting`, so the bundle agrees with what library code reads;
        with nothing registered, the first required field raises
        :class:`RuntimeError`.

        The deprecated ``hide_ingest_eta`` and ``inclusion`` fields are not
        settings, so an assembled config leaves them at their defaults.
        """
        if _core_config_builder is not None and not _setting_getters:
            return _core_config_builder(settings_path)
        if settings_path is not None:
            redirect_settings_path(settings_path)
        return cls(**{name: get_setting(name) for name in sorted(_SETTING_NAMES)})


#: Fields kept only so an out-of-tree construction still works; nothing reads them.
_RETIRED_FIELDS = frozenset({"hide_ingest_eta", "inclusion"})

#: Every knob :func:`get_setting` answers: the :class:`CoreConfig` fields, less the retired ones.
_SETTING_NAMES = frozenset(f.name for f in fields(CoreConfig)) - _RETIRED_FIELDS


def _check_setting_name(name: str) -> None:
    if name not in _SETTING_NAMES:
        raise ValueError(f"{name!r} is not a vtscore setting; expected one of {sorted(_SETTING_NAMES)}")


def _field_default(name: str) -> Any:
    """*name*'s :class:`CoreConfig` default (a fresh one for a mutable field), else :data:`MISSING`."""
    f = CoreConfig.__dataclass_fields__[name]
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return MISSING


def register_setting_getter(name: str, fn: Callable[[], Any]) -> None:
    """Install the host's getter for the knob *name*, a :class:`CoreConfig` field.

    :func:`get_setting` calls *fn* with no arguments on every read of *name*,
    so it should be cheap and answer for whoever is asking (the request's user,
    in a multi-user host).  It wins over a registered builder.  The VTSearch app
    registers one per knob at startup
    (``vtsearch.seam_wiring.register_app_setting_getters``).

    Raises :class:`ValueError` for a name that is not a setting.
    """
    _check_setting_name(name)
    _setting_getters[name] = fn


def get_setting(name: str) -> Any:
    """The current value of the knob *name*, one :class:`CoreConfig` field.

    The read library code makes when it needs a knob: the host's getter for
    *name* if one is registered, else that field of the registered builder's
    :class:`CoreConfig`, else the field's default.  A required field with
    none of the three raises :class:`RuntimeError`, rather than guessing
    where a library-only consumer keeps its detectors.

    Raises :class:`ValueError` for a name that is not a setting.
    """
    _check_setting_name(name)
    getter = _setting_getters.get(name)
    if getter is not None:
        return getter()
    if _core_config_builder is not None:
        return getattr(_core_config_builder(None), name)
    default = _field_default(name)
    if default is MISSING:
        raise RuntimeError(
            f"No value for the setting {name!r}: no getter is registered for it "
            "(register_setting_getter) and no CoreConfig builder is installed "
            "(register_core_config_builder).  The VTSearch app registers its getters "
            "at startup; a library-only consumer registers a builder that returns "
            "its own CoreConfig(...)."
        )
    return default


def register_settings_path_redirect(fn: Callable[[str | Path], None]) -> None:
    """Install the host's hook that points its server-tier settings at another file.

    :func:`redirect_settings_path` calls it.  The VTSearch app installs
    ``vtsearch.settings.set_settings_path``.
    """
    global _settings_path_redirect
    _settings_path_redirect = fn


def redirect_settings_path(settings_path: str | Path) -> None:
    """Read the host's server-tier settings from *settings_path* from now on.

    The CLI calls this for ``--settings``, before anything reads a knob, so
    every :func:`get_setting` after it answers from that file.  With no
    redirect hook registered, a registered builder is called with the path
    instead (the redirect its contract asks of it), and with neither there is
    no settings file to redirect.
    """
    if _settings_path_redirect is not None:
        _settings_path_redirect(settings_path)
    elif _core_config_builder is not None:
        _core_config_builder(settings_path)
