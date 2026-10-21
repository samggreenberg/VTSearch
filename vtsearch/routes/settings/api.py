"""Flask routes for the Settings API.

Endpoints
---------
GET  /api/settings
    Return all persisted settings for the current user (merged
    server-tier + per-user-tier).

PUT  /api/settings
    Update one or more settings fields.  Only supplied keys are changed.

GET  /api/settings/defaults
    Return the default values for all settings.

This module was the **OpenAPI pilot**: schemas in
``vtsearch/schemas/settings.py`` are the source of truth, validation
runs through marshmallow, and the OpenAPI spec is generated
automatically by flask-smorest.
"""

from __future__ import annotations

from typing import Any, Callable

from flask_smorest import Blueprint, abort
from marshmallow import fields

from vtsearch import admin_overrides, import_hooks, settings
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema
from vtsearch.state import (
    set_calibrate_count as _state_set_calibrate_count,
    set_calibration_fraction as _state_set_calibration_fraction,
    set_beta as _state_set_beta,
)

settings_bp = Blueprint(
    "settings",
    __name__,
    description="Read and modify persisted user / server settings.",
)


# The scalar-setter dispatch table (``_SCALAR_SETTERS``) is derived from
# ``SettingsUpdateSchema`` further down, once ``_CUSTOM_SETTERS`` /
# ``_READ_ONLY_KEYS`` / ``_NON_PUT_KEYS`` are defined. See
# ``_build_scalar_setters``.


#: Schema fields declared as dicts (``browse_*``, the per-media-type
#: embedder maps, ``import_defaults_by_media_type`` …). A persisted or
#: hand-edited settings file from an older build can carry a stale scalar
#: where one of these is now expected (e.g. a pre-per-media-type
#: ``browse_colormap: "auto"``). Marshmallow's ``Dict`` field calls
#: ``.items()`` while dumping, so a bare string there would 500 the entire
#: settings endpoint. Derived from the schema so it stays in sync as
#: fields are added.
_DICT_FIELD_NAMES = frozenset(
    name for name, field in AppSettingsSchema().fields.items() if isinstance(field, fields.Dict)
)


def _coerce_dict_fields(data: dict) -> dict:
    """Replace any non-dict value in a schema dict field with ``{}``.

    Mirrors how the per-side getters (``grid_icon_size`` etc.) already coerce a
    junk persisted value back to its default on read: a stale scalar left
    over from an older settings file resolves to "never set" rather than
    crashing the serializer. Mutates and returns *data*.
    """
    for name in _DICT_FIELD_NAMES:
        if name in data and not isinstance(data[name], dict):
            data[name] = {}
    return data


def _with_effective(data: dict) -> dict:
    """Overlay the resolver-computed (read-only) views onto *data*.

    Every process-level **admin override** -- the solo mediaType lock, the
    per-mediaType solo-embedder locks, the plugin hide list, the dataset
    retention window, the support email, the Semantic-only lock -- publishes
    the value *actually in force* here, so the frontend never has to know
    whether a restriction came from a CLI flag, an env var, or the settings
    file. The set is not spelled out: each knob declares its own
    ``effective_key`` (and any JSON coercion) in
    :mod:`vtsearch.admin_overrides`, and this loops the registry, so a new
    override is surfaced without touching this route.

    Shared by the GET and PUT responses. Stale dict fields are coerced to
    ``{}`` first so a corrupt persisted value can't 500 the endpoint (see
    :func:`_coerce_dict_fields`).
    """
    _coerce_dict_fields(data)
    for override in admin_overrides.OVERRIDES.values():
        data[override.effective_key] = override.dump(settings.get_effective_override(override.name))
    # Not an admin override (see :mod:`vtsearch.import_hooks`), so set here:
    # only whether hooks exist, never what they are.
    data["import_hooks_configured"] = import_hooks.configured()
    return data


@settings_bp.route("/api/settings", methods=["GET"])
@settings_bp.response(200, AppSettingsSchema)
def get_settings():
    """Return the merged server + per-user settings dict.

    Augments the persisted dict with the resolver-computed read-only views
    (the effective solo mediaType / embedder locks, effective hidden
    plugins) via :func:`_with_effective`. The frontend reads those keys
    when deciding whether to hide pickers.
    """
    return _with_effective(settings.get_all())


#: Keys whose value is computed on read and silently ignored on write
#: (the raw fields they're derived from go through their own dispatch
#: entry below).
_READ_ONLY_KEYS = frozenset(
    {
        "effective_solo_embedder_per_media_type",
    }
)


#: Keys whose update needs more than the field validator: a check against a
#: live registry, a cleaning pass, or a side effect beside the write (the
#: achievement wipe). Each entry pairs a validator -- which raises
#: ``ValueError`` (a 400 here) and returns the cleaned value -- with the writer
#: that persists it, so a whole PUT body can be validated before any of it is
#: committed (see :func:`update_settings`). The table is the settings module's
#: own, shared with a settings import and a settings-source sync, so neither
#: can store a value this route would refuse (#4965).
_CUSTOM_SETTERS = settings.CUSTOM_USER_WRITES


# The training-relevant settings route through ``vtsearch.state`` rather
# than ``vtsearch.settings`` so the state setter's side-effect fires and the
# stale derived artefacts on every loaded detector context are dropped;
# otherwise ``/api/find-label`` / ``/api/find`` / ``/api/auto-detect`` would
# keep scoring with a threshold computed under the prior setting (M7). These
# override the plain ``settings.set_<key>`` accessor the table would
# otherwise pick up.
#
# ``calibrate_count`` / ``calibration_fraction`` change what a *retrain*
# produces, so their setters invalidate the cached head
# (``invalidate_loaded_detector_models``). ``beta`` (the balance, #4413) is a
# pure cutoff knob, so its setter keeps the head and only re-derives the
# threshold from the cached ranking (``recompute_detector_thresholds``). All
# three persist back to :mod:`vtsearch.settings` through the shim's
# setting-persister hook (:func:`vtsearch.shim.register_app_persistence_hooks`),
# which is why the state-tier setter is the whole write and no
# ``settings.set_<key>`` call belongs beside it.
#
# Routing ``beta`` here rather than through :data:`_CUSTOM_SETTERS` is what
# gives it a *single* clamp: :func:`_plan_one_key` validates every entry in
# this table with ``settings.validate_setting``, which resolves to the
# autogenerated ``settings.validate_beta`` and therefore to the ``[0.25, 4]``
# bound declared once on ``UserSettings.beta``. A bespoke validator here would
# be a second copy of that range (issue #3416).
#
# Only this route dispatches here. A settings import or a settings-source sync
# writes these keys through the plain setters and then hands the change to
# loaded detectors itself (``settings._refresh_loaded_detectors``): ``set_beta``
# also makes the value the *active* detector's own balance, which is a balance
# pick, and a bulk import of the user's settings is not one.
_STATE_TIER_SETTERS: dict[str, Callable[[Any], Any]] = {
    "calibrate_count": _state_set_calibrate_count,
    "calibration_fraction": _state_set_calibration_fraction,
    "beta": _state_set_beta,
}

#: Keys that ``SettingsUpdateSchema`` accepts and that have a
#: ``settings.set_<key>`` accessor, yet are intentionally NOT settable via
#: ``PUT /api/settings`` because they are persisted through other paths:
#:
#: * ``last_embedder_per_media_type`` - written by the load-pipeline
#:   "last embedder" hook (see :mod:`vtsearch.shim`), keyed per media type,
#:   not by the settings form.
#:
#: (``dataset_max_age_days`` is a server-tier retention policy set via the
#: ``--dataset-max-age-days`` CLI flag / settings file, not via PUT, and
#: ``solo_media_type`` is a server-tier restriction set via
#: ``--solo-media-type`` / the settings file. Both are ``dump_only`` in the
#: schema, so neither is "loadable" and neither needs an entry here.)
#:
#: They are listed here so the drift-guard test
#: (``tests/api/test_settings_dispatch.py``) treats their absence from the
#: dispatch tables as deliberate rather than a missed wiring.
_NON_PUT_KEYS = frozenset({"last_embedder_per_media_type"})


def _build_scalar_setters() -> dict[str, Callable[[Any], Any]]:
    """Derive the scalar-setter dispatch table from ``SettingsUpdateSchema``.

    Every loadable (non-``dump_only``) schema field whose update is a plain
    ``settings.set_<key>`` call is wired automatically, so adding a simple
    setting no longer means hand-editing a parallel dispatch table. The
    curated exceptions stay explicit: :data:`_STATE_TIER_SETTERS` overrides
    the plain accessor for the three training settings, and keys in
    :data:`_CUSTOM_SETTERS` (bespoke validation), :data:`_READ_ONLY_KEYS`
    (computed-on-read), and :data:`_NON_PUT_KEYS` (persisted elsewhere) are
    skipped. ``tests/api/test_settings_dispatch.py`` asserts this table and
    the schema cannot silently diverge.
    """
    table: dict[str, Callable[[Any], Any]] = dict(_STATE_TIER_SETTERS)
    loadable = {name for name, field in SettingsUpdateSchema().fields.items() if not field.dump_only}
    for key in sorted(loadable):
        if key in table or key in _CUSTOM_SETTERS or key in _READ_ONLY_KEYS or key in _NON_PUT_KEYS:
            continue
        setter = getattr(settings, f"set_{key}", None)
        if setter is not None:
            table[key] = setter
    return table


_SCALAR_SETTERS: dict[str, Callable[[Any], Any]] = _build_scalar_setters()


def _plan_one_key(key: str, value) -> tuple[Callable[[Any], Any], Any] | None:
    """Validate one settings-body entry into a ``(writer, cleaned value)`` pair.

    Returns ``None`` for entries that are no-ops (computed-on-read keys,
    keys with no wired setter) and aborts 400 on invalid input --
    **without persisting anything**, so :func:`update_settings` can check a
    whole body before committing any of it.

    Keeps :func:`update_settings` simple by hosting the per-key branching
    here. Side effects (registry checks, achievement wipe, state-tier
    setter) are isolated to their helper functions.
    """
    if key in _READ_ONLY_KEYS:
        return None
    try:
        custom = _CUSTOM_SETTERS.get(key)
        if custom is not None:
            return custom.apply, custom.validate(value)
        setter = _SCALAR_SETTERS.get(key)
        if setter is None:
            return None
        return setter, settings.validate_setting(key, value)
    except (TypeError, ValueError) as exc:
        abort(400, message=str(exc))


@settings_bp.route("/api/settings", methods=["PUT"])
@settings_bp.arguments(SettingsUpdateSchema)
@settings_bp.response(200, AppSettingsSchema)
@settings_bp.alt_response(
    400, description="Setter-level validation failure (range, one-of, unknown media type / embedder / exporter)."
)
def update_settings(body: dict):
    """Update one or more settings fields, all-or-nothing.

    Only keys present in *body* are applied. Unknown keys are silently
    dropped (per the schema's ``unknown = "exclude"`` policy); type
    errors raise 422 with the standard error envelope; setter-level
    validation failures (range / one-of / an unknown media type, embedder
    or exporter) raise 400 and leave every key in the body unchanged.
    Server-tier keys are not in the schema, so they are dropped like any
    unknown key: no user can set one through this route (or through an
    import or a sync; see :func:`vtsearch.settings.plan_user_write`).
    """
    # Validate the whole body before writing any of it. Each setter
    # persists immediately, so applying as we went meant a body whose
    # third key was invalid returned 400 with its first two keys already
    # committed -- leaving the client, which reasonably reads a 400 as
    # "nothing changed", out of sync, with JSON key order silently
    # deciding which writes stuck.
    planned = [entry for entry in (_plan_one_key(key, value) for key, value in body.items()) if entry is not None]
    for write, cleaned in planned:
        write(cleaned)

    return _with_effective(settings.get_all())


@settings_bp.route("/api/settings/defaults", methods=["GET"])
@settings_bp.response(200, AppSettingsSchema)
def get_defaults():
    """Return the default values for all settings."""
    return settings.get_defaults()
