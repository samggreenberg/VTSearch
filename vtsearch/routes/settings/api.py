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

from typing import Any, Callable, NamedTuple

from flask_smorest import Blueprint, abort
from marshmallow import fields

from vtsearch import admin_overrides, settings
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema
from vtsearch.state import (
    set_calibrate_count as _state_set_calibrate_count,
    set_calibration_fraction as _state_set_calibration_fraction,
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


class _CustomSetter(NamedTuple):
    """A key whose update needs bespoke validation, split from its write.

    *validate* returns the cleaned value (or aborts 400); *apply* persists
    an already-cleaned value and must not re-validate. Keeping the two
    halves separate is what lets :func:`update_settings` check an entire
    body before committing any of it.
    """

    validate: Callable[[Any], Any]
    apply: Callable[[Any], None]


def _validate_enable_achievements(value) -> bool:
    """Coerce the toggle to a bool (the schema already rejects non-booleans)."""
    return bool(value)


def _apply_enable_achievements(value: bool) -> None:
    """Persist the toggle and wipe stored counters when flipping it off.

    The user-visible promise is that turning the feature off zeroes the
    achievement counters and keeps them there. Wiping on the True→False
    transition (rather than every set) makes the on→off→on cycle
    deterministic: counters reset on opt-out and start fresh if the user
    ever opts back in.
    """
    prev = bool(settings.get_enable_achievements())
    settings.set_enable_achievements(value)
    if prev and not value:
        from vtsearch import achievements

        achievements.wipe_state()


def _validate_inclusion(value) -> int:
    try:
        return int(max(-10, min(10, int(value))))
    except (TypeError, ValueError) as exc:
        abort(400, message=str(exc))


def _apply_inclusion(clamped: int) -> None:
    """``inclusion`` is set via :mod:`vtsearch.state`, not :mod:`settings`."""
    from vtsearch.state import set_inclusion

    set_inclusion(clamped)


def _validate_solo_embedder_per_media_type(value) -> dict[str, str] | None:
    """Validate the ``{media_type: embedder}`` map and return it cleaned.

    ``None`` (which clears every per-type lock) passes through. Otherwise
    *value* must be a dict mapping registered media-type ids to embedder
    names that exist for that type (per
    :func:`vtscore.media.embedders_for_type`). An empty string value is
    preserved as a **per-type opt-out sentinel**; it overrides the
    ``--solo-embedder`` CLI fallback for that type. Any other invalid
    pairing raises 400.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        abort(400, message="solo_embedder_per_media_type must be a dict or null")

    from vtscore.media import all_type_ids, embedders_for_type

    valid_types = set(all_type_ids())
    cleaned: dict[str, str] = {}
    for raw_type, raw_emb in value.items():
        if not isinstance(raw_type, str) or not raw_type.strip():
            abort(400, message="solo_embedder_per_media_type keys must be non-empty media-type ids")
        mt = raw_type.strip()
        if mt not in valid_types:
            abort(400, message=f"Unknown media type: {mt!r}. Valid: {sorted(valid_types)}")
        if raw_emb is None or (isinstance(raw_emb, str) and not raw_emb.strip()):
            # Per-type opt-out sentinel: preserve so it overrides the
            # CLI fallback. ``None`` is normalised to "" here.
            cleaned[mt] = ""
            continue
        if not isinstance(raw_emb, str):
            abort(400, message=f"solo_embedder_per_media_type[{mt!r}] must be a string")
        emb_name = raw_emb.strip()
        valid_embedders = {e.name for e in embedders_for_type(mt)}
        if emb_name not in valid_embedders:
            abort(
                400,
                message=(f"Unknown embedder {emb_name!r} for media type {mt!r}. Valid: {sorted(valid_embedders)}"),
            )
        cleaned[mt] = emb_name
    return cleaned


def _validate_dir(key: str, value: str) -> str:
    """Validate a directory-path setting and return the path to persist.

    Returns the *approved* path rather than the raw input: under multi-user
    confinement a relative dir is checked against the user's data dir but
    would later be opened relative to the process CWD.  Unconfined
    (single-user) values come back verbatim.
    """
    import vtscore.security.path_validation as _paths

    if not value or not value.strip():
        abort(400, message=f"{key} must be a non-empty string")

    try:
        return _paths.confine_server_filepath(value.strip(), _paths.get_file_access_base_dir())
    except ValueError as exc:
        abort(400, message=str(exc))


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


def _validate_autofind_exporter(value) -> str:
    """Validate the Auto-Find results exporter name and return it.

    ``""``/``None`` (which clears auto-export) normalises to ``""``. Any
    other value must name a registered (pickable) exporter; an unknown
    name aborts 400. Field values are validated lazily at export time
    against the chosen plugin's schema, not here.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return ""
    if not isinstance(value, str):
        abort(400, message="autofind_exporter must be a string")
    from vtscore.exporters import list_exporters

    name = value.strip()
    valid = {exp.name for exp in list_exporters() if not getattr(exp, "hidden_from_picker", False)}
    if name not in valid:
        abort(400, message=f"Unknown exporter {name!r}. Available: {sorted(valid)}")
    return name


def _validate_autofind_exporter_field_values(value) -> dict[str, dict[str, str]]:
    """Validate the per-exporter ``{name: {key: value}}`` map and clean it.

    ``None`` (which clears every exporter's stored config) normalises to
    ``{}``. Otherwise the value must be a dict whose entries are
    themselves ``{str: str}`` dicts; non-string values are coerced to
    strings so the persisted shape stays flat (exporter fields are always
    rendered/sent as strings).
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        abort(400, message="autofind_exporter_field_values must be a dict")
    cleaned: dict[str, dict[str, str]] = {}
    for exp_name, fvals in value.items():
        if not isinstance(exp_name, str) or not isinstance(fvals, dict):
            abort(400, message="autofind_exporter_field_values must map exporter names to field dicts")
        cleaned[exp_name] = {str(k): "" if v is None else str(v) for k, v in fvals.items()}
    return cleaned


#: Keys with bespoke side-effects (validation against a registry, path
#: traversal checks, counter wipes, etc.). Each entry pairs a validator
#: -- which raises 400 on invalid input and returns the cleaned value --
#: with the writer that persists it, so a whole PUT body can be validated
#: before any of it is committed (see :func:`update_settings`).
_CUSTOM_SETTERS: dict[str, _CustomSetter] = {
    "inclusion": _CustomSetter(_validate_inclusion, _apply_inclusion),
    "saved_datasets_dir": _CustomSetter(
        lambda v: _validate_dir("saved_datasets_dir", v), settings.set_saved_datasets_dir
    ),
    "detectors_dir": _CustomSetter(lambda v: _validate_dir("detectors_dir", v), settings.set_detectors_dir),
    "enable_achievements": _CustomSetter(_validate_enable_achievements, _apply_enable_achievements),
    "solo_embedder_per_media_type": _CustomSetter(
        _validate_solo_embedder_per_media_type, settings.apply_user_solo_embedder_per_media_type
    ),
    "autofind_exporter": _CustomSetter(_validate_autofind_exporter, settings.set_autofind_exporter),
    "autofind_exporter_field_values": _CustomSetter(
        _validate_autofind_exporter_field_values, settings.set_autofind_exporter_field_values
    ),
}


# The training-relevant settings route through ``vtsearch.state`` rather
# than ``vtsearch.settings`` so the state setter's side-effect
# (``invalidate_loaded_detector_models``) fires and the cached MLP /
# threshold on every loaded detector context is dropped; otherwise
# ``/api/find-label`` / ``/api/find`` / ``/api/auto-detect`` would keep
# scoring with a threshold computed under the prior setting (M7). These
# override the plain ``settings.set_<key>`` accessor the table would
# otherwise pick up.
_STATE_TIER_SETTERS: dict[str, Callable[[Any], Any]] = {
    "calibrate_count": _state_set_calibrate_count,
    "calibration_fraction": _state_set_calibration_fraction,
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
    here. Side effects (path validation, achievement wipe, state-tier
    setter) are isolated to their helper functions.
    """
    if key in _READ_ONLY_KEYS:
        return None
    custom = _CUSTOM_SETTERS.get(key)
    if custom is not None:
        return custom.apply, custom.validate(value)
    setter = _SCALAR_SETTERS.get(key)
    if setter is None:
        return None
    try:
        return setter, settings.validate_setting(key, value)
    except (TypeError, ValueError) as exc:
        abort(400, message=str(exc))


@settings_bp.route("/api/settings", methods=["PUT"])
@settings_bp.arguments(SettingsUpdateSchema)
@settings_bp.response(200, AppSettingsSchema)
@settings_bp.alt_response(400, description="Setter-level validation failure (range, one-of, path traversal).")
def update_settings(body: dict):
    """Update one or more settings fields, all-or-nothing.

    Only keys present in *body* are applied. Unknown keys are silently
    dropped (per the schema's ``unknown = "exclude"`` policy); type
    errors raise 422 with the standard error envelope; setter-level
    validation failures (range / one-of / path traversal) raise 400 and
    leave every key in the body unchanged.
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
