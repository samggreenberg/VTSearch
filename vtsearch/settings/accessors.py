"""The accessors written out by hand, and the factory for the generated ones.

:func:`_make_scalar_accessors` builds the ``get_<key>`` / ``set_<key>`` /
``validate_<key>`` triple for one model field; the package ``__init__`` runs it
over every field of both models and installs the results as its own
attributes.  The rest of this module is what needs more than a field's own
validator:

* :func:`_make_per_side_setting`, the factory for the six per-media-type,
  per-side view settings (``grid_icon_size_left`` … ``panel_pct_right``), whose
  accessors the package installs the same way;
* the last-picked embedder per media type;
* the AutoFind detector list (dedupe, add / remove / membership);
* the two server-tier directory paths;
* the whole-dict reads: :func:`get_defaults`, :func:`get_user_settings` and
  :func:`get_all`.

A function here that needs a generated accessor calls it on the package
(``settings.get_docs_links()``), where it is installed - which is also why a
stub patched onto the package reaches it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtsearch import settings
from vtsearch.settings import tiers
from vtsearch.settings.models import VALID_PANEL_PX
from vtsearch.settings.schema import (
    _EXCLUDE_FROM_DEFAULTS,
    _all_defaults,
    _server_defaults,
    _user_defaults,
    _validate_field,
)
from vtsearch.settings.tiers import (
    _ensure_server_loaded,
    _ensure_user_loaded,
    _mutate_server_locked,
    _read_value,
    _settings_lock,
    _user_caches,
    _write_value,
    mutate_user,
    snapshot_user,
)

# -------------------------------------------------------------------
# Pydantic-driven generation of simple get_<key> / set_<key> pairs.
#
# Each :class:`UserSettings` / :class:`ServerSettings` field becomes a
# pair of module-level accessors on the package. Validation (clamping,
# one-of) lives on the model; the shims below only marshal values between
# the cache dict and the field's validator.
# -------------------------------------------------------------------


def _make_scalar_accessors(model: type, key: str):
    def getter():
        # ``_read_value`` takes ``_settings_lock`` itself, around the cache
        # read only. Wrapping the call in the lock here would put the
        # ensure-loaded file/sync locks underneath it and invert the
        # canonical file_lock → settings_lock order.
        raw = _read_value(key)
        try:
            return _validate_field(model, key, raw)
        except ValueError:
            # Corrupt disk value - fall back to the default so callers
            # never see partially-typed garbage.
            return model.model_fields[key].get_default(call_default_factory=True)

    def validate(value):
        return _validate_field(model, key, value)

    def setter(value):
        # Lock acquisition is delegated to ``_write_value`` →
        # ``_mutate_*_locked``, which takes ``_file_lock`` first and
        # then ``_settings_lock``. Acquiring ``_settings_lock`` here
        # would invert the order and risk an AB-BA deadlock with paths
        # that enter ``_mutate_*_locked`` directly.
        coerced = validate(value)
        _write_value(key, coerced)

    getter.__name__ = f"get_{key}"
    setter.__name__ = f"set_{key}"
    validate.__name__ = f"validate_{key}"
    return getter, setter, validate


# -------------------------------------------------------------------
# Per-media-type per-side settings
#
# These six settings (``grid_icon_size_{left,right}``,
# ``focus_mode_{left,right}``, ``panel_pct_{left,right}``) store a
# ``{media_type_id: value}`` dict.
# Their validation rules differ from the simple settings in two ways:
#
# 1. The set of valid media-type IDs is resolved dynamically from
#    :func:`vtscore.media.all_type_ids` (plugins may register more).
# 2. ``panel_pct_*`` raises on out-of-range writes but clamps on reads
#    (the disk value can become invalid via direct file edits).
#
# We keep a small shim layer that delegates per-element validation to
# the Pydantic models defined in :mod:`vtsearch.settings.models`.
# -------------------------------------------------------------------

_GRID_ICON_SIZE_DEFAULT: str = "M"
_FOCUS_MODE_DEFAULTS: dict[str, str] = {"left": "click", "right": "click"}
_PANEL_PX_DEFAULTS: dict[str, int] = {"left": 260, "right": 300}


def _valid_media_types() -> tuple[str, ...]:
    """Return valid media type IDs from the media registry."""
    from vtscore.media import all_type_ids

    return tuple(all_type_ids())


def _make_per_side_setting(  # noqa: C901
    key_base: str,
    defaults: dict[str, Any],
    *,
    normalize=None,
    value_type: str = "str",
    valid_values: tuple[str, ...] | None = None,
):
    """Build ``get_{base}_left``/``_right`` and ``set_{base}_left``/``_right``.

    *normalize* is applied to string values on read and write (e.g.
    :func:`str.upper`). *valid_values* enables enum membership checks
    for string types. For ``value_type="int"`` the value is clamped to
    :data:`VALID_PANEL_PX` on read and raises on out-of-range writes.
    """
    lo_hi = VALID_PANEL_PX if value_type == "int" else None

    def _validate_entry(v, key: str, tid: str | None = None):
        if value_type == "str":
            if normalize is not None and isinstance(v, str):
                v = normalize(v)
            if valid_values is not None and v not in valid_values:
                label = f"{key} value for {tid}" if tid else key
                raise ValueError(f"Invalid {label}: {v!r}")
            return v
        # int (panel_pct): raise on out-of-range writes
        lo, hi = lo_hi  # type: ignore[misc]
        try:
            v = int(round(float(v)))
        except (ValueError, TypeError):
            label = f"{key} value for {tid}" if tid else key
            raise ValueError(f"Invalid {label}: {v!r}") from None
        if not (lo <= v <= hi):
            label = f"{key} value for {tid}" if tid else key
            raise ValueError(f"Invalid {label}: {v} (must be between {lo} and {hi})")
        return v

    def _get_dict(key: str) -> dict[str, Any]:
        side = key[len(key_base) + 1 :]
        default_val = defaults.get(side, next(iter(defaults.values())))
        # Unlocked: ``_read_value`` locks its own cache read (see
        # ``_make_scalar_accessors.getter``).
        raw = _read_value(key)
        types = _valid_media_types()
        if not isinstance(raw, dict):
            return {tid: default_val for tid in types}

        result: dict[str, Any] = {}
        for tid in types:
            v = raw.get(tid, default_val)
            if normalize is not None and isinstance(v, str):
                v = normalize(v)
            if valid_values is not None and v not in valid_values:
                v = default_val
            elif value_type == "int":
                lo, hi = lo_hi  # type: ignore[misc]
                try:
                    v = max(lo, min(hi, int(round(float(v)))))
                except (ValueError, TypeError):
                    v = default_val
            result[tid] = v
        return result

    def _validate_dict(key: str, value) -> dict[str, Any]:
        valid_types = _valid_media_types()

        # Scalar expansion: "grid" → {"audio": "grid", "image": "grid", ...}
        if value_type == "str" and isinstance(value, str):
            value = {tid: _validate_entry(value, key) for tid in valid_types}
        elif value_type == "int" and isinstance(value, (int, float)):
            value = {tid: _validate_entry(value, key) for tid in valid_types}

        if not isinstance(value, dict):
            expected = "dict or string" if value_type == "str" else "dict or number"
            raise ValueError(f"{key} must be a {expected}")

        coerced: dict[str, Any] = {}
        for tid, v in value.items():
            if tid not in valid_types:
                raise ValueError(f"Invalid media type: {tid!r}")
            coerced[tid] = _validate_entry(v, key, tid)
        return coerced

    def _set_dict(key: str, value) -> None:
        # Locks are taken inside ``_write_value`` in the canonical
        # order (file_lock → settings_lock); see ``_make_scalar_accessors.setter``.
        _write_value(key, _validate_dict(key, value))

    def get_left():
        return _get_dict(f"{key_base}_left")

    def get_right():
        return _get_dict(f"{key_base}_right")

    def set_left(value):
        _set_dict(f"{key_base}_left", value)

    def set_right(value):
        _set_dict(f"{key_base}_right", value)

    def validate_left(value):
        return _validate_dict(f"{key_base}_left", value)

    def validate_right(value):
        return _validate_dict(f"{key_base}_right", value)

    get_left.__name__ = f"get_{key_base}_left"
    get_right.__name__ = f"get_{key_base}_right"
    set_left.__name__ = f"set_{key_base}_left"
    set_right.__name__ = f"set_{key_base}_right"
    validate_left.__name__ = f"validate_{key_base}_left"
    validate_right.__name__ = f"validate_{key_base}_right"
    return get_left, get_right, set_left, set_right, validate_left, validate_right


def get_last_embedder_for_media_type(media_type: str) -> str:
    """Return the user's last picked embedder for *media_type*, or ``""``."""
    if not media_type:
        return ""
    raw = settings.get_last_embedder_per_media_type()
    if isinstance(raw, dict):
        value = raw.get(media_type, "")
        if isinstance(value, str):
            return value
    return ""


def set_last_embedder_for_media_type(media_type: str, embedder: str) -> None:
    """Record *embedder* as the user's last pick for *media_type*."""
    if not media_type or not embedder:
        return
    current = settings.get_last_embedder_per_media_type()
    updated = dict(current) if isinstance(current, dict) else {}
    if updated.get(media_type) == embedder:
        return
    updated[media_type] = embedder
    settings.set_last_embedder_per_media_type(updated)


def get_autofind_detectors() -> list[str]:
    """Return the current user's list of detector names flagged for AutoFind.

    Each name maps to a JSON file under ``data/detectors/``; scoring resolves
    the labelset's origins, re-embeds, trains a head, and applies it to the
    loaded dataset. Per-user (with a read-through to the server file for the
    built-in "default" user; see :func:`~vtsearch.settings.tiers._read_value`).
    """
    raw = _read_value("autofind_detectors")
    return list(raw) if isinstance(raw, list) else []


def validate_autofind_detectors(value: list[str]) -> list[str]:
    """Return the deduped AutoFind detector list, without persisting it."""
    return list(dict.fromkeys(value))  # dedupe, preserve order


def set_autofind_detectors(value: list[str]) -> None:
    """Set and persist the current user's full AutoFind detector list."""
    _write_value("autofind_detectors", validate_autofind_detectors(value))


def add_autofind_detector(name: str) -> None:
    """Add a detector name to the current user's AutoFind list (idempotent).

    Atomic per-user read-modify-write: ``mutate_user`` re-reads the on-disk
    file under the cross-process lock, so a concurrent writer's entry is
    merged rather than clobbered. The ``seed`` captures the effective value
    (including the default-user read-through) for the first write, before the
    user has any entry of its own.
    """
    seed = get_autofind_detectors()

    def _add(cache: dict[str, Any]) -> None:
        base = cache["autofind_detectors"] if "autofind_detectors" in cache else seed
        cache["autofind_detectors"] = list(dict.fromkeys([*base, name]))

    mutate_user(_add)


def remove_autofind_detector(name: str) -> bool:
    """Remove a detector name from the current user's AutoFind list.

    Returns ``True`` if the name was present (pre-read). Atomic per-user RMW,
    same merge semantics as :func:`add_autofind_detector`.
    """
    seed = get_autofind_detectors()
    found = name in seed

    def _remove(cache: dict[str, Any]) -> None:
        base = cache["autofind_detectors"] if "autofind_detectors" in cache else seed
        cache["autofind_detectors"] = [n for n in base if n != name]

    mutate_user(_remove)
    return found


def is_autofind_detector(name: str) -> bool:
    """Check whether a detector name is in the AutoFind list."""
    return name in get_autofind_detectors()


# -------------------------------------------------------------------
# Directory path settings (server tier)
# -------------------------------------------------------------------


def _get_dir(key: str) -> Path:
    """Return a server-tier directory path setting as a :class:`~pathlib.Path`.

    ``_ensure_server_loaded`` runs outside ``_settings_lock`` (it takes the
    server ``file_lock`` on a cold cache); only the cache read is locked.
    """
    _ensure_server_loaded()
    with _settings_lock:
        server_cache = tiers._store.server_cache or {}
        raw = server_cache.get(key, _server_defaults()[key])
    return Path(raw)


def _set_dir(key: str, value: str | Path) -> None:
    """Persist a server-tier directory path setting."""
    _ensure_server_loaded()
    coerced = str(value)
    _mutate_server_locked(lambda c: c.__setitem__(key, coerced))


def get_saved_datasets_dir() -> Path:
    """Return the configured saved-datasets directory."""
    return _get_dir("saved_datasets_dir")


def set_saved_datasets_dir(value: str | Path) -> None:
    """Set the saved-datasets directory."""
    _set_dir("saved_datasets_dir", value)


def get_detectors_dir() -> Path:
    """Return the configured detectors directory."""
    return _get_dir("detectors_dir")


def set_detectors_dir(value: str | Path) -> None:
    """Set the detectors directory."""
    _set_dir("detectors_dir", value)


# -------------------------------------------------------------------
# The whole settings dict
# -------------------------------------------------------------------


def get_defaults() -> dict[str, Any]:
    """Return a copy of the default settings (excluding infrastructure keys)."""
    result = {k: v for k, v in _all_defaults().items() if k not in _EXCLUDE_FROM_DEFAULTS}
    valid_types = _valid_media_types()
    result["grid_icon_size_left"] = {tid: _GRID_ICON_SIZE_DEFAULT for tid in valid_types}
    result["grid_icon_size_right"] = {tid: _GRID_ICON_SIZE_DEFAULT for tid in valid_types}
    result["focus_mode_left"] = {tid: _FOCUS_MODE_DEFAULTS["left"] for tid in valid_types}
    result["focus_mode_right"] = {tid: _FOCUS_MODE_DEFAULTS["right"] for tid in valid_types}
    result["panel_pct_left"] = {tid: _PANEL_PX_DEFAULTS["left"] for tid in valid_types}
    result["panel_pct_right"] = {tid: _PANEL_PX_DEFAULTS["right"] for tid in valid_types}
    return result


def get_user_settings() -> dict[str, Any]:
    """Return the current user's per-user settings (with defaults filled in).

    Used by the settings-export route and the per-user sync source so an
    export only carries this user's preferences - not the shared
    server-tier infrastructure keys (``saved_datasets_dir``,
    ``hidden_plugins`` etc.) and not other users' files.
    """
    from vtsearch.auth import get_current_user

    user_copy = snapshot_user(get_current_user())
    result = dict(_user_defaults())
    result.update(user_copy)
    result["grid_icon_size_left"] = settings.get_grid_icon_size_left()
    result["grid_icon_size_right"] = settings.get_grid_icon_size_right()
    result["focus_mode_left"] = settings.get_focus_mode_left()
    result["focus_mode_right"] = settings.get_focus_mode_right()
    result["panel_pct_left"] = settings.get_panel_pct_left()
    result["panel_pct_right"] = settings.get_panel_pct_right()
    return result


def get_all() -> dict[str, Any]:
    """Return the merged settings dict for the current user.

    Combines the server-tier defaults+overrides with the current user's
    per-user defaults+overrides. The per-side view-mode dicts are
    returned in expanded (per-media-type) form.
    """
    from vtsearch.auth import get_current_user

    username = get_current_user()
    _ensure_server_loaded()
    _ensure_user_loaded(username)
    with _settings_lock:
        server_copy = dict(tiers._store.server_cache or {})
        user_copy = dict(_user_caches.get(username, {}))
    result = dict(_all_defaults())
    result.update(server_copy)
    result.update(user_copy)
    # Always return expanded per-media-type size/focus/panel dicts.
    result["grid_icon_size_left"] = settings.get_grid_icon_size_left()
    result["grid_icon_size_right"] = settings.get_grid_icon_size_right()
    result["focus_mode_left"] = settings.get_focus_mode_left()
    result["focus_mode_right"] = settings.get_focus_mode_right()
    result["panel_pct_left"] = settings.get_panel_pct_left()
    result["panel_pct_right"] = settings.get_panel_pct_right()
    # AutoFind keys go through their accessors so the default-user read-through
    # (and per-user isolation for named users) is applied consistently: the
    # plain ``result.update(server_copy)`` above would otherwise leak the
    # server file's AutoFind list to every named user, not just "default".
    result["autofind_detectors"] = get_autofind_detectors()
    result["autofind_exporter"] = settings.get_autofind_exporter()
    result["autofind_exporter_field_values"] = settings.get_autofind_exporter_field_values()
    result["autofind_cli_delete_dataset"] = settings.get_autofind_cli_delete_dataset()
    # The raw file value can hold entries the normalizer drops (a blank label,
    # a ``javascript:`` URL); read through the accessor so only usable links
    # ever reach the Help modal.
    result["docs_links"] = settings.get_docs_links()
    return result
