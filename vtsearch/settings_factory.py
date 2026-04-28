"""Accessor factories for :mod:`vtsearch.settings`.

These factory functions build pairs/quadruples of ``get_<key>`` /
``set_<key>`` closures.  They were inlined in :mod:`vtsearch.settings`
historically; pulling them here keeps the settings module focused on
the cache, file I/O, and the spec-driven generation loops.

The generated accessors close over :mod:`vtsearch.settings` state
(``_settings_lock``, ``_ensure_loaded``, ``_save``, ``_DEFAULTS``) which
they import lazily at call time to avoid an import cycle.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "clamp",
    "clamp_min",
    "make_accessors",
    "make_per_side_setting",
    "one_of",
]


def clamp(cast: type, lo: float, hi: float) -> Callable:
    """Return a coercion that casts then clamps to ``[lo, hi]``."""
    return lambda v: cast(max(lo, min(hi, cast(v))))


def clamp_min(cast: type, lo: float) -> Callable:
    """Return a coercion that casts then clamps to ``>= lo``."""
    return lambda v: cast(max(lo, cast(v)))


def one_of(key: str, valid) -> Callable:
    """Return a coercion that validates membership in *valid*."""

    def _coerce(v):
        v = str(v)
        if v not in valid:
            raise ValueError(f"Invalid {key}: {v!r}")
        return v

    return _coerce


def make_accessors(key: str, cast: type, coerce=None):
    """Create a ``get_<key>`` / ``set_<key>`` pair for a simple setting.

    *cast* converts the stored value on read (e.g. ``float``, ``int``, ``bool``).
    *coerce* normalises the value on write (e.g. clamping); defaults to *cast*.
    """
    if coerce is None:
        coerce = cast

    def getter():
        from vtsearch.settings import _DEFAULTS, _ensure_loaded, _settings_lock

        with _settings_lock:
            return cast(_ensure_loaded().get(key, _DEFAULTS[key]))

    def setter(value):
        from vtsearch.settings import _ensure_loaded, _save, _settings_lock

        with _settings_lock:
            s = _ensure_loaded()
            s[key] = coerce(value)
            _save(s)

    getter.__name__ = f"get_{key}"
    setter.__name__ = f"set_{key}"
    return getter, setter


def make_per_side_setting(
    key_base: str,
    defaults: dict[str, Any],
    valid_values: tuple[str, ...] | None = None,
    *,
    valid_panel_px: tuple[int, int],
    normalize=None,
    value_type: str = "str",
):
    """Factory for per-media-type per-side settings.

    Generates ``get_<key_base>_left()``, ``get_<key_base>_right()``,
    ``set_<key_base>_left()``, ``set_<key_base>_right()``.

    Parameters
    ----------
    key_base:
        Setting name without the side suffix, e.g. ``"view_mode"``.
    defaults:
        ``{"left": default_value, "right": default_value}``.
    valid_values:
        Tuple of allowed string values (for enum-like settings).
        ``None`` skips membership validation (for numeric settings).
    valid_panel_px:
        ``(min, max)`` integer range used when ``value_type == "int"``.
    normalize:
        Optional ``str -> str`` applied to string values on read and
        write (e.g. ``str.upper`` for grid_icon_size).
    value_type:
        ``"str"`` or ``"int"`` — controls validation and coercion.
    """
    lo_hi = valid_panel_px if value_type == "int" else None

    def _valid_media_types() -> tuple[str, ...]:
        from vtsearch.media import all_type_ids

        return tuple(all_type_ids())

    def _get_dict(key: str) -> dict[str, Any]:
        from vtsearch.settings import _DEFAULTS, _ensure_loaded, _settings_lock

        side = key[len(key_base) + 1 :]  # strip "<key_base>_" prefix
        default_val = defaults.get(side, next(iter(defaults.values())))
        with _settings_lock:
            raw = _ensure_loaded().get(key, _DEFAULTS[key])

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

    def _validate_str(v, key, tid=None):
        if normalize is not None and isinstance(v, str):
            v = normalize(v)
        if valid_values is not None and v not in valid_values:
            label = f"{key} value for {tid}" if tid else key
            raise ValueError(f"Invalid {label}: {v!r}")
        return v

    def _validate_int(v, key, tid=None):
        lo, hi = lo_hi  # type: ignore[misc]
        try:
            v = int(round(float(v)))
        except (ValueError, TypeError):
            label = f"{key} value for {tid}" if tid else key
            raise ValueError(f"Invalid {label}: {v!r}")
        if not (lo <= v <= hi):
            label = f"{key} value for {tid}" if tid else key
            raise ValueError(f"Invalid {label}: {v} (must be between {lo} and {hi})")
        return v

    _validate_entry = _validate_int if value_type == "int" else _validate_str

    def _set_dict(key: str, value) -> None:
        from vtsearch.settings import _ensure_loaded, _save, _settings_lock

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

        with _settings_lock:
            s = _ensure_loaded()
            s[key] = coerced
            _save(s)

    def get_left():
        return _get_dict(f"{key_base}_left")

    def get_right():
        return _get_dict(f"{key_base}_right")

    def set_left(value):
        _set_dict(f"{key_base}_left", value)

    def set_right(value):
        _set_dict(f"{key_base}_right", value)

    get_left.__name__ = f"get_{key_base}_left"
    get_right.__name__ = f"get_{key_base}_right"
    set_left.__name__ = f"set_{key_base}_left"
    set_right.__name__ = f"set_{key_base}_right"
    return get_left, get_right, set_left, set_right
