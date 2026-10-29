"""What the settings are: each key's tier, its default, and how one value is checked.

The two pydantic models in :mod:`vtsearch.settings.models` are the source of
truth for every key's type, default and range.  This module derives what the
rest of the package needs from them, without instantiating either model at
import time:

* the tier key sets (:data:`_SERVER_KEYS`, :data:`_USER_KEYS`), the per-user
  keys the built-in "default" user reads through to the server file
  (:data:`_DEFAULT_USER_FALLBACK_KEYS`), and the keys left out of the Defaults
  button and of a settings-source export;
* the defaults, built on first read (:func:`_server_defaults` and friends);
* checking one value (:func:`_validate_field`), and the read-side clean-up every
  settings dict gets on its way into a cache (:func:`_sanitize_tier`).

It reads nothing else from the package, so every other module can read it.
"""

from __future__ import annotations

from typing import Annotated, Any, cast

from pydantic import TypeAdapter, ValidationError

from vtsearch.settings.models import ServerSettings, UserSettings

# ---------------------------------------------------------------------------
# Defaults, partitioned by tier
# ---------------------------------------------------------------------------

#: Lazy caches for the model-derived defaults dicts.  Instantiating
#: :class:`ServerSettings` fires its pydantic ``default_factory`` callbacks
#: (notably the GPU-aware default for ``max_concurrent_dataset_embeddings``),
#: and that probe imports torch - ~870ms at startup just to pick an int.
#: Defer the instantiation to first read so ``import vtsearch.settings``
#: stays torch-free.
_SERVER_DEFAULTS_CACHE: dict[str, Any] | None = None
_USER_DEFAULTS_CACHE: dict[str, Any] | None = None


def _server_defaults() -> dict[str, Any]:
    global _SERVER_DEFAULTS_CACHE
    if _SERVER_DEFAULTS_CACHE is None:
        _SERVER_DEFAULTS_CACHE = ServerSettings().model_dump()
    return _SERVER_DEFAULTS_CACHE


def _user_defaults() -> dict[str, Any]:
    global _USER_DEFAULTS_CACHE
    if _USER_DEFAULTS_CACHE is None:
        _USER_DEFAULTS_CACHE = UserSettings().model_dump()
    return _USER_DEFAULTS_CACHE


def _all_defaults() -> dict[str, Any]:
    return {**_server_defaults(), **_user_defaults()}


def __getattr__(name: str) -> Any:
    # PEP 562 - expose ``_SERVER_DEFAULTS`` / ``_USER_DEFAULTS`` / ``_DEFAULTS``
    # as attributes for external readers (tests, future callers) without
    # forcing the eager pydantic instantiation at module load.
    if name == "_SERVER_DEFAULTS":
        return _server_defaults()
    if name == "_USER_DEFAULTS":
        return _user_defaults()
    if name == "_DEFAULTS":
        return _all_defaults()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Set of keys that belong to the server tier (everything else is per-user).
#: Reading ``model_fields`` doesn't instantiate the model, so this stays eager.
#: No per-user write path sets one of these - not ``PUT /api/settings``, not a
#: settings import, not a settings-source sync (see
#: :func:`~vtsearch.settings.sync.plan_user_write`):
#: they are the operator's, set through the settings file, a CLI flag or an
#: env var.
_SERVER_KEYS: frozenset[str] = frozenset(ServerSettings.model_fields.keys())

#: The per-user tier's keys: the only ones a user's own write may set.
_USER_KEYS: frozenset[str] = frozenset(UserSettings.model_fields.keys())

#: Per-user keys for which the built-in "default" user reads through to the
#: *server* settings file when the key is absent from its own per-user file.
#: These are the AutoFind knobs: per-user in multi-user deployments, but the
#: single-user GUI (everyone is "default") and the CLI ``--settings`` flat file
#: still expect a value placed in ``settings.json`` to take effect. The
#: read-through (see :func:`~vtsearch.settings.tiers._read_value`) is what makes that documented
#: flat-file workflow work; it is a live tier exception, not a migration.
#: :func:`_sanitize_tier` therefore keeps these keys in the server file
#: (validating them against ``UserSettings``, which owns them) and drops only
#: the reverse direction: a server key stranded in a per-user file.
_DEFAULT_USER_FALLBACK_KEYS: frozenset[str] = frozenset(
    {
        "autofind_detectors",
        "autofind_exporter",
        "autofind_exporter_field_values",
        "autofind_cli_delete_dataset",
    }
)

#: Keys excluded from the "defaults" endpoint (infrastructure settings that
#: should not be reset by the Default button).
_EXCLUDE_FROM_DEFAULTS = {
    "autofind_detectors",
    "autofind_exporter",
    "autofind_exporter_field_values",
    "saved_datasets_dir",
    "detectors_dir",
    "settings_source",
    "default_settings_source",
    "default_labelset_source",
}

#: Keys excluded from source sync export (to avoid circular config). Both the
#: per-user ``settings_source`` and the deployment-wide
#: ``default_settings_source`` name a sync target, so neither is exported into
#: the target itself.
_EXCLUDE_FROM_SOURCE_EXPORT = {
    "settings_source",
    "default_settings_source",
}


# ---------------------------------------------------------------------------
# Checking one value
# ---------------------------------------------------------------------------


#: Cache of per-field validators, keyed by ``(model, key)``. Built lazily on
#: first use; a ``TypeAdapter`` compiles a core-schema once, so caching it
#: makes ``_validate_field`` O(one field) instead of constructing and dumping
#: the whole ~40-field model (which also fires every absent field's
#: ``default_factory``) on every settings read/write. ``None`` marks a field
#: that must keep the whole-model path (see :func:`_build_field_adapter`).
_FIELD_ADAPTERS: dict[tuple[type, str], TypeAdapter[Any] | None] = {}


def _build_field_adapter(model: type, key: str) -> TypeAdapter[Any] | None:
    """Build a validator for a single field of *model*, or ``None``.

    ``None`` means "validate through the whole model": a ``TypeAdapter``
    only sees the field's annotation, so any ``@field_validator`` /
    ``@model_validator`` hook would be silently skipped. Neither settings
    model currently defines such hooks (all validation lives in
    ``Annotated[..., BeforeValidator(...)]`` metadata), so today every
    field gets a fast per-field adapter; the guard keeps a future hook
    from being dropped without anyone noticing.
    """
    decorators = model.__pydantic_decorators__  # type: ignore[attr-defined]
    hooked = {f for dec in decorators.field_validators.values() for f in dec.info.fields}
    if decorators.model_validators or "*" in hooked or key in hooked:
        return None
    field_info = model.model_fields[key]  # type: ignore[attr-defined]
    annotation = field_info.annotation
    if field_info.metadata:
        # Pydantic unpacks ``Annotated[T, meta...]`` into ``annotation=T``
        # + ``metadata=[meta...]``; re-wrap so the ``BeforeValidator``
        # clamps and case-folds still run under the adapter.
        annotation = Annotated[tuple([annotation, *field_info.metadata])]
    # pydantic >= 2.14 types the argument as ``TypeForm[T]``, which an
    # ``Annotated`` built at runtime cannot satisfy statically; the value is a
    # valid type form, so tell the checker so.
    return TypeAdapter(cast(Any, annotation))


def _validate_field(model: type, key: str, value: Any) -> Any:
    """Run *value* through *model*'s validator for *key* and return the result.

    Uses a cached per-field :class:`TypeAdapter` built from the field's
    annotation (metadata re-attached) so per-field ``BeforeValidator``
    clamps and enum checks fire as if the value had been loaded from disk,
    without paying a whole-model construct-and-dump on every read/write.
    A :class:`pydantic.ValidationError` is surfaced as :class:`ValueError`
    with a compact message - matches the error shape callers expect from
    the legacy spec-driven setters.
    """
    cache_key = (model, key)
    if cache_key not in _FIELD_ADAPTERS:
        _FIELD_ADAPTERS[cache_key] = _build_field_adapter(model, key)
    adapter = _FIELD_ADAPTERS[cache_key]
    try:
        if adapter is None:
            return model.model_validate({key: value}).model_dump()[key]  # type: ignore[attr-defined]
        return adapter.validate_python(value)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {"msg": str(exc)}
        raise ValueError(f"Invalid {key}: {first.get('msg', exc)}") from None


def _sanitize_tier(data: dict[str, Any], tier: str) -> dict[str, Any]:
    """Return *data* with values the tier's model would reject removed.

    Applied by :class:`~vtsearch.settings.store.UserSettingsStore` to every
    dict on its way into a cache, so the raw caches that ``get_all`` merges
    hold only values the API can serialise. Two things are dropped:

    * **Uncoercible typed values.** A settings file can carry a value from
      before a field changed shape - a boolean ``show_animations`` from
      before it became a three-way enum, a scalar ``browse_colormap`` from
      before it became a per-media-type dict - or simply a hand-edit that
      is out of range. Dropping it makes the field read as unset, so the
      pydantic default applies (per the "break persisted data freely"
      policy in ``CLAUDE.md``); keeping it would leak the stale value
      straight out of ``GET /api/settings``, and the frontend would echo it
      back into a ``PUT`` that 422s.
    * **Server-tier keys stranded in a per-user file**, left behind when a
      key moved tiers (``browse_signpost_vocab`` did). ``get_all`` layers
      the user cache over the server one, so a stale copy would shadow the
      operator's real value.

    The reverse direction is deliberately *not* dropped: per-user keys in the
    server file are the documented AutoFind read-through (see
    :data:`_DEFAULT_USER_FALLBACK_KEYS`), so those keys are kept - but checked
    against ``UserSettings``, the model that will actually read them, so the
    tier exception is not a hole in the value check. Unknown keys are
    preserved in both tiers - both models set ``extra="allow"`` so a
    forward-compatible or hand-added key round-trips rather than being
    silently deleted.

    This is a read-side view only; nothing here is written back to disk.
    """
    server_tier = tier == "server"
    model = ServerSettings if server_tier else UserSettings
    drop_wrong_tier = frozenset() if server_tier else _SERVER_KEYS
    clean: dict[str, Any] = {}
    for key, value in data.items():
        if key in drop_wrong_tier:
            continue
        # The AutoFind trio is read out of the server file by the "default"
        # user, so validate it there against the model that owns it.
        against = UserSettings if server_tier and key in _DEFAULT_USER_FALLBACK_KEYS else model
        if key in against.model_fields:
            try:
                _validate_field(against, key, value)
            except ValueError:
                continue
        clean[key] = value
    return clean
