"""The admin overrides in force, and the plugins they hide.

An *admin override* is a server-wide restriction an operator sets at startup,
by a CLI flag or an env var, that no user can change.  Each is declared once in
:mod:`vtsearch.settings.admin_overrides`: its flag, env var, the validator both
share, how it combines with the persisted setting, and the key it is published
under at ``/api/settings``.  This module answers "what is in force": the
``set_cli_X`` / ``get_cli_X`` / ``get_effective_X`` functions are thin wrappers
over that registry, kept as named functions because they are what the rest of
the app (and the tests) call, and :func:`get_effective_override` resolves any
knob by name.

The plugin hide list is one of those overrides, and the listing routes filter
through it here (:func:`filter_visible_plugins` and friends), together with the
plugins this server cannot run at all (:func:`unavailable_plugins`).

The persisted settings an override combines with are generated accessors, so
they are read on the package (``settings.get_hidden_plugins()``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vtsearch import settings
from vtsearch.settings import tiers
from vtsearch.settings import admin_overrides as _admin
from vtsearch.settings.models import partition_docs_links


def get_effective_override(name: str) -> Any:
    """Resolve one :mod:`~vtsearch.settings.admin_overrides` knob against its setting.

    Reads the persisted value through the getter the descriptor names, then
    hands both to the descriptor's ``resolve`` rule. The named
    ``get_effective_X`` functions below are one-liners over this, and
    ``/api/settings`` loops the registry through it to build its read-only
    overlay -- so a new knob is published without touching the route.
    """
    override = _admin.OVERRIDES[name]
    return override.resolve(_admin.get_override(name), getattr(settings, override.persisted_getter)())


def set_cli_solo_media_type(value: str | None) -> None:
    """Set the process-level override for the ``solo_media_type`` setting.

    Called once from ``app.py`` startup when ``--solo-media-type`` is
    passed on the command line (or ``VTSEARCH_SOLO_MEDIA_TYPE`` is set).
    The value applies server-wide (every user) and is fixed for the process
    lifetime; :func:`get_effective_solo_media_type` returns it in preference
    to the persisted server setting. Pass ``None`` (or an empty /
    whitespace-only string) to clear the override so reads fall back to the
    persisted file value.

    This stores whatever it is given: validation against the live media-type
    registry happens on the way in, in
    :func:`vtsearch.settings.admin_overrides.apply_flag_values`.
    """
    if value is not None:
        value = value.strip() or None
    _admin.set_override("solo_media_type", value)


def get_cli_solo_media_type() -> str | None:
    """Return the process-level CLI / env override (``None`` if unset)."""
    return _admin.get_override("solo_media_type")


def get_effective_solo_media_type() -> str | None:
    """Return the solo mediaType restriction in force.

    Resolution order:

    1. The process-level override set by :func:`set_cli_solo_media_type`
       (``--solo-media-type`` / ``VTSEARCH_SOLO_MEDIA_TYPE``), which applies
       to every user for the lifetime of the process.
    2. The persisted server-tier setting (``data/settings.json``), which
       defaults to ``None``.

    Returns ``None`` to mean "no solo mode active"; any other return
    value is a mediaType id (e.g. ``"image"``) that the UI locks its
    pickers to. This is an admin-set restriction: there is no per-user
    override, and ``PUT /api/settings`` cannot change it.
    """
    return get_effective_override("solo_media_type")


def set_cli_dataset_max_age_days(value: int | None) -> None:
    """Set the process-level override for the ``dataset_max_age_days`` setting.

    Called once from ``app.py`` startup when ``--dataset-max-age-days`` is
    passed (or ``VTSEARCH_DATASET_MAX_AGE_DAYS`` is set). The value applies
    server-wide (every user) and is fixed for the process lifetime;
    :func:`get_effective_dataset_max_age_days` returns it in preference to
    the persisted server setting. Pass ``None`` to clear the override
    (reads fall back to the persisted file value).
    """
    _admin.set_override("dataset_max_age_days", value)


def get_cli_dataset_max_age_days() -> int | None:
    """Return the process-level CLI / env override (``None`` if unset)."""
    return _admin.get_override("dataset_max_age_days")


def get_effective_dataset_max_age_days() -> int | None:
    """Return the dataset max age (in days) in force.

    Resolution order:

    1. The process-level override set by
       :func:`set_cli_dataset_max_age_days` (``--dataset-max-age-days`` /
       ``VTSEARCH_DATASET_MAX_AGE_DAYS``), which applies to every user for
       the lifetime of the process.
    2. The persisted server-tier setting (``data/settings.json``).

    ``None`` means datasets never expire. This is the value the dataset
    creation pipeline stamps ``expires_at`` from, and the value surfaced
    at ``/api/settings`` so the dashboard's Age-Off column reflects what is
    actually in force.
    """
    return get_effective_override("dataset_max_age_days")


def set_cli_support_email(value: str | None) -> None:
    """Set the process-level override for the ``support_email`` setting.

    Called once from ``app.py`` startup when ``--support-email`` is passed (or
    ``VTSEARCH_SUPPORT_EMAIL`` is set). The value applies server-wide (every
    user) and is fixed for the process lifetime;
    :func:`get_effective_support_email` returns it in preference to the
    persisted server setting. Pass ``None`` (or an empty / whitespace-only
    string) to clear the override so reads fall back to the persisted file
    value.
    """
    if value is not None:
        value = value.strip() or None
    _admin.set_override("support_email", value)


def get_cli_support_email() -> str | None:
    """Return the process-level CLI / env override (``None`` if unset)."""
    return _admin.get_override("support_email")


def get_effective_support_email() -> str:
    """Return the "Email us" support address in force.

    Resolution order:

    1. The process-level override set by :func:`set_cli_support_email`
       (``--support-email`` / ``VTSEARCH_SUPPORT_EMAIL``), which applies to
       every user for the lifetime of the process.
    2. The persisted server-tier setting (``data/settings.json``), which
       defaults to :data:`~vtsearch.settings.models.DEFAULT_SUPPORT_EMAIL`.

    This is the value surfaced at ``/api/settings`` so the Help modal's
    "Email us" link opens a pre-addressed compose window.
    """
    return get_effective_override("support_email")


def get_rejected_docs_links() -> list[Any]:
    """Return the persisted ``docs_links`` entries that ``settings.get_docs_links()`` drops.

    The accessor quietly keeps only the usable links (see
    :func:`~vtsearch.settings.models.partition_docs_links`), so an operator
    whose link is missing from the Help modal would otherwise have nothing to
    go on. ``app.py`` names each of these in the startup log instead.
    """
    raw = tiers._read_value("docs_links")
    return partition_docs_links(raw)[1] if isinstance(raw, list) else []


def set_cli_semantic_only(value: bool | None) -> None:
    """Set the process-level override for the ``semantic_only`` setting.

    Called once from ``app.py`` startup when ``--semantic-only`` is passed (or
    ``VTSEARCH_SEMANTIC_ONLY`` is set). The value applies server-wide (every
    user) and is fixed for the process lifetime;
    :func:`get_effective_semantic_only` returns it in preference to the
    persisted server setting. Pass ``None`` to clear the override so reads fall
    back to the persisted file value.
    """
    _admin.set_override("semantic_only", None if value is None else bool(value))


def get_cli_semantic_only() -> bool | None:
    """Return the process-level CLI / env override (``None`` if unset)."""
    return _admin.get_override("semantic_only")


def get_effective_semantic_only() -> bool:
    """Return whether this instance is locked to Semantic embedders.

    Resolution order:

    1. The process-level override set by :func:`set_cli_semantic_only`
       (``--semantic-only`` / ``VTSEARCH_SEMANTIC_ONLY``), which applies to
       every user for the lifetime of the process.
    2. The persisted server-tier setting (``data/settings.json``), which
       defaults to ``False``.

    When true, the prototype Patch Semantic / Structural embedder types are
    hidden from every picker (``GET /api/embedders`` filters them out) and
    rejected by the dataset-load and detector-create routes.
    """
    return get_effective_override("semantic_only")


def get_cli_autopilot_only() -> bool | None:
    """Return the process-level CLI / env override (``None`` if unset)."""
    return _admin.get_override("autopilot_only")


def get_effective_autopilot_only() -> bool:
    """Return whether Train and Tune are locked to their Autopilot tabs.

    Resolution order:

    1. The process-level override (``--autopilot-only`` /
       ``VTSEARCH_AUTOPILOT_ONLY``, stored by :mod:`vtsearch.settings.admin_overrides`),
       which applies to every user for the lifetime of the process.
    2. The persisted server-tier setting (``data/settings.json``), which
       defaults to ``False``.

    When true, the SPA renders neither Train's Manual / Autopilot tabs nor
    Tune's Autopilot / Review tabs: both views stay on their Autopilot panel
    (#4666).
    """
    return get_effective_override("autopilot_only")


def set_cli_solo_embedder(media_type: str, embedder: str | None) -> None:
    """Set or clear a process-level solo-embedder fallback for *media_type*.

    Called from ``app.py`` startup for each ``--solo-embedder TYPE=EMB``
    pair on the command line (or each comma-separated entry in
    ``VTSEARCH_SOLO_EMBEDDERS``). Pass ``embedder=None`` (or an empty
    string) to clear an entry. Both arguments are stripped; an empty
    *media_type* is silently ignored (the parser already validates).
    """
    mt = (media_type or "").strip()
    if not mt:
        return
    emb = (embedder or "").strip() if embedder else ""
    current = _admin.get_override("solo_embedders")
    if not emb:
        current.pop(mt, None)
    else:
        current[mt] = emb
    _admin.set_override("solo_embedders", current or None)


def get_cli_solo_embedders() -> dict[str, str]:
    """Return a copy of the process-level solo-embedder fallbacks."""
    return _admin.get_override("solo_embedders")


def get_effective_solo_embedders() -> dict[str, str]:
    """Return the merged ``{media_type: embedder}`` dict for the current user.

    Combines the per-user ``get_solo_embedder_per_media_type`` map (user
    explicit) with the process-level ``--solo-embedder`` /
    ``VTSEARCH_SOLO_EMBEDDERS`` fallbacks. The merge rule -- user entries win
    per-key, an empty-string user value is a per-type opt-out sentinel -- lives
    with the descriptor, in
    :func:`vtsearch.settings.admin_overrides._resolve_solo_embedders`.
    """
    return get_effective_override("solo_embedders")


def get_effective_solo_embedder(media_type: str) -> str | None:
    """Return the effective solo embedder for *media_type*, or ``None``."""
    if not media_type:
        return None
    return get_effective_solo_embedders().get(media_type)


def apply_user_solo_embedder_per_media_type(value: dict[str, str] | None) -> None:
    """Replace the per-user ``solo_embedder_per_media_type`` map.

    Used by the settings PUT route. ``None`` clears every entry. Keys
    are stripped and skipped if empty. Values are stripped; an
    empty-string value is preserved as a **per-type opt-out sentinel**
    (overrides the CLI fallback for that type - see
    :func:`get_effective_solo_embedders`). The route layer is
    responsible for validating non-empty ``(media_type, embedder)``
    pairs against the live registries before calling this.
    """
    if value is None:
        settings.set_solo_embedder_per_media_type({})
        return
    cleaned: dict[str, str] = {}
    for mt, emb in value.items():
        if not isinstance(mt, str) or not mt.strip():
            continue
        if isinstance(emb, str) and emb.strip():
            cleaned[mt.strip()] = emb.strip()
        elif isinstance(emb, str):
            # Empty-string sentinel - preserve as opt-out marker.
            cleaned[mt.strip()] = ""
    settings.set_solo_embedder_per_media_type(cleaned)


# -------------------------------------------------------------------
# Plugin hiding (admin-side picker declutter)
# -------------------------------------------------------------------


#: Re-exported for the settings-file readers that normalise a persisted
#: ``hidden_plugins`` value; the canonical definition sits beside the
#: descriptor that uses it.
_normalize_hidden_plugins = _admin.normalize_hidden_plugins


def set_cli_hidden_plugins(value: dict[str, Any] | None) -> None:
    """Replace the process-level ``--hide-plugin`` fallback in one shot.

    Called by tests that want to seed a known hide list. ``None`` or an empty
    dict clears the fallback. The ``--hide-plugin`` /
    ``VTSEARCH_HIDE_PLUGINS`` entry points accumulate through
    :func:`vtsearch.settings.admin_overrides.apply_flag_values` instead, which
    validates each family against the live registry first.
    """
    normalized = _admin.normalize_hidden_plugins(value or {})
    _admin.set_override("hidden_plugins", normalized or None)


def add_cli_hidden_plugin(family: str, name: str) -> None:
    """Append ``(family, name)`` to the process-level hide list (idempotent)."""
    if not family or not name:
        return
    current = _admin.get_override("hidden_plugins")
    current.setdefault(family, set()).add(name)
    _admin.set_override("hidden_plugins", current)


def get_cli_hidden_plugins() -> dict[str, set[str]]:
    """Return a defensive copy of the CLI/env-only hide map."""
    return _admin.get_override("hidden_plugins")


def get_effective_hidden_plugins() -> dict[str, set[str]]:
    """Return the merged ``{family: {name, ...}}`` hide map.

    Combines the persisted ``hidden_plugins`` server setting with the
    process-level ``--hide-plugin`` / ``VTSEARCH_HIDE_PLUGINS`` fallback. The
    union semantics matter: a plugin is hidden if either source asks for it,
    so the flag can only add hides (never un-hide something the settings file
    marks hidden).
    """
    return get_effective_override("hidden_plugins")


def is_plugin_hidden(family: str, name: str) -> bool:
    """Return True if *name* is hidden in *family* by the effective hide map.

    Static ``hidden_from_picker=True`` on a plugin class is **not**
    consulted here - that flag is already serialised on the plugin's
    ``to_dict()`` and filtered client-side. This function only answers
    "is the admin hiding this in this deployment?"
    """
    hidden = _admin.get_override("hidden_plugins").get(family)
    if hidden and name in hidden:
        return True
    persisted = settings.get_hidden_plugins()
    if not isinstance(persisted, dict):
        return False
    family_names = persisted.get(family)
    if not isinstance(family_names, (list, set, tuple)):
        return False
    return name in family_names


def unavailable_plugins() -> dict[str, set[str]]:
    """The ``{family: {name, ...}}`` this server cannot run right now, hidden like an admin hide.

    Face is the one case: on a server without face models
    (:func:`vtscore.media.face.imagekit_bridge.face_models`) the ``face`` media
    type and its embedder drop out of every listing.  The ``image2face``
    converter hides itself (its ``hidden_from_picker``), which also covers the
    listings that read only that flag.  Never persisted and never part of
    :func:`get_effective_hidden_plugins`: it is a fact about the machine, not a
    choice an admin made, so the settings UI does not show it as one.
    """
    from vtscore.media.face import imagekit_bridge  # noqa: PLC0415

    if imagekit_bridge.face_models().available:
        return {}
    return {"media_types": {"face"}, "embedders": {imagekit_bridge.EMBEDDER_NAME}}


def _hidden_in(family: str) -> set[str]:
    """Everything hidden in *family*: the admin's hides plus what this server cannot run."""
    return set(get_effective_hidden_plugins().get(family) or ()) | unavailable_plugins().get(family, set())


def filter_visible_plugins(family: str, plugins: Iterable[Any], *, id_attr: str = "name") -> list[Any]:
    """Drop plugins whose id-attribute is hidden in *family* (or that this server cannot run).

    The single chokepoint for the listing routes. *id_attr* names the
    attribute that carries the plugin's registry id - ``"name"`` for the
    PluginBase families, ``"type_id"`` for :class:`MediaType` (whose
    registry key is the type id, not its human-readable name).
    """
    hidden = _hidden_in(family)
    if not hidden:
        return list(plugins)
    return [p for p in plugins if getattr(p, id_attr, None) not in hidden]


def filter_visible_plugin_dicts(
    family: str, plugin_dicts: Iterable[dict[str, Any]], *, id_key: str = "name"
) -> list[dict[str, Any]]:
    """Same as :func:`filter_visible_plugins` but for pre-serialised dicts.

    Reads ``dict[id_key]`` to compare against the hide set. Entries
    without that key are kept.
    """
    hidden = _hidden_in(family)
    if not hidden:
        return list(plugin_dicts)
    return [d for d in plugin_dicts if d.get(id_key) not in hidden]


def filter_semantic_only_embedder_dicts(embedder_dicts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop patch / structural embedders when the instance is Semantic-locked.

    The single chokepoint for ``GET /api/embedders``: with
    :func:`get_effective_semantic_only` true, an embedder that advertises
    ``supports_patch_regions`` or ``supports_geometric_verification`` is
    withheld from the listing, so every picker fed by that endpoint (the Add
    Dataset "Advanced" block's Embedder / Region embedder / Instance embedder
    selects, the Import tab's defaults) offers Semantic embedders only.

    A no-op when the lock is off, so the ordinary deployment pays one boolean.
    """
    dicts = list(embedder_dicts)
    if not get_effective_semantic_only():
        return dicts
    return [d for d in dicts if not d.get("supports_patch_regions") and not d.get("supports_geometric_verification")]
