"""Declarative registry of the process-level **admin override** knobs.

An *admin override* is a server-wide restriction an operator sets at startup:
it applies to every user, is fixed for the lifetime of the process, and is not
editable through ``PUT /api/settings``. Six exist today -- the solo mediaType
lock, the per-mediaType solo-embedder locks, the plugin hide list, the dataset
retention window, the support-email address, and the Semantic-only embedder
lock.

Each one used to be spelled out four times: a ``set_cli_X`` / ``get_cli_X`` /
``get_effective_X`` triad in :mod:`vtsearch.settings`, an ``_apply_X`` argparse
hook in :mod:`vtsearch.cli_main`, an ``_apply_env_X`` startup hook in
``app.py``, and a line in the ``/api/settings`` effective-value overlay. Nothing
tied the four together, so the set drifted: only three of the six were reachable
from the environment, which meant the gunicorn-launched Docker images (which
never parse ``argv``) could set *some* admin restrictions and not others, for no
reason a reader could recover. The env path also re-implemented validation --
``--dataset-max-age-days 0`` was an argparse error while
``VTSEARCH_DATASET_MAX_AGE_DAYS=0`` warned and carried on.

This module is the single source of truth instead. One :class:`AdminOverride`
descriptor per knob carries the flag spelling and help text, the env-var name,
the parser/validator both entry paths share, the rule for combining the
override with the persisted setting, and the key it surfaces under at
``/api/settings``. Everything else derives from the registry:

* :func:`register_override_flags` builds the argparse arguments.
* :func:`apply_flag_values` validates and stores what argparse parsed.
* :func:`apply_env_overrides` does the same for the environment, honouring an
  explicit flag first.
* ``vtsearch.settings.get_effective_override`` resolves one against its
  persisted value, and the named ``get_effective_X`` wrappers delegate to it.
* ``vtsearch.routes.settings.api._with_effective`` loops the registry to build
  the read-only overlay.

Adding a seventh knob is therefore one :class:`AdminOverride` entry, and
``tests/core/test_admin_overrides.py`` fails if it does not carry a flag, an env
var, and an overlay key -- the coverage can no longer go arbitrary by accident.

This module deliberately imports nothing from :mod:`vtsearch.settings` (the
dependency runs the other way) and touches the plugin/media registries only
lazily, inside the validators, so importing it is free.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

#: Env-var values accepted as "on" for a switch-shaped override.
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class OverrideValueError(ValueError):
    """A flag or env value failed its override's validator.

    Carries a message written for an operator (it names the flag or variable
    and the valid values). :mod:`vtsearch.cli_main` turns it into
    ``parser.error`` (exit 2); the env path prints it as a warning and leaves
    the override unset, because a bad variable should not stop a container from
    booting.
    """


@dataclass(frozen=True)
class AdminOverride:
    """One process-level admin knob, described once for all four consumers.

    :param name: Stable id; also the ``settings.py`` accessor stem
        (``support_email`` → ``set_cli_support_email``).
    :param flag: Long option spelling, e.g. ``--support-email``.
    :param env: Environment variable the gunicorn images use instead.
    :param kind: ``"value"`` (single argument), ``"append"`` (repeatable, one
        entry per occurrence; the env form is a comma-separated list) or
        ``"switch"`` (``store_true``; can only *enable*, never loosen).
    :param persisted_getter: Name of the :mod:`vtsearch.settings` getter for
        the setting this overrides. Resolved lazily by
        ``settings.get_effective_override`` so this module never imports it.
    :param effective_key: Key the resolved value is published under in the
        ``/api/settings`` payload.
    :param parse: ``str -> value`` for one token, raising
        :class:`OverrideValueError`. Returning ``None`` means "no override"
        (used by the whitespace-only cases). Unused for ``"switch"``.
    :param resolve: ``(override, persisted) -> effective``. The override is
        whatever is stored (``empty_factory()`` when unset).
    :param fold: For ``"append"``, ``(store, parsed) -> store``; how one parsed
        token accumulates into the stored value.
    :param dump: ``effective -> JSON-safe``, for the ``/api/settings`` overlay.
    :param empty_factory: Builds the "no override set" value (``None`` for
        scalars, a fresh empty dict for the accumulating ones).
    """

    name: str
    flag: str
    env: str
    kind: str
    persisted_getter: str
    effective_key: str
    resolve: Callable[[Any, Any], Any]
    help: str
    metavar: str | None = None
    parse: Callable[[str], Any] | None = None
    fold: Callable[[Any, Any], Any] | None = None
    dump: Callable[[Any], Any] = lambda value: value
    empty_factory: Callable[[], Any] = lambda: None

    @property
    def dest(self) -> str:
        """argparse ``dest`` for this flag (the registry name)."""
        return self.name

    def parse_env(self, raw: str) -> Any:
        """Parse the *whole* env-var value into a stored override.

        Returns ``None`` when the variable asks for nothing (blank, or a
        switch set to something other than a truthy token). Raises
        :class:`OverrideValueError` with the variable named -- not the flag --
        so the operator is pointed at what they actually set.
        """
        if self.kind == "switch":
            return True if raw.strip().lower() in _TRUTHY else None
        if self.kind == "append":
            store = self.empty_factory()
            found = False
            for token in raw.split(","):
                token = token.strip()
                if not token:
                    continue
                parsed = self.parse_token(token, source=self.env)
                if parsed is None:
                    continue
                assert self.fold is not None  # every "append" override sets it
                store = self.fold(store, parsed)
                found = True
            return store if found else None
        return self.parse_token(raw, source=self.env)

    def parse_token(self, raw: str, *, source: str) -> Any:
        """Run :attr:`parse`, re-labelling the error with *source*.

        The validators are written against the flag spelling (that is what a
        CLI user sees). When the same value arrived from the environment, swap
        the label so the message names the variable instead.
        """
        assert self.parse is not None  # only "switch" overrides omit it
        try:
            return self.parse(raw)
        except OverrideValueError as exc:
            if source == self.flag:
                raise
            raise OverrideValueError(str(exc).replace(self.flag, source)) from None


# ---------------------------------------------------------------------------
# Validators / parsers (registry lookups are imported lazily)
# ---------------------------------------------------------------------------


def _parse_solo_media_type(raw: str) -> str:
    from vtscore.media import all_type_ids

    valid = set(all_type_ids())
    if raw not in valid:
        raise OverrideValueError(f"Unknown --solo-media-type: {raw!r}. Valid values: {sorted(valid)}")
    return raw


def _parse_solo_embedder(raw: str) -> tuple[str, str]:
    from vtscore.media import all_embedders, all_type_ids, embedders_for_type

    if "=" not in raw:
        raise OverrideValueError(f"Invalid --solo-embedder value: {raw!r}. Expected TYPE=EMBEDDER (e.g. image=siglip).")
    mt, _, emb = raw.partition("=")
    mt = mt.strip()
    emb = emb.strip()
    if not mt or not emb:
        raise OverrideValueError(f"Invalid --solo-embedder value: {raw!r}. Both TYPE and EMBEDDER must be non-empty.")
    valid_types = set(all_type_ids())
    if mt not in valid_types:
        raise OverrideValueError(
            f"Unknown mediaType in --solo-embedder {raw!r}: {mt!r}. Valid values: {sorted(valid_types)}"
        )
    if emb not in {e.name for e in all_embedders()}:
        raise OverrideValueError(
            f"Unknown embedder in --solo-embedder {raw!r}: {emb!r}. "
            f"Valid embedder names: {sorted(e.name for e in all_embedders())}"
        )
    valid_for_type = {e.name for e in embedders_for_type(mt)}
    if emb not in valid_for_type:
        raise OverrideValueError(
            f"Embedder {emb!r} is not registered for media type {mt!r}. "
            f"Valid embedders for {mt}: {sorted(valid_for_type)}"
        )
    return mt, emb


def _parse_hidden_plugin(raw: str) -> tuple[str, str]:
    from vtscore.plugins.inventory import FAMILIES

    valid_families = set(FAMILIES)
    if ":" not in raw:
        raise OverrideValueError(
            f"--hide-plugin expects FAMILY:NAME, got {raw!r}. Valid families: {sorted(valid_families)}"
        )
    family, _, plugin_name = raw.partition(":")
    family = family.strip()
    plugin_name = plugin_name.strip()
    if not family or not plugin_name:
        raise OverrideValueError(f"--hide-plugin {raw!r} has an empty family or name")
    if family not in valid_families:
        raise OverrideValueError(f"Unknown --hide-plugin family {family!r}. Valid: {sorted(valid_families)}")
    return family, plugin_name


def _parse_dataset_max_age_days(raw: str) -> int:
    try:
        days = int(raw)
    except (TypeError, ValueError):
        raise OverrideValueError(
            f"--dataset-max-age-days must be a positive integer (number of days), got {raw!r}"
        ) from None
    if days < 1:
        raise OverrideValueError("--dataset-max-age-days must be a positive integer (number of days)")
    return days


def _parse_support_email(raw: str) -> str:
    email = raw.strip()
    if not email:
        raise OverrideValueError("--support-email must be a non-empty address")
    return email


# ---------------------------------------------------------------------------
# Normalisers / resolvers
# ---------------------------------------------------------------------------


def normalize_hidden_plugins(value: Any) -> dict[str, set[str]]:
    """Coerce arbitrary input to ``{family: {name, ...}}`` form.

    Accepts ``dict[str, Iterable[str]]`` shapes and drops empty entries.
    Non-string keys / names and ``None`` values are silently skipped so a
    corrupt settings file doesn't crash plugin listings.
    """
    out: dict[str, set[str]] = {}
    if not isinstance(value, dict):
        return out
    for family, names in value.items():
        if not isinstance(family, str) or not family:
            continue
        if isinstance(names, (str, bytes)):
            continue
        try:
            members = {n for n in names if isinstance(n, str) and n}
        except TypeError:
            continue
        if members:
            out[family] = members
    return out


def _resolve_scalar(override: Any, persisted: Any) -> Any:
    """The plain two-step precedence: an override set at startup, else the file."""
    return persisted if override is None else override


def _resolve_solo_media_type(override: str | None, persisted: Any) -> str | None:
    if override is not None:
        return override
    # Empty string from JSON drift normalises to None.
    if isinstance(persisted, str) and not persisted.strip():
        return None
    return persisted


def _resolve_semantic_only(override: bool | None, persisted: Any) -> bool:
    return bool(persisted) if override is None else override


def _resolve_solo_embedders(override: dict[str, str], persisted: Any) -> dict[str, str]:
    """Layer the per-user map over the process-level fallbacks, per key.

    User entries win per-key; missing user keys fall through to the startup
    value. An **empty-string value** in the user map is a per-type opt-out
    sentinel -- it removes that type from the merged map even if the startup
    fallback has a value for it. (``solo_media_type`` has no such per-user
    opt-out: it is an admin-set server restriction.)

    Validity (does the embedder still exist for this type?) is *not* checked
    here -- the frontend resolves it against the live embedder registry on its
    end and falls back to the normal picker for any entry that no longer
    matches. Keeping validation client-side means a rename or removal never
    blocks the settings UI from rendering.
    """
    merged: dict[str, str] = {}
    for mt, emb in override.items():
        if mt and isinstance(emb, str) and emb.strip():
            merged[mt] = emb.strip()
    if isinstance(persisted, dict):
        for mt, emb in persisted.items():
            if not isinstance(mt, str) or not mt:
                continue
            if isinstance(emb, str) and emb.strip():
                merged[mt] = emb.strip()
            else:
                # Empty-string sentinel - user explicitly opted out for this
                # type, so drop the startup fallback too.
                merged.pop(mt, None)
    return merged


def _resolve_hidden_plugins(override: dict[str, set[str]], persisted: Any) -> dict[str, set[str]]:
    """Union the persisted hide list with the startup one.

    The union semantics matter: a plugin is hidden if either source asks for
    it, so ``--hide-plugin`` can only add hides (never un-hide something the
    settings file marks hidden).
    """
    merged = {family: set(names) for family, names in normalize_hidden_plugins(persisted).items()}
    for family, names in override.items():
        merged.setdefault(family, set()).update(names)
    return merged


def _fold_pair(store: dict[str, str], parsed: tuple[str, str]) -> dict[str, str]:
    key, value = parsed
    store[key] = value
    return store


def _fold_family(store: dict[str, set[str]], parsed: tuple[str, str]) -> dict[str, set[str]]:
    family, name = parsed
    store.setdefault(family, set()).add(name)
    return store


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_REGISTRY: tuple[AdminOverride, ...] = (
    AdminOverride(
        name="solo_media_type",
        flag="--solo-media-type",
        env="VTSEARCH_SOLO_MEDIA_TYPE",
        kind="value",
        persisted_getter="get_solo_media_type",
        effective_key="solo_media_type",
        parse=_parse_solo_media_type,
        resolve=_resolve_solo_media_type,
        help=(
            "Streamline the UI for a single mediaType. Hides mediaType pickers "
            "in the dataset-importer and new-detector flows, locks them to the "
            "given type, filters converter offerings to converters whose output "
            "is this type, and preloads that type's default embedder at startup. "
            "This is an admin-set restriction: it applies to every user, and "
            "users cannot change or opt out of it from the settings UI. "
            "Overrides the persisted ``solo_media_type`` key in the server "
            "settings file for the lifetime of the process. Valid values are "
            "the registered media-type ids (e.g. audio, image, video, text, "
            "document)."
        ),
    ),
    AdminOverride(
        name="solo_embedders",
        flag="--solo-embedder",
        env="VTSEARCH_SOLO_EMBEDDERS",
        kind="append",
        metavar="TYPE=EMBEDDER",
        persisted_getter="get_solo_embedder_per_media_type",
        effective_key="effective_solo_embedder_per_media_type",
        parse=_parse_solo_embedder,
        fold=_fold_pair,
        resolve=_resolve_solo_embedders,
        empty_factory=dict,
        help=(
            "Lock a single embedder for a mediaType so the dataset-importer "
            "modal hides its embedder picker for that type and silently uses "
            "the named embedder. Repeatable, one --solo-embedder per mediaType "
            "(e.g. --solo-embedder image=siglip --solo-embedder audio=clap). "
            "Format is TYPE=EMBEDDER, where TYPE is a registered media-type id "
            "and EMBEDDER is a registered embedder name for that type. Acts as "
            "a per-process fallback. Any user who sets their own value via "
            "the settings UI overrides this flag per-mediaType for themselves. "
            "Other mediaTypes still show the normal embedder picker. Also "
            "settable as a comma-separated VTSEARCH_SOLO_EMBEDDERS."
        ),
    ),
    AdminOverride(
        name="hidden_plugins",
        flag="--hide-plugin",
        env="VTSEARCH_HIDE_PLUGINS",
        kind="append",
        metavar="FAMILY:NAME",
        persisted_getter="get_hidden_plugins",
        effective_key="hidden_plugins",
        parse=_parse_hidden_plugin,
        fold=_fold_family,
        resolve=_resolve_hidden_plugins,
        dump=lambda merged: {family: sorted(names) for family, names in merged.items()},
        empty_factory=dict,
        help=(
            "Hide a plugin from picker / listing API responses for this "
            "process (declutter the UI without editing the codebase). "
            "Repeatable. FAMILY is a plugin-family id (importers, exporters, "
            "label_importers, labelset_sources, converters, media_sources, "
            "media_types, embedders, clippers, settings_importers, "
            "settings_exporters, settings_sources); NAME is the plugin's "
            "registered name. Hidden plugins remain importable and callable "
            "by name via execution endpoints (e.g. autodetect, label "
            "import). This is a UI flag, not a security boundary. Merges "
            "with the persisted ``hidden_plugins`` key in the server "
            "settings file. Use ``--list-plugins --format names`` to see "
            "the available family:name pairs. Also settable as a "
            "comma-separated VTSEARCH_HIDE_PLUGINS."
        ),
    ),
    AdminOverride(
        name="dataset_max_age_days",
        flag="--dataset-max-age-days",
        env="VTSEARCH_DATASET_MAX_AGE_DAYS",
        kind="value",
        metavar="DAYS",
        persisted_getter="get_dataset_max_age_days",
        effective_key="dataset_max_age_days",
        parse=_parse_dataset_max_age_days,
        resolve=_resolve_scalar,
        help=(
            "Stamp every dataset created by this server process with an "
            "expiry DAYS days after creation; expired datasets are aged off "
            "from the registry. Applies to all users and overrides any "
            "persisted dataset_max_age_days in the settings file for the "
            "lifetime of the process; the value is not editable via the "
            "settings API. Must be a positive integer. Omit to use the "
            "persisted value (no expiry if none is set)."
        ),
    ),
    AdminOverride(
        name="support_email",
        flag="--support-email",
        env="VTSEARCH_SUPPORT_EMAIL",
        kind="value",
        metavar="ADDRESS",
        persisted_getter="get_support_email",
        effective_key="support_email",
        parse=_parse_support_email,
        resolve=_resolve_scalar,
        help=(
            "Recipient address for the Help modal's 'Email us' contact link. "
            "Applies to all users and overrides any persisted support_email in "
            "the settings file for the lifetime of the process; the value is "
            "not editable via the settings API. Omit to use the persisted value "
            "(defaults to the built-in project address)."
        ),
    ),
    AdminOverride(
        name="semantic_only",
        flag="--semantic-only",
        env="VTSEARCH_SEMANTIC_ONLY",
        kind="switch",
        persisted_getter="get_semantic_only",
        effective_key="semantic_only",
        resolve=_resolve_semantic_only,
        help=(
            "Lock this instance to Semantic embedders: the prototype Patch "
            "Semantic and Structural embedder types are hidden from every "
            "picker and rejected by the dataset-load and detector-create "
            "routes. Applies to all users for the lifetime of the process and "
            "is not editable via the settings API. There is no "
            "--no-semantic-only: the flag can only enable the lock, never "
            "loosen one the persisted semantic_only setting asked for."
        ),
    ),
)

#: ``name -> AdminOverride``, in registration order. The order is the one the
#: ``/api/settings`` overlay and the startup log walk, so it is worth keeping
#: readable rather than alphabetical.
OVERRIDES: dict[str, AdminOverride] = {ov.name: ov for ov in _REGISTRY}


# ---------------------------------------------------------------------------
# Process-level store
# ---------------------------------------------------------------------------

#: The values in force for this process. Every entry starts at its override's
#: ``empty_factory()`` ("nothing set"); ``settings.get_effective_override``
#: falls through to the persisted setting for those.
_values: dict[str, Any] = {name: ov.empty_factory() for name, ov in OVERRIDES.items()}

#: Where each set value came from (``"--support-email"``,
#: ``"VTSEARCH_SUPPORT_EMAIL"``), for the startup banner. ``None`` while unset.
_sources: dict[str, str | None] = dict.fromkeys(OVERRIDES)


def get_override(name: str) -> Any:
    """Return the stored override for *name* (its empty value when unset).

    Dict-valued overrides are copied defensively, so a caller cannot mutate
    the process-level store by accident.
    """
    value = _values[name]
    if isinstance(value, dict):
        return {k: (set(v) if isinstance(v, set) else v) for k, v in value.items()}
    return value


def set_override(name: str, value: Any, *, source: str | None = None) -> None:
    """Store *value* as the override for *name* (``None`` / empty clears it).

    This is the un-validated entry point the ``settings.set_cli_X`` wrappers
    delegate to: it stores whatever it is given. Values arriving from an
    operator go through :func:`apply_flag_values` or
    :func:`apply_env_overrides`, which validate first.
    """
    ov = OVERRIDES[name]
    _values[name] = ov.empty_factory() if value is None else value
    _sources[name] = source if value is not None else None


def override_source(name: str) -> str | None:
    """Return what set this override (a flag or env-var name), or ``None``."""
    return _sources[name]


def snapshot() -> dict[str, Any]:
    """Return a deep-enough copy of the whole store, for tests to restore."""
    return {
        "values": {name: get_override(name) for name in OVERRIDES},
        "sources": dict(_sources),
    }


def restore(state: dict[str, Any]) -> None:
    """Put back a :func:`snapshot`."""
    for name in OVERRIDES:
        _values[name] = state["values"][name]
    _sources.update(state["sources"])


def reset_overrides() -> None:
    """Clear every override (used by tests between cases)."""
    for name, ov in OVERRIDES.items():
        _values[name] = ov.empty_factory()
        _sources[name] = None


# ---------------------------------------------------------------------------
# Entry points: argparse and the environment
# ---------------------------------------------------------------------------


def register_override_flags(parser: Any) -> None:
    """Add every override's CLI flag to *parser*.

    Value-taking flags are registered as ``type=str`` even where the value is
    numeric, so the descriptor's own validator produces the error message
    rather than argparse's generic "invalid int value" -- and so the flag and
    the env var fail identically.
    """
    for ov in OVERRIDES.values():
        if ov.kind == "switch":
            parser.add_argument(ov.flag, action="store_true", default=False, dest=ov.dest, help=ov.help)
        elif ov.kind == "append":
            parser.add_argument(ov.flag, action="append", default=[], dest=ov.dest, metavar=ov.metavar, help=ov.help)
        else:
            parser.add_argument(ov.flag, type=str, default=None, dest=ov.dest, metavar=ov.metavar, help=ov.help)


def apply_flag_values(args: Any) -> None:
    """Validate and stash whatever argparse parsed for the override flags.

    Raises :class:`OverrideValueError` on the first bad value;
    :mod:`vtsearch.cli_main` turns that into ``parser.error``. Runs before any
    dataset is created or any listing endpoint is served, so a typo fails fast
    instead of silently no-opping the restriction.
    """
    for ov in OVERRIDES.values():
        raw = getattr(args, ov.dest, None)
        if ov.kind == "switch":
            if raw:
                set_override(ov.name, True, source=ov.flag)
            continue
        if ov.kind == "append":
            tokens: Iterable[str] = raw or []
            store = ov.empty_factory()
            found = False
            for token in tokens:
                parsed = ov.parse_token(token, source=ov.flag)
                if parsed is None:
                    continue
                assert ov.fold is not None
                store = ov.fold(store, parsed)
                found = True
            if found:
                set_override(ov.name, store, source=ov.flag)
            continue
        if raw is not None:
            set_override(ov.name, ov.parse_token(raw, source=ov.flag), source=ov.flag)


def apply_env_overrides(*, warn: Callable[[str], None] | None = None) -> list[tuple[AdminOverride, Any]]:
    """Apply the env-var form of every override that no flag already set.

    This is what makes the knobs reachable under Docker: the gunicorn-launched
    images import ``app.py`` without ever parsing ``argv``, so the flags above
    are unreachable there. An explicit flag always wins.

    A malformed value is reported through *warn* (default: ``print``) and the
    override is left unset -- a typo in a container's environment should not
    stop the server booting, which is the opposite of the CLI's fail-fast
    stance where a human is watching.

    Returns the ``(override, value)`` pairs actually applied, so the caller can
    log them.
    """
    emit = warn if warn is not None else (lambda msg: print(msg, flush=True))
    applied: list[tuple[AdminOverride, Any]] = []
    for ov in OVERRIDES.values():
        if override_source(ov.name) is not None:
            continue
        raw = os.environ.get(ov.env)
        if not raw or not raw.strip():
            continue
        try:
            value = ov.parse_env(raw)
        except OverrideValueError as exc:
            emit(f"⚠️  Ignoring {ov.env}={raw!r}: {exc}")
            continue
        if value is None:
            continue
        set_override(ov.name, value, source=ov.env)
        applied.append((ov, value))
    return applied
