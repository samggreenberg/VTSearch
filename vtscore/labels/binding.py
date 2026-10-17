"""Which labelset source a detector syncs with, and with what field values.

A detector's link to a :class:`~vtscore.labels.sources.base.LabelsetSource`
can come from three places, and :func:`resolve_labelset_source` is the one
resolver every sync path reads, so the push, the pull, the routes and the
Dashboard cannot disagree about it:

1. the detector's own **explicit binding**, stored on its registry entry
   under ``labelset_source`` (``{"source_name": ..., "field_values": ...}``);
   ``{"source_name": "none"}`` is an explicit opt-out that wins even over a
   deployment default;
2. otherwise a binding set on its loaded
   :class:`~vtscore.state.core.DetectorContext` (``ctx.labelset_source``),
   which lives only in memory - the library-level way to link a source
   without a registry;
3. otherwise the deployment-wide **default** (the ``default_labelset_source``
   setting, the app's server setting of that name), which the detector
   *inherits*;
4. otherwise no source.

The binding lives on the registry entry, not in the detector JSON, because it
is deployment configuration rather than part of the detector: a copied or
combined detector must not inherit the original's sync target (it would
overwrite the original's labels there), and listing the Dashboard must not
parse every detector file to learn which ones sync.

**Fixed fields.**  Each non-empty value in the default's ``field_values`` is
fixed by the admin.  Whenever a detector syncs to the default's source -
explicitly or by inheritance - those values overlay its own, so a user can
never point a fixed field elsewhere, and the fixed values never need to be
stored per detector (or sent to a browser).  The New Detector form asks only
for the source's *other* fields (:func:`unfixed_fields`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vtscore.labels.sources.base import LabelsetSource
    from vtscore.plugins import PluginField

logger = logging.getLogger(__name__)

__all__ = [
    "OPT_OUT",
    "ResolvedLabelsetSource",
    "deployment_default_labelset_source",
    "fixed_field_values",
    "labelset_source_status",
    "resolve_labelset_source",
    "stored_labelset_source",
    "unfixed_fields",
]

#: ``source_name`` of an explicit opt-out: the detector syncs nowhere, even
#: when the deployment has a default.
OPT_OUT = "none"

# Unknown source names are logged once per name, not once per vote.
_warned_unknown: set[str] = set()


@dataclass(frozen=True)
class ResolvedLabelsetSource:
    """The labelset source a detector syncs with, resolved.

    Attributes:
        source_name: Registered name of the labelset source plugin.
        field_values: The values to hand the source: the detector's own, with
            the deployment default's fixed values laid over them when the
            source is the default's.  Not yet normalized (templates still
            unresolved).
        inherited: ``True`` when the detector has no binding of its own and
            syncs to the deployment default.
        fixed_keys: Field keys whose values the deployment default fixes for
            this source (empty when the source is not the default's).
        missing_fields: Keys of required fields that have neither a value nor
            a declared default.  A detector with any is not synced until they
            are filled in.
    """

    source_name: str
    field_values: dict[str, Any]
    inherited: bool = False
    fixed_keys: frozenset[str] = frozenset()
    missing_fields: tuple[str, ...] = field(default_factory=tuple)

    @property
    def source(self) -> LabelsetSource | None:
        """The registered source plugin, or ``None`` if the name is unknown."""
        from vtscore.labels.sources import get_labelset_source  # noqa: PLC0415

        return get_labelset_source(self.source_name)

    @property
    def ready(self) -> bool:
        """Whether every required field has a value, so the source can be used."""
        return not self.missing_fields


def _valid_config(cfg: Any) -> dict[str, Any] | None:
    """Return *cfg* when it is a ``{"source_name": <non-empty str>, ...}`` dict."""
    if not isinstance(cfg, dict):
        return None
    name = cfg.get("source_name")
    if not isinstance(name, str) or not name.strip():
        return None
    return cfg


def deployment_default_labelset_source() -> dict[str, Any] | None:
    """Return the deployment-wide default binding, or ``None``.

    Read with :func:`~vtscore.config.get_setting` so the library stays
    settings-import-free; a library-only process that configured none simply
    has no default.  An opt-out or malformed value reads as no default.
    """
    from vtscore.config import get_setting  # noqa: PLC0415

    try:
        cfg = get_setting("default_labelset_source")
    except Exception:  # noqa: BLE001 - unreadable settings
        return None
    cfg = _valid_config(cfg)
    if cfg is None or cfg["source_name"].strip() == OPT_OUT:
        return None
    return cfg


def fixed_field_values(default: dict[str, Any] | None) -> dict[str, Any]:
    """The values *default* fixes: every non-blank entry of its ``field_values``."""
    if not default:
        return {}
    values = default.get("field_values") or {}
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items() if v is not None and str(v).strip() != ""}


def unfixed_fields(source: LabelsetSource, default: dict[str, Any] | None) -> list[PluginField]:
    """The fields of *source* a user still fills in under *default*.

    Every field the default does not fix, minus ``hidden`` ones (those carry
    the plugin author's value, never the user's).  When *default* names a
    different source, nothing is fixed and every visible field is returned.
    """
    fixed: set[str] = set()
    if default and default.get("source_name") == source.name:
        fixed = set(fixed_field_values(default))
    return [f for f in source.fields if f.key not in fixed and not f.hidden]


def stored_labelset_source(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """The explicit binding stored on a registry *entry*, or ``None``."""
    if not entry:
        return None
    return _valid_config(entry.get("labelset_source"))


def _missing_required(source: LabelsetSource, values: dict[str, Any]) -> tuple[str, ...]:
    """Keys of *source*'s required fields with no value and no declared default."""
    missing = []
    for f in source.fields:
        if f.field_type == "file" or not f.required or f.default:
            continue
        value = values.get(f.key)
        if value is None or str(value).strip() == "":
            missing.append(f.key)
    return tuple(missing)


def _context_binding(detector_id: str) -> dict[str, Any] | None:
    """The in-memory binding on the loaded context for *detector_id*, if any."""
    if not detector_id:
        return None
    from vtscore.state.core import get_detector_context  # noqa: PLC0415

    ctx = get_detector_context(detector_id)
    return _valid_config(ctx.labelset_source) if ctx is not None else None


#: Sentinel for "read the deployment default yourself".
_READ_DEFAULT: Any = object()


def resolve_labelset_source(
    entry: dict[str, Any] | None,
    *,
    detector_id: str = "",
    default: dict[str, Any] | None = _READ_DEFAULT,
) -> ResolvedLabelsetSource | None:
    """Resolve which labelset source a detector syncs with.

    Args:
        entry: The detector's registry entry (its ``labelset_source`` key is
            the explicit binding), or ``None`` for a detector with no entry.
        detector_id: Id of the detector, for the in-memory context binding.
            Defaults to ``entry["id"]``.
        default: The deployment default to resolve against, when the caller
            already read it (a listing resolving many detectors); read with
            :func:`deployment_default_labelset_source` when omitted.

    Returns:
        The resolved binding, or ``None`` when the detector syncs nowhere: it
        opted out, nothing binds it, or the bound source is not registered
        (logged once per name).
    """
    detector_id = detector_id or (entry or {}).get("id", "") or ""
    if default is _READ_DEFAULT:
        default = deployment_default_labelset_source()

    explicit = stored_labelset_source(entry) or _context_binding(detector_id)
    if explicit is not None:
        source_name = explicit["source_name"].strip()
        if source_name == OPT_OUT:
            return None
        values = dict(explicit.get("field_values") or {})
        inherited = False
    elif default is not None:
        source_name = default["source_name"].strip()
        values = {}
        inherited = True
    else:
        return None

    from vtscore.labels.sources import get_labelset_source  # noqa: PLC0415

    source = get_labelset_source(source_name)
    if source is None:
        if source_name not in _warned_unknown:
            _warned_unknown.add(source_name)
            logger.warning("Unknown labelset source %r; detectors bound to it are not synced", source_name)
        return None

    fixed: dict[str, Any] = {}
    if default is not None and default["source_name"].strip() == source_name:
        fixed = fixed_field_values(default)
        values.update(fixed)

    return ResolvedLabelsetSource(
        source_name=source_name,
        field_values=values,
        inherited=inherited,
        fixed_keys=frozenset(fixed),
        missing_fields=_missing_required(source, values),
    )


def labelset_source_status(
    entry: dict[str, Any] | None,
    *,
    detector_id: str = "",
    default: dict[str, Any] | None = _READ_DEFAULT,
) -> dict[str, Any] | None:
    """A detector's sync status for display: source, inheritance, missing fields.

    ``None`` when the detector syncs nowhere.  Carries no field values, so it
    is safe to list for every user who can see the detector.
    """
    resolved = resolve_labelset_source(entry, detector_id=detector_id, default=default)
    if resolved is None:
        return None
    source = resolved.source
    return {
        "source_name": resolved.source_name,
        "display_name": (source.display_name if source is not None else "") or resolved.source_name,
        "inherited": resolved.inherited,
        "missing_fields": list(resolved.missing_fields),
    }
