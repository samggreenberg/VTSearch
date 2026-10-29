"""The mixin every plugin class inherits from.

:class:`PluginBase` supplies the CLI-flag, validation, notification and
serialisation helpers that are identical across every plugin family, and
auto-derives a plugin's default ``name`` / ``display_name`` /
``description`` / ``icon`` from its class when the author leaves them unset.

Import it from :mod:`vtscore.plugins`, which re-exports it.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Any

from vtscore.concurrency.notifications import Notification, notify
from vtscore.plugins.field import FieldOption, PluginField, inactive_field_keys


# ---------------------------------------------------------------------------
# PluginBase: shared base class with CLI and serialisation helpers
# ---------------------------------------------------------------------------


#: Historical class-name suffixes stripped before snake-casing for the
#: default :attr:`PluginBase.name`.
#:
#: **This tuple is a compatibility contract, not a lookup table to tidy.**
#: A derived ``name`` is a registry key: it is what ``get_exporter("…")``
#: resolves, what a third party writes into an entry-point config, what
#: ``origin.params`` records, and what persisted settings store.  These
#: sixteen literals are the suffixes that have *ever* been stripped, so
#: removing one silently renames every out-of-tree plugin whose class name
#: ends in it — on a user's install, with no error.  Entries may be added
#: (a name that derived nothing before starts deriving something); they may
#: never be removed or edited.
#:
#: Order is **not** load-bearing: :func:`_default_plugin_name` strips the
#: *longest* matching suffix, and no entry here is a proper suffix of
#: another except via the four generic tail entries, which are shorter than
#: everything they appear in.  ``tests_lib/core/test_plugin_name_suffix_contract.py``
#: proves both halves of that.
#:
#: ``LabelsetExporter`` earns its place beside ``ResultsExporter``: it is the
#: results-exporter base class's permanent module-level alias, so an
#: out-of-tree ``FooLabelsetExporter`` must keep deriving ``foo`` even though
#: no class of that name exists.  ``MediaSource`` likewise names an abstract
#: base that is *not* a :class:`PluginBase`, so it can never contribute
#: itself dynamically.
_LEGACY_PLUGIN_NAME_SUFFIXES: tuple[str, ...] = (
    "DataSourceImporter",
    "DatasetImporter",
    "LabelsetExporter",
    "ResultsExporter",
    "LabelImporter",
    "LabelsetSource",
    "SeedImporter",
    "SettingsImporter",
    "SettingsExporter",
    "SettingsSource",
    "MediaConverter",
    "MediaSource",
    "Importer",
    "Exporter",
    "Source",
    "Converter",
)


def _snake_case(name: str) -> str:
    """Convert a CamelCase / PascalCase identifier to snake_case."""
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (not name[i - 1].isupper() or (i + 1 < len(name) and name[i + 1].islower())):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _is_family_base(cls: type) -> bool:
    """Return True if *cls*'s own body marks it a plugin-family base.

    Read from ``__dict__`` rather than via :func:`getattr` so the marker
    never leaks down the MRO: a family base's concrete subclasses are
    ordinary plugins, not bases themselves.
    """
    return bool(cls.__dict__.get("_is_plugin_family_base", False))


def _family_base_suffixes(cls: type) -> list[str]:
    """Return the strippable class names contributed by *cls*'s own bases.

    Every plugin-family base in the MRO contributes its ``__name__`` as a
    suffix its subclasses may strip, so declaring a new family means marking
    exactly one class rather than also editing a central table.  A base sets
    ``_strippable_family_base = False`` to withhold its name — see
    :attr:`PluginBase._strippable_family_base`.

    Scoped to *cls*'s own MRO, not to a global registry, so the result can't
    depend on which unrelated modules happen to have been imported first.
    """
    return [
        base.__name__
        for base in cls.__mro__[1:]
        if _is_family_base(base) and base.__dict__.get("_strippable_family_base", True)
    ]


def _default_plugin_name(cls: type) -> str:
    """Derive the snake_case registry key for *cls* from its class name.

    Strips the **longest** family suffix the name ends in — from
    :data:`_LEGACY_PLUGIN_NAME_SUFFIXES` or from a family base in *cls*'s own
    MRO — and snake-cases what is left.  A name that *is* a suffix keeps it
    (``Exporter`` derives ``exporter``, not ``""``) but may still shed a
    shorter one (``MediaSource`` sheds ``Source`` and derives ``media``).

    Longest-match rather than first-match: two suffixes can both match a
    class name only when one is a suffix of the other, and there the more
    specific one is always what the author meant.
    """
    raw = cls.__name__
    candidates = set(_LEGACY_PLUGIN_NAME_SUFFIXES) | set(_family_base_suffixes(cls))
    matches = [s for s in candidates if raw.endswith(s) and raw != s]
    if matches:
        raw = raw[: -len(max(matches, key=len))]
    return _snake_case(raw)


def _default_plugin_display_name(name: str) -> str:
    return " ".join(word.capitalize() for word in name.split("_") if word)


def _default_plugin_description(cls: type) -> str:
    doc = (cls.__doc__ or "").strip()
    if not doc:
        return ""
    return doc.splitlines()[0].strip()


def _default_plugin_letter_icon(cls: type) -> str:
    """Return the first alphabetic character of *cls*'s display name (or
    its snake_case ``name`` as a fallback), upper-cased.

    Gives every plugin a distinguishing default icon (a boxed capital
    letter, rendered by the frontend's ``vt-icon`` component) without its
    author having to design or pick one.  Returns ``""`` when neither
    resolves to a usable string, e.g. a
    :class:`~vtscore.converters.base.MediaConverter` subclass, whose
    ``name`` / ``display_name`` are computed properties rather than plain
    class attributes at ``__init_subclass__`` time.
    """
    for attr in ("display_name", "name"):
        value = getattr(cls, attr, None)
        if isinstance(value, str):
            for ch in value:
                if ch.isalpha():
                    return ch.upper()
    return ""


def _mro_provides(cls: type, attr: str) -> bool:
    """Return True if any ancestor (above *cls*) already provides *attr*
    as a non-empty string or a descriptor (e.g. a ``property``).

    Used to decide whether the auto-default should fire; we never
    overwrite an inherited descriptor or a concrete string supplied by
    a parent (e.g. :class:`MediaConverter.name`, which is a property).
    """
    for base in cls.__mro__[1:]:
        if attr in base.__dict__:
            val = base.__dict__[attr]
            if isinstance(val, str):
                if val:
                    return True
            elif hasattr(val, "__get__"):
                return True
    return False


def _inherits_family_stock_icon(cls: type) -> bool:
    """Return True if *cls*'s inherited :attr:`icon` comes from a family base.

    A plugin family stamps a generic emoji on its abstract base (a plug for
    dataset importers, an outbox tray for exporters, …) so the family is
    recognisable before anyone writes a plugin.  Inheriting it means the
    author never picked an icon, so :func:`_autoderive_plugin_metadata`
    replaces it with a distinguishing letter glyph.

    Decided by *where the icon is defined* rather than by comparing its
    codepoints against a table of the emoji we happen to ship: a table
    cannot see a third-party family's stock icon, and gets it wrong in the
    other direction too when a plugin deliberately picks an emoji that a
    base elsewhere also uses.
    """
    for base in cls.__mro__[1:]:
        if "icon" in base.__dict__:
            return _is_family_base(base)
    return False


def _autoderive_plugin_metadata(cls: type) -> None:
    """Fill in default :attr:`name` / :attr:`display_name` /
    :attr:`description` / :attr:`icon` on *cls* when neither *cls* itself
    nor any ancestor already provides them.

    Called from :meth:`PluginBase.__init_subclass__`.  Classes that mark
    themselves a family base with ``_is_plugin_family_base = True`` — the
    in-tree abstract bases and any third-party intermediate — skip
    auto-derivation entirely, so they don't leak a derived name down to
    their concrete subclasses.
    """
    if _is_family_base(cls):
        return
    if "name" not in cls.__dict__ and not _mro_provides(cls, "name"):
        cls.name = _default_plugin_name(cls)
    if "display_name" not in cls.__dict__ and not _mro_provides(cls, "display_name"):
        derived_name = getattr(cls, "name", None)
        cls.display_name = _default_plugin_display_name(derived_name) if isinstance(derived_name, str) else ""
    if "description" not in cls.__dict__ and not _mro_provides(cls, "description"):
        cls.description = _default_plugin_description(cls)
    if "icon" not in cls.__dict__:
        if not getattr(cls, "icon", "") or _inherits_family_stock_icon(cls):
            letter = _default_plugin_letter_icon(cls)
            if letter:
                cls.icon = letter


class PluginBase:
    """Mixin providing the CLI-argument, validation, and serialisation helpers
    that are identical across every plugin family.

    Default metadata
    ----------------
    Subclasses that don't declare :attr:`name`, :attr:`display_name`,
    :attr:`description`, or :attr:`icon` get auto-derived defaults via
    :meth:`__init_subclass__`:

    - :attr:`name`: class name with the trailing family suffix
      (``DatasetImporter`` / ``ResultsExporter`` / ``MediaConverter`` /
      etc.) stripped and the remainder snake-cased.  E.g.
      ``MyShinyExporter`` → ``"my_shiny"``.  The strippable suffixes are
      :data:`_LEGACY_PLUGIN_NAME_SUFFIXES` plus the ``__name__`` of every
      family base in the class's own MRO, so declaring a new family needs
      no edit here — see :attr:`_is_plugin_family_base`.
    - :attr:`display_name`: title-cased :attr:`name`.
    - :attr:`description`: first line of the class docstring.
    - :attr:`icon`: the first letter of :attr:`display_name`, upper-cased
      (e.g. ``MyShinyExporter`` → ``"M"``).  The frontend renders any
      single capital letter as a boxed letter-glyph icon, so a plugin
      author who hasn't designed a custom icon still gets a
      distinguishing default instead of every plugin in the family
      sharing the same generic emoji.  This only fires when *cls* hasn't
      set its own ``icon`` and would otherwise inherit either nothing
      (``""``) or an icon defined by a family base (e.g.
      :attr:`~vtscore.datasets.importers.base.core.ImporterBase.icon`),
      which is the family's stock glyph rather than a chosen one; an
      author is always free to set a fancier emoji or SVG-type string.

    Explicit declarations always win.  The defaults only fire when
    nothing further up the MRO already provides a string value or a
    descriptor (e.g. :class:`~vtscore.converters.base.MediaConverter`
    declares ``name`` as a property, so concrete converter subclasses
    inherit the property rather than getting a stomped string).
    """

    #: Set to ``True`` in a class's *own* body to mark it a plugin-family
    #: base: an abstract intermediate that groups concrete plugins rather
    #: than being one.  A family base gets no auto-derived metadata (a
    #: derived ``name`` on a base would be inherited by every concrete
    #: subclass that doesn't declare its own, shadowing the per-subclass
    #: default before it can fire), its ``icon`` is treated as the family's
    #: stock glyph rather than a chosen one, and its ``__name__`` becomes a
    #: suffix its subclasses strip when deriving their own names.  Never
    #: inherited — it is read from ``__dict__``, so each base declares it.
    _is_plugin_family_base: bool = False

    #: Whether this family base's ``__name__`` is strippable by its
    #: subclasses.  Set to ``False`` on a base whose name a third-party
    #: subclass may already end in *without* it having been stripped
    #: historically: making it strippable would rename that plugin.  Only
    #: meaningful alongside :attr:`_is_plugin_family_base`.
    _strippable_family_base: bool = True

    #: Internal snake_case identifier used in API routes.
    name: str
    #: Human-readable label shown in the UI.
    display_name: str
    #: One-sentence description shown as a subtitle in the UI.
    description: str
    #: Emoji or icon string shown next to the display name.  Left unset,
    #: this defaults to a boxed capital-letter glyph derived from
    #: :attr:`display_name` (see "Default metadata" above).
    icon: str = ""
    #: Ordered list of fields the user must fill.
    fields: list[PluginField]

    #: A listing filter: ``GET /api/dataset/importers`` lists only importers
    #: whose value is ``"form"`` (the default; a generic form built from
    #: :attr:`fields` drives them).  Shipped importers with a bespoke import
    #: path set ``"custom"`` (demo, local folder / files, combine datasets)
    #: or ``"file_upload"`` (pickle) to stay out of that list.  The value is
    #: serialised by :meth:`to_dict`, but no frontend code reads it.
    ui_mode: str = "form"

    #: When ``True``, this plugin is excluded from the generic picker list
    #: in the frontend.  Useful for plugins that are always invoked through
    #: a dedicated code path (e.g. the GUI exporter).
    hidden_from_picker: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _autoderive_plugin_metadata(cls)

    def resolve_display_name(self, field_values: dict[str, Any] | None) -> str:
        """Return a human-readable name for a dataset loaded with *field_values*.

        The default returns :attr:`display_name`.  Subclasses (e.g. the demo
        importer) can override this to return a dataset-specific label.
        """
        return self.display_name

    # -- Dynamic field options ----------------------------------------------

    def get_field_options(self, field_key: str, current_values: dict[str, Any]) -> list[FieldOption]:
        """Return the dropdown options for a ``dynamic_options`` field.

        Override this on any plugin declaring a :class:`PluginField` with
        ``dynamic_options=True``.  The frontend calls it through that
        family's options route (e.g.
        ``POST /api/exporters/field-options/<name>``) when the form is
        first built and again whenever a field listed in this field's
        :attr:`~PluginField.depends_on` changes.

        Args:
            field_key: :attr:`PluginField.key` of the field whose options
                are being requested.
            current_values: Snapshot of every form field's current value,
                keyed by :attr:`PluginField.key`.  Values are plain
                strings (empty for unfilled fields).

        Returns:
            The allowed options.  Each is either a plain string (shown
            verbatim) or a ``(value, label)`` tuple (the option submits
            the opaque ``value`` while displaying ``label``).

        Raises:
            NotImplementedError: When the plugin declares no dynamic
                fields, or does not handle *field_key*.  Subclasses should
                delegate to ``super()`` for keys they do not recognise.
        """
        raise NotImplementedError(f"{type(self).__name__}.get_field_options({field_key!r}) is not implemented")

    # -- User notifications -------------------------------------------------

    def notify(
        self,
        message: str,
        *,
        level: str = "info",
        detail: str | None = None,
        items: Iterable[str] | None = None,
    ) -> Notification:
        """Put a toast in front of every user currently watching the app.

        The escape hatch for a problem that is worth telling the user about
        but not worth failing over: a source that returned fewer rows than
        asked for, a handful of files that would not decode, a remote API
        that rate-limited us into partial results.  Raising would throw away
        the work that *did* succeed; logging alone means nobody finds out.

        The call cannot fail, so it is safe inside an ``except`` block or a
        tight loop, and it never needs its own ``try``.  :attr:`display_name`
        is attached as the notification's source, so the toast says which
        plugin spoke.  See :mod:`vtscore.concurrency.notifications` for the
        delivery semantics (live broadcast, no replay).

        Args:
            message: Headline, one short sentence.
            level: ``"info"`` (default), ``"success"``, ``"warning"``, or
                ``"error"``.  The first two fade on their own; the last two
                stay until the user dismisses them.
            detail: Optional second line carrying the specifics.
            items: Optional list of the things the message is about, one
                entry each (every skipped file, not the first ten).  The app
                shows them behind a *Details* toggle with a *Copy List*
                button; see :func:`vtscore.concurrency.notifications.notify`
                for the caps.

        Example::

            if skipped:
                self.notify(
                    f"Skipped {len(skipped)} unreadable files",
                    level="warning",
                    items=skipped,
                )
        """
        return notify(message, level=level, detail=detail, source=self.display_name, items=items)

    # -- CLI support --------------------------------------------------------

    def add_cli_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register this plugin's fields as ``argparse`` arguments.

        The default implementation converts each :class:`PluginField` into a
        CLI flag (e.g. a field with ``key="media_type"`` becomes
        ``--media-type``).  ``"select"`` fields gain a ``choices`` constraint.
        """
        for f in self.fields:
            arg_name = f"--{f.key.replace('_', '-')}"
            kwargs: dict[str, Any] = {
                "dest": f.key,
                "help": f.description or f.label,
            }
            if f.field_type == "checkbox":
                # ``--<key>`` / ``--no-<key>`` boolean flag.
                kwargs["action"] = argparse.BooleanOptionalAction
                kwargs["default"] = str(f.default).lower() == "true"
                parser.add_argument(arg_name, **kwargs)
                continue
            if f.default:
                kwargs["default"] = f.default
            if f.field_type == "select" and f.options and not f.allow_free_text and not f.dynamic_options:
                # A dynamic-options select computes its list at runtime, so its
                # declared ``options`` are only a seed - pinning argparse to them
                # would reject every value the plugin resolves later.
                kwargs["choices"] = f.options
            if f.field_type == "number":
                kwargs["type"] = int if f.is_integer_number() else float
                if f.default:
                    kwargs["default"] = kwargs["type"](f.default)
            parser.add_argument(arg_name, **kwargs)

    def validate_cli_field_values(self, field_values: dict[str, Any]) -> None:
        """Raise ``ValueError`` if any required field is missing or empty.

        Also runs the shared
        :func:`vtscore.plugins.normalize.normalize_field_values` pass
        (whitespace strip, template variable substitution, and
        field-type-driven security validation such as
        :func:`~vtscore.security.url_validation.validate_url` for
        ``url`` fields and
        :func:`~vtscore.security.path_validation.confine_server_filepath`
        for ``server_path`` fields) so CLI invocations get the same
        guarantees the HTTP path does.  Required-field rejection is
        delegated to the normalize pass so the two ingress points raise
        identically on the same input.

        ``--`` flag names are preferred in the surfaced error so CLI
        users see the same identifier they typed; the normalize pass's
        generic ``"<Label> is required."`` is rewritten when the
        rejected field comes from missing CLI input.

        A field switched off by its :attr:`~PluginField.visible_when` is
        never missing: ``--email`` is not required while ``--no-enable-email``
        is in force.
        """
        inactive = inactive_field_keys(self.fields, field_values)
        for f in self.fields:
            # Booleans are always populated by argparse (default included).
            if f.field_type == "checkbox" or f.key in inactive:
                continue
            # A field with a declared default is never "missing": the
            # normalize pass below fills a blank one from it, so rejecting
            # it here would make ``--email ""`` fail where omitting the
            # flag entirely succeeds.
            if f.default:
                continue
            value = field_values.get(f.key)
            if f.required and (value is None or (isinstance(value, str) and not value.strip())):
                cli_flag = f"--{f.key.replace('_', '-')}"
                raise ValueError(f"Missing required argument: {cli_flag}")

        from vtscore.plugins.normalize import normalize_field_values  # noqa: PLC0415

        normalize_field_values(self, field_values)

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise plugin metadata for API endpoints."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "fields": [f.to_dict() for f in self.fields],
            "ui_mode": self.ui_mode,
            "hidden_from_picker": self.hidden_from_picker,
        }
