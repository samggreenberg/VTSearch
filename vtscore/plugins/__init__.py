"""Generic plugin registry with auto-discovery.

Provides :class:`PluginRegistry`, a reusable registry that scans a package
directory for sub-packages exposing a sentinel attribute, and
:class:`PluginField` / :class:`PluginBase`, shared base types that eliminate
the duplicated field-dataclass and CLI / serialisation boilerplate across the
four plugin families (dataset importers, exporters, label importers, processor
importers).

Usage for creating a registry::

    from vtscore.plugins import PluginRegistry

    _registry: PluginRegistry[MyPlugin] = PluginRegistry(
        package="vtscore.exporters",
        sentinel="EXPORTER",
        label="exporter",
    )

    get_exporter    = _registry.get
    list_exporters  = _registry.list

Usage for defining a field / base class::

    from vtscore.plugins import PluginBase, PluginField

    class MyExporter(PluginBase):
        ...
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import sys
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, TypeVar

from vtscore.concurrency.notifications import Notification, notify

FieldType = Literal[
    "file",
    "folder",
    "url",
    "text",
    "password",
    "email",
    "number",
    "select",
    "server_path",
    "checkbox",
]

#: A single dropdown option returned by ``get_field_options``.  Either a
#: plain string (the value is shown verbatim as the label) or a
#: ``(value, label)`` tuple (the option submits the opaque ``value`` while
#: displaying the friendly ``label``).  The API layer coerces both shapes
#: to ``{"value", "label"}`` before serialising them to the frontend.
FieldOption = str | tuple[str, str]

__all__ = [
    "EntryPointTombstone",
    "FieldOption",
    "FieldType",
    "Notification",
    "PluginBase",
    "PluginField",
    "PluginRegistry",
    "make_plugin_registry",
    "notify",
]


# ---------------------------------------------------------------------------
# PluginField: shared field descriptor
# ---------------------------------------------------------------------------


@dataclass
class PluginField:
    """Describes a single configurable input for a plugin.

    The ``field_type`` value drives how the frontend renders it:

    - ``"file"``     – OS file-picker; value arrives as an
      :class:`~vtscore.plugins.uploads.UploadedFile` (Werkzeug
      ``FileStorage`` from a Flask request, a ``CliUploadedFile``
      wrapping a path argument from the CLI, or a ``BytesIOUploadedFile``
      for background-thread reads).
    - ``"folder"``   – Path text-input or OS folder-picker.
    - ``"url"``      – Text input pre-validated as a URL.
    - ``"text"``     – Generic single-line text input.
    - ``"password"`` – Text input whose characters are masked.
    - ``"email"``    – Text input pre-validated as an e-mail address.
    - ``"number"``   – Numeric input.  Use :attr:`min`, :attr:`max`, and
      :attr:`step` to constrain the allowed values; an integer ``step``
      (and integer :attr:`default` / :attr:`min` / :attr:`max`) tells the
      CLI parser to coerce values with :class:`int`, otherwise
      :class:`float` is used.  Values still arrive at :meth:`run` as
      strings from web requests, so plugins should ``int()`` or
      ``float()`` them as needed.
    - ``"select"``   – Drop-down; ``options`` must be populated (or
      :attr:`dynamic_options` set, in which case options are fetched at
      runtime from the plugin's ``get_field_options`` method).
    - ``"server_path"`` – File-browser picker for server filesystem paths.
    - ``"checkbox"`` – Boolean tick-box.  ``default`` should be ``"true"`` or
      ``"false"``; values arrive at :meth:`run` as plain strings (or already
      coerced bools) and should be parsed via ``str(value).lower() == "true"``.

    Dynamic option fields
    ---------------------
    Set ``dynamic_options=True`` on a ``"select"`` field whose options must
    be computed at runtime (e.g. by querying a remote service).  The plugin
    must implement ``get_field_options(field_key, current_values)`` to return
    the list.  The frontend re-fetches options every time any field listed
    in :attr:`depends_on` changes value.  Honoured by dataset importers
    (``POST /api/dataset/import/<name>/options``), label importers
    (``POST /api/label-importers/field-options/<name>``), seed importers,
    datasource importers, and results exporters
    (``POST /api/exporters/field-options/<name>``).
    """

    key: str
    label: str
    field_type: FieldType
    description: str = ""
    #: For ``"file"`` fields: comma-separated extensions, e.g. ``".pkl"``.
    accept: str = ""
    #: For ``"select"`` fields: the list of allowed values.
    options: list[str] = field(default_factory=list)
    #: Pre-filled default value shown in the UI.
    default: str = ""
    required: bool = True
    #: Hint shown as placeholder text inside the input widget.
    placeholder: str = ""
    #: Inline format-hint text rendered as a visible chip below the input,
    #: separate from :attr:`description` (which feeds the placeholder).  Use
    #: this for format / schema hints the user needs to see at a glance even
    #: after they start typing (e.g. accepted file extensions, expected
    #: column schema, a short sample of the file layout).  Newlines and
    #: leading-space indented lines render verbatim, so multi-line samples
    #: are fine.
    hint: str = ""
    #: When ``True``, :attr:`options` is computed at runtime by the plugin's
    #: ``get_field_options(field_key, current_values)`` method.  Static
    #: :attr:`options` (if any) are still served as the initial list.
    dynamic_options: bool = False
    #: Field keys whose values this field's options depend on.  When any
    #: listed field changes, the frontend re-fetches options for this field.
    #: Only meaningful when :attr:`dynamic_options` is ``True``.
    depends_on: list[str] = field(default_factory=list)
    #: For ``"select"`` fields: when ``True``, the dropdown renders as a
    #: combobox the user can type an arbitrary value into, even one the
    #: option list doesn't include.  When the option list refreshes, a
    #: typed value that isn't in the new list is kept (a strict ``select``
    #: — the default — clears such a value instead).
    allow_free_text: bool = False
    #: For ``"number"`` fields: minimum allowed value (string form, empty = no min).
    min: str = ""
    #: For ``"number"`` fields: maximum allowed value (string form, empty = no max).
    max: str = ""
    #: For ``"number"`` fields: step increment (string form).  If empty or
    #: ``"any"``, falls back to ``"1"`` for integer-looking defaults and
    #: ``"any"`` for floats.  A non-integer step (e.g. ``"0.05"``) tells the
    #: CLI parser to use :class:`float`; an integer step uses :class:`int`.
    step: str = ""

    #: Field keys this field is mutually exclusive with.  When the user
    #: enters a non-empty value into this field, the frontend blanks every
    #: field listed here (and they should list this field back, so the
    #: relationship is symmetric).  Use for "supply A *or* B" inputs where
    #: only one can be active at a time, e.g. video frame sampling by
    #: frames-per-video *or* seconds-per-frame.  Purely a UI affordance:
    #: the backend still reads whichever value arrives non-empty.
    clears: list[str] = field(default_factory=list)

    #: Whether this field's value should be copied into the importer's
    #: persisted origin dict (see :meth:`DatasetImporter.build_origin`).
    #: ``None`` means "use the field-type default": ``False`` for
    #: ``"file"`` and ``"password"`` fields (don't persist file uploads or
    #: secrets), ``True`` for every other type.  Set explicitly to
    #: override the default, e.g. ``include_in_origin=False`` on a noisy
    #: text field that doesn't belong in the persisted origin.
    include_in_origin: bool | None = None

    #: Optional callable that converts the field's value to the string
    #: form persisted in the origin dict.  Receives the raw value (as
    #: provided by the request or CLI) and must return a ``str``.  Use
    #: this for list/dict-typed values whose default ``str(...)``
    #: representation isn't round-trip safe (e.g. a list field that should
    #: be serialised as a comma-joined string).  Ignored when
    #: :attr:`include_in_origin` resolves to ``False``.
    origin_serializer: Callable[[Any], str] | None = None

    #: Template variables the framework should substitute into this
    #: field's value before the plugin's ``run`` / ``export`` receives
    #: it.  Each name (e.g. ``"detector_name"``) is replaced everywhere
    #: it appears as ``{name}``; the substituted value is run through
    #: :func:`vtscore.security.path_validation.sanitize_template_value`
    #: so attacker-controlled values cannot escape the directory implied
    #: by an admin-configured template.  Supported names:
    #: ``"YYYYMMDD-HHMMSS"``, ``"YYYYMMDD"``, ``"YYYY"``, ``"MM"``,
    #: ``"DD"``, ``"detector_name"``, ``"detector_id"``,
    #: ``"username"``.  Empty tuple (the default) means the framework
    #: performs no substitution and the value reaches the plugin
    #: verbatim.
    template_vars: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "field_type": self.field_type,
            "description": self.description,
            "accept": self.accept,
            "options": self.options,
            "default": self.default,
            "required": self.required,
            "placeholder": self.placeholder,
            "hint": self.hint,
            "dynamic_options": self.dynamic_options,
            "depends_on": list(self.depends_on),
            "allow_free_text": self.allow_free_text,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "clears": list(self.clears),
            "template_vars": list(self.template_vars),
        }

    def is_integer_number(self) -> bool:
        """Return True for a ``"number"`` field that represents an integer.

        A number field is treated as an integer when its :attr:`step`,
        :attr:`default`, :attr:`min`, and :attr:`max` all lack a decimal
        point.  Otherwise it is treated as a float.
        """
        if self.field_type != "number":
            return False
        for val in (self.step, self.default, self.min, self.max):
            if val and "." in str(val):
                return False
        return True


# ---------------------------------------------------------------------------
# PluginBase: shared base class with CLI and serialisation helpers
# ---------------------------------------------------------------------------


#: Class-name suffixes stripped before snake-casing for the default
#: :attr:`PluginBase.name`.  Order matters; longer / more-specific
#: suffixes come first so ``HolderResultsExporter`` strips
#: ``ResultsExporter`` rather than just ``Exporter``.  ``LabelsetExporter``
#: stays listed beside it: it is the results-exporter base class's permanent
#: alias, so an out-of-tree ``FooLabelsetExporter`` must keep deriving ``foo``.
_PLUGIN_NAME_SUFFIXES: tuple[str, ...] = (
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


#: Framework-level abstract plugin bases that should never get
#: auto-derived metadata stamped onto them.  Setting a derived
#: ``name`` on, say, :class:`LabelImporter` itself would pollute every
#: concrete subclass that doesn't declare its own ``name``; concrete
#: subclasses inherit the polluted value from the MRO before the
#: per-subclass auto-default can fire.  Third-party intermediates that
#: should behave the same way can opt out by setting
#: ``_is_plugin_family_base = True`` in their own ``__dict__``.
_PLUGIN_FAMILY_BASE_NAMES: frozenset[str] = frozenset(
    {
        "ImporterBase",
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
        "SyncSource",
    }
)


#: Generic emoji stamped directly onto each plugin-family's abstract base
#: (e.g. :attr:`ImporterBase.icon`).  A concrete subclass that hasn't set
#: its own ``icon`` inherits one of these through the MRO; treated as "no
#: icon chosen" so :func:`_autoderive_plugin_metadata` can replace it with
#: a distinguishing letter glyph instead (see :func:`_default_plugin_letter_icon`).
_FAMILY_STOCK_ICONS: frozenset[str] = frozenset(
    {
        "\U0001f50c",  # dataset importer (plug)
        "\U0001f4e5",  # datasource importer (inbox tray)
        "\U0001f4e4",  # labelset / settings exporter (outbox tray)
        "\U0001f3f7️",  # label importer (label)
        "\U0001f331",  # seed importer (seedling)
        "\U0001f504",  # sync source (counterclockwise arrows)
        "⚙️",  # settings importer (gear)
    }
)


def _snake_case(name: str) -> str:
    """Convert a CamelCase / PascalCase identifier to snake_case."""
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (not name[i - 1].isupper() or (i + 1 < len(name) and name[i + 1].islower())):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _default_plugin_name(cls: type) -> str:
    raw = cls.__name__
    for suffix in _PLUGIN_NAME_SUFFIXES:
        if raw.endswith(suffix) and raw != suffix:
            raw = raw[: -len(suffix)]
            break
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


def _autoderive_plugin_metadata(cls: type) -> None:
    """Fill in default :attr:`name` / :attr:`display_name` /
    :attr:`description` / :attr:`icon` on *cls* when neither *cls* itself
    nor any ancestor already provides them.

    Called from :meth:`PluginBase.__init_subclass__`.  Framework-level
    abstract bases (named in :data:`_PLUGIN_FAMILY_BASE_NAMES`) and
    third-party intermediates that set
    ``_is_plugin_family_base = True`` skip auto-derivation entirely so
    they don't leak a derived name down to their concrete subclasses.
    """
    if cls.__name__ in _PLUGIN_FAMILY_BASE_NAMES:
        return
    if cls.__dict__.get("_is_plugin_family_base", False):
        return
    if "name" not in cls.__dict__ and not _mro_provides(cls, "name"):
        cls.name = _default_plugin_name(cls)
    if "display_name" not in cls.__dict__ and not _mro_provides(cls, "display_name"):
        derived_name = getattr(cls, "name", None)
        cls.display_name = _default_plugin_display_name(derived_name) if isinstance(derived_name, str) else ""
    if "description" not in cls.__dict__ and not _mro_provides(cls, "description"):
        cls.description = _default_plugin_description(cls)
    if "icon" not in cls.__dict__:
        current_icon = getattr(cls, "icon", "")
        if not current_icon or current_icon in _FAMILY_STOCK_ICONS:
            letter = _default_plugin_letter_icon(cls)
            if letter:
                cls.icon = letter


class PluginBase:
    """Mixin providing the CLI-argument, validation, and serialisation helpers
    that are identical across all four plugin families.

    Default metadata
    ----------------
    Subclasses that don't declare :attr:`name`, :attr:`display_name`,
    :attr:`description`, or :attr:`icon` get auto-derived defaults via
    :meth:`__init_subclass__`:

    - :attr:`name`: class name with the trailing family suffix
      (``DatasetImporter`` / ``ResultsExporter`` / ``MediaConverter`` /
      etc.) stripped and the remainder snake-cased.  E.g.
      ``MyShinyExporter`` → ``"my_shiny"``.
    - :attr:`display_name`: title-cased :attr:`name`.
    - :attr:`description`: first line of the class docstring.
    - :attr:`icon`: the first letter of :attr:`display_name`, upper-cased
      (e.g. ``MyShinyExporter`` → ``"M"``).  The frontend renders any
      single capital letter as a boxed letter-glyph icon, so a plugin
      author who hasn't designed a custom icon still gets a
      distinguishing default instead of every plugin in the family
      sharing the same generic emoji.  This only fires when *cls* hasn't
      set its own ``icon`` and would otherwise inherit either nothing
      (``""``) or one of the generic family defaults (e.g.
      :attr:`~vtscore.datasets.importers.base.core.ImporterBase.icon`);
      an author is always free to set a fancier emoji or SVG-type string.

    Explicit declarations always win.  The defaults only fire when
    nothing further up the MRO already provides a string value or a
    descriptor (e.g. :class:`~vtscore.converters.base.MediaConverter`
    declares ``name`` as a property, so concrete converter subclasses
    inherit the property rather than getting a stomped string).
    """

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

    #: How the frontend should render this plugin's UI.
    #: ``"form"``: generic form built from :attr:`fields` (default).
    #: ``"file_upload"``: the frontend should use its native file picker.
    #: ``"custom"``: the plugin has a dedicated UI section in the frontend.
    #: ``"none"``: no user-facing UI (e.g. the GUI exporter is handled
    #:   automatically by the frontend results view).
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

        Example::

            if skipped:
                self.notify(
                    f"Skipped {len(skipped)} unreadable files",
                    level="warning",
                    detail=", ".join(skipped[:10]),
                )
        """
        return notify(message, level=level, detail=detail, source=self.display_name)

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
        """
        for f in self.fields:
            # Booleans are always populated by argparse (default included).
            if f.field_type == "checkbox":
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


# ---------------------------------------------------------------------------
# Entry-point tombstone: defer a plugin's own ImportError to first use
# ---------------------------------------------------------------------------


class EntryPointTombstone:
    """Placeholder for an entry-point plugin whose *own* import raised.

    When a third-party entry point fails to load (e.g. a missing optional
    dependency such as ``open_clip`` for a ``siglip_l`` embedder), we don't
    want that single failure to crash discovery for every other plugin.
    Instead the registry registers one of these under the entry point's name
    so the plugin still *resolves* via :meth:`PluginRegistry.get`, and the
    original error is re-raised only when the plugin is actually invoked
    (any attribute access other than :attr:`name` / private / dunder names).

    Tombstones are deliberately kept out of :meth:`PluginRegistry.list` so
    listing/serialising the plugin family (which touches ``to_dict``,
    ``display_name``, ``ui_mode``, ...) never trips over them; the broken
    plugin only bites the caller who deliberately asks for it by name.

    The object is a plain, picklable value: :attr:`name`, the entry-point
    ``value`` string, and the captured exception all live in ``__dict__``,
    so ``pickle.loads(pickle.dumps(tombstone))`` round-trips and the restored
    copy re-raises the same error on use.
    """

    def __init__(self, name: str, value: str, error: BaseException) -> None:
        self.name = name
        self._ep_value = value
        self._error = error

    def __getattr__(self, attr: str) -> Any:
        # Private / dunder lookups (``__reduce_ex__``, ``__getstate__``,
        # ``__deepcopy__``, ...) must fall through with AttributeError so
        # pickle and copy keep working; only "real" plugin attribute/method
        # access re-raises the deferred import error.
        if attr.startswith("_"):
            raise AttributeError(attr)
        raise ImportError(
            f"Plugin {self.name!r} (entry point {self._ep_value!r}) is unavailable "
            f"because its import failed: {self._error}"
        ) from self._error


# ---------------------------------------------------------------------------
# PluginRegistry: generic auto-discovery registry
# ---------------------------------------------------------------------------

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """Auto-discovering plugin registry.

    Parameters
    ----------
    package:
        Fully-qualified dotted name of the package whose sub-packages will be
        scanned, e.g. ``"vtscore.exporters"``.
    sentinel:
        Module-level attribute name to look for in each sub-package, e.g.
        ``"EXPORTER"``.
    label:
        Human-readable noun used in warning messages, e.g. ``"exporter"``.
    discover_modules:
        When ``True``, also scan flat ``.py`` files (not just sub-packages)
        for the sentinel.  Useful for plugin families where each plugin is
        a single module rather than a sub-package (e.g. converters, sources).
    entry_point_group:
        Optional :mod:`importlib.metadata` entry-point group to scan after
        the local package scan, e.g. ``"vtscore.importers"``.  Third-party
        packages can register a plugin by adding an entry to this group in
        their own ``pyproject.toml`` / ``setup.cfg``::

            [project.entry-points."vtscore.importers"]
            my_importer = "my_pkg.my_module:IMPORTER"

        The entry point must resolve to an already-instantiated plugin
        object (same shape as a sentinel attribute); typically you point
        directly at the module's ``IMPORTER`` / ``EXPORTER`` / ... sentinel.
    eager:
        When ``True`` (the default) discovery runs at construction time so
        the registry is populated by the time the constructor returns.
        Set ``False`` to defer discovery until the first :meth:`get` /
        :meth:`list` call; useful in tests that want to inspect the
        pre-discovery state or simulate concurrent first access.
    """

    def __init__(
        self,
        package: str,
        sentinel: str,
        label: str,
        *,
        discover_modules: bool = False,
        entry_point_group: str | None = None,
        eager: bool = True,
    ) -> None:
        self._package = package
        self._sentinel = sentinel
        self._label = label
        self._discover_modules = discover_modules
        self._entry_point_group = entry_point_group
        self._items: dict[str, T] = {}
        #: Entry-point plugins whose own import raised, keyed by entry-point
        #: name.  Resolvable via :meth:`get` (re-raising the original error on
        #: use) but deliberately excluded from :meth:`list`.
        self._tombstones: dict[str, EntryPointTombstone] = {}
        self._discovered = False
        self._discovering = False
        #: Re-entrant by design: a module scanned during :meth:`_discover` may
        #: call :meth:`get` / :meth:`list` on this same registry at import
        #: time, on the discovering thread.  An ``RLock`` lets that call back
        #: in so the ``_discovering`` guard in :meth:`_ensure_discovered` can
        #: hand it the partial registry; a plain ``Lock`` would deadlock the
        #: process before the guard was ever reached.
        self._lock = threading.RLock()
        if eager:
            self._ensure_discovered()

    # -- Discovery ----------------------------------------------------------

    def _discover(self) -> None:
        """Scan sub-packages (and optionally flat modules) for sentinel objects.

        Uses direct filesystem scanning so that symlinked directories are
        reliably discovered.  A symlink to a package directory (containing
        ``__init__.py``) is treated identically to a regular sub-package.

        When :attr:`_discover_modules` is ``True``, also scans ``.py`` files
        (excluding ``__init__.py`` and ``base.py``) for the sentinel.
        """
        parent = importlib.import_module(self._package)
        if parent.__file__ is None:
            raise RuntimeError(f"Cannot discover plugins under namespace package {self._package!r}")
        package_dir = Path(parent.__file__).parent
        for entry in sorted(package_dir.iterdir()):
            if entry.name.startswith((".", "_")):
                continue

            # Sub-packages (directories with __init__.py)
            if entry.is_dir():
                # Skip names containing dots; they aren't valid Python
                # identifiers and would be misinterpreted as nested module
                # paths by importlib (e.g. "foo.symbolic_link" would try to
                # import package "foo" first).  This commonly happens with
                # symlinks whose names include an extension or suffix.
                if "." in entry.name:
                    continue
                init_path = entry / "__init__.py"
                if not init_path.exists():
                    continue
                self._try_load(entry.name, file_path=init_path if entry.is_symlink() else None)
            # Flat modules (.py files)
            elif self._discover_modules and entry.is_file() and entry.suffix == ".py":
                if entry.name in ("__init__.py", "base.py"):
                    continue
                self._try_load(entry.stem, file_path=entry if entry.is_symlink() else None)

        if self._entry_point_group:
            self._discover_entry_points()

    def _discover_entry_points(self) -> None:
        """Load third-party plugins registered via :mod:`importlib.metadata`.

        Each entry point in :attr:`_entry_point_group` is resolved to an
        object that's treated like a sentinel value; its ``.name`` is the
        registry key.  When an entry point's *own* import raises (e.g. a
        missing optional dependency), the failure is surfaced as a warning
        and a :class:`EntryPointTombstone` is registered under the entry-point
        name so the plugin still resolves via :meth:`get` and the original
        error is deferred to first use — a single bad third-party plugin can't
        break discovery for the rest of the registry.

        Built-in plugins (discovered by the package scan above) take
        precedence: an entry point whose name clashes with a built-in is
        skipped.  This prevents an installed third-party package from
        accidentally shadowing a core plugin.
        """
        group = self._entry_point_group
        if group is None:
            return
        try:
            eps = importlib.metadata.entry_points(group=group)
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to read entry-point group {self._entry_point_group!r}: {exc}",
                stacklevel=2,
            )
            return
        for ep in eps:
            try:
                plugin = ep.load()
            except Exception as exc:
                # The plugin's *own* import raised (e.g. a missing optional
                # dependency).  Don't crash discovery: register a tombstone
                # under the entry-point name so the plugin still resolves and
                # the original error is deferred to first use.  A built-in of
                # the same name still wins (it's already in ``_items``).
                warnings.warn(
                    f"Failed to load {self._label} entry point {ep.name!r} from {ep.value!r}: {exc}; "
                    f"deferring the error to first use of {ep.name!r}",
                    stacklevel=2,
                )
                if ep.name not in self._items:
                    self._tombstones[ep.name] = EntryPointTombstone(ep.name, ep.value, exc)
                continue
            plugin_name = getattr(plugin, "name", None)
            if not plugin_name:
                warnings.warn(
                    f"{self._label} entry point {ep.name!r} from {ep.value!r} has no 'name' attribute; skipped",
                    stacklevel=2,
                )
                continue
            if plugin_name in self._items:
                warnings.warn(
                    f"{self._label} entry point {ep.name!r} from "
                    f"{ep.value!r} clashes with built-in plugin "
                    f"{plugin_name!r}; skipped",
                    stacklevel=2,
                )
                continue
            self._items[plugin_name] = plugin

    def _try_load(self, module_name: str, *, file_path: Path | None = None) -> None:
        """Import *module_name* under this registry's package and register its sentinel.

        When *file_path* is given (symlinked entries), uses
        :func:`importlib.util.spec_from_file_location` to load the module
        directly from the resolved path.  Python's default ``FileFinder`` can
        miss symlinked packages on some platforms because its directory cache
        may not follow symlinks consistently.
        """
        full_name = f"{self._package}.{module_name}"
        try:
            if file_path is not None:
                resolved = file_path.resolve()
                is_package = resolved.name == "__init__.py"
                spec = importlib.util.spec_from_file_location(
                    full_name,
                    str(resolved),
                    submodule_search_locations=[str(resolved.parent)] if is_package else None,
                )
                if spec is None or spec.loader is None:  # pragma: no cover
                    return
                mod = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = mod
                spec.loader.exec_module(mod)
            else:
                mod = importlib.import_module(full_name)
            plugin = getattr(mod, self._sentinel, None)
            if plugin is not None:
                if plugin.name in self._items:
                    warnings.warn(
                        f"{self._label} module {module_name!r} declares name "
                        f"{plugin.name!r} which is already registered; skipped",
                        stacklevel=2,
                    )
                    return
                self._items[plugin.name] = plugin
        except Exception as exc:  # pragma: no cover
            # Clean up partially-registered module on failure.
            sys.modules.pop(full_name, None)
            warnings.warn(
                f"Failed to load {self._label} '{module_name}': {exc}",
                stacklevel=2,
            )

    def _ensure_discovered(self) -> None:
        if self._discovered:
            return
        with self._lock:
            if not self._discovered:
                # Guard against re-entrant discovery on the discovering
                # thread.  When discover_modules is True, importing a sibling
                # module may trigger get()/list() on this registry before
                # discovery finishes (e.g. runner.py importing from its own
                # package's __init__).  ``self._lock`` is an RLock precisely
                # so that call re-enters here instead of deadlocking; we
                # return early with a partial registry, and the ongoing
                # discovery completes shortly and fills in the rest.  Other
                # threads never see _discovering True: they block on the
                # RLock until discovery is done and _discovered is set.
                if self._discovering:
                    return
                self._discovering = True
                try:
                    self._discover()
                finally:
                    self._discovering = False
                self._discovered = True

    # -- Public API ---------------------------------------------------------

    def get(self, name: str) -> T | None:
        """Return the registered plugin with *name*, or ``None``.

        Falls back to a tombstone for an entry-point plugin whose own import
        failed (see :class:`EntryPointTombstone`); the returned object
        re-raises the original error the moment it is actually invoked.
        """
        self._ensure_discovered()
        item = self._items.get(name)
        if item is not None:
            return item
        return self._tombstones.get(name)  # type: ignore[return-value]

    def list(self) -> list[T]:
        """Return all registered plugins in discovery order.

        Tombstones for failed entry-point imports are excluded so that
        listing/serialising a plugin family never trips over a broken plugin;
        such plugins are reachable only by an explicit :meth:`get`.
        """
        self._ensure_discovered()
        return list(self._items.values())


# ---------------------------------------------------------------------------
# Factory helper: collapses the per-package boilerplate into one call
# ---------------------------------------------------------------------------


def make_plugin_registry(
    package: str,
    sentinel: str,
    label: str,
    *,
    discover_modules: bool = False,
    entry_point_group: str | None = None,
    eager: bool = True,
) -> tuple[Callable[[str], Any], Callable[[], list[Any]]]:
    """Create a :class:`PluginRegistry` and return its ``(get, list)`` accessors.

    Shorthand for the boilerplate repeated across every plugin ``__init__.py``::

        from vtscore.plugins import make_plugin_registry

        get_importer, list_importers = make_plugin_registry(
            package=__name__,
            sentinel="IMPORTER",
            label="dataset importer",
            entry_point_group="vtscore.importers",
        )

    Parameters are forwarded to :class:`PluginRegistry`.
    """
    registry: PluginRegistry = PluginRegistry(
        package=package,
        sentinel=sentinel,
        label=label,
        discover_modules=discover_modules,
        entry_point_group=entry_point_group,
        eager=eager,
    )
    return registry.get, registry.list
