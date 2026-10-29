"""The field descriptor every plugin declares its inputs with.

:class:`PluginField` describes one configurable input (its widget, its CLI
flag, its validation and persistence rules); :data:`FieldType`,
:data:`FieldOption` and :data:`ConditionValue` are the value types its
attributes use.  :func:`inactive_field_keys` is the one evaluator of
:attr:`PluginField.visible_when`, shared by the HTTP schema, the CLI and the
normalize pass, and :func:`parse_checkbox` is the one reading of a
``"checkbox"`` value.

Import these from :mod:`vtscore.plugins`, which re-exports them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

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

#: One value a :attr:`PluginField.visible_when` condition accepts: a string
#: (compared with the controlling field's value after stripping whitespace)
#: or a ``bool`` (the natural spelling for a ``"checkbox"`` controller).
ConditionValue = str | bool


def parse_checkbox(value: Any) -> bool:
    """Coerce a ``"checkbox"`` field's value to ``bool``.

    Checkbox values reach a plugin in three shapes depending on the caller: a
    native ``bool`` (the CLI, whose ``argparse.BooleanOptionalAction`` already
    coerced it), the ``"true"``/``"false"`` strings a form submission
    serialises to, or ``None`` when the field was omitted entirely.  Anything
    that is not recognisably true reads as ``False``.

    Shared rather than re-derived per call site: the GUI and the CLI must agree
    on what a ticked box means, and a checkbox whose two readers disagree is
    the kind of divergence that only shows up as different results from the
    same input (issue #3556).
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


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
    - ``"checkbox"`` – Boolean tick-box rendered beside the field's label.
      ``default`` should be ``"true"`` or ``"false"`` (either case; a
      Python-style ``"False"`` reads the way its author meant it).  Values
      arrive at :meth:`run` as plain strings or as already-coerced bools, so
      parse them with :func:`parse_checkbox` rather than by hand — it is the
      one reading the GUI, the CLI and the frontend all share.

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
    #: Pre-filled default value.  It is a **value**, not a placeholder (use
    #: :attr:`placeholder` for grey hint text): the GUI renders it into the
    #: widget so the user sees and can edit it, the CLI hands it to
    #: ``argparse`` as the flag's default, and
    #: :func:`~vtscore.plugins.normalize.normalize_field_values` fills it in
    #: for any field that still arrives blank — so a field the user never
    #: touched reaches :meth:`run` as its default rather than as ``""``.
    #: That also means a declared default satisfies :attr:`required`.
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
    #: persisted origin dict (see :meth:`DatasetImporter.build_origin`), or,
    #: for a label importer, into the import record a detector it creates
    #: keeps (see :mod:`vtscore.detectors.import_record`).
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

    #: When ``True``, no GUI form renders a widget for this field: the
    #: value is the plugin author's to supply, not the user's.  Pair it
    #: with :attr:`default` — the framework fills a hidden field in from
    #: that default like any other blank field, so the plugin body reads
    #: it exactly as if the user had typed it.  Use this for a value the
    #: deployment fixes and the user must not edit or even see: the
    #: canonical case is an ``open_url`` subclass whose ``url_template``
    #: points at one in-house site, which should present as a single
    #: button rather than as a URL box the user is invited to retype.
    #:
    #: Hiding is a **GUI affordance only**, the field-level sibling of
    #: :attr:`~vtscore.plugins.PluginBase.hidden_from_picker`.  The field
    #: keeps its CLI flag (so a scripted or AutoFind run can still override
    #: the fixed value), keeps its place in ``to_dict()`` on the wire, and is
    #: validated, normalised and template-substituted exactly like a
    #: visible one.  It is *not* an access control: the value ships to
    #: the browser in the plugin listing either way, so never hide a
    #: secret behind it — use ``field_type="password"`` for that.
    #:
    #: A hidden field that is :attr:`required` with no :attr:`default`
    #: is unfillable from the GUI and will fail validation on submit.
    #: That combination is only meaningful for a CLI-only field; give a
    #: GUI-reachable plugin a default, or mark the field
    #: ``required=False``.
    hidden: bool = False

    #: Show this field only while other fields hold particular values:
    #: ``{controller_key: accepted}``, where *accepted* is one value or a
    #: list of them.  Every entry must match (AND); list several values
    #: under one key for OR.  The canonical case is a checkbox that
    #: switches a group of inputs on::
    #:
    #:     PluginField(key="enable_email", label="Send e-mail", field_type="checkbox", default="false"),
    #:     PluginField(key="email", label="Address", field_type="email", visible_when={"enable_email": True}),
    #:
    #: and a select whose choice decides which inputs apply
    #: (``visible_when={"source": ["s3", "gcs"]}``).  Any field type can be
    #: conditional, and any field type can control one.
    #:
    #: The controller's value is compared as a string after stripping, so
    #: ``True`` / ``False`` mean ``"true"`` / ``"false"``; a ``"checkbox"``
    #: controller reads both sides through :func:`parse_checkbox`, so
    #: ``"True"`` matches a ticked box too.  A blank controller counts as
    #: its :attr:`default`, and so does a controller that is itself switched
    #: off by its own condition — the value the plugin would receive for it.
    #: A key naming no field of the plugin is ignored.
    #:
    #: While its condition fails the field is **inactive**, and that is
    #: more than presentation: no form renders it, it is not
    #: :attr:`required`, and :func:`~vtscore.plugins.normalize.normalize_field_values`
    #: replaces whatever was submitted for it with its :attr:`default` (or
    #: blank; ``None`` for a file), so the plugin never acts on a value the
    #: user could no longer see.  :func:`inactive_field_keys` is the one
    #: evaluator, shared by the HTTP schema, the CLI and the normalize pass;
    #: ``frontend/src/app/utils/plugin-fields.ts`` mirrors it for the forms.
    #: Empty (the default) means always shown, exactly as before.
    visible_when: dict[str, ConditionValue | list[ConditionValue]] = field(default_factory=dict)

    #: For ``"url"`` fields: ``True`` declares that the URL is opened by the
    #: **user's browser** (handed back as an exporter's ``open_url``) and is
    #: never fetched by the server.  Such a value is validated with
    #: :func:`~vtscore.security.url_validation.validate_browser_url` — a
    #: scheme allowlist that rejects ``javascript:`` / ``data:`` / ``file:``
    #: — instead of the SSRF guard
    #: :func:`~vtscore.security.url_validation.validate_url`, which refuses
    #: ``localhost`` and private addresses.  Those are legitimate targets
    #: for a browser (a companion viewer or mock server on the user's own
    #: machine or LAN), and refusing them buys no protection when no
    #: server-side request is made.
    #:
    #: Defaults to ``False`` so every ``url`` field stays SSRF-guarded
    #: unless its author says otherwise.  Only set it on a field whose
    #: value your plugin never passes to ``requests`` / ``urllib`` itself;
    #: a URL the server fetches must keep the SSRF guard.  Ignored for
    #: every other :attr:`field_type` (a ``"text"`` field is not
    #: URL-validated at all).
    opened_in_browser: bool = False

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
            "hidden": self.hidden,
            "visible_when": {key: list(_condition_tokens(accepted)) for key, accepted in self.visible_when.items()},
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
# visible_when: which fields a condition switches off
# ---------------------------------------------------------------------------


def _condition_token(value: Any) -> str:
    """The string a ``visible_when`` comparison sees for *value*."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value).strip()


def _condition_tokens(accepted: Any) -> tuple[str, ...]:
    """The accepted values of one ``visible_when`` entry, as strings."""
    items = accepted if isinstance(accepted, (list, tuple)) else [accepted]
    return tuple(_condition_token(v) for v in items)


def _field_token(f: PluginField, value: Any) -> str:
    """*value* as a comparison token for the controlling field *f*.

    A checkbox reads through :func:`parse_checkbox`, so ``"True"``, ``True``
    and ``"true"`` are one token and a blank box is ``"false"``.
    """
    if f.field_type == "checkbox":
        return "true" if parse_checkbox(value) else "false"
    return _condition_token(value)


def _submitted_token(f: PluginField, values: Mapping[str, Any]) -> str:
    """The token for the value *f* would reach the plugin with: the caller's,
    or :attr:`PluginField.default` when the caller left it blank."""
    raw = values.get(f.key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = f.default
    return _field_token(f, raw)


def _condition_holds(
    f: PluginField,
    by_key: Mapping[str, PluginField],
    values: Mapping[str, Any],
    chain: frozenset[str],
) -> bool:
    """Whether every entry of *f*'s ``visible_when`` matches *values*.

    A controller that is itself switched off counts as its default.  *chain*
    holds the fields already being resolved above this one, so a cycle (an
    author error) stops at the field it re-enters, which counts as active.
    """
    if not f.visible_when:
        return True
    chain = chain | {f.key}
    for ctrl_key, accepted in f.visible_when.items():
        ctrl = by_key.get(ctrl_key)
        if ctrl is None:
            continue
        if ctrl.key in chain or _condition_holds(ctrl, by_key, values, chain):
            actual = _submitted_token(ctrl, values)
        else:
            actual = _field_token(ctrl, ctrl.default)
        if actual not in {_field_token(ctrl, v) for v in _condition_tokens(accepted)}:
            return False
    return True


def inactive_field_keys(fields: Iterable[PluginField], values: Mapping[str, Any]) -> frozenset[str]:
    """Keys of the fields whose :attr:`PluginField.visible_when` fails on *values*.

    *values* is the form as submitted, in any of the shapes a plugin's field
    values arrive in (form strings, JSON, argparse's already-coerced values);
    a missing or blank key reads as that field's default.  An inactive field
    is not rendered, not required, and is reset to its default before the
    plugin runs: see :attr:`PluginField.visible_when` for the rules.

    This is the single evaluator every ingress point uses — the HTTP schema,
    the CLI, :func:`~vtscore.plugins.normalize.normalize_field_values` — and
    ``frontend/src/app/utils/plugin-fields.ts`` is its TypeScript twin, so the
    form a user sees and the values the server accepts cannot disagree.
    """
    fields = list(fields)
    if not any(f.visible_when for f in fields):
        return frozenset()
    by_key = {f.key: f for f in fields}
    return frozenset(f.key for f in fields if not _condition_holds(f, by_key, values, frozenset()))
