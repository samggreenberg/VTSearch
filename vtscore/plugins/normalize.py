"""Framework-side normalization of plugin ``field_values``.

After marshmallow / argparse validates the *shape* of incoming field
values, this module's :func:`normalize_field_values` applies the
behaviours that used to be plugin-author responsibility:

1. **Whitespace strip** on every text-like value, so plugin bodies can
   trust ``field_values[key]`` is the trimmed form.
2. **Template variable substitution** for fields that declare
   :attr:`PluginField.template_vars`.  Recognised names -
   ``YYYYMMDD-HHMMSS``, ``detector_name``, ``detector_id``,
   ``username`` - are resolved and run through
   :func:`~vtscore.security.path_validation.sanitize_template_value`
   so attacker-controlled values cannot escape the directory implied
   by an admin-configured template.
3. **Field-type-driven security validation**.
   ``field_type="url"`` values are passed through
   :func:`~vtscore.security.url_validation.validate_url`;
   ``field_type="server_path"`` values are passed through
   :func:`~vtscore.security.path_validation.validate_server_filepath`
   anchored at the per-user data dir.

Plugin bodies are expected to trust the resulting dict - no more
``if not foo: raise ValueError`` boilerplate, no more manual
``validate_url`` / ``validate_server_filepath`` calls, no more bespoke
``str.replace("{detector_name}", ...)`` chains.

Wired into both ingress points:

- HTTP path: ``vtsearch/routes/_shared.py:validate_plugin_args`` calls
  this after marshmallow loads the body and file uploads are populated.
- CLI path: :meth:`PluginBase.validate_cli_field_values` calls this
  after the presence check.

External plugins that still call the validators by hand keep working -
re-validation is idempotent on already-validated values, and
``sanitize_template_value`` is idempotent on already-sanitised strings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vtscore.plugins import PluginBase

_TEXT_LIKE_TYPES = frozenset({"text", "url", "email", "password", "folder", "server_path", "select"})

_KNOWN_TEMPLATE_VARS = frozenset({"YYYYMMDD-HHMMSS", "detector_name", "detector_id", "username"})


def _resolve_template_var(name: str) -> str:
    """Resolve a single ``{name}`` placeholder to its current value.

    Raises :class:`ValueError` for unknown names so a typo in a plugin
    schema fails fast at the first request.
    """
    if name == "YYYYMMDD-HHMMSS":
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if name in ("detector_name", "detector_id"):
        from vtscore.state.core import get_active_detector_context  # noqa: PLC0415

        ctx = get_active_detector_context()
        return ctx.name if name == "detector_name" else ctx.detector_id

    if name == "username":
        # Lazy import: ``vtscore.plugins`` is library-tier and must not
        # take a hard dependency on ``vtsearch.auth`` at import time.
        from vtsearch.auth import get_current_user  # noqa: PLC0415

        return get_current_user() or "default"

    raise ValueError(f"Unknown template variable: {{{name}}}")


def _apply_templates(value: str, template_vars: tuple[str, ...]) -> str:
    """Substitute every ``{var}`` in *value* using the framework resolver.

    Each resolved value is sanitised so user-controlled template values
    cannot escape the directory implied by the admin-configured
    template.  No-op when *template_vars* is empty.
    """
    if not template_vars:
        return value

    # Local import - sanitize lives in the security package; keep the
    # module import light.
    from vtscore.security.path_validation import sanitize_template_value  # noqa: PLC0415

    for var in template_vars:
        if var not in _KNOWN_TEMPLATE_VARS:
            raise ValueError(f"Unknown template variable declared: {{{var}}}")
        placeholder = "{" + var + "}"
        if placeholder in value:
            value = value.replace(placeholder, sanitize_template_value(_resolve_template_var(var)))
    return value


def _validate_field_value(field_type: str, value: str) -> None:
    """Run the field-type-driven security validator on a non-empty value."""
    if field_type == "url":
        from vtscore.security.url_validation import validate_url  # noqa: PLC0415

        validate_url(value)
    elif field_type in ("server_path", "folder"):
        from vtscore.security.path_validation import (  # noqa: PLC0415
            get_file_access_base_dir,
            validate_server_filepath,
        )

        validate_server_filepath(value, base_dir=get_file_access_base_dir())


def normalize_field_values(plugin: PluginBase, field_values: dict[str, Any]) -> dict[str, Any]:
    """Normalize *field_values* against *plugin*'s declared fields.

    Mutates *field_values* in place and returns it.  Skips file uploads
    (those don't carry strings) and non-string values (numbers,
    booleans).  Raises :class:`ValueError` for an empty / missing
    required field, an invalid URL, a path-traversal attempt, or an
    unknown template variable.

    Idempotent: running the pass twice on the same dict produces the
    same result and never raises differently the second time.
    """
    for f in plugin.fields:
        if f.field_type == "file":
            continue
        if f.field_type not in _TEXT_LIKE_TYPES:
            continue

        raw = field_values.get(f.key)
        if raw is None:
            value = ""
        elif isinstance(raw, str):
            value = raw.strip()
        else:
            # Non-string value already present (e.g. tests passing an int)
            # - leave it alone and let the plugin handle it.
            continue

        if not value and f.required:
            raise ValueError(f"{f.label} is required.")

        if value:
            value = _apply_templates(value, tuple(f.template_vars))
            _validate_field_value(f.field_type, value)
        field_values[f.key] = value

    return field_values
