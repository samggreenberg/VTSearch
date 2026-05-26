"""Abstract base class for media converters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vtscore.plugins import PluginBase, PluginField


class MediaConverter(PluginBase, ABC):
    """Convert a media dict of one :class:`~vtscore.media.base.MediaType`
    into one or more media dicts of a *different* media type.

    Subclasses must implement:

    * :attr:`source_type` — the ``type_id`` of the input media type.
    * :attr:`target_type` — the ``type_id`` of the output media type.
    * :meth:`convert` — perform the actual conversion.

    Per-converter parameters
    ------------------------
    Converters can declare user-configurable parameters via the
    :attr:`fields` class attribute (a list of
    :class:`~vtscore.plugins.PluginField`) — the same mechanism
    every other plugin family uses.  Values flow in through the ``params``
    dict on :meth:`convert`.

    Framework-side call sites use :meth:`convert_normalized` instead of
    :meth:`convert` directly.  That wrapper validates ``params`` against
    the declared :attr:`fields` schema (rejecting out-of-range numbers,
    unknown ``select`` values, etc.) and fills missing or empty-string
    keys with the field's declared :attr:`~vtscore.plugins.PluginField.default`,
    so :meth:`convert` receives a fully-populated, non-``None`` dict
    where every declared field key is present.  Subclasses can therefore
    read ``params[key]`` directly without juggling ``None`` / missing /
    empty-string cases — :meth:`get_param` remains as a thin shim for
    third-party converters not yet migrated.

    The returned media dicts contain the fields produced by the target
    media type's :meth:`~vtscore.media.base.MediaType.load_media_data`
    (e.g. ``media_bytes``, ``duration``, ``width``, ``height``) plus a
    ``filename`` key.  They do **not** include ``id``, ``embedding``, or
    ``md5`` — the caller is responsible for assigning IDs, computing
    embeddings, and hashing.
    """

    #: Human-readable label shown in the converter chooser UI.
    #: Subclasses may override; the default is derived from the source
    #: and target type IDs.
    display_name: str = ""

    #: Short description of what this converter does.
    description: str = ""

    #: One-line preview with ``{key}`` placeholders for the converter's
    #: parameter values.  The frontend substitutes each ``{key}`` with the
    #: current value of the field named ``key`` (see :attr:`fields`) when
    #: rendering the import row preview.  Subclasses with configurable
    #: parameters should override to surface the active values — e.g.
    #: ``"Pull the audio track from video. Timeout {ffmpeg_timeout}sec."``.
    #: Defaults to empty (falls back to :attr:`description`).
    summary_template: str = ""

    #: User-configurable parameters.  Same :class:`PluginField` system
    #: every plugin family uses.  Empty by default — converters with no
    #: tunables don't have to declare anything.
    fields: list[PluginField] = []

    @property
    def name(self) -> str:
        """Unique identifier, e.g. ``'video2image'``."""
        return f"{self.source_type}2{self.target_type}"

    @property
    @abstractmethod
    def source_type(self) -> str:
        """The ``type_id`` of the media type this converter reads from."""

    @property
    @abstractmethod
    def target_type(self) -> str:
        """The ``type_id`` of the media type this converter produces."""

    @abstractmethod
    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Convert *media* and return a list of new media dicts.

        Framework call sites invoke :meth:`convert_normalized` (not
        :meth:`convert` directly), so when ``self`` is reached through
        the framework, *params* is guaranteed to be a non-``None`` dict
        with every declared field key populated.  Subclasses can therefore
        index ``params[key]`` directly.  Third-party call sites that
        invoke :meth:`convert` themselves should call
        :meth:`convert_normalized` instead, or route their reads through
        :meth:`get_param` to keep working with ``None`` / missing keys.

        Args:
            media: The source media dict (target of conversion).  Must
                contain at minimum ``media_bytes`` or ``media_path`` for
                binary types, or ``media_string`` for text.  ``filename``
                is used to derive output names.
            params: Mapping of :attr:`PluginField.key` → user-supplied
                value.  When reached via :meth:`convert_normalized`
                (the framework path), this is a fully-populated dict;
                otherwise it may be ``None`` or partially populated.

        Each returned dict must contain at minimum:

        * ``"filename"`` — a descriptive filename for the converted media.
        * The data fields expected by the target media type (e.g.
          ``"media_bytes"`` and ``"duration"`` for image/audio/video,
          ``"media_string"`` for text).

        Returns an empty list if the conversion fails or produces no
        output (e.g. an empty document).
        """

    def convert_normalized(
        self,
        media: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Validate and default-fill *params*, then dispatch to :meth:`convert`.

        This is the framework's entry point — every in-tree call site
        (importer multi-media ingestion, converter-folder runner,
        clipper-chain runner) routes through here so :meth:`convert`
        receives a fully-populated dict every time.

        On a validation failure (out-of-range number, unknown ``select``
        value, etc.) this raises :class:`ValueError` rather than
        :class:`marshmallow.ValidationError`, matching the rest of the
        framework's plugin-arg error contract.
        """
        normalized = self.normalize_params(params)
        return self.convert(media, normalized)

    # ------------------------------------------------------------------
    # Param helpers
    # ------------------------------------------------------------------

    def normalize_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Validate *params* and fill missing / empty-string keys with declared defaults.

        Pre-strips empty-string values for any declared field that has a
        non-empty :attr:`PluginField.default`, then runs
        :meth:`validate_params` (which loads through the per-plugin
        marshmallow schema, enforcing declared :attr:`PluginField.min` /
        :attr:`max` ranges and :attr:`options` whitelists).  This
        preserves the legacy :meth:`get_param` semantics where ``""``
        was treated as "unset" — marshmallow's ``Number`` fields would
        otherwise reject an empty string as "Not a valid integer".

        Subclasses rarely call this directly; :meth:`convert_normalized`
        wraps it.  Raises :class:`ValueError` on a validation failure.
        """
        from marshmallow import ValidationError  # noqa: PLC0415

        if params:
            scrubbed: dict[str, Any] = {}
            default_for: dict[str, str] = {f.key: f.default for f in self.fields if f.default}
            for key, value in params.items():
                if isinstance(value, str) and not value and key in default_for:
                    continue
                scrubbed[key] = value
            params = scrubbed

        try:
            validated = self.validate_params(params)
        except ValidationError as exc:
            raise ValueError(f"Invalid params for converter {self.name!r}: {exc.messages}") from exc

        for f in self.fields:
            existing = validated.get(f.key)
            if existing is None or (isinstance(existing, str) and existing == ""):
                if f.default:
                    validated[f.key] = f.default
        return validated

    def validate_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Validate *params* against this converter's :attr:`fields` schema.

        Converters are usually invoked from ``source_specs`` /
        ``clipper_chain`` dicts that flow through the API as pass-through
        payloads — they don't go through :func:`validate_plugin_args`
        like importer/exporter form bodies do.  Framework call sites
        invoke :meth:`convert_normalized`, which wraps this method;
        most plugin code never calls ``validate_params`` directly.
        Raises :class:`marshmallow.ValidationError` on a bad value.
        """
        from vtscore.plugins.schema import get_plugin_arg_schema  # noqa: PLC0415

        schema = get_plugin_arg_schema(self)
        loaded = schema.load(params or {})
        # ``Schema.load(<dict>)`` returns a dict (the ``many=True`` overload
        # returns a list); marshmallow's typing is too permissive to narrow
        # this automatically.
        assert isinstance(loaded, dict)
        return loaded

    def get_param(self, params: dict[str, Any] | None, key: str) -> Any:
        """Return the value for *key* from *params*, falling back to the
        declared :attr:`PluginField.default` for that key.

        Empty strings are treated as "unset" so a UI that submits empty
        inputs still gets the default.  Returns ``""`` if no field with
        *key* is declared.

        Framework-routed :meth:`convert` calls receive params already
        default-filled by :meth:`convert_normalized`, so this helper is
        mostly redundant for in-tree converters; it remains for
        third-party converters whose call sites bypass the framework
        wrapper.
        """
        if params is not None:
            value = params.get(key, None)
            if value is not None and value != "":
                return value
        for f in self.fields:
            if f.key == key:
                return f.default
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise converter metadata for API endpoints."""
        d: dict[str, Any] = {
            "name": self.name,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "display_name": self.display_name or f"{self.source_type.title()} → {self.target_type.title()}",
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
        }
        if self.summary_template:
            d["summary_template"] = self.summary_template
        return d
