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
    dict on :meth:`convert`.  When no params are supplied (or an unknown
    key is passed), the field's declared :attr:`default` applies.

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
    converter_description: str = ""

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

        Args:
            media: The source media dict (target of conversion).  Must
                contain at minimum ``media_bytes`` or ``media_path`` for
                binary types, or ``media_string`` for text.  ``filename``
                is used to derive output names.
            params: Mapping of :attr:`PluginField.key` → user-supplied
                value, or ``None`` for "use declared defaults".  Implementers
                should always read params through :meth:`get_param` so
                missing keys fall back to the field defaults.

        Each returned dict must contain at minimum:

        * ``"filename"`` — a descriptive filename for the converted media.
        * The data fields expected by the target media type (e.g.
          ``"media_bytes"`` and ``"duration"`` for image/audio/video,
          ``"media_string"`` for text).

        Returns an empty list if the conversion fails or produces no
        output (e.g. an empty document).
        """

    # ------------------------------------------------------------------
    # Param helpers
    # ------------------------------------------------------------------

    def get_param(self, params: dict[str, Any] | None, key: str) -> Any:
        """Return the value for *key* from *params*, falling back to the
        declared :attr:`PluginField.default` for that key.

        Empty strings are treated as "unset" so a UI that submits empty
        inputs still gets the default.  Returns ``""`` if no field with
        *key* is declared.
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
        return {
            "name": self.name,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "display_name": self.display_name or f"{self.source_type.title()} → {self.target_type.title()}",
            "description": self.converter_description,
            "fields": [f.to_dict() for f in self.fields],
        }
