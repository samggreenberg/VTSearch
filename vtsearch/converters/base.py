"""Abstract base class for media converters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MediaConverter(ABC):
    """Convert a media dict of one :class:`~vtsearch.media.base.MediaType`
    into one or more media dicts of a *different* media type.

    Subclasses must implement:

    * :attr:`source_type` — the ``type_id`` of the input media type.
    * :attr:`target_type` — the ``type_id`` of the output media type.
    * :meth:`convert` — perform the actual conversion.

    The returned media dicts contain the fields produced by the target
    media type's :meth:`~vtsearch.media.base.MediaType.load_media_data`
    (e.g. ``media_bytes``, ``duration``, ``width``, ``height``) plus a
    ``filename`` key.  They do **not** include ``id``, ``embedding``, or
    ``md5`` — the caller is responsible for assigning IDs, computing
    embeddings, and hashing.
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """The ``type_id`` of the media type this converter reads from."""

    @property
    @abstractmethod
    def target_type(self) -> str:
        """The ``type_id`` of the media type this converter produces."""

    @abstractmethod
    def convert(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert *media* and return a list of new media dicts.

        Each returned dict must contain at minimum:

        * ``"filename"`` — a descriptive filename for the converted media.
        * The data fields expected by the target media type (e.g.
          ``"media_bytes"`` and ``"duration"`` for image/audio/video,
          ``"media_string"`` for text).

        Returns an empty list if the conversion fails or produces no
        output (e.g. an empty document).
        """
