"""Media converters — transform media of one type into media of another type.

A :class:`~vtsearch.converters.base.MediaConverter` takes a single media dict
of one :class:`~vtsearch.media.base.MediaType` and returns one or more media
dicts of a *different* media type.

Built-in converters are auto-discovered via the ``CONVERTER`` sentinel
attribute, just like exporters and importers.
"""

from vtsearch.converters.audio2image import Audio2ImageMediaConverter
from vtsearch.converters.base import MediaConverter
from vtsearch.converters.document2image import Document2ImageMediaConverter
from vtsearch.converters.document2text import Document2TextMediaConverter
from vtsearch.converters.image2text import Image2TextMediaConverter
from vtsearch.converters.video2audio import Video2AudioMediaConverter
from vtsearch.converters.video2image import Video2ImageMediaConverter
from vtsearch.plugins import PluginRegistry

# ---------------------------------------------------------------------------
# Converter registry (auto-discovered via CONVERTER sentinel)
# ---------------------------------------------------------------------------

_registry: PluginRegistry[MediaConverter] = PluginRegistry(
    package="vtsearch.converters",
    sentinel="CONVERTER",
    label="media converter",
    discover_modules=True,
    entry_point_group="vtsearch.converters",
)


def list_converters() -> list[MediaConverter]:
    """Return all registered converters."""
    return _registry.list()


def get_converter(name: str) -> MediaConverter | None:
    """Return the converter with *name*, or ``None``."""
    return _registry.get(name)


def list_converters_for_target(target_type: str) -> list[MediaConverter]:
    """Return converters that produce *target_type* (a ``type_id``)."""
    return [c for c in _registry.list() if c.target_type == target_type]


def list_converters_for_source(source_type: str) -> list[MediaConverter]:
    """Return converters that consume *source_type* (a ``type_id``)."""
    return [c for c in _registry.list() if c.source_type == source_type]


__all__ = [
    "MediaConverter",
    "Audio2ImageMediaConverter",
    "Document2ImageMediaConverter",
    "Document2TextMediaConverter",
    "Image2TextMediaConverter",
    "Video2AudioMediaConverter",
    "Video2ImageMediaConverter",
    "list_converters",
    "get_converter",
    "list_converters_for_target",
    "list_converters_for_source",
]
