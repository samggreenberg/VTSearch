"""Media converters - transform media of one type into media of another type.

A :class:`~vtscore.converters.base.MediaConverter` takes a single media dict
of one :class:`~vtscore.media.base.MediaType` and returns one or more media
dicts of a *different* media type.

Built-in converters are auto-discovered via the ``CONVERTER`` sentinel
attribute, just like exporters and importers.
"""

from vtscore.converters.audio2image import Audio2ImageMediaConverter
from vtscore.converters.audio2text import Audio2TextMediaConverter
from vtscore.converters.base import MediaConverter
from vtscore.converters.document2image import Document2ImageMediaConverter
from vtscore.converters.document2text import Document2TextMediaConverter
from vtscore.converters.image2text import Image2TextMediaConverter
from vtscore.converters.video2audio import Video2AudioMediaConverter
from vtscore.converters.video2image import Video2ImageMediaConverter
from vtscore.plugins import make_plugin_registry

# ---------------------------------------------------------------------------
# Converter registry (auto-discovered via CONVERTER sentinel)
# ---------------------------------------------------------------------------

get_converter, list_converters = make_plugin_registry(
    package=__name__,
    sentinel="CONVERTER",
    label="media converter",
    discover_modules=True,
    entry_point_group="vtscore.converters",
)


def list_converters_for_target(target_type: str) -> list[MediaConverter]:
    """Return converters that produce *target_type* (a ``type_id``)."""
    return [c for c in list_converters() if c.target_type == target_type]


def list_converters_for_source(source_type: str) -> list[MediaConverter]:
    """Return converters that consume *source_type* (a ``type_id``)."""
    return [c for c in list_converters() if c.source_type == source_type]


def default_converter_for(source_type: str, preferred_target: str = "") -> MediaConverter | None:
    """The converter a *convert-out* half type takes before it can be embedded.

    A half type (``document``: files to ingest, no embedder of its own) names
    the types it can become in
    :attr:`~vtscore.media.base.MediaType.converts_to`.  *preferred_target* wins
    when it is one of them (a text dataset fed PDFs extracts their text);
    otherwise the first entry does (a PDF's pages, as images).

    ``None`` for a type that embeds as it is (image, audio, video, text,
    face), for an unknown type, and for a half type no registered converter
    leaves.
    """
    from vtscore.media import get as get_media_type  # noqa: PLC0415

    try:
        mt = get_media_type(source_type)
    except KeyError:
        return None
    targets = list(mt.converts_to)
    if mt.embeddable or not targets:
        return None
    target = preferred_target if preferred_target in targets else targets[0]
    candidates = [c for c in list_converters_for_source(mt.type_id) if c.target_type == target]
    # A converter the pickers hide is still a converter; prefer one they show.
    candidates.sort(key=lambda c: bool(c.hidden_from_picker))
    return candidates[0] if candidates else None


__all__ = [
    "MediaConverter",
    "Audio2ImageMediaConverter",
    "Audio2TextMediaConverter",
    "Document2ImageMediaConverter",
    "Document2TextMediaConverter",
    "Image2TextMediaConverter",
    "Video2AudioMediaConverter",
    "Video2ImageMediaConverter",
    "default_converter_for",
    "list_converters",
    "get_converter",
    "list_converters_for_target",
    "list_converters_for_source",
]
