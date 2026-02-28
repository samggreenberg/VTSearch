"""Media converters — transform media of one type into media of another type.

A :class:`~vtsearch.converters.base.MediaConverter` takes a single media dict
of one :class:`~vtsearch.media.base.MediaType` and returns one or more media
dicts of a *different* media type.

Built-in converters
-------------------
* :class:`Document2ImageMediaConverter` — render document pages as images.
* :class:`Document2TextMediaConverter` — extract embedded text from documents.
* :class:`Video2AudioMediaConverter` — extract the audio track from a video.
* :class:`Video2ImageMediaConverter` — sample frames from a video as images.
"""

from vtsearch.converters.base import MediaConverter
from vtsearch.converters.document2image import Document2ImageMediaConverter
from vtsearch.converters.document2text import Document2TextMediaConverter
from vtsearch.converters.video2audio import Video2AudioMediaConverter
from vtsearch.converters.video2image import Video2ImageMediaConverter

__all__ = [
    "MediaConverter",
    "Document2ImageMediaConverter",
    "Document2TextMediaConverter",
    "Video2AudioMediaConverter",
    "Video2ImageMediaConverter",
]
