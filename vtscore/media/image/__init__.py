from vtscore.media.image.clipper import (
    ImageDefaultClipper,
    ImageFaceClipper,
    ImageObjectClipper,
    ImageTilingClipper,
)
from vtscore.media.image.media_type import ImageMediaType

MEDIA_TYPE = ImageMediaType()
CLIPPERS = [
    ImageDefaultClipper(),
    ImageTilingClipper(),
    ImageObjectClipper(),
    ImageFaceClipper(),
]
