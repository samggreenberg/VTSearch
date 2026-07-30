from vtscore.media.image.cleaner import ImageEdgeTrimCleaner, ImageExifOrientCleaner
from vtscore.media.image.clipper import (
    ImageDefaultClipper,
    ImageObjectClipper,
    ImageTilingClipper,
)
from vtscore.media.image.media_type import ImageMediaType

MEDIA_TYPE = ImageMediaType()
CLIPPERS = [
    ImageDefaultClipper(),
    ImageTilingClipper(),
    ImageObjectClipper(),
]
# Registration order is run order: bake the display rotation in *before*
# looking for solid margins, so the edge box is measured on the upright frame.
CLEANERS = [ImageExifOrientCleaner(), ImageEdgeTrimCleaner()]
