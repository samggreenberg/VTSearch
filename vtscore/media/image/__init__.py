from vtscore.media.image.cleaner import ImageEdgeTrimCleaner, ImageExifOrientCleaner
from vtscore.media.image.clipper import (
    ImageDefaultClipper,
    ImageObjectClipper,
    ImageTilingClipper,
)
from vtscore.media.image.decode import configure_pil_limits
from vtscore.media.image.media_type import ImageMediaType

# The media registry imports this package eagerly while auto-discovering media
# types, so lifting Pillow's decompression-bomb ceiling here puts it in place
# before any image is opened anywhere in the process — including from code that
# reaches for ``PIL.Image.open`` directly.
configure_pil_limits()

MEDIA_TYPE = ImageMediaType()
CLIPPERS = [
    ImageDefaultClipper(),
    ImageTilingClipper(),
    ImageObjectClipper(),
]
# Registration order is run order: bake the display rotation in *before*
# looking for solid margins, so the edge box is measured on the upright frame.
CLEANERS = [ImageExifOrientCleaner(), ImageEdgeTrimCleaner()]
