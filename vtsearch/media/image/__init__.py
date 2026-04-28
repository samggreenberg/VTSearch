from vtsearch.media.image.clipper import ImageDefaultClipper, ImageTilingClipper
from vtsearch.media.image.media_type import ImageMediaType

MEDIA_TYPE = ImageMediaType()
CLIPPERS = [ImageDefaultClipper(), ImageTilingClipper()]
