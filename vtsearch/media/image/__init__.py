from vtsearch.media.image.clipper import ImageDefaultClipper, ImageTilingClipper
from vtsearch.media.image.embedder import ImageClipEmbedder
from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder
from vtsearch.media.image.media_type import ImageMediaType

MEDIA_TYPE = ImageMediaType()
EMBEDDERS = [ImageClipEmbedder(), ImageSiglipEmbedder()]
CLIPPERS = [ImageDefaultClipper(), ImageTilingClipper()]
