from vtsearch.media.text.clipper import TextDefaultClipper, TextSentenceClipper
from vtsearch.media.text.embedder import TextE5Embedder
from vtsearch.media.text.embedder_bge import TextBGEEmbedder
from vtsearch.media.text.media_type import TextMediaType

MEDIA_TYPE = TextMediaType()
EMBEDDERS = [TextE5Embedder(), TextBGEEmbedder()]
CLIPPERS = [TextDefaultClipper(), TextSentenceClipper()]
