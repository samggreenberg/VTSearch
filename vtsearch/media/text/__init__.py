from vtsearch.media.text.clipper import TextDefaultClipper, TextParagraphClipper, TextSentenceClipper
from vtsearch.media.text.media_type import TextMediaType

MEDIA_TYPE = TextMediaType()
CLIPPERS = [TextDefaultClipper(), TextParagraphClipper(), TextSentenceClipper()]
