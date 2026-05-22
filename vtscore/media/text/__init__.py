from vtscore.media.text.clipper import TextDefaultClipper, TextParagraphClipper, TextSentenceClipper
from vtscore.media.text.media_type import TextMediaType

MEDIA_TYPE = TextMediaType()
CLIPPERS = [TextDefaultClipper(), TextParagraphClipper(), TextSentenceClipper()]
