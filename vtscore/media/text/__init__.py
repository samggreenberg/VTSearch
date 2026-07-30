from vtscore.media.text.cleaner import TextMarkupStripCleaner, TextWhitespaceCleaner
from vtscore.media.text.clipper import TextDefaultClipper, TextParagraphClipper, TextSentenceClipper
from vtscore.media.text.media_type import TextMediaType

MEDIA_TYPE = TextMediaType()
CLIPPERS = [TextDefaultClipper(), TextParagraphClipper(), TextSentenceClipper()]
# Registration order is run order: pull the markup out first, then normalise the
# whitespace the strip leaves behind.
CLEANERS = [TextMarkupStripCleaner(), TextWhitespaceCleaner()]
