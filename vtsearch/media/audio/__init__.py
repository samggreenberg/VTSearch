from vtsearch.media.audio.clipper import SoundDefaultClipper, SoundTilingClipper
from vtsearch.media.audio.media_type import AudioMediaType

MEDIA_TYPE = AudioMediaType()
CLIPPERS = [SoundDefaultClipper(), SoundTilingClipper(2.0)]
