from vtsearch.media.audio.clipper import SoundAutoClipper, SoundDefaultClipper, SoundTilingClipper
from vtsearch.media.audio.media_type import AudioMediaType

MEDIA_TYPE = AudioMediaType()
CLIPPERS = [SoundAutoClipper(), SoundDefaultClipper(), SoundTilingClipper(2.0)]
