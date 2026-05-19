from vtsearch.media.audio.clipper import (
    SoundAutoClipper,
    SoundDefaultClipper,
    SoundSilenceClipper,
    SoundSpeechActivityClipper,
    SoundTilingClipper,
)
from vtsearch.media.audio.media_type import AudioMediaType

MEDIA_TYPE = AudioMediaType()
CLIPPERS = [
    SoundAutoClipper(),
    SoundDefaultClipper(),
    SoundTilingClipper(2.0),
    SoundSilenceClipper(),
    SoundSpeechActivityClipper(),
]
