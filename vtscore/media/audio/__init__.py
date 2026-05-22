from vtscore.media.audio.clipper import (
    SoundAutoClipper,
    SoundDefaultClipper,
    SoundSilenceClipper,
    SoundSpeechActivityClipper,
    SoundTilingClipper,
)
from vtscore.media.audio.media_type import AudioMediaType

MEDIA_TYPE = AudioMediaType()
CLIPPERS = [
    SoundAutoClipper(),
    SoundDefaultClipper(),
    SoundTilingClipper(2.0),
    SoundSilenceClipper(),
    SoundSpeechActivityClipper(),
]
