from vtscore.media.audio.clipper import (
    SoundDefaultClipper,
    SoundSilenceClipper,
    SoundSpeechActivityClipper,
    SoundTilingClipper,
)
from vtscore.media.audio.media_type import AudioMediaType

MEDIA_TYPE = AudioMediaType()
CLIPPERS = [
    SoundTilingClipper(10.0, 1.0),
    SoundDefaultClipper(),
    SoundSilenceClipper(),
    SoundSpeechActivityClipper(),
]
