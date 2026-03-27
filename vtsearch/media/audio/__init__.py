from vtsearch.media.audio.clipper import SoundDefaultClipper, SoundTilingClipper
from vtsearch.media.audio.embedder import AudioClapEmbedder
from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder
from vtsearch.media.audio.media_type import AudioMediaType

MEDIA_TYPE = AudioMediaType()
EMBEDDERS = [AudioClapEmbedder(), AudioClapMusicEmbedder()]
CLIPPERS = [SoundDefaultClipper(), SoundTilingClipper(2.0)]
