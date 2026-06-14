"""Audio embedder - CLAP Music & Speech (laion/larger_clap_music_and_speech)."""

from __future__ import annotations

from vtscore.config import CLAP_MUSIC_MODEL_ID
from vtscore.media.audio._clap_shared import _ClapBase


class AudioClapMusicEmbedder(_ClapBase):
    """Embeds audio files using the larger CLAP model (laion/larger_clap_music_and_speech).

    * Audio files → 512-dimensional vectors via CLAP's audio encoder.
    * Text queries → 512-dimensional vectors via CLAP's text encoder.

    This variant is trained on music and speech data, providing better
    performance for music retrieval and genre classification compared to
    the unfused CLAP model.
    """

    @property
    def name(self) -> str:
        return "clap_music"

    @property
    def display_name(self) -> str:
        return "CLAP (music)"

    @property
    def label(self) -> str:
        return "CLAP Music"

    @property
    def model_id(self) -> str:
        return CLAP_MUSIC_MODEL_ID


EMBEDDER = AudioClapMusicEmbedder()
