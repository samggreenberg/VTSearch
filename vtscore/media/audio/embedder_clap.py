"""Audio embedder - CLAP (laion/clap-htsat-unfused)."""

from __future__ import annotations

from vtscore.config import CLAP_MODEL_ID
from vtscore.media.audio._clap_shared import _ClapBase


class AudioClapEmbedder(_ClapBase):
    """Embeds audio files using the CLAP model (laion/clap-htsat-unfused).

    * Audio files → 512-dimensional vectors via CLAP's audio encoder.
    * Text queries → 512-dimensional vectors via CLAP's text encoder.
    """

    @property
    def name(self) -> str:
        return "clap"

    @property
    def display_name(self) -> str:
        return "CLAP (general audio)"

    @property
    def label(self) -> str:
        return "CLAP"

    @property
    def model_id(self) -> str:
        return CLAP_MODEL_ID

    @property
    def is_default(self) -> bool:
        return True


EMBEDDER = AudioClapEmbedder()
