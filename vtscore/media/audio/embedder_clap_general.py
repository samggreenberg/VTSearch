"""Audio embedder - CLAP General 2024 (laion/larger_clap_general)."""

from __future__ import annotations

from vtscore.config import CLAP_GENERAL_MODEL_ID
from vtscore.media.audio._clap_shared import _ClapBase


class AudioClapGeneralEmbedder(_ClapBase):
    """Embeds audio files using the larger CLAP general checkpoint (laion/larger_clap_general).

    * Audio files → 512-dimensional vectors via CLAP's audio encoder.
    * Text queries → 512-dimensional vectors via CLAP's text encoder.

    Compared to the original ``laion/clap-htsat-unfused`` baseline, this
    larger general-purpose checkpoint is trained on a broader audio mix
    and tends to give stronger zero-shot transfer for general sounds.
    """

    @property
    def name(self) -> str:
        return "clap_general"

    @property
    def display_name(self) -> str:
        return "CLAP (general 2024)"

    @property
    def label(self) -> str:
        return "CLAP General"

    @property
    def model_id(self) -> str:
        return CLAP_GENERAL_MODEL_ID


EMBEDDER = AudioClapGeneralEmbedder()
