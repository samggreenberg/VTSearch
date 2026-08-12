"""Audio embedder - CLAP (laion/clap-htsat-unfused), the cheap/fast tier."""

from __future__ import annotations

from vtscore.config import CLAP_MODEL_ID
from vtscore.media.audio._clap_shared import _ClapBase


class AudioClapEmbedder(_ClapBase):
    """Embeds audio files using the CLAP model (laion/clap-htsat-unfused).

    * Audio files → 512-dimensional vectors via CLAP's audio encoder.
    * Text queries → 512-dimensional vectors via CLAP's text encoder.

    The smaller of the two general-purpose CLAP checkpoints (HTSAT 768-d
    hidden, depths ``[2, 2, 6, 2]``, ~614 MB of weights).  It loses to
    :class:`~vtscore.media.audio.embedder_clap_general.AudioClapGeneralEmbedder`
    on every measured retrieval metric, but embeds roughly 2x faster for
    ~20% less disk, which is why it stays available as the cheap tier
    rather than being the default.
    """

    @property
    def name(self) -> str:
        return "clap"

    @property
    def display_name(self) -> str:
        return "CLAP (general, faster)"

    @property
    def label(self) -> str:
        return "CLAP Fast"

    @property
    def model_id(self) -> str:
        return CLAP_MODEL_ID


EMBEDDER = AudioClapEmbedder()
