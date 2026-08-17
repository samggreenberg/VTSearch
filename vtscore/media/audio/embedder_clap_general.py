"""Audio embedder - CLAP General (laion/larger_clap_general), the audio default."""

from __future__ import annotations

from vtscore.config import CLAP_GENERAL_MODEL_ID
from vtscore.media.audio._clap_shared import _ClapBase


class AudioClapGeneralEmbedder(_ClapBase):
    """Embeds audio files using the larger CLAP general checkpoint (laion/larger_clap_general).

    * Audio files → 512-dimensional vectors via CLAP's audio encoder.
    * Text queries → 512-dimensional vectors via CLAP's text encoder.

    The default audio embedder.  Compared to the original
    ``laion/clap-htsat-unfused`` baseline (still shipped as ``clap``), this
    larger general-purpose checkpoint is trained on a broader audio mix and
    wins every measured comparison on the full ESC-50: text-sort mAP 0.869-0.895
    vs 0.850-0.866, learned-sort mean F1 0.523-0.564 vs 0.457-0.529, and
    leave-one-out 1-NN accuracy 0.973 vs 0.958.  The cost is ~2.1x the embedding
    time and ~776 MB of weights against ``clap``'s ~614 MB, so ``clap`` remains
    the explicit cheap/fast choice.
    """

    @property
    def name(self) -> str:
        return "clap_general"

    @property
    def display_name(self) -> str:
        return "CLAP (general, larger)"

    @property
    def label(self) -> str:
        return "CLAP General"

    @property
    def model_id(self) -> str:
        return CLAP_GENERAL_MODEL_ID

    @property
    def is_default(self) -> bool:
        return True


EMBEDDER = AudioClapGeneralEmbedder()
