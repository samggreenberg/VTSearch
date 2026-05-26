"""Image embedder - DINOv3 single-vector mode (``dinov3_single``).

Produces one CLS-pooled vector per image (768-dim).  Fast, small storage,
no region search - uses the same fixed-size representation the DINOv3
paper linear-probes for classification benchmarks.

For the patch-region variant of the same backbone see
``embedder_dinov3_patch.py``.  Both share weights via
``_dinov3_shared.py``.

DINOv3 weights are gated on Hugging Face - running this embedder requires
the ``HF_TOKEN`` env var to be set to a user token that has accepted the
DINOv3 licence at
``https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m``.
No text encoder, so :attr:`supports_text` is ``False``.
"""

from __future__ import annotations

from vtscore.media.image._dinov3_shared import _Dinov3Base


class ImageDinov3SingleEmbedder(_Dinov3Base):
    """Embeds images using DINOv3 ViT-B/16 with CLS-token pooling.

    Output dimension: 768.  No patch-region pipeline - use
    :class:`~vtscore.media.image.embedder_dinov3_patch.ImageDinov3PatchEmbedder`
    if you need region search.
    """

    @property
    def name(self) -> str:
        return "dinov3_single"

    @property
    def display_name(self) -> str:
        return "DINOv3 single (image vector)"


EMBEDDER = ImageDinov3SingleEmbedder()
