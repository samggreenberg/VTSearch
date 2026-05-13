"""Image embedder — DINOv2 single-vector mode (``dinov2_single``).

Produces one CLS-pooled vector per image (768-dim).  Fast, small storage,
no region search — picks the same fixed-size representation used by the
DINOv2 paper for linear-probe classification benchmarks.

For the patch-region variant of the same backbone see
``embedder_dinov2_patch.py``.  Both share weights via
``_dinov2_shared.py``.

DINOv2 weights are **ungated** on Hugging Face — anyone can download
them without an HF account, which makes this embedder a friendly
out-of-the-box option for the bundled-image Docker build.  There is no
text encoder, so :attr:`supports_text` is ``False``.
"""

from __future__ import annotations

from vtsearch.media.image._dinov2_shared import _Dinov2Base


class ImageDinov2SingleEmbedder(_Dinov2Base):
    """Embeds images using DINOv2 ViT-B/14 with CLS-token pooling.

    Output dimension: 768.  No patch-region pipeline — use
    :class:`~vtsearch.media.image.embedder_dinov2_patch.ImageDinov2PatchEmbedder`
    if you need region search.
    """

    @property
    def name(self) -> str:
        return "dinov2_single"


EMBEDDER = ImageDinov2SingleEmbedder()
