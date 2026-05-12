"""Image embedder — EUPE single-vector mode (``eupe_single``).

Produces one CLS-pooled vector per image (768-dim) using the
``facebookresearch/EUPE`` ViT-B/16 backbone.  Fast, small storage, no
region search.

For the patch-region variant of the same backbone see
``embedder_eupe_patch.py``.  Both share weights via ``_eupe_shared.py``.

EUPE's HF mirror at ``facebook/EUPE-ViT-B`` is ungated (no HF token
needed), but outputs are bound by Meta's FAIR Noncommercial Research
Licence — see :attr:`license_notice`.  No text encoder, so
:attr:`supports_text` is ``False``.
"""

from __future__ import annotations

from vtsearch.media.image._eupe_shared import _EupeBase


class ImageEupeSingleEmbedder(_EupeBase):
    """Embeds images using facebookresearch/EUPE ViT-B/16 with CLS pooling.

    Output dimension: 768.  No patch-region pipeline — use
    :class:`~vtsearch.media.image.embedder_eupe_patch.ImageEupePatchEmbedder`
    if you need region search.
    """

    @property
    def name(self) -> str:
        return "eupe_single"


EMBEDDER = ImageEupeSingleEmbedder()
