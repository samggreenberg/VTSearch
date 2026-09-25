"""SIFT/VLAD tuned for document scans — same backend, a document-sized keypoint budget.

The shipped :mod:`~vtscore.media.image.embedder_sift_vlad` keeps
``DEFAULT_MAX_FEATURES`` (1,024) keypoints per image, which is right for photos
and wrong for scanned pages: a page is mostly text, the text wins the response
ranking, and the mark that is actually being searched for gets none of the
budget (#3911).  This embedder is that one with
:data:`~vtscore.media.structural.DOCUMENT_MAX_FEATURES` (8,192) instead.

**It is a separate embedder rather than a flag because the app already
expresses "which structural settings did this dataset use" as an embedder
name.**  A dataset records the embedder it bound, cells are keyed on that name,
and a detector reads it back — so binding ``sift_vlad_doc`` states the choice
where every consumer already looks, and no existing ``sift_vlad`` cell changes
meaning or needs rebuilding.

**Measured on FullMarks** (23 roster classes, 721 instances; #3911, #4021):
retrieval AP **0.88** at 8,192 keypoints against **0.12** for SigLIP and 0.16
for the shipped 1,024-keypoint pairing.

**The budget and the detection cap are coupled, so do not move one alone**
(#4021).  At 8,192 keypoints the shipped 2 MP
:data:`~vtscore.config.MAX_STRUCTURAL_DETECT_PIXELS` is the optimum (1 MP ties
at 0.87, 4 MP 0.76, uncapped 0.74, 0.25 MP 0.50); at 1,024 keypoints the
optimum moves to 0.5 MP and 2 MP collapses to 0.16.  This embedder therefore
takes the cap from the shared config rather than pinning its own, so an
operator who moves the cap moves it for both — but anyone who does should
re-measure the pair, not just the knob they touched.

**Storage is the price, and it is large.**  Local features scale with the
budget: a tier-`l` FullMarks cell (200k pages) is **~167 GB** at 2 MP / 8,192
against **~78 GB** at 1 MP / 8,192, where the 1,024-keypoint cell is a small
fraction of either (#3911 measured ~169 KB/page at 8,192; #4021 priced the
tier-`l` build at ~5.3 h on 24 CPUs).  Bind this on a large document corpus
only with that number in front of you.
"""

from __future__ import annotations

from vtscore.media.image._structural_shared import _StructuralImageBase
from vtscore.media.structural import DOCUMENT_MAX_FEATURES, SiftMatcher, StructuralMatcher


class ImageSiftVladDocEmbedder(_StructuralImageBase):
    """Document scans → VLAD vector (Stage 1) + SIFT keypoints/descriptors (Stage 2)."""

    @property
    def name(self) -> str:
        return "sift_vlad_doc"

    @property
    def display_name(self) -> str:
        return "SIFT/VLAD (document scans)"

    @property
    def embedding_dim(self) -> int:
        # Same shipped VLAD codebook as `sift_vlad`: the budget changes how many
        # descriptors are aggregated, never the width they aggregate into.
        return 8192

    @property
    def max_features(self) -> int:
        return DOCUMENT_MAX_FEATURES

    def _make_matcher(self) -> StructuralMatcher:
        # No `max_detect_pixels` override: the cap stays the shared, env-tunable
        # `MAX_STRUCTURAL_DETECT_PIXELS`, whose 2 MP default is the measured
        # optimum at this budget (see the module docstring on the coupling).
        return SiftMatcher()


EMBEDDER = ImageSiftVladDocEmbedder()
