"""SIFT/VLAD structural embedder - v1 instance-matching backend.

Classic SIFT + VLAD aggregation for Stage-1 retrieval and SIFT +
similarity-transform RANSAC for Stage-2 geometric verification (design:
``docs/plans/structural-embedder.md``).  CPU-friendly, deterministic, ungated,
and light on dependencies (SIFT is in mainline OpenCV since 4.4).

This is a *structural* embedder: it searches for specific instances ("this exact
logo") rather than semantic categories ("a cola can").  It sets
``supports_geometric_verification = True`` (so the loader stores
``media["local_features"]``) and ``supports_text = False`` (no text encoder maps
into VLAD space; the text sort greys out via the existing gate).
"""

from __future__ import annotations

from vtscore.media.image._structural_shared import _StructuralImageBase
from vtscore.media.structural import SiftMatcher, StructuralMatcher


class ImageSiftVladEmbedder(_StructuralImageBase):
    """Images → VLAD vector (Stage 1) + SIFT keypoints/descriptors (Stage 2)."""

    @property
    def name(self) -> str:
        return "sift_vlad"

    @property
    def display_name(self) -> str:
        return "SIFT/VLAD (instance matching)"

    def _make_matcher(self) -> StructuralMatcher:
        return SiftMatcher()


EMBEDDER = ImageSiftVladEmbedder()
