"""The document keypoint budget (#3908/#3911/#4021): a profile, not a global change.

The shipped ``sift_vlad`` budget of 1,024 keypoints starves a document scan —
text wins the response ranking and the mark gets none of it — so document work
binds ``sift_vlad_doc`` at 8,192 instead.  What these tests protect is that the
*photo* path is untouched by that, since changing it would silently re-score
every existing ``sift_vlad`` dataset.
"""

from __future__ import annotations


class TestPhotoPathUnchanged:
    def test_sift_vlad_still_keeps_1024(self):
        from vtscore.media.image.embedder_sift_vlad import ImageSiftVladEmbedder
        from vtscore.media.structural import DEFAULT_MAX_FEATURES

        assert DEFAULT_MAX_FEATURES == 1024
        assert ImageSiftVladEmbedder().max_features == 1024

    def test_a_structural_embedder_that_opts_out_keeps_the_shipped_budget(self):
        """The no-profile case: not overriding ``max_features`` means today's value."""
        from vtscore.media.image._structural_shared import _StructuralImageBase
        from vtscore.media.structural import DEFAULT_MAX_FEATURES

        class _NoProfile(_StructuralImageBase):
            @property
            def name(self) -> str:
                return "_no_profile"

            @property
            def display_name(self) -> str:
                return "no profile"

        assert _NoProfile().max_features == DEFAULT_MAX_FEATURES


class TestDocumentProfile:
    def test_budget_is_8192(self):
        from vtscore.media.image.embedder_sift_vlad_doc import ImageSiftVladDocEmbedder
        from vtscore.media.structural import DOCUMENT_MAX_FEATURES

        assert DOCUMENT_MAX_FEATURES == 8192
        assert ImageSiftVladDocEmbedder().max_features == 8192

    def test_identity_separates_it_from_the_photo_cell(self):
        """Cells are keyed on the embedder name, so a distinct name is what keeps
        a document build from overwriting or being read as a ``sift_vlad`` one."""
        from vtscore.media.image.embedder_sift_vlad import ImageSiftVladEmbedder
        from vtscore.media.image.embedder_sift_vlad_doc import ImageSiftVladDocEmbedder

        photo, doc = ImageSiftVladEmbedder(), ImageSiftVladDocEmbedder()
        assert doc.name == "sift_vlad_doc" != photo.name
        assert doc.media_type_id == photo.media_type_id == "image"
        # Same shipped VLAD codebook: the budget changes how many descriptors are
        # aggregated, never the width.
        assert doc.embedding_dim == photo.embedding_dim == 8192
        assert doc.supports_geometric_verification is True
        assert doc.supports_text is False
        assert doc.is_default is False

    def test_registered_and_listed_for_images(self):
        from vtscore.media import embedders_for_type, get_embedder
        from vtscore.media.image._structural_shared import _StructuralImageBase

        registered = get_embedder("sift_vlad_doc")
        # `get_embedder` is typed to the MediaEmbedder base; `max_features` is a
        # structural-embedder property, so narrow before reading it.
        assert isinstance(registered, _StructuralImageBase)
        assert registered.max_features == 8192
        assert "sift_vlad_doc" in [e.name for e in embedders_for_type("image")]

    def test_detection_cap_stays_the_shared_one(self):
        """#4021: the cap and the budget are coupled, and 2 MP is the measured
        optimum AT this budget.  The profile must not pin its own cap, or an
        operator moving ``MAX_STRUCTURAL_DETECT_PIXELS`` would move it for photos
        only and silently leave documents on a stale pairing."""
        from vtscore.config import MAX_STRUCTURAL_DETECT_PIXELS
        from vtscore.media.image.embedder_sift_vlad_doc import ImageSiftVladDocEmbedder
        from vtscore.media.structural import SiftMatcher

        matcher = ImageSiftVladDocEmbedder()._make_matcher()
        # `_make_matcher` is typed to the StructuralMatcher protocol; the cap is
        # SiftMatcher's own state, so narrow before reading it.
        assert isinstance(matcher, SiftMatcher)
        assert matcher._max_detect_pixels == MAX_STRUCTURAL_DETECT_PIXELS
