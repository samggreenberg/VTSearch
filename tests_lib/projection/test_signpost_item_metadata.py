"""Library-tier tests for signpost texts surfacing as item metadata.

A dataset prepped for Browse already carries a model-generated text per media
(the caption / tag list that letters the map).  ``MediaType.display_metadata``
surfaces it in the labeling UI's metadata grid under an explicitly hedged
label — "AI Caption" / "AI Tags", never a bare "Description" — so a user reads
it as a model's guess rather than curated truth.

These tests lock the wiring (every media type inherits the row through the
base implementation, without disturbing the type-specific fields) and the
honesty of the labeling.
"""

from __future__ import annotations

from vtscore.media.audio.media_type import AudioMediaType
from vtscore.media.image.media_type import ImageMediaType
from vtscore.media.text.media_type import TextMediaType
from vtscore.projection import signpost_texts as st


def _image_media(**extra) -> dict:
    return {
        "id": 1,
        "media_type": "image",
        "category": "dogs",
        "file_size": 4096,
        "width": 800,
        "height": 600,
        **extra,
    }


class TestCaptionsAppearAsItemMetadata:
    def test_caption_row_is_titled_ai_caption(self):
        media = _image_media(**{st.TEXT_FIELD: "a golden retriever on a beach", st.KIND_FIELD: st.KIND_CAPTION})
        meta = ImageMediaType().display_metadata(media)
        assert meta["AI Caption"] == "a golden retriever on a beach"

    def test_tag_row_is_titled_ai_tags(self):
        media = {
            "id": 2,
            "media_type": "audio",
            "category": "field",
            "file_size": 1024,
            st.TEXT_FIELD: "Rain, Thunderstorm, Wind",
            st.KIND_FIELD: st.KIND_TAGS,
        }
        meta = AudioMediaType().display_metadata(media)
        assert meta["AI Tags"] == "Rain, Thunderstorm, Wind"

    def test_type_specific_fields_are_untouched(self):
        media = _image_media(**{st.TEXT_FIELD: "a cat", st.KIND_FIELD: st.KIND_CAPTION})
        meta = ImageMediaType().display_metadata(media)
        assert meta["Dimensions"] == "800×600"
        assert meta["Category"] == "dogs"
        assert meta["File Size"] == 4096

    def test_unprepped_media_gets_no_extra_row(self):
        # The overwhelmingly common case: a dataset that never ran the signpost
        # stage must produce exactly the metadata it did before.
        meta = ImageMediaType().display_metadata(_image_media())
        assert set(meta) == {"Category", "Dimensions", "File Size"}

    def test_text_media_own_content_is_not_surfaced(self):
        # The text provider's "text" is the item's own content, which the
        # viewer already shows — and no model wrote it, so no "AI …" row.
        media = {
            "id": 3,
            "media_type": "text",
            "word_count": 12,
            st.TEXT_FIELD: "the paragraph itself",
            st.KIND_FIELD: st.KIND_CONTENT,
        }
        meta = TextMediaType().display_metadata(media)
        assert not [key for key in meta if key.startswith("AI ")]
        assert meta["Word Count"] == 12

    def test_no_label_claims_to_be_authoritative(self):
        media = _image_media(**{st.TEXT_FIELD: "a cat", st.KIND_FIELD: st.KIND_CAPTION})
        meta = ImageMediaType().display_metadata(media)
        generated = [key for key in meta if key not in {"Category", "Dimensions", "File Size"}]
        assert generated == ["AI Caption"]
