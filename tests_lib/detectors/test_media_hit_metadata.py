"""``build_media_hit`` carries the importer's ``custom_metadata`` onto the hit.

An exporter's whole job is often to hand results back to the system the media
came from, and the only thing on a media that names a row *in that system* is
the importer-supplied ``custom_metadata`` (asset ids, catalogue keys, …).  It
therefore travels with the hit, minus the ``embedding`` plumbing key.
"""

from __future__ import annotations

import json

import numpy as np

from vtscore.utils.hits import build_media_hit, hit_custom_metadata


class TestCustomMetadataOnHit:
    def test_included_when_media_has_it(self):
        media = {"filename": "a.wav", "category": "audio", "md5": "abc", "custom_metadata": {"asset_id": "XY-7"}}
        hit = build_media_hit(1, media, 0.9)
        assert hit["custom_metadata"] == {"asset_id": "XY-7"}

    def test_absent_when_media_has_none(self):
        hit = build_media_hit(1, {"filename": "a.wav", "category": "audio"}, 0.9)
        assert "custom_metadata" not in hit

    def test_absent_when_empty_or_null(self):
        for value in ({}, None, "not-a-dict"):
            hit = build_media_hit(1, {"filename": "a.wav", "custom_metadata": value}, 0.9)
            assert "custom_metadata" not in hit, value

    def test_embedding_key_is_stripped(self):
        """A pre-computed vector shipped via ``custom_metadata_map`` stays out of the hit."""
        media = {
            "filename": "a.wav",
            "category": "audio",
            "custom_metadata": {"asset_id": "XY-7", "embedding": np.zeros(4, dtype=np.float32)},
        }
        hit = build_media_hit(1, media, 0.9)
        assert hit["custom_metadata"] == {"asset_id": "XY-7"}
        # The JSON exporters call json.dumps with no default=; a numpy array
        # riding along inside custom_metadata would fail the whole export.
        assert json.loads(json.dumps(hit))["custom_metadata"] == {"asset_id": "XY-7"}

    def test_only_embedding_means_no_custom_metadata_key(self):
        media = {"filename": "a.wav", "custom_metadata": {"embedding": np.zeros(4, dtype=np.float32)}}
        assert "custom_metadata" not in build_media_hit(1, media, 0.9)

    def test_hit_metadata_is_a_copy(self):
        media = {"filename": "a.wav", "custom_metadata": {"asset_id": "XY-7"}}
        hit = build_media_hit(1, media, 0.9)
        hit["custom_metadata"]["asset_id"] = "mutated"
        assert media["custom_metadata"] == {"asset_id": "XY-7"}

    def test_extra_kwarg_still_wins(self):
        media = {"filename": "a.wav", "custom_metadata": {"asset_id": "XY-7"}}
        hit = build_media_hit(1, media, 0.9, custom_metadata={"asset_id": "override"})
        assert hit["custom_metadata"] == {"asset_id": "override"}

    def test_other_hit_fields_are_unchanged(self):
        media = {
            "filename": "a.wav",
            "category": "audio",
            "md5": "abc",
            "origin": {"importer": "server_folder"},
            "origin_name": "a.wav",
            "custom_metadata": {"asset_id": "XY-7"},
        }
        hit = build_media_hit(7, media, 0.87312, label="good")
        assert hit["id"] == 7
        assert hit["filename"] == "a.wav"
        assert hit["category"] == "audio"
        assert hit["score"] == 0.8731
        assert hit["md5"] == "abc"
        assert hit["origin"] == {"importer": "server_folder"}
        assert hit["origin_name"] == "a.wav"
        assert hit["label"] == "good"


class TestHitCustomMetadataDirectly:
    """The sanitiser is public because the app's route helpers reuse it.

    ``vtsearch.routes._shared.media_info_for_response``, ``POST
    /api/medias/batch`` and the enriched label export all call it, so its
    contract is pinned here rather than only through ``build_media_hit``.
    """

    def test_strips_the_embedding_channel(self):
        media = {"custom_metadata": {"asset_id": "XY-7", "embedding": np.zeros(4, dtype=np.float32)}}
        assert hit_custom_metadata(media) == {"asset_id": "XY-7"}

    def test_empty_dict_for_a_media_without_usable_metadata(self):
        for value in ({}, None, "not-a-dict", 7):
            assert hit_custom_metadata({"custom_metadata": value}) == {}, value
        assert hit_custom_metadata({}) == {}

    def test_result_is_a_fresh_dict(self):
        media = {"custom_metadata": {"asset_id": "XY-7"}}
        out = hit_custom_metadata(media)
        out["asset_id"] = "mutated"
        assert media["custom_metadata"] == {"asset_id": "XY-7"}
