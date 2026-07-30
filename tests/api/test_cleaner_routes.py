"""API-tier tests for MediaCleaners: the listing route and variant serving.

See ``docs/plans/media-cleaners.md``.

Covers:
- ``GET /api/cleaners`` (all, filtered, empty for a type with no cleaners).
- Cleaners never leak into ``GET /api/clippers``.
- ``has_original`` on the media batch payload.
- ``?variant=original`` on the byte / text / thumbnail routes, including the
  fallbacks for an item without a snapshot.
"""

from __future__ import annotations

import io

import pytest
from vtsearch.state import medias


def _png_bytes(color: str, size: tuple[int, int] = (8, 8)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def cleaned_image_media():
    """Install one cleaned image media (canonical red, original blue).

    ``medias`` is intentionally not reset between tests (regenerating the test
    corpus is expensive), so the entry is added and removed here.
    """
    from vtscore.utils.hashing import content_md5

    canonical = _png_bytes("red")
    original = _png_bytes("blue", (16, 16))
    media_id = max(medias.keys(), default=0) + 1
    medias[media_id] = {
        "id": media_id,
        "media_type": "image",
        "filename": "photo.png",
        "media_bytes": canonical,
        "original_media_bytes": original,
        "duration": 0,
        "file_size": len(canonical),
        "md5": content_md5(canonical),
        "embeddings": {},
        "origin": {"importer": "server_folder", "params": {}},
        "origin_name": "photo.png",
    }
    try:
        yield media_id, canonical, original
    finally:
        medias.pop(media_id, None)


@pytest.fixture
def cleaned_text_media():
    """Install one cleaned text media (canonical upper, original lower)."""
    from vtscore.utils.hashing import content_md5

    media_id = max(medias.keys(), default=0) + 1
    medias[media_id] = {
        "id": media_id,
        "media_type": "text",
        "filename": "doc.txt",
        "media_string": "HELLO WORLD",
        "original_media_string": "hello  world",
        "duration": 0,
        "file_size": 11,
        "word_count": 2,
        "character_count": 11,
        "md5": content_md5(b"HELLO WORLD"),
        "embeddings": {},
        "origin": {"importer": "server_folder", "params": {}},
        "origin_name": "doc.txt",
    }
    try:
        yield media_id
    finally:
        medias.pop(media_id, None)


# ---------------------------------------------------------------------------
# GET /api/cleaners
# ---------------------------------------------------------------------------


class TestCleanersApiEndpoint:
    def test_list_all_cleaners(self, client):
        resp = client.get("/api/cleaners")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [c["name"] for c in data["cleaners"]]
        assert "image_exif_orient" in names
        for c in data["cleaners"]:
            assert "display_name" in c
            assert "media_type" in c
            # The flag drives which import checkboxes come up pre-checked, so
            # every entry must carry it explicitly.
            assert isinstance(c["default_enabled"], bool)

    def test_filter_by_type_id(self, client):
        resp = client.get("/api/cleaners?media_type=image")
        assert resp.status_code == 200
        cleaners = resp.get_json()["cleaners"]
        assert cleaners
        assert all(c["media_type"] == "image" for c in cleaners)

    def test_filter_with_no_registered_cleaners_is_empty(self, client):
        resp = client.get("/api/cleaners?media_type=audio")
        assert resp.status_code == 200
        assert resp.get_json()["cleaners"] == []

    def test_exif_cleaner_defaults_on(self, client):
        resp = client.get("/api/cleaners?media_type=image")
        exif = next(c for c in resp.get_json()["cleaners"] if c["name"] == "image_exif_orient")
        assert exif["default_enabled"] is True
        assert exif["description"]

    def test_cleaners_are_absent_from_the_clipper_listing(self, client):
        """A clipper chooser is a radio choice; a cleaner must never appear in it."""
        resp = client.get("/api/clippers")
        assert resp.status_code == 200
        assert "image_exif_orient" not in {c["name"] for c in resp.get_json()["clippers"]}


# ---------------------------------------------------------------------------
# has_original on the batch payload
# ---------------------------------------------------------------------------


class TestHasOriginalFlag:
    def test_flag_set_for_cleaned_media(self, client, cleaned_image_media):
        media_id, _canonical, _original = cleaned_image_media
        resp = client.post("/api/medias/batch", json={"ids": [media_id]})
        assert resp.status_code == 200
        assert resp.get_json()[0]["has_original"] is True

    def test_flag_absent_for_uncleaned_media(self, client):
        some_id = next(iter(medias))
        resp = client.post("/api/medias/batch", json={"ids": [some_id]})
        assert resp.status_code == 200
        assert "has_original" not in resp.get_json()[0]


# ---------------------------------------------------------------------------
# ?variant=original
# ---------------------------------------------------------------------------


class TestVariantServing:
    def test_image_route_serves_each_variant(self, client, cleaned_image_media):
        media_id, canonical, original = cleaned_image_media
        assert client.get(f"/api/medias/{media_id}/image").data == canonical
        assert client.get(f"/api/medias/{media_id}/image?variant=original").data == original

    def test_generic_media_route_serves_each_variant(self, client, cleaned_image_media):
        media_id, canonical, original = cleaned_image_media
        assert client.get(f"/api/medias/{media_id}/media").data == canonical
        assert client.get(f"/api/medias/{media_id}/media?variant=original").data == original

    def test_thumbnail_variant_differs_from_canonical(self, client, cleaned_image_media):
        """The stored thumbnail describes the cleaned item, so the original
        variant has to regenerate rather than reuse it."""
        media_id, _canonical, _original = cleaned_image_media
        clean = client.get(f"/api/medias/{media_id}/thumbnail")
        original = client.get(f"/api/medias/{media_id}/thumbnail?variant=original")
        assert clean.status_code == original.status_code == 200
        assert clean.data != original.data

    def test_text_route_serves_each_variant_with_recounted_stats(self, client, cleaned_text_media):
        media_id = cleaned_text_media
        clean = client.get(f"/api/medias/{media_id}/text").get_json()
        assert clean["content"] == "HELLO WORLD"
        assert clean["character_count"] == 11

        original = client.get(f"/api/medias/{media_id}/text?variant=original").get_json()
        assert original["content"] == "hello  world"
        # Counts describe what was served, not the canonical payload.
        assert original["character_count"] == 12
        assert original["word_count"] == 2

    def test_variant_original_falls_back_when_there_is_no_snapshot(self, client, cleaned_image_media):
        """A stale link should show the item, not 404."""
        media_id, canonical, _original = cleaned_image_media
        medias[media_id].pop("original_media_bytes")
        assert client.get(f"/api/medias/{media_id}/image?variant=original").data == canonical

    def test_unknown_variant_is_rejected(self, client, cleaned_image_media):
        media_id, _canonical, _original = cleaned_image_media
        assert client.get(f"/api/medias/{media_id}/image?variant=bogus").status_code == 422

    def test_context_query_params_still_ride_along(self, client, cleaned_image_media):
        """Browser-native ``src`` requests smuggle ``dataset_id`` in as a query
        arg; declaring ``variant`` must not start rejecting them."""
        media_id, canonical, original = cleaned_image_media
        resp = client.get(f"/api/medias/{media_id}/image?dataset_id=_test_default")
        assert resp.status_code == 200
        assert resp.data == canonical
        resp = client.get(f"/api/medias/{media_id}/image?dataset_id=_test_default&variant=original")
        assert resp.status_code == 200
        assert resp.data == original

    def test_variant_response_is_not_etag_confusable_with_canonical(self, client, cleaned_text_media):
        """The variant view drops ``md5`` so an ETag-emitting route hashes the
        bytes it actually serves instead of labelling the original with the
        cleaned item's hash."""
        from vtsearch.routes.media.list import _variant_media

        media = medias[cleaned_text_media]
        with client.application.test_request_context(f"/api/medias/{media['id']}/text?variant=original"):
            view = _variant_media(media)
        assert view is not media
        assert view["media_string"] == "hello  world"
        assert view.get("md5") is None
        assert view.get("media_path") is None
        # The live media is untouched.
        assert media["media_string"] == "HELLO WORLD"
        assert media["md5"]

    def test_canonical_request_returns_the_live_media_dict(self, client, cleaned_image_media):
        """No copy on the default path, so request-time memoisation (thumbnail
        bytes, transcoded MP4) still lands on the loaded media."""
        from vtsearch.routes.media.list import _variant_media

        media = medias[cleaned_image_media[0]]
        with client.application.test_request_context(f"/api/medias/{media['id']}/image"):
            assert _variant_media(media) is media
