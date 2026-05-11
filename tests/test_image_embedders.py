"""Tests for the new image embedders: CLIP, SigLIP 2, DINOv3, Perception Encoder.

Mirrors the patterns in ``test_new_embedders.py`` — verifies class properties,
registration, ``to_dict`` shape, and ``supports_text`` reporting without
downloading any model weights. The Dockerfile (``Dockerfile.image-embedders``)
exercises the real weight downloads.
"""

from unittest.mock import MagicMock, patch


class TestImageClipEmbedder:
    def test_name(self):
        from vtsearch.media.image.embedder_clip import ImageClipEmbedder

        assert ImageClipEmbedder().name == "clip"

    def test_media_type_id(self):
        from vtsearch.media.image.embedder_clip import ImageClipEmbedder

        assert ImageClipEmbedder().media_type_id == "image"

    def test_is_not_default(self):
        """SigLIP remains the default; CLIP must not steal that role."""
        from vtsearch.media.image.embedder_clip import ImageClipEmbedder

        assert ImageClipEmbedder().is_default is False

    def test_supports_text(self):
        from vtsearch.media.image.embedder_clip import ImageClipEmbedder

        assert ImageClipEmbedder().supports_text is True

    def test_to_dict(self):
        from vtsearch.media.image.embedder_clip import ImageClipEmbedder

        assert ImageClipEmbedder().to_dict() == {
            "name": "clip",
            "media_type_id": "image",
            "supports_text": True,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_registered_in_registry(self):
        from vtsearch.media import get_embedder

        emb = get_embedder("clip")
        assert emb.name == "clip"
        assert emb.media_type_id == "image"

    def test_uses_correct_model_id(self):
        from vtsearch.config import CLIP_MODEL_ID

        assert CLIP_MODEL_ID == "openai/clip-vit-base-patch32"

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtsearch.media.image.embedder_clip import ImageClipEmbedder

        emb = ImageClipEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None

    def test_embed_text_returns_none_when_not_loaded(self):
        from vtsearch.media.image.embedder_clip import ImageClipEmbedder

        emb = ImageClipEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text("a cat")
        assert result is None

    def test_load_models_idempotent(self):
        from vtsearch.media.image.embedder_clip import ImageClipEmbedder

        emb = ImageClipEmbedder()
        emb._model = MagicMock()
        emb._processor = MagicMock()
        emb.load_models()
        assert isinstance(emb._model, MagicMock)


class TestImageSiglip2Embedder:
    def test_name(self):
        from vtsearch.media.image.embedder_siglip2 import ImageSiglip2Embedder

        assert ImageSiglip2Embedder().name == "siglip2"

    def test_media_type_id(self):
        from vtsearch.media.image.embedder_siglip2 import ImageSiglip2Embedder

        assert ImageSiglip2Embedder().media_type_id == "image"

    def test_is_not_default(self):
        from vtsearch.media.image.embedder_siglip2 import ImageSiglip2Embedder

        assert ImageSiglip2Embedder().is_default is False

    def test_supports_text(self):
        from vtsearch.media.image.embedder_siglip2 import ImageSiglip2Embedder

        assert ImageSiglip2Embedder().supports_text is True

    def test_to_dict(self):
        from vtsearch.media.image.embedder_siglip2 import ImageSiglip2Embedder

        assert ImageSiglip2Embedder().to_dict() == {
            "name": "siglip2",
            "media_type_id": "image",
            "supports_text": True,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_registered_in_registry(self):
        from vtsearch.media import get_embedder

        emb = get_embedder("siglip2")
        assert emb.name == "siglip2"

    def test_uses_correct_model_id(self):
        from vtsearch.config import SIGLIP2_MODEL_ID

        assert SIGLIP2_MODEL_ID == "google/siglip2-base-patch16-224"

    def test_description_wrappers_non_empty(self):
        from vtsearch.media.image.embedder_siglip2 import ImageSiglip2Embedder

        wrappers = ImageSiglip2Embedder().description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtsearch.media.image.embedder_siglip2 import ImageSiglip2Embedder

        emb = ImageSiglip2Embedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None


class TestImageDinov2Embedder:
    def test_name(self):
        from vtsearch.media.image.embedder_dinov2 import ImageDinov2Embedder

        assert ImageDinov2Embedder().name == "dinov2"

    def test_media_type_id(self):
        from vtsearch.media.image.embedder_dinov2 import ImageDinov2Embedder

        assert ImageDinov2Embedder().media_type_id == "image"

    def test_is_not_default(self):
        from vtsearch.media.image.embedder_dinov2 import ImageDinov2Embedder

        assert ImageDinov2Embedder().is_default is False

    def test_supports_text_false(self):
        """DINOv2 has no text encoder."""
        from vtsearch.media.image.embedder_dinov2 import ImageDinov2Embedder

        assert ImageDinov2Embedder().supports_text is False

    def test_to_dict(self):
        from vtsearch.media.image.embedder_dinov2 import ImageDinov2Embedder

        assert ImageDinov2Embedder().to_dict() == {
            "name": "dinov2",
            "media_type_id": "image",
            "supports_text": False,
            "supports_patch_regions": True,
            "license_notice": None,
        }

    def test_embed_text_returns_none(self):
        from vtsearch.media.image.embedder_dinov2 import ImageDinov2Embedder

        emb = ImageDinov2Embedder()
        assert emb.embed_text("anything") is None

    def test_no_description_wrappers(self):
        from vtsearch.media.image.embedder_dinov2 import ImageDinov2Embedder

        assert ImageDinov2Embedder().description_wrappers == []

    def test_registered_in_registry(self):
        from vtsearch.media import get_embedder

        emb = get_embedder("dinov2")
        assert emb.name == "dinov2"

    def test_uses_correct_model_id(self):
        """DINOv2's HF repo is ungated — any builder can download it without
        an account, which is the whole reason it's bundled in the Docker
        image alongside the gated DINOv3."""
        from vtsearch.config import DINOV2_MODEL_ID

        assert DINOV2_MODEL_ID == "facebook/dinov2-base"

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtsearch.media.image.embedder_dinov2 import ImageDinov2Embedder

        emb = ImageDinov2Embedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None


class TestImageDinov3Embedder:
    def test_name(self):
        from vtsearch.media.image.embedder_dinov3 import ImageDinov3Embedder

        assert ImageDinov3Embedder().name == "dinov3"

    def test_media_type_id(self):
        from vtsearch.media.image.embedder_dinov3 import ImageDinov3Embedder

        assert ImageDinov3Embedder().media_type_id == "image"

    def test_is_not_default(self):
        from vtsearch.media.image.embedder_dinov3 import ImageDinov3Embedder

        assert ImageDinov3Embedder().is_default is False

    def test_supports_text_false(self):
        """DINOv3 has no text encoder — supports_text must be False."""
        from vtsearch.media.image.embedder_dinov3 import ImageDinov3Embedder

        assert ImageDinov3Embedder().supports_text is False

    def test_to_dict(self):
        from vtsearch.media.image.embedder_dinov3 import ImageDinov3Embedder

        assert ImageDinov3Embedder().to_dict() == {
            "name": "dinov3",
            "media_type_id": "image",
            "supports_text": False,
            "supports_patch_regions": True,
            "license_notice": None,
        }

    def test_embed_text_returns_none(self):
        """Base-class ``embed_text`` default kicks in (DINOv3 doesn't override)."""
        from vtsearch.media.image.embedder_dinov3 import ImageDinov3Embedder

        emb = ImageDinov3Embedder()
        assert emb.embed_text("anything") is None

    def test_no_description_wrappers(self):
        """Without a text encoder, description-enrichment wrappers are meaningless."""
        from vtsearch.media.image.embedder_dinov3 import ImageDinov3Embedder

        assert ImageDinov3Embedder().description_wrappers == []

    def test_registered_in_registry(self):
        from vtsearch.media import get_embedder

        emb = get_embedder("dinov3")
        assert emb.name == "dinov3"

    def test_uses_correct_model_id(self):
        from vtsearch.config import DINOV3_MODEL_ID

        assert DINOV3_MODEL_ID == "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtsearch.media.image.embedder_dinov3 import ImageDinov3Embedder

        emb = ImageDinov3Embedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None


class TestImageEupeEmbedder:
    def test_name(self):
        from vtsearch.media.image.embedder_eupe import ImageEupeEmbedder

        assert ImageEupeEmbedder().name == "eupe"

    def test_media_type_id(self):
        from vtsearch.media.image.embedder_eupe import ImageEupeEmbedder

        assert ImageEupeEmbedder().media_type_id == "image"

    def test_is_not_default(self):
        from vtsearch.media.image.embedder_eupe import ImageEupeEmbedder

        assert ImageEupeEmbedder().is_default is False

    def test_supports_text_false(self):
        from vtsearch.media.image.embedder_eupe import ImageEupeEmbedder

        assert ImageEupeEmbedder().supports_text is False

    def test_to_dict(self):
        from vtsearch.media.image.embedder_eupe import ImageEupeEmbedder

        assert ImageEupeEmbedder().to_dict() == {
            "name": "eupe",
            "media_type_id": "image",
            "supports_text": False,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_embed_text_returns_none(self):
        from vtsearch.media.image.embedder_eupe import ImageEupeEmbedder

        emb = ImageEupeEmbedder()
        assert emb.embed_text("anything") is None

    def test_registered_in_registry(self):
        from vtsearch.media import get_embedder

        emb = get_embedder("eupe")
        assert emb.name == "eupe"

    def test_uses_correct_model_id(self):
        from vtsearch.config import EUPE_MODEL_ID

        assert EUPE_MODEL_ID == "facebook/PE-Core-B16-224"

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtsearch.media.image.embedder_eupe import ImageEupeEmbedder

        emb = ImageEupeEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None


class TestApiEmbeddersResponseShape:
    """``GET /api/embedders`` must include ``supports_text`` on every entry
    so the frontend can decide whether to hide text-search UI per dataset."""

    def test_image_embedders_endpoint_includes_supports_text(self, client):
        resp = client.get("/api/embedders?media_type=image")
        assert resp.status_code == 200
        body = resp.get_json()
        entries = {e["name"]: e for e in body["embedders"]}
        # All six image embedders must be present.
        assert set(entries) == {"siglip", "siglip2", "clip", "dinov2", "dinov3", "eupe"}
        # Shape: every entry has the three capability fields, with bool /
        # Optional[str] types.
        for entry in entries.values():
            assert isinstance(entry["supports_text"], bool)
            assert isinstance(entry["supports_patch_regions"], bool)
            assert entry["license_notice"] is None or isinstance(entry["license_notice"], str)
        # Specific expectations.
        assert entries["siglip"]["supports_text"] is True
        assert entries["dinov2"]["supports_text"] is False
        assert entries["dinov3"]["supports_text"] is False
        assert entries["eupe"]["supports_text"] is False
        # DINOv2 and DINOv3 produce patch regions; the others don't.
        # (EUPE will flip True once its loader is reworked off the broken
        # AutoModel path onto real facebookresearch/EUPE.)
        assert entries["dinov2"]["supports_patch_regions"] is True
        assert entries["dinov3"]["supports_patch_regions"] is True
        for name in ("siglip", "siglip2", "clip", "eupe"):
            assert entries[name]["supports_patch_regions"] is False
        # No embedder currently carries a license notice (real-EUPE will set
        # one once its loader is reworked off the broken AutoModel path).
        for entry in entries.values():
            assert entry["license_notice"] is None


class TestSortRouteRejectsTextWhenUnsupported:
    """``POST /api/sort`` must return a structured error when the active
    media's embedder is vision-only (``supports_text=False``), instead of
    surfacing a generic 500."""

    def test_sort_rejects_text_for_dinov3(self, client):
        from vtsearch.utils import medias

        # Seed a fake DINOv3-embedded media so the sort route picks up the
        # active embedder name. We don't need real vectors — the route
        # exits at the embed_text_query None-check before touching them.
        saved = dict(medias)
        medias.clear()
        try:
            medias[1] = {
                "id": 1,
                "type": "image",
                "filename": "x.jpg",
                "md5": "deadbeef",
                "embedder": "dinov3",
            }
            resp = client.post("/api/sort", json={"text": "a cat"})
            assert resp.status_code == 400
            body = resp.get_json()
            assert body.get("supports_text") is False
            assert "dinov3" in body["error"]
        finally:
            medias.clear()
            medias.update(saved)
