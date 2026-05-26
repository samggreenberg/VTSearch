"""Tests for the new image embedders: CLIP, SigLIP 2, DINOv3, Perception Encoder.

Mirrors the patterns in ``test_new_embedders.py`` - verifies class properties,
registration, ``to_dict`` shape, and ``supports_text`` reporting without
downloading any model weights. The Dockerfile (``docker/Dockerfile.image-embedders``)
exercises the real weight downloads.
"""

from unittest.mock import MagicMock, patch


class TestImageClipEmbedder:
    def test_name(self):
        from vtscore.media.image.embedder_clip import ImageClipEmbedder

        assert ImageClipEmbedder().name == "clip"

    def test_media_type_id(self):
        from vtscore.media.image.embedder_clip import ImageClipEmbedder

        assert ImageClipEmbedder().media_type_id == "image"

    def test_is_not_default(self):
        """SigLIP remains the default; CLIP must not steal that role."""
        from vtscore.media.image.embedder_clip import ImageClipEmbedder

        assert ImageClipEmbedder().is_default is False

    def test_supports_text(self):
        from vtscore.media.image.embedder_clip import ImageClipEmbedder

        assert ImageClipEmbedder().supports_text is True

    def test_to_dict(self):
        from vtscore.media.image.embedder_clip import ImageClipEmbedder

        assert ImageClipEmbedder().to_dict() == {
            "name": "clip",
            "display_name": "CLIP (general images)",
            "media_type_id": "image",
            "is_default": False,
            "supports_text": True,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("clip")
        assert emb.name == "clip"
        assert emb.media_type_id == "image"

    def test_uses_correct_model_id(self):
        from vtscore.config import CLIP_MODEL_ID

        assert CLIP_MODEL_ID == "openai/clip-vit-base-patch32"

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.image.embedder_clip import ImageClipEmbedder

        emb = ImageClipEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None

    def test_embed_text_returns_none_when_not_loaded(self):
        from vtscore.media.image.embedder_clip import ImageClipEmbedder

        emb = ImageClipEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text("a cat")
        assert result is None

    def test_load_models_idempotent(self):
        from vtscore.media.image.embedder_clip import ImageClipEmbedder

        emb = ImageClipEmbedder()
        emb._model = MagicMock()
        emb._processor = MagicMock()
        emb.load_models()
        assert isinstance(emb._model, MagicMock)


class TestImageSiglip2Embedder:
    def test_name(self):
        from vtscore.media.image.embedder_siglip2 import ImageSiglip2Embedder

        assert ImageSiglip2Embedder().name == "siglip2"

    def test_media_type_id(self):
        from vtscore.media.image.embedder_siglip2 import ImageSiglip2Embedder

        assert ImageSiglip2Embedder().media_type_id == "image"

    def test_is_not_default(self):
        from vtscore.media.image.embedder_siglip2 import ImageSiglip2Embedder

        assert ImageSiglip2Embedder().is_default is False

    def test_supports_text(self):
        from vtscore.media.image.embedder_siglip2 import ImageSiglip2Embedder

        assert ImageSiglip2Embedder().supports_text is True

    def test_to_dict(self):
        from vtscore.media.image.embedder_siglip2 import ImageSiglip2Embedder

        assert ImageSiglip2Embedder().to_dict() == {
            "name": "siglip2",
            "display_name": "SigLIP 2 (general images)",
            "media_type_id": "image",
            "is_default": False,
            "supports_text": True,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("siglip2")
        assert emb.name == "siglip2"

    def test_uses_correct_model_id(self):
        from vtscore.config import SIGLIP2_MODEL_ID

        assert SIGLIP2_MODEL_ID == "google/siglip2-base-patch16-224"

    def test_description_wrappers_non_empty(self):
        from vtscore.media.image.embedder_siglip2 import ImageSiglip2Embedder

        wrappers = ImageSiglip2Embedder().description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.image.embedder_siglip2 import ImageSiglip2Embedder

        emb = ImageSiglip2Embedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None


class TestImageDinov2SingleEmbedder:
    def test_name(self):
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        assert ImageDinov2SingleEmbedder().name == "dinov2_single"

    def test_media_type_id(self):
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        assert ImageDinov2SingleEmbedder().media_type_id == "image"

    def test_is_not_default(self):
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        assert ImageDinov2SingleEmbedder().is_default is False

    def test_supports_text_false(self):
        """DINOv2 has no text encoder."""
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        assert ImageDinov2SingleEmbedder().supports_text is False

    def test_supports_patch_regions_false(self):
        """Single-vector variant: region pipeline disabled."""
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        assert ImageDinov2SingleEmbedder().supports_patch_regions is False

    def test_to_dict(self):
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        assert ImageDinov2SingleEmbedder().to_dict() == {
            "name": "dinov2_single",
            "display_name": "DINOv2 single (image vector)",
            "media_type_id": "image",
            "is_default": False,
            "supports_text": False,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_embed_text_returns_none(self):
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        emb = ImageDinov2SingleEmbedder()
        assert emb.embed_text("anything") is None

    def test_no_description_wrappers(self):
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        assert ImageDinov2SingleEmbedder().description_wrappers == []

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("dinov2_single")
        assert emb.name == "dinov2_single"

    def test_uses_correct_model_id(self):
        """DINOv2's HF repo is ungated - any builder can download it without
        an account, which is the whole reason it's bundled in the Docker
        image alongside the gated DINOv3."""
        from vtscore.config import DINOV2_MODEL_ID

        assert DINOV2_MODEL_ID == "facebook/dinov2-base"

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        emb = ImageDinov2SingleEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None

    def test_patch_forward_returns_none(self):
        """Single-vector variant inherits the ABC default and returns None."""
        from vtscore.media.image.embedder_dinov2_single import ImageDinov2SingleEmbedder

        emb = ImageDinov2SingleEmbedder()
        assert emb.patch_forward({"media_path": "/nonexistent.jpg"}) is None


class TestImageDinov2PatchEmbedder:
    def test_name(self):
        from vtscore.media.image.embedder_dinov2_patch import ImageDinov2PatchEmbedder

        assert ImageDinov2PatchEmbedder().name == "dinov2_patch"

    def test_supports_patch_regions_true(self):
        from vtscore.media.image.embedder_dinov2_patch import ImageDinov2PatchEmbedder

        assert ImageDinov2PatchEmbedder().supports_patch_regions is True

    def test_to_dict(self):
        from vtscore.media.image.embedder_dinov2_patch import ImageDinov2PatchEmbedder

        assert ImageDinov2PatchEmbedder().to_dict() == {
            "name": "dinov2_patch",
            "display_name": "DINOv2 patch (region-aware images)",
            "media_type_id": "image",
            "is_default": False,
            "supports_text": False,
            "supports_patch_regions": True,
            "license_notice": None,
        }

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("dinov2_patch")
        assert emb.name == "dinov2_patch"


class TestImageDinov3SingleEmbedder:
    def test_name(self):
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        assert ImageDinov3SingleEmbedder().name == "dinov3_single"

    def test_media_type_id(self):
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        assert ImageDinov3SingleEmbedder().media_type_id == "image"

    def test_is_not_default(self):
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        assert ImageDinov3SingleEmbedder().is_default is False

    def test_supports_text_false(self):
        """DINOv3 has no text encoder - supports_text must be False."""
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        assert ImageDinov3SingleEmbedder().supports_text is False

    def test_supports_patch_regions_false(self):
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        assert ImageDinov3SingleEmbedder().supports_patch_regions is False

    def test_to_dict(self):
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        assert ImageDinov3SingleEmbedder().to_dict() == {
            "name": "dinov3_single",
            "display_name": "DINOv3 single (image vector)",
            "media_type_id": "image",
            "is_default": False,
            "supports_text": False,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_embed_text_returns_none(self):
        """Base-class ``embed_text`` default kicks in (DINOv3 doesn't override)."""
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        emb = ImageDinov3SingleEmbedder()
        assert emb.embed_text("anything") is None

    def test_no_description_wrappers(self):
        """Without a text encoder, description-enrichment wrappers are meaningless."""
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        assert ImageDinov3SingleEmbedder().description_wrappers == []

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("dinov3_single")
        assert emb.name == "dinov3_single"

    def test_uses_correct_model_id(self):
        from vtscore.config import DINOV3_MODEL_ID

        assert DINOV3_MODEL_ID == "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        emb = ImageDinov3SingleEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None

    def test_patch_forward_returns_none(self):
        from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder

        emb = ImageDinov3SingleEmbedder()
        assert emb.patch_forward({"media_path": "/nonexistent.jpg"}) is None


class TestImageDinov3PatchEmbedder:
    def test_name(self):
        from vtscore.media.image.embedder_dinov3_patch import ImageDinov3PatchEmbedder

        assert ImageDinov3PatchEmbedder().name == "dinov3_patch"

    def test_supports_patch_regions_true(self):
        from vtscore.media.image.embedder_dinov3_patch import ImageDinov3PatchEmbedder

        assert ImageDinov3PatchEmbedder().supports_patch_regions is True

    def test_to_dict(self):
        from vtscore.media.image.embedder_dinov3_patch import ImageDinov3PatchEmbedder

        assert ImageDinov3PatchEmbedder().to_dict() == {
            "name": "dinov3_patch",
            "display_name": "DINOv3 patch (region-aware images)",
            "media_type_id": "image",
            "is_default": False,
            "supports_text": False,
            "supports_patch_regions": True,
            "license_notice": None,
        }

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("dinov3_patch")
        assert emb.name == "dinov3_patch"


class TestImageEupeSingleEmbedder:
    def test_name(self):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        assert ImageEupeSingleEmbedder().name == "eupe_single"

    def test_media_type_id(self):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        assert ImageEupeSingleEmbedder().media_type_id == "image"

    def test_is_not_default(self):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        assert ImageEupeSingleEmbedder().is_default is False

    def test_supports_text_false(self):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        assert ImageEupeSingleEmbedder().supports_text is False

    def test_supports_patch_regions_false(self):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        assert ImageEupeSingleEmbedder().supports_patch_regions is False

    def test_license_notice_set(self):
        """EUPE outputs are bound by FAIR Noncommercial - surface that on
        both variants."""
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        notice = ImageEupeSingleEmbedder().license_notice
        assert isinstance(notice, str)
        assert "Noncommercial" in notice or "noncommercial" in notice.lower()

    def test_to_dict(self):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        d = ImageEupeSingleEmbedder().to_dict()
        assert d["name"] == "eupe_single"
        assert d["media_type_id"] == "image"
        assert d["supports_text"] is False
        assert d["supports_patch_regions"] is False
        assert isinstance(d["license_notice"], str)
        assert "noncommercial" in d["license_notice"].lower()

    def test_embed_text_returns_none(self):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        emb = ImageEupeSingleEmbedder()
        assert emb.embed_text("anything") is None

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("eupe_single")
        assert emb.name == "eupe_single"

    def test_uses_correct_model_id(self):
        """EUPE_MODEL_ID is the HF weight URL torch.hub fetches.

        The previous "eupe" slug pointed at facebook/PE-Core-B16-224 via a
        broken AutoModel.from_pretrained path; this PR replaces it with the
        real facebookresearch/EUPE model whose ungated weights live at
        the URL below.
        """
        from vtscore.config import EUPE_MODEL_ID

        assert EUPE_MODEL_ID == "https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt"

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        emb = ImageEupeSingleEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None

    def test_patch_forward_returns_none(self):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        emb = ImageEupeSingleEmbedder()
        assert emb.patch_forward({"media_path": "/nonexistent.jpg"}) is None


class TestImageEupePatchEmbedder:
    def test_name(self):
        from vtscore.media.image.embedder_eupe_patch import ImageEupePatchEmbedder

        assert ImageEupePatchEmbedder().name == "eupe_patch"

    def test_supports_patch_regions_true(self):
        from vtscore.media.image.embedder_eupe_patch import ImageEupePatchEmbedder

        assert ImageEupePatchEmbedder().supports_patch_regions is True

    def test_license_notice_set(self):
        from vtscore.media.image.embedder_eupe_patch import ImageEupePatchEmbedder

        notice = ImageEupePatchEmbedder().license_notice
        assert isinstance(notice, str)
        assert "noncommercial" in notice.lower()

    def test_to_dict(self):
        from vtscore.media.image.embedder_eupe_patch import ImageEupePatchEmbedder

        d = ImageEupePatchEmbedder().to_dict()
        assert d["name"] == "eupe_patch"
        assert d["supports_text"] is False
        assert d["supports_patch_regions"] is True
        assert isinstance(d["license_notice"], str)

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("eupe_patch")
        assert emb.name == "eupe_patch"


class TestApiEmbeddersResponseShape:
    """``GET /api/embedders`` must include ``supports_text`` on every entry
    so the frontend can decide whether to hide text-search UI per dataset."""

    def test_image_embedders_endpoint_includes_supports_text(self, client):
        resp = client.get("/api/embedders?media_type=image")
        assert resp.status_code == 200
        body = resp.get_json()
        entries = {e["name"]: e for e in body["embedders"]}
        # All ten image embedders must be present - three CLIP-family
        # bimodal models, single/patch pairs for DINOv2/DINOv3/EUPE, and
        # the FaceNet face-identity embedder.
        assert set(entries) == {
            "siglip",
            "siglip2",
            "clip",
            "dinov2_single",
            "dinov2_patch",
            "dinov3_single",
            "dinov3_patch",
            "eupe_single",
            "eupe_patch",
            "face",
        }
        # Shape: every entry has the three capability fields, with bool /
        # Optional[str] types.
        for entry in entries.values():
            assert isinstance(entry["supports_text"], bool)
            assert isinstance(entry["supports_patch_regions"], bool)
            assert entry["license_notice"] is None or isinstance(entry["license_notice"], str)
        # Specific expectations.
        assert entries["siglip"]["supports_text"] is True
        for name in (
            "dinov2_single",
            "dinov2_patch",
            "dinov3_single",
            "dinov3_patch",
            "eupe_single",
            "eupe_patch",
        ):
            assert entries[name]["supports_text"] is False
        # Patch variants produce patch regions; single variants and the
        # bimodal CLIP-style embedders don't.
        for name in ("dinov2_patch", "dinov3_patch", "eupe_patch"):
            assert entries[name]["supports_patch_regions"] is True
        for name in (
            "siglip",
            "siglip2",
            "clip",
            "dinov2_single",
            "dinov3_single",
            "eupe_single",
        ):
            assert entries[name]["supports_patch_regions"] is False
        # Only EUPE carries a licence warning today (FAIR Noncommercial) -
        # on both variants.
        for name in (
            "siglip",
            "siglip2",
            "clip",
            "dinov2_single",
            "dinov2_patch",
            "dinov3_single",
            "dinov3_patch",
        ):
            assert entries[name]["license_notice"] is None
        for name in ("eupe_single", "eupe_patch"):
            notice = entries[name]["license_notice"]
            assert isinstance(notice, str)
            assert "noncommercial" in notice.lower()


class TestSortRouteRejectsTextWhenUnsupported:
    """``POST /api/sort`` must return a structured error when the active
    media's embedder is vision-only (``supports_text=False``), instead of
    surfacing a generic 500."""

    def test_sort_rejects_text_for_dinov3(self, client):
        from vtsearch.state import medias

        # Seed a fake DINOv3-embedded media so the sort route picks up the
        # active embedder name. We don't need real vectors - the route
        # exits at the embed_text_query None-check before touching them.
        saved = dict(medias)
        medias.clear()
        try:
            medias[1] = {
                "id": 1,
                "media_type": "image",
                "filename": "x.jpg",
                "md5": "deadbeef",
                "embedder": "dinov3_single",
            }
            resp = client.post("/api/sort", json={"text": "a cat"})
            assert resp.status_code == 400
            body = resp.get_json()
            # Migrated to flask-smorest: handler-level rejects surface
            # under ``message``. The frontend already reads the
            # ``supports_text`` flag from each embedder's
            # ``EmbedderInfo`` directly, so the legacy body field that
            # used to ride along on this error has been dropped.
            assert "dinov3_single" in body["message"]
        finally:
            medias.clear()
            medias.update(saved)


class TestImageFaceEmbedder:
    def test_name(self):
        from vtscore.media.image.embedder_face import ImageFaceEmbedder

        assert ImageFaceEmbedder().name == "face"

    def test_media_type_id(self):
        from vtscore.media.image.embedder_face import ImageFaceEmbedder

        assert ImageFaceEmbedder().media_type_id == "image"

    def test_is_not_default(self):
        from vtscore.media.image.embedder_face import ImageFaceEmbedder

        assert ImageFaceEmbedder().is_default is False

    def test_does_not_support_text(self):
        """FaceNet has no text branch - face-identity space has no
        analogue of 'a photo of a cat', so text queries must be hidden
        in the UI for face-embedder datasets."""
        from vtscore.media.image.embedder_face import ImageFaceEmbedder

        assert ImageFaceEmbedder().supports_text is False

    def test_does_not_support_patch_regions(self):
        from vtscore.media.image.embedder_face import ImageFaceEmbedder

        assert ImageFaceEmbedder().supports_patch_regions is False

    def test_to_dict(self):
        from vtscore.media.image.embedder_face import ImageFaceEmbedder

        assert ImageFaceEmbedder().to_dict() == {
            "name": "face",
            "display_name": "FaceNet (face identity, 512d)",
            "media_type_id": "image",
            "is_default": False,
            "supports_text": False,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("face")
        assert emb.name == "face"
        assert emb.media_type_id == "image"

    def test_embed_text_returns_none(self):
        """FaceNet's text branch should always yield ``None`` - there is no
        text encoder in face-identity space."""
        from vtscore.media.image.embedder_face import ImageFaceEmbedder

        assert ImageFaceEmbedder().embed_text("a face") is None

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.image.embedder_face import ImageFaceEmbedder

        emb = ImageFaceEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None

    def test_load_models_idempotent(self):
        from vtscore.media.image.embedder_face import ImageFaceEmbedder

        emb = ImageFaceEmbedder()
        emb._model = MagicMock()
        emb.load_models()
        assert isinstance(emb._model, MagicMock)

    def test_l2_normalises_output(self):
        """Vectors out of FaceNet must be unit-norm so cosine ranking works."""
        import numpy as np

        from vtscore.media.image.embedder_face import ImageFaceEmbedder

        emb = ImageFaceEmbedder()
        v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        out = emb._l2_normalise(v)
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6
        # Zero vector survives without NaN.
        z = emb._l2_normalise(np.zeros(3, dtype=np.float32))
        assert not np.isnan(z).any()
        # Bulk: each row independently normalised.
        batch = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
        out_b = emb._l2_normalise(batch)
        assert abs(float(np.linalg.norm(out_b[0])) - 1.0) < 1e-6
        assert abs(float(np.linalg.norm(out_b[1])) - 1.0) < 1e-6
