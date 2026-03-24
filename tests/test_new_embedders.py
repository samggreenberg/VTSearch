"""Tests for new media embedders: ImageSiglipEmbedder, AudioClapMusicEmbedder, TextBGEEmbedder.

These tests verify class structure, registration, and property correctness
without downloading model weights.
"""

from unittest.mock import MagicMock, patch

import numpy as np


class TestImageSiglipEmbedderProperties:
    """Verify ImageSiglipEmbedder class properties and registration."""

    def test_name(self):
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        assert emb.name == "siglip"

    def test_media_type_id(self):
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        assert emb.media_type_id == "image"

    def test_description_wrappers_non_empty(self):
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_registered_in_registry(self):
        from vtsearch.media import get_embedder

        emb = get_embedder("siglip")
        assert emb.name == "siglip"
        assert emb.media_type_id == "image"

    def test_listed_in_embedders_for_type(self):
        from vtsearch.media import embedders_for_type

        embedders = embedders_for_type("image")
        names = [e.name for e in embedders]
        assert "siglip" in names

    def test_to_dict(self):
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        d = emb.to_dict()
        assert d == {"name": "siglip", "media_type_id": "image"}

    def test_load_models_idempotent(self):
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        # Simulate already-loaded model
        emb._model = MagicMock()
        emb._processor = MagicMock()
        # Calling load_models again should be a no-op (not re-download)
        emb.load_models()
        # Model should be the same mock
        assert isinstance(emb._model, MagicMock)

    def test_embed_media_returns_none_when_not_loaded(self):
        from pathlib import Path

        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        # Patch load_models to not actually load anything
        with patch.object(emb, "load_models"):
            result = emb.embed_media(Path("/nonexistent.jpg"))
        assert result is None

    def test_embed_text_returns_none_when_not_loaded(self):
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text("a cat")
        assert result is None

    def test_embed_pil_image_returns_none_when_not_loaded(self):
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        with patch.object(emb, "load_models"):
            mock_image = MagicMock()
            result = emb.embed_pil_image(mock_image)
        assert result is None

    def test_uses_correct_model_id(self):
        from vtsearch.config import SIGLIP_MODEL_ID

        assert SIGLIP_MODEL_ID == "google/siglip-base-patch16-224"


class TestAudioClapMusicEmbedderProperties:
    """Verify AudioClapMusicEmbedder class properties and registration."""

    def test_name(self):
        from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        assert emb.name == "clap_music"

    def test_media_type_id(self):
        from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        assert emb.media_type_id == "audio"

    def test_description_wrappers_non_empty(self):
        from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_registered_in_registry(self):
        from vtsearch.media import get_embedder

        emb = get_embedder("clap_music")
        assert emb.name == "clap_music"
        assert emb.media_type_id == "audio"

    def test_listed_in_embedders_for_type(self):
        from vtsearch.media import embedders_for_type

        embedders = embedders_for_type("audio")
        names = [e.name for e in embedders]
        assert "clap_music" in names

    def test_to_dict(self):
        from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        d = emb.to_dict()
        assert d == {"name": "clap_music", "media_type_id": "audio"}

    def test_load_models_idempotent(self):
        from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        emb._model = MagicMock()
        emb._processor = MagicMock()
        emb.load_models()
        assert isinstance(emb._model, MagicMock)

    def test_embed_media_returns_none_when_not_loaded(self):
        from pathlib import Path

        from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media(Path("/nonexistent.wav"))
        assert result is None

    def test_embed_text_returns_none_when_not_loaded(self):
        from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text("a dog barking")
        assert result is None

    def test_uses_correct_model_id(self):
        from vtsearch.config import CLAP_MUSIC_MODEL_ID

        assert CLAP_MUSIC_MODEL_ID == "laion/larger_clap_music_and_speech"


class TestTextBGEEmbedderProperties:
    """Verify TextBGEEmbedder class properties and registration."""

    def test_name(self):
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        assert emb.name == "bge"

    def test_media_type_id(self):
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        assert emb.media_type_id == "text"

    def test_description_wrappers_non_empty(self):
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_registered_in_registry(self):
        from vtsearch.media import get_embedder

        emb = get_embedder("bge")
        assert emb.name == "bge"
        assert emb.media_type_id == "text"

    def test_listed_in_embedders_for_type(self):
        from vtsearch.media import embedders_for_type

        embedders = embedders_for_type("text")
        names = [e.name for e in embedders]
        assert "bge" in names

    def test_to_dict(self):
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        d = emb.to_dict()
        assert d == {"name": "bge", "media_type_id": "text"}

    def test_load_models_idempotent(self):
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        emb._model = MagicMock()
        emb.load_models()
        assert isinstance(emb._model, MagicMock)

    def test_embed_media_returns_none_when_not_loaded(self):
        from pathlib import Path

        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media(Path("/nonexistent.txt"))
        assert result is None

    def test_embed_text_returns_none_when_not_loaded(self):
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text("hello world")
        assert result is None

    def test_embed_text_passage_returns_none_when_not_loaded(self):
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text_passage("hello world")
        assert result is None

    def test_uses_correct_model_id(self):
        from vtsearch.config import BGE_MODEL_ID

        assert BGE_MODEL_ID == "BAAI/bge-base-en-v1.5"

    def test_embed_text_uses_query_prefix(self):
        """BGE uses 'Represent this sentence: ' prefix for query embedding."""
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(768)
        emb._model = mock_model

        emb.embed_text("test query")
        mock_model.encode.assert_called_once_with("Represent this sentence: test query", normalize_embeddings=True)

    def test_embed_media_reads_file(self, tmp_path):
        """embed_media should read the text file and encode it."""
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(768)
        emb._model = mock_model

        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello world", encoding="utf-8")

        result = emb.embed_media(text_file)
        assert result is not None
        mock_model.encode.assert_called_once_with("Hello world", normalize_embeddings=True)

    def test_embed_text_passage_no_prefix(self):
        """BGE passage embedding should not add a query prefix."""
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(768)
        emb._model = mock_model

        emb.embed_text_passage("some passage text")
        mock_model.encode.assert_called_once_with("some passage text", normalize_embeddings=True)


class TestAllEmbeddersRegistration:
    """Verify all expected embedders are registered."""

    def test_total_embedder_count(self):
        from vtsearch.media import all_embedders

        embedders = all_embedders()
        assert len(embedders) == 7

    def test_all_expected_names_present(self):
        from vtsearch.media import all_embedders

        names = {e.name for e in all_embedders()}
        expected = {"clap", "clap_music", "clip", "siglip", "e5", "bge", "xclip"}
        assert names == expected

    def test_embedders_for_audio(self):
        from vtsearch.media import embedders_for_type

        names = {e.name for e in embedders_for_type("audio")}
        assert names == {"clap", "clap_music"}

    def test_embedders_for_image(self):
        from vtsearch.media import embedders_for_type

        names = {e.name for e in embedders_for_type("image")}
        assert names == {"clip", "siglip"}

    def test_embedders_for_text(self):
        from vtsearch.media import embedders_for_type

        names = {e.name for e in embedders_for_type("text")}
        assert names == {"e5", "bge"}

    def test_embedders_for_video(self):
        from vtsearch.media import embedders_for_type

        names = {e.name for e in embedders_for_type("video")}
        assert names == {"xclip"}

    def test_all_embedders_dict(self):
        from vtsearch.media import all_embedders_dict

        dicts = all_embedders_dict()
        assert len(dicts) == 7
        for d in dicts:
            assert "name" in d
            assert "media_type_id" in d


class TestNewEmbeddersInheritance:
    """Verify new embedders correctly extend MediaEmbedder."""

    def test_siglip_is_media_embedder(self):
        from vtsearch.media.base import MediaEmbedder
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        assert issubclass(ImageSiglipEmbedder, MediaEmbedder)

    def test_clap_music_is_media_embedder(self):
        from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder
        from vtsearch.media.base import MediaEmbedder

        assert issubclass(AudioClapMusicEmbedder, MediaEmbedder)

    def test_bge_is_media_embedder(self):
        from vtsearch.media.base import MediaEmbedder
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        assert issubclass(TextBGEEmbedder, MediaEmbedder)

    def test_embed_text_enriched_works(self):
        """embed_text_enriched (inherited from base) should work with mocked embed_text."""
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        fake_vec = np.random.RandomState(42).randn(768).astype(np.float32)
        with patch.object(emb, "embed_text", return_value=fake_vec):
            result = emb.embed_text_enriched("a cat")
        assert result is not None
        assert result.shape == (768,)
