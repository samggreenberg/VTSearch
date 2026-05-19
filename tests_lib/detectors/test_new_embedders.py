"""Tests for new media embedders: ImageSiglipEmbedder, AudioClapMusicEmbedder, TextBGEEmbedder.

These tests verify class structure, registration, and property correctness
without downloading model weights.
"""

import threading
from unittest.mock import MagicMock, patch

import numpy as np


class TestImageSiglipEmbedderProperties:
    """Verify ImageSiglipEmbedder class properties and registration."""

    def test_name(self):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        assert emb.name == "siglip"

    def test_media_type_id(self):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        assert emb.media_type_id == "image"

    def test_description_wrappers_non_empty(self):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("siglip")
        assert emb.name == "siglip"
        assert emb.media_type_id == "image"

    def test_listed_in_embedders_for_type(self):
        from vtscore.media import embedders_for_type

        embedders = embedders_for_type("image")
        names = [e.name for e in embedders]
        assert "siglip" in names

    def test_to_dict(self):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        d = emb.to_dict()
        assert d == {
            "name": "siglip",
            "display_name": "SigLIP (general images)",
            "media_type_id": "image",
            "is_default": True,
            "supports_text": True,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_load_models_idempotent(self):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        # Simulate already-loaded model
        emb._model = MagicMock()
        emb._processor = MagicMock()
        # Calling load_models again should be a no-op (not re-download)
        emb.load_models()
        # Model should be the same mock
        assert isinstance(emb._model, MagicMock)

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        # Patch load_models to not actually load anything
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
        assert result is None

    def test_embed_text_returns_none_when_not_loaded(self):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text("a cat")
        assert result is None

    def test_embed_pil_image_returns_none_when_not_loaded(self):
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        with patch.object(emb, "load_models"):
            mock_image = MagicMock()
            result = emb.embed_pil_image(mock_image)
        assert result is None

    def test_uses_correct_model_id(self):
        from vtscore.config import SIGLIP_MODEL_ID

        assert SIGLIP_MODEL_ID == "google/siglip-base-patch16-224"


class TestAudioClapMusicEmbedderProperties:
    """Verify AudioClapMusicEmbedder class properties and registration."""

    def test_name(self):
        from vtscore.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        assert emb.name == "clap_music"

    def test_media_type_id(self):
        from vtscore.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        assert emb.media_type_id == "audio"

    def test_description_wrappers_non_empty(self):
        from vtscore.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("clap_music")
        assert emb.name == "clap_music"
        assert emb.media_type_id == "audio"

    def test_listed_in_embedders_for_type(self):
        from vtscore.media import embedders_for_type

        embedders = embedders_for_type("audio")
        names = [e.name for e in embedders]
        assert "clap_music" in names

    def test_to_dict(self):
        from vtscore.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        d = emb.to_dict()
        assert d == {
            "name": "clap_music",
            "display_name": "CLAP (music)",
            "media_type_id": "audio",
            "is_default": False,
            "supports_text": True,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_load_models_idempotent(self):
        from vtscore.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        emb._model = MagicMock()
        emb._processor = MagicMock()
        emb.load_models()
        assert isinstance(emb._model, MagicMock)

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.wav"})
        assert result is None

    def test_embed_text_returns_none_when_not_loaded(self):
        from vtscore.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text("a dog barking")
        assert result is None

    def test_uses_correct_model_id(self):
        from vtscore.config import CLAP_MUSIC_MODEL_ID

        assert CLAP_MUSIC_MODEL_ID == "laion/larger_clap_music_and_speech"


class TestAudioClapGeneralEmbedderProperties:
    """Verify AudioClapGeneralEmbedder class properties and registration."""

    def test_name(self):
        from vtscore.media.audio.embedder_clap_general import AudioClapGeneralEmbedder

        emb = AudioClapGeneralEmbedder()
        assert emb.name == "clap_general"

    def test_media_type_id(self):
        from vtscore.media.audio.embedder_clap_general import AudioClapGeneralEmbedder

        emb = AudioClapGeneralEmbedder()
        assert emb.media_type_id == "audio"

    def test_description_wrappers_non_empty(self):
        from vtscore.media.audio.embedder_clap_general import AudioClapGeneralEmbedder

        emb = AudioClapGeneralEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("clap_general")
        assert emb.name == "clap_general"
        assert emb.media_type_id == "audio"

    def test_to_dict(self):
        from vtscore.media.audio.embedder_clap_general import AudioClapGeneralEmbedder

        emb = AudioClapGeneralEmbedder()
        d = emb.to_dict()
        assert d == {
            "name": "clap_general",
            "display_name": "CLAP (general 2024)",
            "media_type_id": "audio",
            "is_default": False,
            "supports_text": True,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_uses_correct_model_id(self):
        from vtscore.config import CLAP_GENERAL_MODEL_ID

        assert CLAP_GENERAL_MODEL_ID == "laion/larger_clap_general"


class TestAudioASTEmbedderProperties:
    """Verify AudioASTEmbedder class properties and registration."""

    def test_name(self):
        from vtscore.media.audio.embedder_ast import AudioASTEmbedder

        emb = AudioASTEmbedder()
        assert emb.name == "ast"

    def test_media_type_id(self):
        from vtscore.media.audio.embedder_ast import AudioASTEmbedder

        emb = AudioASTEmbedder()
        assert emb.media_type_id == "audio"

    def test_supports_text_is_false(self):
        """AST has no text encoder."""
        from vtscore.media.audio.embedder_ast import AudioASTEmbedder

        emb = AudioASTEmbedder()
        assert emb.supports_text is False

    def test_inherits_default_embed_text(self):
        """Without a custom override, ``embed_text`` returns ``None``."""
        from vtscore.media.audio.embedder_ast import AudioASTEmbedder

        emb = AudioASTEmbedder()
        assert emb.embed_text("any query") is None

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("ast")
        assert emb.name == "ast"
        assert emb.media_type_id == "audio"

    def test_to_dict(self):
        from vtscore.media.audio.embedder_ast import AudioASTEmbedder

        emb = AudioASTEmbedder()
        d = emb.to_dict()
        assert d == {
            "name": "ast",
            "display_name": "AST (audio spectrogram)",
            "media_type_id": "audio",
            "is_default": False,
            "supports_text": False,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_uses_correct_model_id(self):
        from vtscore.config import AST_MODEL_ID, AST_SAMPLE_RATE

        assert AST_MODEL_ID == "MIT/ast-finetuned-audioset-10-10-0.4593"
        assert AST_SAMPLE_RATE == 16000


class TestAudioWhisperEncoderEmbedderProperties:
    """Verify AudioWhisperEncoderEmbedder class properties and registration."""

    def test_name(self):
        from vtscore.media.audio.embedder_whisper import AudioWhisperEncoderEmbedder

        emb = AudioWhisperEncoderEmbedder()
        assert emb.name == "whisper_encoder"

    def test_media_type_id(self):
        from vtscore.media.audio.embedder_whisper import AudioWhisperEncoderEmbedder

        emb = AudioWhisperEncoderEmbedder()
        assert emb.media_type_id == "audio"

    def test_supports_text_is_false(self):
        """Whisper's decoder is text-OUT, not a text encoder; no shared space."""
        from vtscore.media.audio.embedder_whisper import AudioWhisperEncoderEmbedder

        emb = AudioWhisperEncoderEmbedder()
        assert emb.supports_text is False

    def test_inherits_default_embed_text(self):
        from vtscore.media.audio.embedder_whisper import AudioWhisperEncoderEmbedder

        emb = AudioWhisperEncoderEmbedder()
        assert emb.embed_text("any query") is None

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("whisper_encoder")
        assert emb.name == "whisper_encoder"
        assert emb.media_type_id == "audio"

    def test_to_dict(self):
        from vtscore.media.audio.embedder_whisper import AudioWhisperEncoderEmbedder

        emb = AudioWhisperEncoderEmbedder()
        d = emb.to_dict()
        assert d == {
            "name": "whisper_encoder",
            "display_name": "Whisper encoder (speech)",
            "media_type_id": "audio",
            "is_default": False,
            "supports_text": False,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_uses_correct_model_id(self):
        from vtscore.config import WHISPER_MODEL_ID, WHISPER_SAMPLE_RATE

        assert WHISPER_MODEL_ID == "openai/whisper-base"
        assert WHISPER_SAMPLE_RATE == 16000


class TestTextBGEEmbedderProperties:
    """Verify TextBGEEmbedder class properties and registration."""

    def test_name(self):
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        assert emb.name == "bge"

    def test_media_type_id(self):
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        assert emb.media_type_id == "text"

    def test_description_wrappers_non_empty(self):
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("bge")
        assert emb.name == "bge"
        assert emb.media_type_id == "text"

    def test_listed_in_embedders_for_type(self):
        from vtscore.media import embedders_for_type

        embedders = embedders_for_type("text")
        names = [e.name for e in embedders]
        assert "bge" in names

    def test_to_dict(self):
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        d = emb.to_dict()
        assert d == {
            "name": "bge",
            "display_name": "BGE (text)",
            "media_type_id": "text",
            "is_default": False,
            "supports_text": True,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_load_models_idempotent(self):
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        emb._model = MagicMock()
        emb.load_models()
        assert isinstance(emb._model, MagicMock)

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.txt"})
        assert result is None

    def test_embed_text_returns_none_when_not_loaded(self):
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text("hello world")
        assert result is None

    def test_embed_text_passage_returns_none_when_not_loaded(self):
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text_passage("hello world")
        assert result is None

    def test_uses_correct_model_id(self):
        from vtscore.config import BGE_MODEL_ID

        assert BGE_MODEL_ID == "BAAI/bge-base-en-v1.5"

    def test_embed_text_uses_query_prefix(self):
        """BGE uses 'Represent this sentence: ' prefix for query embedding."""
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(768)
        emb._model = mock_model

        emb.embed_text("test query")
        mock_model.encode.assert_called_once_with("Represent this sentence: test query", normalize_embeddings=True)

    def test_embed_media_reads_file(self, tmp_path):
        """embed_media should read the text file and encode it."""
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(768)
        emb._model = mock_model

        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello world", encoding="utf-8")

        result = emb.embed_media({"media_path": str(text_file)})
        assert result is not None
        mock_model.encode.assert_called_once_with("Hello world", normalize_embeddings=True)

    def test_embed_text_passage_no_prefix(self):
        """BGE passage embedding should not add a query prefix."""
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(768)
        emb._model = mock_model

        emb.embed_text_passage("some passage text")
        mock_model.encode.assert_called_once_with("some passage text", normalize_embeddings=True)


class TestVideoLanguageBindEmbedderProperties:
    """Verify VideoLanguageBindEmbedder class properties and registration."""

    def test_name(self):
        from vtscore.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        assert emb.name == "languagebind"

    def test_media_type_id(self):
        from vtscore.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        assert emb.media_type_id == "video"

    def test_description_wrappers_non_empty(self):
        from vtscore.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("languagebind")
        assert emb.name == "languagebind"
        assert emb.media_type_id == "video"

    def test_listed_in_embedders_for_type(self):
        from vtscore.media import embedders_for_type

        embedders = embedders_for_type("video")
        names = [e.name for e in embedders]
        assert "languagebind" in names

    def test_to_dict(self):
        from vtscore.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        d = emb.to_dict()
        assert d == {
            "name": "languagebind",
            "display_name": "LanguageBind (video)",
            "media_type_id": "video",
            "is_default": False,
            "supports_text": True,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_load_models_idempotent(self):
        from vtscore.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        # Simulate already-loaded model
        emb._model = MagicMock()
        emb._tokenizer = MagicMock()
        # Calling load_models again should be a no-op (not re-download)
        emb.load_models()
        # Model should be the same mock
        assert isinstance(emb._model, MagicMock)

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.mp4"})
        assert result is None

    def test_embed_text_returns_none_when_not_loaded(self):
        from vtscore.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text("a cat playing")
        assert result is None

    def test_uses_correct_model_id(self):
        from vtscore.config import LANGUAGEBIND_VIDEO_MODEL_ID

        assert LANGUAGEBIND_VIDEO_MODEL_ID == "LanguageBind/LanguageBind_Video_V1.5_FT"


class TestVideoMAEEmbedderProperties:
    """Verify VideoVideoMAEEmbedder class properties and registration."""

    def test_name(self):
        from vtscore.media.video.embedder_videomae import VideoVideoMAEEmbedder

        emb = VideoVideoMAEEmbedder()
        assert emb.name == "videomae"

    def test_media_type_id(self):
        from vtscore.media.video.embedder_videomae import VideoVideoMAEEmbedder

        emb = VideoVideoMAEEmbedder()
        assert emb.media_type_id == "video"

    def test_supports_text_is_false(self):
        """VideoMAE is vision-only — no text encoder, so text queries are unsupported."""
        from vtscore.media.video.embedder_videomae import VideoVideoMAEEmbedder

        emb = VideoVideoMAEEmbedder()
        assert emb.supports_text is False

    def test_is_not_default(self):
        """X-CLIP remains the default video embedder; VideoMAE is opt-in."""
        from vtscore.media.video.embedder_videomae import VideoVideoMAEEmbedder

        emb = VideoVideoMAEEmbedder()
        assert emb.is_default is False

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        emb = get_embedder("videomae")
        assert emb.name == "videomae"
        assert emb.media_type_id == "video"

    def test_listed_in_embedders_for_type(self):
        from vtscore.media import embedders_for_type

        embedders = embedders_for_type("video")
        names = [e.name for e in embedders]
        assert "videomae" in names

    def test_to_dict(self):
        from vtscore.media.video.embedder_videomae import VideoVideoMAEEmbedder

        emb = VideoVideoMAEEmbedder()
        d = emb.to_dict()
        assert d == {
            "name": "videomae",
            "display_name": "VideoMAE v2 (action features)",
            "media_type_id": "video",
            "is_default": False,
            "supports_text": False,
            "supports_patch_regions": False,
            "license_notice": None,
        }

    def test_load_models_idempotent(self):
        from vtscore.media.video.embedder_videomae import VideoVideoMAEEmbedder

        emb = VideoVideoMAEEmbedder()
        # Simulate already-loaded model
        emb._model = MagicMock()
        # Calling load_models again should be a no-op (not re-download)
        emb.load_models()
        # Model should be the same mock
        assert isinstance(emb._model, MagicMock)

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.video.embedder_videomae import VideoVideoMAEEmbedder

        emb = VideoVideoMAEEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.mp4"})
        assert result is None

    def test_embed_text_always_returns_none(self):
        """VideoMAE has no text tower — embed_text returns None even when loaded."""
        from vtscore.media.video.embedder_videomae import VideoVideoMAEEmbedder

        emb = VideoVideoMAEEmbedder()
        # Loaded or not, text embedding is unsupported.
        emb._model = MagicMock()
        assert emb.embed_text("an action") is None

    def test_uses_correct_model_id(self):
        from vtscore.config import VIDEOMAE_MODEL_ID

        assert VIDEOMAE_MODEL_ID == "OpenGVLab/VideoMAEv2-Base"


class TestPreprocessFrames:
    """Verify the _preprocess_frames helper produces correct shapes and values."""

    def test_output_shape(self):
        from PIL import Image

        from vtscore.media.video.embedder_languagebind import _preprocess_frames

        # Create 8 dummy RGB frames of varying sizes.
        rng = np.random.default_rng(42)
        frames = [Image.fromarray(rng.integers(0, 255, (320, 240, 3), dtype=np.uint8)) for _ in range(8)]
        result = _preprocess_frames(frames)
        # Expected shape: (C, T, H, W) = (3, 8, 224, 224).
        assert result.shape == (3, 8, 224, 224)

    def test_output_dtype(self):
        from PIL import Image

        from vtscore.media.video.embedder_languagebind import _preprocess_frames

        rng = np.random.default_rng(42)
        frames = [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(4)]
        result = _preprocess_frames(frames)
        assert result.dtype == np.float32

    def test_single_frame(self):
        from PIL import Image

        from vtscore.media.video.embedder_languagebind import _preprocess_frames

        frame = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8))
        result = _preprocess_frames([frame])
        assert result.shape == (3, 1, 224, 224)


class TestVideoMAEPreprocessFrames:
    """Verify VideoMAE's frame preprocessing produces the expected shape.

    VideoMAE expects ``(T, C, H, W)`` per video — unlike LanguageBind which
    transposes to ``(C, T, H, W)`` — so the helper differs by one axis order.
    """

    def test_output_shape(self):
        from PIL import Image

        from vtscore.media.video.embedder_videomae import _preprocess_frames

        rng = np.random.default_rng(42)
        frames = [Image.fromarray(rng.integers(0, 255, (320, 240, 3), dtype=np.uint8)) for _ in range(16)]
        result = _preprocess_frames(frames)
        # VideoMAE expects (T, C, H, W).
        assert result.shape == (16, 3, 224, 224)

    def test_output_dtype(self):
        from PIL import Image

        from vtscore.media.video.embedder_videomae import _preprocess_frames

        rng = np.random.default_rng(42)
        frames = [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(4)]
        result = _preprocess_frames(frames)
        assert result.dtype == np.float32

    def test_single_frame(self):
        from PIL import Image

        from vtscore.media.video.embedder_videomae import _preprocess_frames

        frame = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8))
        result = _preprocess_frames([frame])
        assert result.shape == (1, 3, 224, 224)


class TestAllEmbeddersRegistration:
    """Verify all expected embedders are registered."""

    def test_total_embedder_count(self):
        from vtscore.media import all_embedders

        embedders = all_embedders()
        # 7 original + 8 image embedders (clip, siglip2, plus single/patch
        # variants for dinov2, dinov3, eupe) + 1 face embedder
        # + 1 vision-only video embedder (videomae)
        # + 3 audio embedders (ast, clap_general, whisper_encoder).
        assert len(embedders) == 20

    def test_all_embedders_dict_includes_supports_text(self):
        """The new ``supports_text`` flag must round-trip through ``to_dict``
        so the frontend can hide text-search UI for vision-only embedders."""
        from vtscore.media import all_embedders_dict

        dicts = all_embedders_dict()
        by_name = {d["name"]: d for d in dicts}
        # Cross-modal embedders advertise text support.
        for name in ("siglip", "siglip2", "clip", "clap", "clap_general", "xclip"):
            assert by_name[name]["supports_text"] is True, name
        # Vision-only / patch-based and speech-only embedders do not.
        for name in (
            "dinov2_single",
            "dinov2_patch",
            "dinov3_single",
            "dinov3_patch",
            "eupe_single",
            "eupe_patch",
            "videomae",
            "ast",
            "whisper_encoder",
        ):
            assert by_name[name]["supports_text"] is False, name

    def test_all_expected_names_present(self):
        from vtscore.media import all_embedders

        names = {e.name for e in all_embedders()}
        expected = {
            "clap",
            "clap_music",
            "clap_general",
            "ast",
            "whisper_encoder",
            "siglip",
            "siglip2",
            "clip",
            "dinov2_single",
            "dinov2_patch",
            "dinov3_single",
            "dinov3_patch",
            "eupe_single",
            "eupe_patch",
            "e5",
            "bge",
            "xclip",
            "languagebind",
            "videomae",
            "face",
        }
        assert names == expected

    def test_embedders_for_audio(self):
        from vtscore.media import embedders_for_type

        names = {e.name for e in embedders_for_type("audio")}
        assert names == {"clap", "clap_music", "clap_general", "ast", "whisper_encoder"}

    def test_embedders_for_image(self):
        from vtscore.media import embedders_for_type

        names = {e.name for e in embedders_for_type("image")}
        assert names == {
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

    def test_siglip_is_still_default_image_embedder(self):
        """SigLIP must remain the first (default) image embedder so callers
        using ``embedders_for_type('image')[0]`` keep the historical default
        even after the new image embedders land."""
        from vtscore.media import embedders_for_type

        ordered = embedders_for_type("image")
        assert ordered[0].name == "siglip"
        assert ordered[0].is_default is True

    def test_embedders_for_text(self):
        from vtscore.media import embedders_for_type

        names = {e.name for e in embedders_for_type("text")}
        assert names == {"e5", "bge"}

    def test_embedders_for_video(self):
        from vtscore.media import embedders_for_type

        names = {e.name for e in embedders_for_type("video")}
        assert names == {"xclip", "languagebind", "videomae"}

    def test_all_embedders_dict(self):
        from vtscore.media import all_embedders_dict

        dicts = all_embedders_dict()
        # 7 original + 8 image embedders (clip, siglip2, plus single/patch
        # variants for dinov2, dinov3, eupe) + 1 face embedder
        # + 1 vision-only video embedder (videomae)
        # + 3 audio embedders (ast, clap_general, whisper_encoder).
        assert len(dicts) == 20
        for d in dicts:
            assert "name" in d
            assert "media_type_id" in d
            assert "is_default" in d
            assert "supports_text" in d
            assert "supports_patch_regions" in d
            assert "license_notice" in d


class TestEmbedderSentinelDiscovery:
    """Verify built-in embedder modules expose the ``EMBEDDER`` sentinel
    so that auto-discovery picks them up with no edits to any
    ``__init__.py`` — the same pattern used by exporters, dataset
    importers, label importers, processor importers, settings importers /
    exporters, and sync sources.
    """

    def test_every_builtin_embedder_module_has_sentinel(self):
        from vtscore.media.audio import embedder_ast
        from vtscore.media.audio import embedder_clap
        from vtscore.media.audio import embedder_clap_general
        from vtscore.media.audio import embedder_clap_music
        from vtscore.media.audio import embedder_whisper
        from vtscore.media.image import embedder_siglip
        from vtscore.media.text import embedder_bge
        from vtscore.media.text import embedder_e5
        from vtscore.media.video import embedder_languagebind
        from vtscore.media.video import embedder_videomae
        from vtscore.media.video import embedder_xclip

        from vtscore.media.embedder import MediaEmbedder

        modules = [
            embedder_ast,
            embedder_clap,
            embedder_clap_general,
            embedder_clap_music,
            embedder_whisper,
            embedder_siglip,
            embedder_bge,
            embedder_e5,
            embedder_languagebind,
            embedder_videomae,
            embedder_xclip,
        ]
        for mod in modules:
            sentinel = getattr(mod, "EMBEDDER", None)
            assert sentinel is not None, f"{mod.__name__} is missing an EMBEDDER sentinel"
            assert isinstance(sentinel, MediaEmbedder), f"{mod.__name__}.EMBEDDER must be a MediaEmbedder instance"

    def test_sentinel_identity_matches_registry(self):
        """The registered embedder for each name should be the module's EMBEDDER sentinel."""
        from vtscore.media import get_embedder
        from vtscore.media.audio.embedder_clap import EMBEDDER as clap_sentinel
        from vtscore.media.text.embedder_bge import EMBEDDER as bge_sentinel
        from vtscore.media.video.embedder_languagebind import EMBEDDER as lb_sentinel
        from vtscore.media.video.embedder_videomae import EMBEDDER as vm_sentinel

        assert get_embedder("clap") is clap_sentinel
        assert get_embedder("bge") is bge_sentinel
        assert get_embedder("languagebind") is lb_sentinel
        assert get_embedder("videomae") is vm_sentinel

    def test_media_type_init_no_longer_lists_embedders(self):
        """Media-type package ``__init__.py`` files should not expose an
        ``EMBEDDERS`` attribute — embedders are discovered per-module.
        """
        from vtscore.media import audio, document, image, text, video

        for pkg in (audio, image, text, video, document):
            assert not hasattr(pkg, "EMBEDDERS"), (
                f"{pkg.__name__} still exposes an EMBEDDERS list — "
                "embedders should be discovered via per-module EMBEDDER sentinels"
            )

    def test_folder_embedder_auto_discovered(self, tmp_path):
        """A new ``embedder_*/`` sub-package (directory with ``__init__.py``)
        dropped into a media-type package should be auto-discovered, just
        like a flat ``embedder_*.py`` module.
        """
        import importlib
        import importlib.util
        import sys
        from pathlib import Path

        from vtscore.media import _discover_embedders_in
        from vtscore.media.embedder import MediaEmbedder

        # Fake media-type package containing an embedder *sub-package*
        # (not a flat module) exposing the EMBEDDER sentinel from its
        # __init__.py.
        fake_pkg = tmp_path / "fakemedia_folder"
        fake_pkg.mkdir()
        (fake_pkg / "__init__.py").write_text("")

        embedder_pkg = fake_pkg / "embedder_folder"
        embedder_pkg.mkdir()
        (embedder_pkg / "__init__.py").write_text(
            "from vtscore.media.embedder import MediaEmbedder\n"
            "\n"
            "class _FolderEmbedder(MediaEmbedder):\n"
            "    @property\n"
            "    def name(self):\n"
            "        return 'folder_discoverable'\n"
            "    @property\n"
            "    def media_type_id(self):\n"
            "        return 'fake'\n"
            "    def _load_models_impl(self):\n"
            "        pass\n"
            "    def _embed_media_impl(self, media):\n"
            "        return None\n"
            "\n"
            "EMBEDDER = _FolderEmbedder()\n"
        )

        package_name = "vtscore.media._fakemedia_folder_test"
        spec = importlib.util.spec_from_file_location(
            package_name,
            str(fake_pkg / "__init__.py"),
            submodule_search_locations=[str(fake_pkg)],
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = mod
        try:
            spec.loader.exec_module(mod)
            from vtscore.media import _embedder_registry

            saved = dict(_embedder_registry)
            try:
                _discover_embedders_in(Path(fake_pkg), package_name)
                assert "folder_discoverable" in _embedder_registry
                discovered = _embedder_registry["folder_discoverable"]
                assert isinstance(discovered, MediaEmbedder)
                assert discovered.media_type_id == "fake"
            finally:
                _embedder_registry.clear()
                _embedder_registry.update(saved)
        finally:
            for key in list(sys.modules):
                if key == package_name or key.startswith(package_name + "."):
                    sys.modules.pop(key, None)

    def test_custom_embedder_auto_discovered(self, tmp_path, monkeypatch):
        """A new ``embedder_*.py`` file dropped into a media-type package
        should be auto-discovered with no ``__init__.py`` edits.
        """
        import importlib
        import importlib.util
        import sys
        from pathlib import Path

        from vtscore.media import _discover_embedders_in
        from vtscore.media.embedder import MediaEmbedder

        # Create a throwaway media-type package under tmp_path with a single
        # embedder module exposing the EMBEDDER sentinel.
        fake_pkg = tmp_path / "fakemedia"
        fake_pkg.mkdir()
        (fake_pkg / "__init__.py").write_text("")
        embedder_src = (
            "from vtscore.media.embedder import MediaEmbedder\n"
            "\n"
            "class _FakeEmbedder(MediaEmbedder):\n"
            "    @property\n"
            "    def name(self):\n"
            "        return 'fake_discoverable'\n"
            "    @property\n"
            "    def media_type_id(self):\n"
            "        return 'fake'\n"
            "    def _load_models_impl(self):\n"
            "        pass\n"
            "    def _embed_media_impl(self, media):\n"
            "        return None\n"
            "\n"
            "EMBEDDER = _FakeEmbedder()\n"
        )
        (fake_pkg / "embedder_fake.py").write_text(embedder_src)

        # Make the fake package importable as if it lived under vtscore.media.
        package_name = "vtscore.media._fakemedia_test"
        spec = importlib.util.spec_from_file_location(
            package_name,
            str(fake_pkg / "__init__.py"),
            submodule_search_locations=[str(fake_pkg)],
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = mod
        try:
            spec.loader.exec_module(mod)
            # Snapshot the registry so we can restore it after the test.
            from vtscore.media import _embedder_registry

            saved = dict(_embedder_registry)
            try:
                _discover_embedders_in(Path(fake_pkg), package_name)
                assert "fake_discoverable" in _embedder_registry
                discovered = _embedder_registry["fake_discoverable"]
                assert isinstance(discovered, MediaEmbedder)
                assert discovered.media_type_id == "fake"
            finally:
                _embedder_registry.clear()
                _embedder_registry.update(saved)
        finally:
            # Remove the fake package and its submodule from sys.modules.
            for key in list(sys.modules):
                if key == package_name or key.startswith(package_name + "."):
                    sys.modules.pop(key, None)


class TestNewEmbeddersInheritance:
    """Verify new embedders correctly extend MediaEmbedder."""

    def test_siglip_is_media_embedder(self):
        from vtscore.media.embedder import MediaEmbedder
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        assert issubclass(ImageSiglipEmbedder, MediaEmbedder)

    def test_clap_music_is_media_embedder(self):
        from vtscore.media.audio.embedder_clap_music import AudioClapMusicEmbedder
        from vtscore.media.embedder import MediaEmbedder

        assert issubclass(AudioClapMusicEmbedder, MediaEmbedder)

    def test_bge_is_media_embedder(self):
        from vtscore.media.embedder import MediaEmbedder
        from vtscore.media.text.embedder_bge import TextBGEEmbedder

        assert issubclass(TextBGEEmbedder, MediaEmbedder)

    def test_languagebind_is_media_embedder(self):
        from vtscore.media.embedder import MediaEmbedder
        from vtscore.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        assert issubclass(VideoLanguageBindEmbedder, MediaEmbedder)

    def test_videomae_is_media_embedder(self):
        from vtscore.media.embedder import MediaEmbedder
        from vtscore.media.video.embedder_videomae import VideoVideoMAEEmbedder

        assert issubclass(VideoVideoMAEEmbedder, MediaEmbedder)

    def test_clap_general_is_media_embedder(self):
        from vtscore.media.audio.embedder_clap_general import AudioClapGeneralEmbedder
        from vtscore.media.embedder import MediaEmbedder

        assert issubclass(AudioClapGeneralEmbedder, MediaEmbedder)

    def test_ast_is_media_embedder(self):
        from vtscore.media.audio.embedder_ast import AudioASTEmbedder
        from vtscore.media.embedder import MediaEmbedder

        assert issubclass(AudioASTEmbedder, MediaEmbedder)

    def test_whisper_encoder_is_media_embedder(self):
        from vtscore.media.audio.embedder_whisper import AudioWhisperEncoderEmbedder
        from vtscore.media.embedder import MediaEmbedder

        assert issubclass(AudioWhisperEncoderEmbedder, MediaEmbedder)

    def test_embed_text_enriched_works(self):
        """embed_text_enriched (inherited from base) should work with mocked embed_text."""
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        fake_vec = np.random.RandomState(42).randn(768).astype(np.float32)
        with patch.object(emb, "embed_text", return_value=fake_vec):
            result = emb.embed_text_enriched("a cat")
        assert result is not None
        assert result.shape == (768,)


class TestEmbedMediaLock:
    """Verify that embed_media serialises concurrent calls via _embed_lock."""

    def test_concurrent_embed_media_serialised(self):
        """Two threads calling embed_media must not overlap (global lock)."""
        from vtscore.media.embedder import MediaEmbedder

        inside = threading.Event()
        proceed = threading.Event()
        overlap_detected = False

        class SlowEmbedder(MediaEmbedder):
            @property
            def name(self):
                return "slow_test"

            @property
            def media_type_id(self):
                return "test"

            def _load_models_impl(self):
                pass

            def _embed_media_impl(self, media):
                nonlocal overlap_detected
                if inside.is_set():
                    overlap_detected = True
                inside.set()
                proceed.wait(timeout=5)
                inside.clear()
                return np.zeros(8, dtype=np.float32)

            def embed_text(self, text):
                return None

        emb = SlowEmbedder()
        results = [None, None]

        def call_embed(idx):
            results[idx] = emb.embed_media({"media_path": "/fake"})

        t1 = threading.Thread(target=call_embed, args=(0,))
        t2 = threading.Thread(target=call_embed, args=(1,))

        # Save and replace the lock with a fresh one so we don't interfere
        # with other tests (the class attr is shared).
        original_lock = MediaEmbedder._embed_lock
        MediaEmbedder._embed_lock = threading.Lock()
        try:
            t1.start()
            # Wait for t1 to be inside _embed_media_impl
            inside.wait(timeout=5)
            assert inside.is_set(), "t1 should be inside _embed_media_impl"

            t2.start()
            # Give t2 a moment to hit the lock
            import time

            time.sleep(0.1)
            # t2 should be blocked on the lock — inside should still be set by t1 only
            assert not overlap_detected, "t2 entered _embed_media_impl while t1 was still inside"

            # Let t1 finish
            proceed.set()
            t1.join(timeout=5)
            t2.join(timeout=5)

            assert not overlap_detected, "Concurrent embed_media calls overlapped"
            assert results[0] is not None
            assert results[1] is not None
        finally:
            MediaEmbedder._embed_lock = original_lock

    def test_embed_media_delegates_to_impl(self):
        """embed_media() should call _embed_media_impl() and return its result."""
        from vtscore.media.embedder import MediaEmbedder

        class SimpleEmbedder(MediaEmbedder):
            @property
            def name(self):
                return "simple_test"

            @property
            def media_type_id(self):
                return "test"

            def _load_models_impl(self):
                pass

            def _embed_media_impl(self, media):
                return np.ones(4, dtype=np.float32)

            def embed_text(self, text):
                return None

        emb = SimpleEmbedder()
        result = emb.embed_media({"media_path": "/fake"})
        assert result is not None
        np.testing.assert_array_equal(result, np.ones(4, dtype=np.float32))
