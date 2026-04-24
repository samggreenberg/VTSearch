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
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        emb = ImageSiglipEmbedder()
        # Patch load_models to not actually load anything
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.jpg"})
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
        from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder

        emb = AudioClapMusicEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.wav"})
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
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        emb = TextBGEEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.txt"})
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

        result = emb.embed_media({"media_path": str(text_file)})
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


class TestVideoLanguageBindEmbedderProperties:
    """Verify VideoLanguageBindEmbedder class properties and registration."""

    def test_name(self):
        from vtsearch.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        assert emb.name == "languagebind"

    def test_media_type_id(self):
        from vtsearch.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        assert emb.media_type_id == "video"

    def test_description_wrappers_non_empty(self):
        from vtsearch.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) > 0
        assert all("{text}" in w for w in wrappers)

    def test_registered_in_registry(self):
        from vtsearch.media import get_embedder

        emb = get_embedder("languagebind")
        assert emb.name == "languagebind"
        assert emb.media_type_id == "video"

    def test_listed_in_embedders_for_type(self):
        from vtsearch.media import embedders_for_type

        embedders = embedders_for_type("video")
        names = [e.name for e in embedders]
        assert "languagebind" in names

    def test_to_dict(self):
        from vtsearch.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        d = emb.to_dict()
        assert d == {"name": "languagebind", "media_type_id": "video"}

    def test_load_models_idempotent(self):
        from vtsearch.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        # Simulate already-loaded model
        emb._model = MagicMock()
        emb._tokenizer = MagicMock()
        # Calling load_models again should be a no-op (not re-download)
        emb.load_models()
        # Model should be the same mock
        assert isinstance(emb._model, MagicMock)

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtsearch.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_media({"media_path": "/nonexistent.mp4"})
        assert result is None

    def test_embed_text_returns_none_when_not_loaded(self):
        from vtsearch.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        with patch.object(emb, "load_models"):
            result = emb.embed_text("a cat playing")
        assert result is None

    def test_uses_correct_model_id(self):
        from vtsearch.config import LANGUAGEBIND_VIDEO_MODEL_ID

        assert LANGUAGEBIND_VIDEO_MODEL_ID == "LanguageBind/LanguageBind_Video_V1.5_FT"


class TestPreprocessFrames:
    """Verify the _preprocess_frames helper produces correct shapes and values."""

    def test_output_shape(self):
        from PIL import Image

        from vtsearch.media.video.embedder_languagebind import _preprocess_frames

        # Create 8 dummy RGB frames of varying sizes.
        rng = np.random.default_rng(42)
        frames = [Image.fromarray(rng.integers(0, 255, (320, 240, 3), dtype=np.uint8)) for _ in range(8)]
        result = _preprocess_frames(frames)
        # Expected shape: (C, T, H, W) = (3, 8, 224, 224).
        assert result.shape == (3, 8, 224, 224)

    def test_output_dtype(self):
        from PIL import Image

        from vtsearch.media.video.embedder_languagebind import _preprocess_frames

        rng = np.random.default_rng(42)
        frames = [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(4)]
        result = _preprocess_frames(frames)
        assert result.dtype == np.float32

    def test_single_frame(self):
        from PIL import Image

        from vtsearch.media.video.embedder_languagebind import _preprocess_frames

        frame = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8))
        result = _preprocess_frames([frame])
        assert result.shape == (3, 1, 224, 224)


class TestAllEmbeddersRegistration:
    """Verify all expected embedders are registered."""

    def test_total_embedder_count(self):
        from vtsearch.media import all_embedders

        embedders = all_embedders()
        assert len(embedders) == 8

    def test_all_expected_names_present(self):
        from vtsearch.media import all_embedders

        names = {e.name for e in all_embedders()}
        expected = {"clap", "clap_music", "clip", "siglip", "e5", "bge", "xclip", "languagebind"}
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
        assert names == {"xclip", "languagebind"}

    def test_all_embedders_dict(self):
        from vtsearch.media import all_embedders_dict

        dicts = all_embedders_dict()
        assert len(dicts) == 8
        for d in dicts:
            assert "name" in d
            assert "media_type_id" in d


class TestEmbedderSentinelDiscovery:
    """Verify built-in embedder modules expose the ``EMBEDDER`` sentinel
    so that auto-discovery picks them up with no edits to any
    ``__init__.py`` — the same pattern used by exporters, dataset
    importers, label importers, processor importers, settings importers /
    exporters, and sync sources.
    """

    def test_every_builtin_embedder_module_has_sentinel(self):
        from vtsearch.media.audio import embedder as audio_clap
        from vtsearch.media.audio import embedder_clap_music
        from vtsearch.media.image import embedder as image_clip
        from vtsearch.media.image import embedder_siglip
        from vtsearch.media.text import embedder as text_e5
        from vtsearch.media.text import embedder_bge
        from vtsearch.media.video import embedder as video_xclip
        from vtsearch.media.video import embedder_languagebind

        from vtsearch.media.embedder import MediaEmbedder

        modules = [
            audio_clap,
            embedder_clap_music,
            image_clip,
            embedder_siglip,
            text_e5,
            embedder_bge,
            video_xclip,
            embedder_languagebind,
        ]
        for mod in modules:
            sentinel = getattr(mod, "EMBEDDER", None)
            assert sentinel is not None, f"{mod.__name__} is missing an EMBEDDER sentinel"
            assert isinstance(sentinel, MediaEmbedder), f"{mod.__name__}.EMBEDDER must be a MediaEmbedder instance"

    def test_sentinel_identity_matches_registry(self):
        """The registered embedder for each name should be the module's EMBEDDER sentinel."""
        from vtsearch.media import get_embedder
        from vtsearch.media.audio.embedder import EMBEDDER as clap_sentinel
        from vtsearch.media.text.embedder_bge import EMBEDDER as bge_sentinel
        from vtsearch.media.video.embedder_languagebind import EMBEDDER as lb_sentinel

        assert get_embedder("clap") is clap_sentinel
        assert get_embedder("bge") is bge_sentinel
        assert get_embedder("languagebind") is lb_sentinel

    def test_media_type_init_no_longer_lists_embedders(self):
        """Media-type package ``__init__.py`` files should not expose an
        ``EMBEDDERS`` attribute — embedders are discovered per-module.
        """
        from vtsearch.media import audio, document, image, text, video

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
        import sys
        from pathlib import Path

        from vtsearch.media import _discover_embedders_in
        from vtsearch.media.embedder import MediaEmbedder

        # Fake media-type package containing an embedder *sub-package*
        # (not a flat module) exposing the EMBEDDER sentinel from its
        # __init__.py.
        fake_pkg = tmp_path / "fakemedia_folder"
        fake_pkg.mkdir()
        (fake_pkg / "__init__.py").write_text("")

        embedder_pkg = fake_pkg / "embedder_folder"
        embedder_pkg.mkdir()
        (embedder_pkg / "__init__.py").write_text(
            "from vtsearch.media.embedder import MediaEmbedder\n"
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

        package_name = "vtsearch.media._fakemedia_folder_test"
        spec = importlib.util.spec_from_file_location(
            package_name,
            str(fake_pkg / "__init__.py"),
            submodule_search_locations=[str(fake_pkg)],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = mod
        try:
            spec.loader.exec_module(mod)
            from vtsearch.media import _embedder_registry

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
        import sys
        from pathlib import Path

        from vtsearch.media import _discover_embedders_in
        from vtsearch.media.embedder import MediaEmbedder

        # Create a throwaway media-type package under tmp_path with a single
        # embedder module exposing the EMBEDDER sentinel.
        fake_pkg = tmp_path / "fakemedia"
        fake_pkg.mkdir()
        (fake_pkg / "__init__.py").write_text("")
        embedder_src = (
            "from vtsearch.media.embedder import MediaEmbedder\n"
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

        # Make the fake package importable as if it lived under vtsearch.media.
        package_name = "vtsearch.media._fakemedia_test"
        spec = importlib.util.spec_from_file_location(
            package_name,
            str(fake_pkg / "__init__.py"),
            submodule_search_locations=[str(fake_pkg)],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = mod
        try:
            spec.loader.exec_module(mod)
            # Snapshot the registry so we can restore it after the test.
            from vtsearch.media import _embedder_registry

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
        from vtsearch.media.embedder import MediaEmbedder
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

        assert issubclass(ImageSiglipEmbedder, MediaEmbedder)

    def test_clap_music_is_media_embedder(self):
        from vtsearch.media.audio.embedder_clap_music import AudioClapMusicEmbedder
        from vtsearch.media.embedder import MediaEmbedder

        assert issubclass(AudioClapMusicEmbedder, MediaEmbedder)

    def test_bge_is_media_embedder(self):
        from vtsearch.media.embedder import MediaEmbedder
        from vtsearch.media.text.embedder_bge import TextBGEEmbedder

        assert issubclass(TextBGEEmbedder, MediaEmbedder)

    def test_languagebind_is_media_embedder(self):
        from vtsearch.media.embedder import MediaEmbedder
        from vtsearch.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        assert issubclass(VideoLanguageBindEmbedder, MediaEmbedder)

    def test_embed_text_enriched_works(self):
        """embed_text_enriched (inherited from base) should work with mocked embed_text."""
        from vtsearch.media.image.embedder_siglip import ImageSiglipEmbedder

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
        from vtsearch.media.embedder import MediaEmbedder

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
        from vtsearch.media.embedder import MediaEmbedder

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
