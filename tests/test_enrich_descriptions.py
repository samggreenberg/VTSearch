"""Tests for the Enrich Sort Descriptions feature.

Covers:
- description_wrappers property on each media type
- embed_text_enriched method (base class logic)
- enrich_descriptions setting (get/set/persist)
- embed_text_query with enrich=True
- eval runner with enrich=True
- settings API route for enrich_descriptions
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from vtsearch.eval.config import EvalQuery
from vtsearch.eval.runner import eval_text_sort
from vtsearch.media.audio.embedder import AudioClapEmbedder
from vtsearch.media.base import MediaEmbedder
from vtsearch.media.image.embedder import ImageClipEmbedder
from vtsearch.media.text.embedder import TextE5Embedder
from vtsearch.media.video.embedder import VideoXClipEmbedder
from vtsearch.models.embeddings import embed_text_query


# =====================================================================
# description_wrappers property
# =====================================================================


class TestDescriptionWrappers:
    def test_audio_has_wrappers(self):
        emb = AudioClapEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) >= 3
        for w in wrappers:
            assert "{text}" in w

    def test_image_has_wrappers(self):
        emb = ImageClipEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) >= 3
        for w in wrappers:
            assert "{text}" in w

    def test_text_has_wrappers(self):
        emb = TextE5Embedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) >= 3
        for w in wrappers:
            assert "{text}" in w

    def test_video_has_wrappers(self):
        emb = VideoXClipEmbedder()
        wrappers = emb.description_wrappers
        assert len(wrappers) >= 3
        for w in wrappers:
            assert "{text}" in w

    def test_wrappers_include_bare_text(self):
        """Each embedder should include a plain '{text}' wrapper."""
        for emb_cls in (AudioClapEmbedder, ImageClipEmbedder, TextE5Embedder, VideoXClipEmbedder):
            emb = emb_cls()
            assert "{text}" in emb.description_wrappers, f"{emb_cls.__name__} missing bare '{{text}}' wrapper"

    def test_wrappers_format_correctly(self):
        """All wrappers should format without errors."""
        for emb_cls in (AudioClapEmbedder, ImageClipEmbedder, TextE5Embedder, VideoXClipEmbedder):
            emb = emb_cls()
            for wrapper in emb.description_wrappers:
                result = wrapper.format(text="test query")
                assert "test query" in result


# =====================================================================
# embed_text_enriched (base class logic)
# =====================================================================


class TestEmbedTextEnriched:
    def _make_mock_embedder(self, wrappers, embed_fn):
        """Create a minimal concrete MediaEmbedder subclass for testing."""

        class MockEmbedder(MediaEmbedder):
            @property
            def name(self):
                return "mock"

            @property
            def media_type_id(self):
                return "mock"

            def _load_models_impl(self):
                pass

            def _embed_media_impl(self, file_path):
                return None

            def embed_text(self, text):
                return embed_fn(text)

            @property
            def description_wrappers(self):
                return wrappers

        return MockEmbedder()

    def test_enriched_averages_wrapper_embeddings(self):
        """embed_text_enriched should average embeddings across wrappers."""
        call_log = []

        def mock_embed(text):
            call_log.append(text)
            # Return different vectors for different wrapped texts
            if text.startswith("a photo"):
                return np.array([1.0, 0.0, 0.0])
            elif text.startswith("an image"):
                return np.array([0.0, 1.0, 0.0])
            else:
                return np.array([0.0, 0.0, 1.0])

        mt = self._make_mock_embedder(
            wrappers=["a photo of {text}", "an image of {text}", "{text}"],
            embed_fn=mock_embed,
        )

        result = mt.embed_text_enriched("cats")
        assert result is not None
        assert len(call_log) == 3
        assert "a photo of cats" in call_log
        assert "an image of cats" in call_log
        assert "cats" in call_log

        # Result should be the L2-normalised mean of the three vectors
        expected = np.mean([[1, 0, 0], [0, 1, 0], [0, 0, 1]], axis=0)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_enriched_falls_back_when_no_wrappers(self):
        """If no wrappers, embed_text_enriched should fall back to embed_text."""

        def mock_embed(text):
            return np.array([1.0, 2.0, 3.0])

        mt = self._make_mock_embedder(wrappers=[], embed_fn=mock_embed)
        result = mt.embed_text_enriched("test")
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_enriched_falls_back_when_all_fail(self):
        """If all wrapper embeddings fail, fall back to plain embed_text."""
        calls = {"count": 0}

        def mock_embed(text):
            calls["count"] += 1
            if calls["count"] <= 2:
                return None  # Fail for wrapped texts
            return np.array([1.0, 0.0])  # Succeed for plain fallback

        mt = self._make_mock_embedder(
            wrappers=["wrapper1 {text}", "wrapper2 {text}"],
            embed_fn=mock_embed,
        )
        result = mt.embed_text_enriched("test")
        assert result is not None
        np.testing.assert_array_equal(result, np.array([1.0, 0.0]))

    def test_enriched_skips_failed_wrappers(self):
        """Wrappers that fail to embed should be skipped, not crash."""

        def mock_embed(text):
            if "bad" in text:
                return None
            return np.array([1.0, 0.0])

        mt = self._make_mock_embedder(
            wrappers=["good {text}", "bad {text}"],
            embed_fn=mock_embed,
        )
        result = mt.embed_text_enriched("query")
        assert result is not None
        # Only one embedding succeeded, so result is just that one (normalised)
        expected = np.array([1.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_enriched_result_is_normalised(self):
        """The enriched embedding should be L2-normalised."""

        def mock_embed(text):
            return np.array([3.0, 4.0])

        mt = self._make_mock_embedder(
            wrappers=["w1 {text}", "w2 {text}"],
            embed_fn=mock_embed,
        )
        result = mt.embed_text_enriched("test")
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6


# =====================================================================
# embed_text_query with enrich flag
# =====================================================================


class TestEmbedTextQueryEnrich:
    def test_enrich_false_calls_embed_text(self):
        """enrich=False should call embed_text, not embed_text_enriched."""
        mock_vec = np.array([1.0, 0.0])

        class FakeEmbedder:
            media_type_id = "audio"

            def embed_text(self, text):
                return mock_vec

            def embed_text_enriched(self, text):
                raise AssertionError("Should not be called")

        with patch("vtsearch.models.embeddings._get_embedder_for_media_type", return_value=FakeEmbedder()):
            result = embed_text_query("test", "audio", enrich=False)
        np.testing.assert_array_equal(result, mock_vec)

    def test_enrich_true_calls_embed_text_enriched(self):
        """enrich=True should call embed_text_enriched."""
        mock_vec = np.array([0.0, 1.0])

        class FakeEmbedder:
            media_type_id = "audio"

            def embed_text(self, text):
                raise AssertionError("Should not be called")

            def embed_text_enriched(self, text):
                return mock_vec

        with patch("vtsearch.models.embeddings._get_embedder_for_media_type", return_value=FakeEmbedder()):
            result = embed_text_query("test", "audio", enrich=True)
        np.testing.assert_array_equal(result, mock_vec)


# =====================================================================
# enrich_descriptions setting
# =====================================================================


class TestEnrichDescriptionsSetting:
    def test_default_is_false(self):
        from vtsearch import settings as settings_mod

        assert settings_mod.get_enrich_descriptions() is False

    def test_set_and_get(self):
        from vtsearch import settings as settings_mod

        settings_mod.set_enrich_descriptions(True)
        assert settings_mod.get_enrich_descriptions() is True

        settings_mod.set_enrich_descriptions(False)
        assert settings_mod.get_enrich_descriptions() is False

    def test_persists_to_disk(self, isolated_settings):
        import json

        from vtsearch import settings as settings_mod

        settings_mod.set_enrich_descriptions(True)
        raw = json.loads(isolated_settings.read_text())
        assert raw["enrich_descriptions"] is True

    def test_in_get_all(self):
        from vtsearch import settings as settings_mod

        data = settings_mod.get_all()
        assert "enrich_descriptions" in data
        assert data["enrich_descriptions"] is False


# =====================================================================
# Settings API route
# =====================================================================


class TestEnrichDescriptionsAPI:
    @pytest.fixture
    def client(self):
        import app as app_module

        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c

    def test_get_includes_enrich_descriptions(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "enrich_descriptions" in data
        assert data["enrich_descriptions"] is False

    def test_put_enrich_descriptions(self, client):
        res = client.put("/api/settings", json={"enrich_descriptions": True})
        assert res.status_code == 200
        data = res.get_json()
        assert data["enrich_descriptions"] is True

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["enrich_descriptions"] is True


# =====================================================================
# Eval runner with enrich flag
# =====================================================================


class TestEvalTextSortEnrich:
    def _make_synthetic_clips(self):
        rng = np.random.RandomState(0)
        medias = {}
        media_id = 1
        cat_dir = np.zeros(16)
        cat_dir[0] = 1.0
        for _ in range(10):
            emb = cat_dir + rng.normal(0, 0.05, 16)
            emb /= np.linalg.norm(emb)
            medias[media_id] = {"id": media_id, "embedding": emb, "category": "cat", "type": "image"}
            media_id += 1
        dog_dir = np.zeros(16)
        dog_dir[1] = 1.0
        for _ in range(10):
            emb = dog_dir + rng.normal(0, 0.05, 16)
            emb /= np.linalg.norm(emb)
            medias[media_id] = {"id": media_id, "embedding": emb, "category": "dog", "type": "image"}
            media_id += 1
        return medias, cat_dir, dog_dir

    def test_eval_text_sort_with_enrich(self):
        """eval_text_sort should pass enrich to embed_text_query."""
        medias, cat_dir, dog_dir = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat")]

        call_kwargs = []

        def mock_embed(text, media_type, enrich=False):
            call_kwargs.append({"enrich": enrich})
            if "cat" in text:
                return cat_dir.copy()
            return dog_dir.copy()

        with patch("vtsearch.models.embeddings.embed_text_query", side_effect=mock_embed):
            results = eval_text_sort(medias, queries, "image", k_values=[5], enrich=True)

        assert len(results) == 1
        assert all(kw["enrich"] is True for kw in call_kwargs)

    def test_eval_text_sort_without_enrich(self):
        """eval_text_sort with enrich=False should pass enrich=False."""
        medias, cat_dir, dog_dir = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat")]

        call_kwargs = []

        def mock_embed(text, media_type, enrich=False):
            call_kwargs.append({"enrich": enrich})
            return cat_dir.copy()

        with patch("vtsearch.models.embeddings.embed_text_query", side_effect=mock_embed):
            eval_text_sort(medias, queries, "image", k_values=[5], enrich=False)

        assert all(kw["enrich"] is False for kw in call_kwargs)
