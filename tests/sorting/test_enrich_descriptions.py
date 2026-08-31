"""Tests for the Enrich Sort Descriptions feature.

Covers:
- description_wrappers: the per-embedder split measured by #3127/#3341
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

from vtscore.eval.config import EvalQuery
from vtscore.eval.runner import eval_text_sort
from vtscore.media.embedder import MediaEmbedder
from vtscore.embedding.helpers import embed_text_query


# =====================================================================
# description_wrappers property
# =====================================================================


class TestDescriptionWrappers:
    """Which embedders enrich, and which embed the typed query plainly.

    Enrichment is a property of the **embedder**, not of the media type.
    #3127 measured the ensemble on/off across 22 eval datasets and 560 paired
    queries (paired Delta in text-sort average precision, SEs clustered on
    (corpus, category)) and the answer split by model:

    ======================  ==================
    embedder                Delta AP
    ======================  ==================
    ``clap_general``        +0.014 +/- 0.009
    ``xclip``               +0.008 +/- 0.014
    ``siglip``              -0.001 +/- 0.002
    ``clap``                -0.010 +/- 0.008
    ``e5``                  -0.057 +/- 0.009
    ``bge``                 -0.059 +/- 0.009
    ======================  ==================

    Every individual template was negative on the bottom four, so #3341 gave
    them an empty wrapper list: ``embed_text_enriched`` degrades to
    ``embed_text`` there and the Enrich Sort Descriptions setting can no longer
    cost anything.  These tests pin that split so the templates are not
    re-added by a later "every embedder should have wrappers" sweep.
    """

    #: Measured worse than the typed query on every template (#3127).
    PLAIN_QUERY_EMBEDDERS = ("siglip", "clap", "e5", "bge")

    #: The only two embedders with a positive point estimate on their own model.
    ENRICHING_EMBEDDERS = ("clap_general", "xclip")

    def test_measured_negative_embedders_have_no_wrappers(self):
        from vtscore.media import get_embedder

        for name in self.PLAIN_QUERY_EMBEDDERS:
            emb = get_embedder(name)
            assert emb is not None, f"{name} is not registered"
            assert emb.description_wrappers == [], (
                f"{name} measured worse than the typed query on every wrapper (#3127); "
                "an empty list here is the measured answer, not an unfilled slot"
            )

    def test_measured_positive_embedders_keep_their_wrappers(self):
        from vtscore.media import get_embedder

        for name in self.ENRICHING_EMBEDDERS:
            emb = get_embedder(name)
            assert emb is not None, f"{name} is not registered"
            wrappers = emb.description_wrappers
            assert len(wrappers) >= 3, f"{name} lost the wrappers #3127 measured a gain from"
            for w in wrappers:
                assert "{text}" in w

    def test_enriching_embedders_include_bare_text(self):
        """The ensemble must average the plain query back in."""
        from vtscore.media import get_embedder

        for name in self.ENRICHING_EMBEDDERS:
            assert "{text}" in get_embedder(name).description_wrappers, f"{name} missing bare '{{text}}' wrapper"

    def test_wrappers_format_correctly(self):
        """Every wrapper still in the tree must format without errors."""
        from vtscore.media import all_embedders

        for emb in all_embedders():
            for wrapper in emb.description_wrappers:
                assert "test query" in wrapper.format(text="test query")

    def test_no_wrappers_means_setting_is_a_no_op(self):
        """With no wrappers, enrich=True and enrich=False are the same query."""

        class PlainEmbedder(MediaEmbedder):
            @property
            def name(self):
                return "plain"

            @property
            def media_type_id(self):
                return "plain"

            def _load_models_impl(self):
                pass

            def _embed_media_impl(self, media):
                return None

            def embed_text(self, text):
                assert text == "boats", "the typed query must reach embed_text unwrapped"
                return np.array([1.0, 0.0])

        emb = PlainEmbedder()
        assert emb.description_wrappers == []
        np.testing.assert_array_equal(emb.embed_text_enriched("boats"), emb.embed_text("boats"))


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

            def _embed_media_impl(self, media):
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
        assert result is not None
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

        with patch("vtscore.embedding.helpers._get_embedder_for_media_type", return_value=FakeEmbedder()):
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

        with patch("vtscore.embedding.helpers._get_embedder_for_media_type", return_value=FakeEmbedder()):
            result = embed_text_query("test", "audio", enrich=True)
        np.testing.assert_array_equal(result, mock_vec)


# =====================================================================
# enrich_descriptions setting
# =====================================================================


class TestEnrichDescriptionsSetting:
    def test_default_is_false(self):
        """Off is a measured decision, not an oversight.

        Issue #3127 re-measured enrichment against every media-type default
        after #3077 moved the audio default to ``clap_general``: it is worth
        +0.014 +/- 0.009 AP there, inert on ``siglip``, and **-0.057 +/- 0.009
        on ``e5``** -- worse on 45 of 45 text categories, and on ``bge`` too.

        #3341 then removed the wrappers from the four embedders that measured
        negative, so the setting is no longer net-negative -- but the default
        still stays off, for a different reason: the only gain left standing,
        ``clap_general``'s +0.014, does not clear its own 2 SE, and ESC-50's 50
        categories are already fully used, so resolving it needs a second audio
        corpus.  See
        ``docs/experiments/2026-08-31-enrich-descriptions-3127/REPORT.md``
        before flipping this.
        """
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
            medias[media_id] = {
                "id": media_id,
                "embedder": "siglip",
                "embeddings": {"siglip": emb},
                "category": "cat",
                "media_type": "image",
            }
            media_id += 1
        dog_dir = np.zeros(16)
        dog_dir[1] = 1.0
        for _ in range(10):
            emb = dog_dir + rng.normal(0, 0.05, 16)
            emb /= np.linalg.norm(emb)
            medias[media_id] = {
                "id": media_id,
                "embedder": "siglip",
                "embeddings": {"siglip": emb},
                "category": "dog",
                "media_type": "image",
            }
            media_id += 1
        return medias, cat_dir, dog_dir

    def test_eval_text_sort_with_enrich(self):
        """eval_text_sort should pass enrich to embed_text_query."""
        medias, cat_dir, dog_dir = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat")]

        call_kwargs = []

        def mock_embed(text, media_type, enrich=False, embedder_name=""):
            call_kwargs.append({"enrich": enrich, "embedder_name": embedder_name})
            if "cat" in text:
                return cat_dir.copy()
            return dog_dir.copy()

        with patch("vtscore.embedding.helpers.embed_text_query", side_effect=mock_embed):
            results = eval_text_sort(medias, queries, "image", k_values=[5], enrich=True)

        assert len(results) == 1
        assert all(kw["enrich"] is True for kw in call_kwargs)
        # Called directly with no embedder_name, so it forwards the empty
        # default. run_eval is what fills it in; that is covered by
        # tests_lib/detectors/test_eval.py.
        assert all(kw["embedder_name"] == "" for kw in call_kwargs)

    def test_eval_text_sort_without_enrich(self):
        """eval_text_sort with enrich=False should pass enrich=False."""
        medias, cat_dir, dog_dir = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat")]

        call_kwargs = []

        def mock_embed(text, media_type, enrich=False, embedder_name=""):
            call_kwargs.append({"enrich": enrich, "embedder_name": embedder_name})
            return cat_dir.copy()

        with patch("vtscore.embedding.helpers.embed_text_query", side_effect=mock_embed):
            eval_text_sort(medias, queries, "image", k_values=[5], enrich=False)

        assert all(kw["enrich"] is False for kw in call_kwargs)


# =====================================================================
# The #3127 study harness vs. the tree it measures
# =====================================================================


class TestEnrichStudyCandidateWrappers:
    """`scripts/experiments/enrich/run_enrich.py` keeps its own wrapper table.

    It has to: #3341 emptied the list on the four embedders this very study
    measured negative, so a harness that read `description_wrappers` would
    find nothing on them, emit an `enriched` arm identical to `plain`, and
    report a flat zero while looking healthy. The table holds the pre-#3341
    templates so the report stays reproducible and a new checkpoint can be
    measured against a set it does not yet ship.

    That copy can rot, which is what these tests are for: on every media type
    where an embedder still ships wrappers, the table must be that same list.
    """

    @staticmethod
    def _module():
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "enrich" / "run_enrich.py"
        spec = importlib.util.spec_from_file_location("_enrich_run_enrich", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @pytest.mark.parametrize(
        ("media_type", "embedder_name"),
        [("audio", "clap_general"), ("image", "siglip2"), ("video", "xclip")],
    )
    def test_table_matches_the_embedders_that_still_ship_wrappers(self, media_type, embedder_name):
        from vtscore.media import get_embedder

        table = self._module().CANDIDATE_WRAPPERS
        assert table[media_type] == get_embedder(embedder_name).description_wrappers, (
            f"the study's {media_type} templates have drifted from {embedder_name}'s; "
            "the harness would measure a set nobody ships"
        )

    def test_every_media_type_has_a_candidate_set_with_the_identity(self):
        """The bare `{text}` arm is the analyzer's planted answer."""
        table = self._module().CANDIDATE_WRAPPERS
        assert set(table) == {"audio", "image", "text", "video"}
        for media_type, wrappers in table.items():
            assert len(wrappers) >= 3, media_type
            assert "{text}" in wrappers, f"{media_type} lost the identity arm"

    def test_emptied_embedders_resolve_to_the_candidate_set(self):
        """The four #3341 emptied are still measurable, and flagged as such."""
        from vtscore.media import get_embedder

        module = self._module()
        for name, media_type in (("siglip", "image"), ("clap", "audio"), ("e5", "text"), ("bge", "text")):
            wrappers, source = module._wrappers_for(get_embedder(name))
            assert source == "candidate", name
            assert wrappers == module.CANDIDATE_WRAPPERS[media_type], name

    def test_override_restores_the_embedders_own_property(self):
        """The class patch must not leak past the cell."""
        from vtscore.media import get_embedder

        module = self._module()
        emb = get_embedder("siglip")
        assert emb.description_wrappers == []
        with module._wrappers_override(emb, ["a photo of {text}"]):
            assert emb.description_wrappers == ["a photo of {text}"]
        assert emb.description_wrappers == []
