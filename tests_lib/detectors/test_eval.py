"""Tests for the vtscore.eval evaluation framework.

These tests exercise the metrics, config, and runner modules using
synthetic data; no real model downloads or embeddings required.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from vtscore.eval.config import EVAL_DATASETS, EvalQuery
from vtscore.eval.metrics import (
    DatasetResult,
    LearnedSortMetrics,
    QueryMetrics,
    compute_average_precision,
    compute_binary_classification_metrics,
    compute_metrics,
    compute_precision_recall_at_k,
)
from vtscore.eval.runner import (
    _cosine_similarity,
    eval_learned_sort,
    eval_text_sort,
    format_results_json,
    run_eval,
)


# =====================================================================
# Metrics: compute_average_precision
# =====================================================================


class TestAveragePrecision:
    def test_perfect_ranking(self):
        """All relevant items at the top -> AP = 1.0."""
        ranked = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert compute_average_precision(ranked, relevant) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """All relevant items at the bottom -> AP < 1.0."""
        ranked = [4, 5, 1, 2, 3]
        relevant = {1, 2, 3}
        ap = compute_average_precision(ranked, relevant)
        assert ap < 1.0
        assert ap > 0.0

    def test_single_relevant(self):
        """One relevant item at position k -> AP = 1/k."""
        ranked = [10, 20, 30, 1, 40]
        relevant = {1}
        assert compute_average_precision(ranked, relevant) == pytest.approx(1 / 4)

    def test_no_relevant(self):
        ranked = [1, 2, 3]
        relevant: set[int] = set()
        assert compute_average_precision(ranked, relevant) == 0.0

    def test_empty_ranking(self):
        ranked: list[int] = []
        relevant = {1, 2}
        assert compute_average_precision(ranked, relevant) == 0.0

    def test_interleaved(self):
        """Relevant at positions 1, 3, 5 out of 6 items."""
        ranked = [1, 10, 2, 20, 3, 30]
        relevant = {1, 2, 3}
        # P@1 = 1/1, P@3 = 2/3, P@5 = 3/5
        expected = (1 / 1 + 2 / 3 + 3 / 5) / 3
        assert compute_average_precision(ranked, relevant) == pytest.approx(expected)


# =====================================================================
# Metrics: precision / recall at k
# =====================================================================


class TestPrecisionRecallAtK:
    def test_default_k_values(self):
        ranked = list(range(1, 101))
        relevant = set(range(1, 11))  # first 10
        p, r = compute_precision_recall_at_k(ranked, relevant)
        assert set(p.keys()) == {5, 10, 20}
        assert p[5] == pytest.approx(1.0)
        assert p[10] == pytest.approx(1.0)
        assert p[20] == pytest.approx(0.5)
        assert r[5] == pytest.approx(0.5)
        assert r[10] == pytest.approx(1.0)
        assert r[20] == pytest.approx(1.0)

    def test_custom_k(self):
        ranked = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        p, r = compute_precision_recall_at_k(ranked, relevant, k_values=[2, 4])
        assert p[2] == pytest.approx(1 / 2)  # {1, 2} & {1, 3, 5} = {1}
        assert p[4] == pytest.approx(2 / 4)  # {1, 2, 3, 4} & {1, 3, 5} = {1, 3}

    def test_no_relevant(self):
        ranked = [1, 2, 3]
        relevant: set[int] = set()
        p, r = compute_precision_recall_at_k(ranked, relevant, k_values=[2])
        assert p[2] == 0.0
        assert r[2] == 0.0


# =====================================================================
# Metrics: binary classification
# =====================================================================


class TestBinaryClassification:
    def test_perfect(self):
        preds = [1, 1, 0, 0]
        labels = [1, 1, 0, 0]
        acc, prec, rec, f1 = compute_binary_classification_metrics(preds, labels)
        assert acc == 1.0
        assert prec == 1.0
        assert rec == 1.0
        assert f1 == 1.0

    def test_all_wrong(self):
        preds = [0, 0, 1, 1]
        labels = [1, 1, 0, 0]
        acc, prec, rec, f1 = compute_binary_classification_metrics(preds, labels)
        assert acc == 0.0
        assert prec == 0.0
        assert rec == 0.0
        assert f1 == 0.0

    def test_mixed(self):
        preds = [1, 0, 1, 0]
        labels = [1, 1, 0, 0]
        acc, prec, rec, f1 = compute_binary_classification_metrics(preds, labels)
        assert acc == pytest.approx(0.5)
        # tp=1, fp=1, fn=1, tn=1
        assert prec == pytest.approx(0.5)
        assert rec == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)

    def test_empty(self):
        acc, prec, rec, f1 = compute_binary_classification_metrics([], [])
        assert acc == 0.0

    def test_empty_logs_degenerate_warning(self, caplog):
        """An empty prediction set returns all-zero metrics, but logs a warning
        so the zeros aren't mistaken for a real evaluation (L7)."""
        import logging

        with caplog.at_level(logging.WARNING, logger="vtscore.eval.metrics"):
            acc, prec, rec, f1 = compute_binary_classification_metrics([], [])
        assert (acc, prec, rec, f1) == (0.0, 0.0, 0.0, 0.0)
        assert any("empty prediction set" in r.message for r in caplog.records)

    def test_nonempty_does_not_warn(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="vtscore.eval.metrics"):
            compute_binary_classification_metrics([1, 0], [1, 0])
        assert not any("empty prediction set" in r.message for r in caplog.records)


# =====================================================================
# Metrics: compute_metrics (integration)
# =====================================================================


class TestComputeMetrics:
    def test_returns_query_metrics(self):
        ranked = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        qm = compute_metrics(ranked, relevant, "test query", "test_cat", k_values=[2, 3])
        assert isinstance(qm, QueryMetrics)
        assert qm.query_text == "test query"
        assert qm.target_category == "test_cat"
        assert qm.average_precision == pytest.approx(1.0)
        assert qm.num_relevant == 2
        assert qm.num_total == 5
        assert 2 in qm.precision_at_k
        assert 3 in qm.recall_at_k


# =====================================================================
# Metrics: DatasetResult
# =====================================================================


class TestDatasetResult:
    def test_mean_average_precision(self):
        dr = DatasetResult(dataset_id="test", media_type="audio")
        dr.text_sort = [
            QueryMetrics("q1", "cat1", average_precision=0.8, num_relevant=5, num_total=20),
            QueryMetrics("q2", "cat2", average_precision=0.6, num_relevant=5, num_total=20),
        ]
        assert dr.mean_average_precision == pytest.approx(0.7)

    def test_mean_learned_f1(self):
        dr = DatasetResult(dataset_id="test", media_type="image")
        dr.learned_sort = [
            LearnedSortMetrics(accuracy=0.9, precision=0.8, recall=0.7, f1=0.75, num_train=10, num_test=10),
            LearnedSortMetrics(accuracy=0.85, precision=0.8, recall=0.9, f1=0.85, num_train=10, num_test=10),
        ]
        assert dr.mean_learned_f1 == pytest.approx(0.8)

    def test_empty_results(self):
        dr = DatasetResult(dataset_id="empty", media_type="audio")
        assert dr.mean_average_precision == 0.0
        assert dr.mean_learned_f1 == 0.0

    def test_to_dict_has_expected_keys(self):
        dr = DatasetResult(dataset_id="test", media_type="audio")
        dr.text_sort = [
            QueryMetrics(
                "q1",
                "cat1",
                average_precision=0.8,
                num_relevant=5,
                num_total=20,
                precision_at_k={5: 0.6},
                recall_at_k={5: 0.3},
            ),
        ]
        d = dr.to_dict()
        assert d["dataset_id"] == "test"
        assert d["media_type"] == "audio"
        assert "text_sort" in d
        assert d["text_sort"]["mAP"] == 0.8
        assert len(d["text_sort"]["per_query"]) == 1
        assert "elapsed_seconds" in d["text_sort"]["per_query"][0]

    def test_to_dict_learned(self):
        dr = DatasetResult(dataset_id="test", media_type="image")
        dr.learned_sort = [
            LearnedSortMetrics(
                accuracy=0.9, precision=0.8, recall=0.7, f1=0.75, num_train=10, num_test=10, target_category="cat1"
            ),
        ]
        d = dr.to_dict()
        assert "learned_sort" in d
        assert d["learned_sort"]["mean_f1"] == 0.75
        assert "elapsed_seconds" in d["learned_sort"]["per_category"][0]


# =====================================================================
# Config
# =====================================================================


class TestEvalConfig:
    def test_all_eval_datasets_have_queries(self):
        for ds_id, ds_cfg in EVAL_DATASETS.items():
            assert "queries" in ds_cfg, f"{ds_id} missing queries"
            assert len(ds_cfg["queries"]) > 0, f"{ds_id} has no queries"
            for q in ds_cfg["queries"]:
                assert isinstance(q, EvalQuery)
                assert q.text.strip(), f"{ds_id}: empty query text"
                assert q.target_category.strip(), f"{ds_id}: empty target_category"

    def test_all_eval_datasets_reference_demo_datasets(self):
        from vtscore.datasets.config import DEMO_DATASETS

        for ds_id, ds_cfg in EVAL_DATASETS.items():
            demo_id = ds_cfg["demo_dataset"]
            assert demo_id in DEMO_DATASETS, f"eval {ds_id} references missing demo dataset {demo_id}"

    def test_query_categories_match_demo_categories(self):
        """Every query's target_category must appear in the demo dataset's category list."""
        from vtscore.datasets.config import DEMO_DATASETS

        for ds_id, ds_cfg in EVAL_DATASETS.items():
            demo_id = ds_cfg["demo_dataset"]
            demo_cats = set(DEMO_DATASETS[demo_id]["categories"])
            for q in ds_cfg["queries"]:
                assert q.target_category in demo_cats, (
                    f"eval {ds_id}: query target {q.target_category!r} not in demo categories {demo_cats}"
                )


# =====================================================================
# Runner: _cosine_similarity
# =====================================================================


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert _cosine_similarity(a, b) == 0.0


# =====================================================================
# Runner: eval_text_sort with synthetic medias
# =====================================================================


class TestEvalTextSort:
    """Test eval_text_sort using mocked embeddings."""

    def _make_synthetic_clips(self):
        """Create medias with known embeddings: cats point one way, dogs another."""
        rng = np.random.RandomState(0)
        medias = {}
        media_id = 1
        # "cat" medias cluster around [1, 0, 0, ...]
        cat_dir = np.zeros(16)
        cat_dir[0] = 1.0
        for _ in range(10):
            emb = cat_dir + rng.normal(0, 0.05, 16)
            emb /= np.linalg.norm(emb)
            medias[media_id] = {"id": media_id, "embeddings": {"siglip": emb}, "category": "cat", "media_type": "image"}
            media_id += 1
        # "dog" medias cluster around [0, 1, 0, ...]
        dog_dir = np.zeros(16)
        dog_dir[1] = 1.0
        for _ in range(10):
            emb = dog_dir + rng.normal(0, 0.05, 16)
            emb /= np.linalg.norm(emb)
            medias[media_id] = {"id": media_id, "embeddings": {"siglip": emb}, "category": "dog", "media_type": "image"}
            media_id += 1
        return medias, cat_dir, dog_dir

    def test_text_sort_separates_categories(self):
        medias, cat_dir, dog_dir = self._make_synthetic_clips()

        queries = [
            EvalQuery("a cat", "cat"),
            EvalQuery("a dog", "dog"),
        ]

        # Mock embed_text_query to return the cluster centre
        def mock_embed(text, media_type, enrich=False, embedder_name=""):
            if "cat" in text:
                return cat_dir.copy()
            return dog_dir.copy()

        with patch("vtscore.embedding.helpers.embed_text_query", side_effect=mock_embed):
            results = eval_text_sort(medias, queries, "image", k_values=[5, 10])

        assert len(results) == 2
        # With clean clusters, AP should be very high
        for qm in results:
            assert qm.average_precision > 0.9
            assert qm.num_relevant == 10

    def test_text_sort_returns_correct_fields(self):
        medias, cat_dir, _ = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat")]

        with patch("vtscore.embedding.helpers.embed_text_query", return_value=cat_dir.copy()):
            results = eval_text_sort(medias, queries, "image", k_values=[5])

        qm = results[0]
        assert qm.query_text == "a cat"
        assert qm.target_category == "cat"
        assert 5 in qm.precision_at_k
        assert 5 in qm.recall_at_k
        assert qm.num_total == 20

    def test_text_sort_elapsed_seconds_without_start_time(self):
        """Without start_time, elapsed_seconds defaults to 0."""
        medias, cat_dir, _ = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat")]
        with patch("vtscore.embedding.helpers.embed_text_query", return_value=cat_dir.copy()):
            results = eval_text_sort(medias, queries, "image", k_values=[5])
        assert results[0].elapsed_seconds == 0.0

    def test_text_sort_elapsed_seconds_with_start_time(self):
        """With start_time, elapsed_seconds is populated and non-negative."""
        import time

        medias, cat_dir, _ = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat"), EvalQuery("a dog", "dog")]

        def mock_embed(text, media_type, enrich=False, embedder_name=""):
            return cat_dir.copy()

        start = time.monotonic()
        with patch("vtscore.embedding.helpers.embed_text_query", side_effect=mock_embed):
            results = eval_text_sort(medias, queries, "image", k_values=[5], start_time=start)
        for qm in results:
            assert qm.elapsed_seconds >= 0.0
        # Second query should have equal or later timestamp
        assert results[1].elapsed_seconds >= results[0].elapsed_seconds

    def test_query_is_embedded_with_the_datasets_embedder(self):
        """The query must land in the *dataset's* space, not the default one.

        Without this the harness embedded queries with the media type's
        default embedder while the medias carried vectors from whatever
        ``--embedder`` asked for. Cosine between two unrelated 512-d spaces
        is noise, so every non-default arm silently reported near-chance mAP
        while looking like a real measurement.
        """
        medias, cat_dir, _ = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat")]
        seen = []

        def mock_embed(text, media_type, enrich=False, embedder_name=""):
            seen.append(embedder_name)
            return cat_dir.copy()

        with patch("vtscore.embedding.helpers.embed_text_query", side_effect=mock_embed):
            eval_text_sort(medias, queries, "image", k_values=[5], embedder_name="clap_general")
        assert seen == ["clap_general"]

    def test_loaded_embedder_name_is_read_off_the_medias(self):
        """Resolved from the medias, not the flag, which is empty by default."""
        from vtscore.eval.runner import _loaded_embedder_name

        medias = {
            1: {"embedder": "clap_general", "embeddings": {"clap_general": np.zeros(4, dtype=np.float32)}},
        }
        assert _loaded_embedder_name(medias) == "clap_general"
        assert _loaded_embedder_name({}) == ""


# =====================================================================
# Runner: eval_learned_sort with synthetic medias
# =====================================================================


class TestEvalLearnedSort:
    def _make_synthetic_clips(self, dim=16, n_per_cat=20):
        """Two categories with separable embeddings."""
        rng = np.random.RandomState(42)
        medias = {}
        media_id = 1
        for _ in range(n_per_cat):
            emb = rng.normal(1.0, 0.3, dim).astype(np.float32)
            medias[media_id] = {
                "id": media_id,
                "embeddings": {"siglip": emb},
                "category": "cat_a",
                "media_type": "image",
            }
            media_id += 1
        for _ in range(n_per_cat):
            emb = rng.normal(-1.0, 0.3, dim).astype(np.float32)
            medias[media_id] = {
                "id": media_id,
                "embeddings": {"siglip": emb},
                "category": "cat_b",
                "media_type": "image",
            }
            media_id += 1
        return medias

    def test_learned_sort_returns_metrics(self):
        medias = self._make_synthetic_clips()
        queries = [EvalQuery("category a stuff", "cat_a")]
        results = eval_learned_sort(medias, queries, train_fraction=0.5, seed=42)
        assert len(results) == 1
        lm = results[0]
        assert lm.target_category == "cat_a"
        assert 0.0 <= lm.accuracy <= 1.0
        assert 0.0 <= lm.f1 <= 1.0
        assert lm.num_train > 0
        assert lm.num_test > 0

    def test_learned_sort_well_separated_categories(self):
        """With well-separated embeddings, learned sort should get high F1."""
        medias = self._make_synthetic_clips(n_per_cat=30)
        queries = [EvalQuery("a", "cat_a")]
        results = eval_learned_sort(medias, queries, train_fraction=0.5, seed=42)
        assert results[0].f1 > 0.7  # generous threshold for small synthetic data

    def test_learned_sort_skips_tiny_categories(self):
        """Categories with < 2 medias should be skipped."""
        medias = {
            1: {
                "id": 1,
                "embeddings": {"siglip": np.ones(8, dtype=np.float32)},
                "category": "rare",
                "media_type": "image",
            },
            2: {
                "id": 2,
                "embeddings": {"siglip": -np.ones(8, dtype=np.float32)},
                "category": "common",
                "media_type": "image",
            },
        }
        queries = [EvalQuery("rare stuff", "rare")]
        results = eval_learned_sort(medias, queries, train_fraction=0.5)
        assert len(results) == 0  # skipped due to too few medias

    def test_learned_sort_elapsed_seconds(self):
        """With start_time, elapsed_seconds is populated on results."""
        import time

        medias = self._make_synthetic_clips()
        queries = [EvalQuery("category a stuff", "cat_a")]
        start = time.monotonic()
        results = eval_learned_sort(medias, queries, train_fraction=0.5, seed=42, start_time=start)
        assert len(results) == 1
        assert results[0].elapsed_seconds > 0.0


# =====================================================================
# Runner: format_results_json
# =====================================================================


class TestFormatResults:
    def test_valid_json(self):
        dr = DatasetResult(dataset_id="test", media_type="audio")
        dr.text_sort = [
            QueryMetrics(
                "q1",
                "cat1",
                average_precision=0.9,
                num_relevant=5,
                num_total=20,
                precision_at_k={5: 0.8},
                recall_at_k={5: 0.4},
            ),
        ]
        result = format_results_json([dr])
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["dataset_id"] == "test"


# =====================================================================
# Multi-label ground truth (Visual Genome)
# =====================================================================


class TestMediaIsPositive:
    """The shared membership test used by every evaluator."""

    def test_single_label_exact_match(self):
        from vtscore.eval.labels import media_is_positive

        media = {"category": "dog"}
        assert media_is_positive(media, "dog") is True
        assert media_is_positive(media, "cat") is False

    def test_multi_label_set_membership(self):
        from vtscore.eval.labels import media_is_positive

        media = {"category": "man", "categories": ["man", "apple"]}
        assert media_is_positive(media, "man") is True
        assert media_is_positive(media, "apple") is True
        # Closed-world: a category absent from the list is a negative.
        assert media_is_positive(media, "banana") is False

    def test_empty_categories_list_is_all_negative(self):
        from vtscore.eval.labels import media_is_positive

        # A present-but-empty list means the image positively matches nothing;
        # it is NOT treated as single-label fallback.
        media = {"category": "man", "categories": []}
        assert media_is_positive(media, "man") is False


class TestEvalMultiLabel:
    """eval_text_sort / eval_learned_sort over multi-label medias."""

    def _make_multilabel_clips(self, n_per_cat=30):
        """man/apple images point one way, banana images another.

        Each "man eating an apple" image is a positive for BOTH man and apple;
        banana images are positive for banana only.  This is the wrinkle that
        the single-label datasets can't express.  Clusters are well-separated
        (mean +1 vs -1 across all dims) so the learned sort is reliable.
        """
        rng = np.random.RandomState(42)
        medias = {}
        media_id = 1
        for _ in range(n_per_cat):
            emb = rng.normal(1.0, 0.3, 16).astype(np.float32)
            medias[media_id] = {
                "id": media_id,
                "embeddings": {"siglip": emb},
                "category": "man",
                "categories": ["man", "apple"],
                "media_type": "image",
            }
            media_id += 1
        for _ in range(n_per_cat):
            emb = rng.normal(-1.0, 0.3, 16).astype(np.float32)
            medias[media_id] = {
                "id": media_id,
                "embeddings": {"siglip": emb},
                "category": "banana",
                "categories": ["banana"],
                "media_type": "image",
            }
            media_id += 1
        return medias

    def test_text_sort_counts_overlapping_positives(self):
        medias = self._make_multilabel_clips()
        # "man" and "apple" target the SAME 30 images even though they are
        # different category strings — the multi-label win the single-label
        # model can't represent.
        queries = [EvalQuery("a man", "man"), EvalQuery("an apple", "apple")]

        with patch("vtscore.embedding.helpers.embed_text_query", return_value=np.ones(16, dtype=np.float32)):
            results = eval_text_sort(medias, queries, "image", k_values=[5])

        assert len(results) == 2
        for qm in results:
            assert qm.num_relevant == 30
            assert qm.average_precision > 0.9

    def test_learned_sort_splits_on_multilabel(self):
        medias = self._make_multilabel_clips()
        # "apple" has 30 positives (the man+apple images) and 30 negatives
        # (the banana images) under closed-world — the split is driven by the
        # categories list, not the single "category" string.
        results = eval_learned_sort(medias, [EvalQuery("an apple", "apple")], seed=42)
        assert len(results) == 1
        assert results[0].f1 > 0.7  # generous threshold for small synthetic data


class TestRegionBoxForCategory:
    """Ground-truth box lookup that feeds region voting in the eval harness."""

    def test_single_box(self):
        from vtscore.eval.labels import region_box_for_category

        media = {"regions": [{"box": [0.1, 0.2, 0.3, 0.4], "label": "apple"}]}
        assert region_box_for_category(media, "apple") == (0.1, 0.2, 0.3, 0.4)

    def test_multiple_boxes_returns_minimal_cover(self):
        from vtscore.eval.labels import region_box_for_category

        # Two apples in one image -> the smallest box covering both, not an
        # arbitrary pick of one.
        media = {
            "regions": [
                {"box": [0.10, 0.20, 0.30, 0.40], "label": "apple"},
                {"box": [0.50, 0.10, 0.90, 0.60], "label": "apple"},
                {"box": [0.00, 0.00, 1.00, 1.00], "label": "man"},  # different label, ignored
            ]
        }
        assert region_box_for_category(media, "apple") == (0.10, 0.10, 0.90, 0.60)

    def test_no_regions_returns_none(self):
        from vtscore.eval.labels import region_box_for_category

        assert region_box_for_category({"category": "apple"}, "apple") is None
        assert region_box_for_category({"regions": []}, "apple") is None

    def test_no_matching_label_returns_none(self):
        from vtscore.eval.labels import region_box_for_category

        media = {"regions": [{"box": [0.1, 0.2, 0.3, 0.4], "label": "man"}]}
        assert region_box_for_category(media, "apple") is None


class TestVotedBoxScale:
    """Scale of the region a Good vote actually drags (the union box).

    The distinction from per-instance area is the whole point: a multi-instance
    category has tiny instances but a near-frame union box, and the union is
    what the detector trains and scores against.
    """

    def test_voted_area_is_the_union_not_an_instance(self):
        from vtscore.eval.labels import voted_box_area

        # Two 1%-area instances at opposite corners -> a ~64% union box.
        media = {
            "regions": [
                {"box": [0.10, 0.10, 0.20, 0.20], "label": "arm"},
                {"box": [0.80, 0.80, 0.90, 0.90], "label": "arm"},
            ]
        }
        assert voted_box_area(media, "arm") == pytest.approx(0.64)

    def test_voted_area_none_without_a_box(self):
        from vtscore.eval.labels import voted_box_area

        assert voted_box_area({"category": "apple"}, "apple") is None
        assert voted_box_area({"regions": [{"box": [0, 0, 1, 1], "label": "man"}]}, "apple") is None

    def test_instance_areas_are_per_box(self):
        from vtscore.eval.labels import instance_box_areas

        media = {
            "regions": [
                {"box": [0.10, 0.10, 0.20, 0.20], "label": "arm"},
                {"box": [0.80, 0.80, 0.90, 0.90], "label": "arm"},
                {"box": [0.00, 0.00, 1.00, 1.00], "label": "sky"},
            ]
        }
        assert instance_box_areas(media, "arm") == pytest.approx([0.01, 0.01])
        assert instance_box_areas(media, "sky") == pytest.approx([1.0])

    def test_union_inflation_separates_scattered_from_single_object(self):
        from vtscore.eval.labels import category_scale_stats

        scattered = {
            i: {
                "regions": [
                    {"box": [0.10, 0.10, 0.20, 0.20], "label": "arm"},
                    {"box": [0.80, 0.80, 0.90, 0.90], "label": "arm"},
                ]
            }
            for i in range(5)
        }
        single = {i: {"regions": [{"box": [0.30, 0.30, 0.70, 0.70], "label": "bed"}]} for i in range(5)}

        s_arm = category_scale_stats(scattered, "arm")
        s_bed = category_scale_stats(single, "bed")
        assert s_arm is not None and s_bed is not None
        # A single-object category's vote IS its instance: inflation ~1.
        assert s_bed["union_inflation"] == pytest.approx(1.0)
        assert s_bed["voted_area"] == pytest.approx(0.16)
        # The scattered category looks tiny per instance but votes a huge box -
        # exactly the case that used to plot at the small end of the scale axis.
        assert s_arm["instance_area"] == pytest.approx(0.01)
        assert s_arm["voted_area"] == pytest.approx(0.64)
        assert s_arm["union_inflation"] == pytest.approx(64.0)

    def test_stats_none_when_category_unboxed(self):
        from vtscore.eval.labels import category_scale_stats

        assert category_scale_stats({1: {"category": "apple"}}, "apple") is None


class TestRegionVotingLearnedSort:
    """eval_learned_sort wires ground-truth boxes into train_and_score."""

    def _make_region_clips(self, n_per_cat=10):
        """Positive (apple) images carry a ground-truth box; negatives don't."""
        rng = np.random.RandomState(7)
        medias = {}
        media_id = 1
        for _ in range(n_per_cat):
            emb = rng.normal(1.0, 0.3, 16).astype(np.float32)
            medias[media_id] = {
                "id": media_id,
                "embeddings": {"dinov3_patch": emb},
                "category": "apple",
                "media_type": "image",
                "regions": [{"box": [0.2, 0.2, 0.6, 0.6], "label": "apple"}],
            }
            media_id += 1
        for _ in range(n_per_cat):
            emb = rng.normal(-1.0, 0.3, 16).astype(np.float32)
            medias[media_id] = {
                "id": media_id,
                "embeddings": {"dinov3_patch": emb},
                "category": "other",
                "media_type": "image",
            }
            media_id += 1
        return medias

    def test_passes_groundtruth_boxes_only_when_region_voting(self):
        captured: list[dict | None] = []

        def _fake_train_and_score(medias, good, bad, **kwargs):
            captured.append(kwargs.get("vote_region_boxes"))
            # Score every media as its embedding's first component (separable).
            scored = [{"id": cid, "score": 1.0 if medias[cid]["category"] == "apple" else 0.0} for cid in medias]
            return scored, 0.5, None

        medias = self._make_region_clips()
        query = [EvalQuery("an apple", "apple")]

        with patch("vtscore.detectors.training.train_and_score", _fake_train_and_score):
            eval_learned_sort(medias, query, train_fraction=0.5, seed=1, region_voting=True)
            eval_learned_sort(medias, query, train_fraction=0.5, seed=1, region_voting=False)

        boxes_on, boxes_off = captured
        # Region voting: every trained Good (apple) media contributes its box.
        assert boxes_on
        assert all(b == (0.2, 0.2, 0.6, 0.6) for b in boxes_on.values())
        # Baseline: no boxes passed at all.
        assert boxes_off is None


# =====================================================================
# Runner: failure paths
# =====================================================================


class TestEvalTextSortFailures:
    """eval_text_sort's error branches: unembeddable query, empty medias."""

    def _one_cat_clips(self):
        rng = np.random.RandomState(0)
        medias = {}
        for media_id in range(1, 6):
            emb = rng.standard_normal(16).astype(np.float32)
            medias[media_id] = {
                "id": media_id,
                "embeddings": {"siglip": emb},
                "category": "cat",
                "media_type": "image",
            }
        return medias

    def test_unembeddable_query_raises(self):
        """When ``embed_text_query`` returns ``None`` (no embedder for the media
        type, or an embed failure), the query cannot be ranked and the runner
        raises rather than silently scoring everything at zero."""
        medias = self._one_cat_clips()
        queries = [EvalQuery("a cat", "cat")]

        with patch("vtscore.embedding.helpers.embed_text_query", return_value=None):
            with pytest.raises(RuntimeError, match="Could not embed query"):
                eval_text_sort(medias, queries, "image")

    def test_empty_medias_yields_zero_metrics(self):
        """An empty media set (e.g. a category with nothing loaded) must not
        crash: the query ranks an empty list and reports zeroed metrics."""
        queries = [EvalQuery("a cat", "cat")]
        with patch("vtscore.embedding.helpers.embed_text_query", return_value=np.ones(16, dtype=np.float32)):
            results = eval_text_sort({}, queries, "image", k_values=[5])
        assert len(results) == 1
        qm = results[0]
        assert qm.num_total == 0
        assert qm.num_relevant == 0
        assert qm.average_precision == 0.0

    def test_query_targeting_absent_category_is_all_negative(self):
        """A query whose target category has no medias reports zero relevant
        and AP 0 (closed-world), without raising."""
        medias = self._one_cat_clips()
        queries = [EvalQuery("a dog", "dog")]  # no "dog" medias exist
        with patch("vtscore.embedding.helpers.embed_text_query", return_value=np.ones(16, dtype=np.float32)):
            results = eval_text_sort(medias, queries, "image", k_values=[5])
        assert results[0].num_relevant == 0
        assert results[0].average_precision == 0.0


class TestEvalLearnedSortFailures:
    """eval_learned_sort's skip branches: empty test set, absent category."""

    def _two_by_two_clips(self):
        """Exactly 2 target + 2 other medias — the minimum that passes the
        ``< 2`` guard, so the empty-test-set branch is what does the skipping."""
        rng = np.random.RandomState(3)
        medias = {}
        media_id = 1
        for _ in range(2):
            medias[media_id] = {
                "id": media_id,
                "embeddings": {"siglip": rng.standard_normal(8).astype(np.float32)},
                "category": "cat_a",
                "media_type": "image",
            }
            media_id += 1
        for _ in range(2):
            medias[media_id] = {
                "id": media_id,
                "embeddings": {"siglip": rng.standard_normal(8).astype(np.float32)},
                "category": "cat_b",
                "media_type": "image",
            }
            media_id += 1
        return medias

    def test_empty_test_set_is_skipped(self):
        """With ``train_fraction=1.0`` every media lands in the train split, so
        the held-out test set is empty and the category is skipped before any
        model is trained."""
        medias = self._two_by_two_clips()
        queries = [EvalQuery("category a", "cat_a")]
        results = eval_learned_sort(medias, queries, train_fraction=1.0, seed=42)
        assert results == []

    def test_absent_category_is_skipped(self):
        """A query whose target category has zero positives (fewer than 2
        medias) is skipped, not evaluated on an all-negative pool."""
        medias = self._two_by_two_clips()
        queries = [EvalQuery("a rare thing", "cat_zzz")]
        results = eval_learned_sort(medias, queries, train_fraction=0.5, seed=42)
        assert results == []


class TestRunEvalFailures:
    """run_eval's dataset-resolution and load-failure branches all *skip* the
    offending dataset and keep going, returning results only for the datasets
    that resolved and loaded."""

    def test_unknown_eval_dataset_id_is_skipped(self, capsys):
        results = run_eval(dataset_ids=["definitely_not_a_dataset"], mode="text")
        assert results == []
        assert "unknown eval dataset" in capsys.readouterr().err

    def test_missing_demo_dataset_is_skipped(self, capsys):
        """An eval config that references a demo dataset which isn't registered
        is skipped with a warning (guards against a stale EVAL_DATASETS entry)."""
        from unittest.mock import patch as _patch

        from vtscore.eval.config import EVAL_DATASETS

        fake = {"demo_dataset": "no_such_demo", "queries": [EvalQuery("x", "y")]}
        with _patch.dict(EVAL_DATASETS, {"fake_eval": fake}, clear=False):
            results = run_eval(dataset_ids=["fake_eval"], mode="text")
        assert results == []
        assert "not found" in capsys.readouterr().err

    def test_dataset_load_failure_is_skipped(self, capsys):
        """If ``load_demo_dataset`` raises (download/embed failure), the dataset
        is skipped with an ERROR line rather than aborting the whole run."""
        from unittest.mock import patch as _patch

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated download failure")

        with _patch("vtscore.datasets.loader.load_demo_dataset", side_effect=_boom):
            results = run_eval(dataset_ids=["esc50_s"], mode="text")
        assert results == []
        assert "ERROR loading dataset" in capsys.readouterr().err
