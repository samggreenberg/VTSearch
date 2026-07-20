"""Unit tests for the windowed sort-results cache (scalability.md S3/S17/S19).

Library tier: the cache is Flask-free process-global state, so it belongs in
``tests_lib``.  Covers score-key handling, above-threshold counting, window
slicing, LRU eviction, the sort-generation (token) guard, and the per-dataset
gate.
"""

from __future__ import annotations

from vtscore.state.sort_results_cache import (
    SORT_WINDOW_HEAD,
    SORT_WINDOW_TAIL,
    SortResultsCache,
    count_above_threshold,
    initial_window_end,
    result_score,
)


class TestInitialWindowEnd:
    def test_boundary_within_head_includes_tail(self):
        # 300 above (< HEAD): window = 300 + TAIL, boundary visible.
        assert initial_window_end(total=10000, above_threshold=300) == 300 + SORT_WINDOW_TAIL

    def test_boundary_past_head_caps_at_head_plus_tail(self):
        # 100k above (> HEAD): window caps at HEAD + TAIL, boundary paged to.
        assert initial_window_end(total=1_000_000, above_threshold=100_000) == SORT_WINDOW_HEAD + SORT_WINDOW_TAIL

    def test_clamped_to_total(self):
        assert initial_window_end(total=5, above_threshold=2) == 5

    def test_zero_above_is_tail_only(self):
        assert initial_window_end(total=10000, above_threshold=0) == SORT_WINDOW_TAIL


class TestResultScore:
    def test_prefers_score_key(self):
        assert result_score({"id": 1, "score": 0.9, "similarity": 0.1}) == 0.9

    def test_falls_back_to_similarity(self):
        assert result_score({"id": 1, "similarity": 0.7}) == 0.7

    def test_missing_both_sorts_below_any_threshold(self):
        assert result_score({"id": 1}) == float("-inf")


class TestCountAboveThreshold:
    def test_counts_ge_threshold(self):
        results = [{"id": i, "score": s} for i, s in enumerate([0.9, 0.8, 0.5, 0.2])]
        assert count_above_threshold(results, 0.5) == 3  # 0.9, 0.8, 0.5 (>=)

    def test_none_threshold_counts_all(self):
        results = [{"id": 1, "score": 0.1}, {"id": 2, "score": 0.2}]
        assert count_above_threshold(results, None) == 2

    def test_similarity_rows_counted(self):
        results = [{"id": 1, "similarity": 0.8}, {"id": 2, "similarity": 0.3}]
        assert count_above_threshold(results, 0.5) == 1


def _ranking(n: int) -> list[dict]:
    """A descending ranking of *n* rows: id i has score (n - i) / n."""
    return [{"id": i, "score": (n - i) / n} for i in range(n)]


class TestSortResultsCache:
    def test_store_returns_distinct_tokens(self):
        cache = SortResultsCache()
        t1 = cache.store(_ranking(3), 0.5)
        t2 = cache.store(_ranking(3), 0.5)
        assert t1 != t2

    def test_page_slices_the_window(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(10), 0.5)
        page = cache.page(token, offset=2, limit=3)
        assert page is not None
        assert [r["id"] for r in page["results"]] == [2, 3, 4]
        assert page["offset"] == 2
        assert page["limit"] == 3
        assert page["total"] == 10
        assert page["threshold"] == 0.5
        assert page["has_more"] is True

    def test_page_last_window_has_no_more(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(10), 0.5)
        page = cache.page(token, offset=8, limit=5)
        assert page is not None
        assert [r["id"] for r in page["results"]] == [8, 9]
        assert page["has_more"] is False

    def test_page_offset_past_end_is_empty(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(3), 0.5)
        page = cache.page(token, offset=99, limit=10)
        assert page is not None
        assert page["results"] == []
        assert page["has_more"] is False

    def test_unknown_token_returns_none(self):
        cache = SortResultsCache()
        assert cache.page("does-not-exist", offset=0, limit=10) is None

    def test_lru_evicts_oldest(self):
        cache = SortResultsCache(max_entries=2)
        t1 = cache.store(_ranking(1), 0.5)
        t2 = cache.store(_ranking(1), 0.5)
        t3 = cache.store(_ranking(1), 0.5)  # evicts t1
        assert cache.page(t1, 0, 10) is None
        assert cache.page(t2, 0, 10) is not None
        assert cache.page(t3, 0, 10) is not None

    def test_paging_keeps_entry_warm_against_eviction(self):
        cache = SortResultsCache(max_entries=2)
        t1 = cache.store(_ranking(1), 0.5)
        cache.store(_ranking(1), 0.5)  # t2
        cache.page(t1, 0, 10)  # touch t1 -> now most-recently-used
        cache.store(_ranking(1), 0.5)  # t3 evicts the LRU, which is now t2 not t1
        assert cache.page(t1, 0, 10) is not None

    def test_dataset_gate_blocks_cross_dataset_paging(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(3), 0.5, dataset_id="ds-A")
        assert cache.page(token, 0, 10, dataset_id="ds-B") is None
        assert cache.page(token, 0, 10, dataset_id="ds-A") is not None

    def test_dataset_gate_skipped_when_entry_unscoped(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(3), 0.5)  # no dataset_id
        assert cache.page(token, 0, 10, dataset_id="ds-anything") is not None

    def test_reset_for_tests_clears(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(3), 0.5)
        cache.reset_for_tests()
        assert cache.page(token, 0, 10) is None
