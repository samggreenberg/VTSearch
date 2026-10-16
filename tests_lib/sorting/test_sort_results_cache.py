"""Unit tests for the windowed sort-results cache (scalability.md S3/S17/S19).

Library tier: the cache is Flask-free process-global state, so it belongs in
``tests_lib``.  Covers score-key handling, above-threshold counting, window
slicing, LRU eviction by entries and by rows, the sort-generation (token)
guard, and the per-dataset gate.
"""

from __future__ import annotations

from vtscore.state.sort_results_cache import (
    SORT_CACHE_MAX_ROWS,
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


class TestRowCap:
    """The rows held across all rankings are bounded too, not only their count (#4978)."""

    def test_evicts_the_oldest_until_under_the_cap(self):
        cache = SortResultsCache(max_rows=10)
        t1 = cache.store(_ranking(3), 0.5)
        t2 = cache.store(_ranking(3), 0.5)
        t3 = cache.store(_ranking(3), 0.5)
        t4 = cache.store(_ranking(6), 0.5)  # 15 rows: t1 and t2 go, leaving 9
        assert cache.page(t1, 0, 10) is None
        assert cache.page(t2, 0, 10) is None
        assert cache.page(t3, 0, 10) is not None
        assert cache.page(t4, 0, 10) is not None

    def test_the_newest_survives_alone_over_the_cap(self):
        cache = SortResultsCache(max_rows=10)
        t1 = cache.store(_ranking(3), 0.5)
        t2 = cache.store(_ranking(25), 0.5)
        assert cache.page(t1, 0, 10) is None
        page = cache.page(t2, offset=20, limit=10)
        assert page is not None
        assert page["total"] == 25
        assert [r["id"] for r in page["results"]] == [20, 21, 22, 23, 24]

    def test_paging_keeps_a_ranking_warm_against_the_row_cap(self):
        cache = SortResultsCache(max_rows=10)
        t1 = cache.store(_ranking(4), 0.5)
        t2 = cache.store(_ranking(4), 0.5)
        cache.page(t1, 0, 10)  # touch t1 -> now most-recently-used
        cache.store(_ranking(4), 0.5)  # 12 rows: the LRU, now t2, goes
        assert cache.page(t1, 0, 10) is not None
        assert cache.page(t2, 0, 10) is None

    def test_the_entry_cap_still_applies_under_the_row_cap(self):
        cache = SortResultsCache(max_entries=2, max_rows=100)
        t1 = cache.store(_ranking(1), 0.5)
        cache.store(_ranking(1), 0.5)
        cache.store(_ranking(1), 0.5)
        assert cache.page(t1, 0, 10) is None

    def test_the_default_cap_is_sort_cache_max_rows(self):
        # One shared row repeated: a list of pointers, so this stays cheap.
        cache = SortResultsCache()
        t1 = cache.store([{"id": 0, "score": 1.0}] * SORT_CACHE_MAX_ROWS, 0.5)
        t2 = cache.store(_ranking(1), 0.5)  # one row over the cap: t1 goes
        assert cache.page(t1, 0, 10) is None
        assert cache.page(t2, 0, 10) is not None


class _Line:
    """A stand-in for a sort's line: at *beta* it sits at ``beta / 10``."""

    def __init__(self) -> None:
        self.asked: list[float | None] = []

    def threshold_at(self, beta: float | None) -> float:
        self.asked.append(beta)
        return (beta or 0.0) / 10


class TestRedraw:
    """A text or example sort's line, redrawn over the stored ranking at another balance (#4760)."""

    def test_redraws_the_stored_line_at_the_balance(self):
        cache = SortResultsCache()
        line = _Line()
        token = cache.store(_ranking(10), 0.9, line=line)
        drawn = cache.redraw(token, 5.0)
        # Scores run 1.0, 0.9, ... 0.1: six sit at or above 0.5.
        assert drawn == {"threshold": 0.5, "above_threshold": 6, "total": 10}
        assert line.asked == [5.0]

    def test_a_page_reports_the_redrawn_line_over_the_same_ranking(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(10), 0.9, line=_Line())
        cache.redraw(token, 3.0)
        page = cache.page(token, offset=0, limit=10)
        assert page is not None
        assert page["threshold"] == 0.3
        assert [r["id"] for r in page["results"]] == list(range(10))

    def test_a_sort_stored_without_a_line_is_not_redrawn(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(3), 0.5)
        assert cache.redraw(token, 1.0) is None
        page = cache.page(token, 0, 10)
        assert page is not None
        assert page["threshold"] == 0.5

    def test_unknown_token_is_not_redrawn(self):
        assert SortResultsCache().redraw("does-not-exist", 1.0) is None

    def test_dataset_gate_blocks_a_cross_dataset_redraw(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(3), 0.5, dataset_id="ds-A", line=_Line())
        assert cache.redraw(token, 1.0, dataset_id="ds-B") is None
        assert cache.redraw(token, 1.0, dataset_id="ds-A") is not None

    def test_redrawing_keeps_the_entry_warm_against_eviction(self):
        cache = SortResultsCache(max_entries=2)
        t1 = cache.store(_ranking(1), 0.5, line=_Line())
        cache.store(_ranking(1), 0.5)  # t2
        cache.redraw(t1, 1.0)  # touch t1
        cache.store(_ranking(1), 0.5)  # t3 evicts t2, not t1
        assert cache.page(t1, 0, 10) is not None


class TestDerived:
    """A view built once over a stored ranking and evicted with it (#4751)."""

    def test_builds_once_per_token_and_key(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(4), 0.5)
        calls: list[int] = []

        def build(results: list[dict]) -> list[int]:
            calls.append(len(results))
            return [r["id"] for r in results]

        assert cache.derived(token, "ids", build) == (True, [0, 1, 2, 3])
        assert cache.derived(token, "ids", build) == (True, [0, 1, 2, 3])
        assert calls == [4]
        cache.derived(token, "other", build)
        assert calls == [4, 4]

    def test_a_new_sort_builds_afresh(self):
        cache = SortResultsCache()
        t1 = cache.store(_ranking(2), 0.5)
        t2 = cache.store(_ranking(3), 0.5)
        assert cache.derived(t1, "n", len) == (True, 2)
        assert cache.derived(t2, "n", len) == (True, 3)

    def test_a_failed_build_keeps_nothing(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(2), 0.5)

        def boom(results: list[dict]) -> int:
            raise LookupError("not yet")

        try:
            cache.derived(token, "n", boom)
        except LookupError:
            pass
        assert cache.derived(token, "n", len) == (True, 2)

    def test_forget_drops_the_view(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(2), 0.5)
        cache.derived(token, "n", lambda results: "first")
        cache.forget_derived(token, "n")
        assert cache.derived(token, "n", lambda results: "second") == (True, "second")

    def test_unknown_token_and_the_dataset_gate(self):
        cache = SortResultsCache()
        token = cache.store(_ranking(2), 0.5, dataset_id="ds-A")
        assert cache.derived("does-not-exist", "n", len) == (False, None)
        assert cache.derived(token, "n", len, dataset_id="ds-B") == (False, None)
        assert cache.derived(token, "n", len, dataset_id="ds-A") == (True, 2)

    def test_evicted_with_its_ranking(self):
        cache = SortResultsCache(max_entries=1)
        t1 = cache.store(_ranking(2), 0.5)
        cache.derived(t1, "n", len)
        cache.store(_ranking(1), 0.5)  # evicts t1
        assert cache.derived(t1, "n", len) == (False, None)
