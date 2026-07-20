"""Process-global cache of full sorted result lists for windowed paging.

A sort API call ranks the *whole* dataset but, at 100 k / 1 M items, must not
ship the entire ordered list to the browser in one JSON response
(``docs/plans/scalability.md`` S3/S17/S19).  This cache lets a sort route stash
its full descending ``results`` list server-side and hand the client an opaque
``sort_token``; the client then pulls deeper windows via
``GET /api/sort/page?token=…``.

Only in-memory, and only the lightweight ranking rows (``{"id", "score"}`` or
``{"id", "similarity"[, "best_region"]}``) — never embeddings or MLP weights, so
this stays within the "No Persisted Vectors or MLPs" rule (nothing is
serialised; the list is re-derived on the next sort).  The cache is bounded to
the most recent ``max_entries`` sorts (LRU), so it cannot grow without bound as
users re-sort.

The ``sort_token`` doubles as the **sort-generation token**: a client holding a
token from one ranking cannot accidentally page into a newer ranking, because a
re-sort mints a fresh token and the old one either still points at its own
(now-stale-but-consistent) list or has been evicted (→ 404, refetch from the
top).
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any


def result_score(result: dict) -> float:
    """Read a ranking row's score regardless of which sort path produced it.

    Text / example sort emit ``{"id", "similarity"}``; learned / find sort emit
    ``{"id", "score"}``.  Prefer ``score`` then ``similarity``; a row carrying
    neither sorts as ``-inf`` so it lands below any real threshold.
    """
    if "score" in result:
        return result["score"]
    if "similarity" in result:
        return result["similarity"]
    return float("-inf")


def count_above_threshold(results: list[dict], threshold: float | None) -> int:
    """Number of rows scoring at or above *threshold* (all of them when ``None``)."""
    if threshold is None:
        return len(results)
    return sum(1 for r in results if result_score(r) >= threshold)


class SortResultsCache:
    """LRU cache of full sorted result lists, keyed by an opaque token.

    Thread-safe: sort routes ``store`` from request threads (and background
    learned-sort workers), while ``/api/sort/page`` reads concurrently.
    """

    def __init__(self, max_entries: int = 8) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max_entries

    def store(
        self,
        results: list[dict],
        threshold: float | None,
        *,
        dataset_id: str = "",
        detector_id: str = "",
    ) -> str:
        """Store *results* under a fresh token and return it.

        Evicts the least-recently-used entries beyond ``max_entries``.  The
        stored list is held by reference (not copied): callers must not mutate a
        results list after handing it off.
        """
        token = uuid.uuid4().hex
        with self._lock:
            self._entries[token] = {
                "results": results,
                "threshold": threshold,
                "dataset_id": dataset_id,
                "detector_id": detector_id,
            }
            self._entries.move_to_end(token)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return token

    def page(
        self,
        token: str,
        offset: int,
        limit: int,
        *,
        dataset_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return a window of the stored list, or ``None`` if the token is unknown.

        ``None`` also results when *dataset_id* is given and doesn't match the
        dataset the sort was stored against — a defence against paging one
        dataset's ranking with another dataset's active context.
        """
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return None
            if dataset_id is not None and entry["dataset_id"] and entry["dataset_id"] != dataset_id:
                return None
            # Touch: paging keeps a still-in-use ranking warm against eviction.
            self._entries.move_to_end(token)
            results: list[dict] = entry["results"]
            threshold = entry["threshold"]

        total = len(results)
        start = max(0, offset)
        end = total if limit < 0 else min(total, start + limit)
        window = results[start:end]
        return {
            "results": window,
            "offset": start,
            "limit": limit,
            "total": total,
            "threshold": threshold,
            "has_more": end < total,
        }

    def reset_for_tests(self) -> None:
        """Drop all cached lists (called from the autouse test-reset fixtures)."""
        with self._lock:
            self._entries.clear()


#: Application-wide singleton used by the sort routes and ``/api/sort/page``.
sort_results_cache = SortResultsCache()
