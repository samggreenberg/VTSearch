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

A text or example sort also leaves its display line's rule here
(:class:`~vtscore.training.query_sort.SortLine`), so a balance change can redraw
that line over the ranking already on screen (:meth:`SortResultsCache.redraw`,
``GET /api/sort/line``, #4760) rather than re-scoring the haystack.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any, Protocol


class RedrawableLine(Protocol):
    """A sort's display line that can be drawn again at another balance (#4760)."""

    def threshold_at(self, beta: float | None) -> float: ...


# Window shape for the *initial* sort response (scalability.md S3/S17/S19).
#
# Below ``SORT_WINDOW_THRESHOLD`` items the full ranking is transmitted unchanged,
# so small / medium sorts keep every existing behaviour — including the stripe
# minimap, which is gated off at the same size (``STRIPE_MAX_ITEMS``). At or above
# it, only a head window rides the initial response and the client pages the rest
# through ``/api/sort/page``. Aligning the two thresholds means a windowed sort is
# always one whose stripe is already disabled, so no client code needs the full
# order once windowing engages.
SORT_WINDOW_THRESHOLD = 20000
#: K_ABOVE — cap on above-threshold rows in the initial window.
SORT_WINDOW_HEAD = 500
#: K_BELOW — rows just past the boundary to include for context.
SORT_WINDOW_TAIL = 200


def initial_window_end(total: int, above_threshold: int) -> int:
    """End index (exclusive) of the initial transmitted window.

    Includes up to ``SORT_WINDOW_HEAD`` above-threshold rows plus
    ``SORT_WINDOW_TAIL`` rows just past the boundary, clamped to *total*.  When
    the above-threshold count exceeds the head cap the boundary falls outside the
    window (the user reaches it by paging) — the strongest matches lead.
    """
    return min(total, min(above_threshold, SORT_WINDOW_HEAD) + SORT_WINDOW_TAIL)


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
        line: RedrawableLine | None = None,
    ) -> str:
        """Store *results* under a fresh token and return it.

        Evicts the least-recently-used entries beyond ``max_entries``.  The
        stored list is held by reference (not copied): callers must not mutate a
        results list after handing it off.  *line* is how to redraw *threshold*
        at another balance (see :meth:`redraw`); ``None`` for a sort whose line
        is not redrawn here.
        """
        token = uuid.uuid4().hex
        with self._lock:
            self._entries[token] = {
                "results": results,
                "threshold": threshold,
                "dataset_id": dataset_id,
                "detector_id": detector_id,
                "line": line,
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

    def redraw(
        self,
        token: str,
        beta: float | None,
        *,
        dataset_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Redraw a stored sort's display line at *beta* (#4760).

        Returns ``{"threshold", "above_threshold", "total"}``: the line its
        :class:`RedrawableLine` draws at *beta*, and how many rows of the whole
        ranking sit at or above it.  The entry keeps the new line, so a later
        :meth:`page` reports it.  The ranking does not move.

        ``None`` when the token is unknown, belongs to another dataset (as in
        :meth:`page`), or names a sort stored without a line - a learned sort,
        whose line moves with its detector instead, or one whose line takes no
        balance.
        """
        with self._lock:
            entry = self._entries.get(token)
            if entry is None or entry["line"] is None:
                return None
            if dataset_id is not None and entry["dataset_id"] and entry["dataset_id"] != dataset_id:
                return None
            self._entries.move_to_end(token)
            line: RedrawableLine = entry["line"]
            results: list[dict] = entry["results"]

        # Outside the lock: a text line refits the score mixture.
        threshold = line.threshold_at(beta)
        with self._lock:
            entry["threshold"] = threshold
        return {
            "threshold": threshold,
            "above_threshold": count_above_threshold(results, threshold),
            "total": len(results),
        }

    def reset_for_tests(self) -> None:
        """Drop all cached lists (called from the autouse test-reset fixtures)."""
        with self._lock:
            self._entries.clear()


#: Application-wide singleton used by the sort routes and ``/api/sort/page``.
sort_results_cache = SortResultsCache()
