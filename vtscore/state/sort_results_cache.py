"""Process-global cache of full sorted result lists for windowed paging.

A sort API call ranks the *whole* dataset but, at 100 k / 1 M items, must not
ship the entire ordered list to the browser in one JSON response
(``docs/plans/scalability.md`` S3/S17/S19).  This cache lets a sort route stash
its full descending ``results`` list server-side and hand the client an opaque
``sort_token``; the client then pulls deeper windows via
``GET /api/sort/page?token=…``.

Only in-memory, and only the lightweight ranking rows (``{"id", "score"}`` or
``{"id", "similarity"[, "best_region"]}``) — never embeddings or head weights, so
this stays within the "No Persisted Vectors or MLPs" rule (nothing is
serialised; the list is re-derived on the next sort).  A ranking is held as
columns rather than a dict per row, about 20 bytes a row instead of 250 (#5012),
and rows are built back into dicts only for the window asked for.  The cache is
bounded to the most recent ``max_entries`` sorts and to ``max_rows`` rows across
them (LRU), so re-sorting neither grows it without bound nor, on a 1 M-item
dataset, pins a large share of memory (#4978).

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
from array import array
from collections import OrderedDict
from itertools import repeat
from operator import itemgetter
from typing import Any, Callable, Hashable, Iterable, Protocol

import numpy as np


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

#: Cap on the ranking rows the cache holds across all its sorts (#4978).  A row
#: is held in columns, 16 bytes (20 with ``best_region``, #5012), so this keeps
#: the cache's rankings to about 160 MB while all eight of ``max_entries``
#: rankings of a 1 M-item dataset fit.  The newest ranking is kept even when it
#: alone is longer, so a fresh sort can page.
SORT_CACHE_MAX_ROWS = 8_000_000


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


#: The keys a stored row may carry and still be held in columns (#5012).
_SCORE_KEYS = frozenset({"score", "similarity"})
_REGION_KEY = "best_region"
_COLUMN_KEYS = frozenset({"id", _REGION_KEY}) | _SCORE_KEYS
#: Score values a column holds exactly (a float64 is a float either way).
_FLOAT_TYPES = frozenset({float, np.float64})
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1

#: How a row's keys map onto the columns: its score key (``None`` for none) and
#: whether it has a ``best_region``.
_Layout = tuple[str | None, bool]


def _layout(keys: tuple[Any, ...]) -> _Layout | None:
    """The layout of a row with *keys*, or ``None`` when the columns cannot hold it.

    They cannot when ``id`` is missing, another key is present, or both score
    keys are.
    """
    if "id" not in keys or not _COLUMN_KEYS.issuperset(keys):
        return None
    score_keys = [key for key in keys if key in _SCORE_KEYS]
    if len(score_keys) > 1:
        return None
    return (score_keys[0] if score_keys else None, _REGION_KEY in keys)


def _cells(row: dict, layout: _Layout, boxes: dict[tuple[Any, ...], int]) -> tuple[int, float, int] | None:
    """A row's ``(id, score, box index)``, or ``None`` when a value does not fit its column.

    A row without a score scores ``-inf``, as :func:`result_score` reads it.
    A new box joins *boxes*.
    """
    score_key, has_region = layout
    rid = row["id"]
    score = row[score_key] if score_key is not None else float("-inf")
    if type(rid) is not int or not _INT64_MIN <= rid <= _INT64_MAX or type(score) not in _FLOAT_TYPES:
        return None
    if not has_region:
        return rid, score, -1
    region = row[_REGION_KEY]
    if type(region) is not list:
        return None
    try:
        return rid, score, boxes.setdefault(tuple(region), len(boxes))
    except TypeError:  # a value that cannot be hashed
        return None


class _Ranking:
    """A stored ranking held as columns, not one dict per row (#5012).

    A row dict costs about 250 bytes (336 with ``best_region``).  Here a row
    is an int64 id and a float64 score, plus, when the ranking has regions,
    the index of its box in a table of the distinct boxes (a region sort's
    boxes are the cells of a small grid): 16 bytes a row, or 20, and one more
    when rows of one ranking differ in their keys.  Scores stay float64 so a
    row compares with a threshold exactly as its dict did.

    :meth:`rows` builds fresh dicts equal to the ones stored, keys in the
    order ``id``, score key, ``best_region``.  A row the columns cannot hold
    is kept whole, as a shallow copy, in a side table: one with a key other
    than those, with both score keys, or whose ``id`` is not an int64, score
    not a float, or ``best_region`` not a list of hashable values.  The sorts
    emit none of these, so the table stays empty.

    Never changed once built, so a reader needs no lock.
    """

    __slots__ = ("ids", "scores", "layouts", "layout_of", "boxes", "box_of", "whole")

    ids: np.ndarray
    scores: np.ndarray
    #: The layouts the rows use, and each row's index into them (``None``: all use the one).
    layouts: list[_Layout]
    layout_of: np.ndarray | None
    #: The distinct boxes, and each row's index into them (``None``: no row has one).
    boxes: list[tuple[Any, ...]]
    box_of: np.ndarray | None
    #: Rows kept whole, by position.
    whole: dict[int, dict]

    def __init__(self, results: Iterable[dict]) -> None:
        rows = results if isinstance(results, list) else list(results)
        if not self._fill_uniform(rows):
            self._fill_rowwise(rows)

    def _fill_uniform(self, rows: list[dict]) -> bool:
        """Fill the columns a column at a time, if every row has the first row's keys.

        The fast path, for the rankings the sorts emit.  ``False``, with
        nothing set, when a row has other keys or a value its column cannot
        hold.
        """
        if not rows:
            return False
        keys = tuple(rows[0])
        layout = _layout(keys)
        # Every row as long as the first and holding each of its keys has exactly its keys.
        if layout is None or set(map(len, rows)) != {len(keys)}:
            return False
        score_key, has_region = layout
        try:
            ids = list(map(itemgetter("id"), rows))
            scores = list(map(itemgetter(score_key), rows)) if score_key is not None else None
            regions = list(map(itemgetter(_REGION_KEY), rows)) if has_region else None
        except KeyError:
            return False
        if set(map(type, ids)) != {int}:
            return False
        if scores is not None and not _FLOAT_TYPES.issuperset(map(type, scores)):
            return False
        boxes: dict[tuple[Any, ...], int] = {}
        box_of = None
        if regions is not None:
            if set(map(type, regions)) != {list}:
                return False
            try:
                box_of = np.array([boxes.setdefault(box, len(boxes)) for box in map(tuple, regions)], dtype=np.int32)
            except TypeError:  # a value that cannot be hashed
                return False
        try:
            self.ids = np.array(ids, dtype=np.int64)
        except OverflowError:
            return False
        self.scores = np.array(scores, dtype=np.float64) if scores is not None else np.full(len(rows), -np.inf)
        self.layouts = [layout]
        self.layout_of = None
        self.boxes = list(boxes)
        self.box_of = box_of
        self.whole = {}
        return True

    def _fill_rowwise(self, rows: list[dict]) -> None:
        """Fill the columns a row at a time, keeping whole each row they cannot hold."""
        ids, scores, layout_of, box_of = array("q"), array("d"), array("B"), array("i")
        # Each key tuple seen, with its layout and that layout's index (``None``: it does not fit).
        layout_at: dict[tuple[Any, ...], tuple[int, _Layout] | None] = {}
        layouts: list[_Layout] = []
        boxes: dict[tuple[Any, ...], int] = {}
        whole: dict[int, dict] = {}
        for i, row in enumerate(rows):
            keys = tuple(row)
            if keys not in layout_at:
                layout = _layout(keys)
                if layout is not None and layout not in layouts:
                    layouts.append(layout)  # six at most, so a byte indexes them
                layout_at[keys] = None if layout is None else (layouts.index(layout), layout)
            found = layout_at[keys]
            if found is not None:
                at, layout = found
                cells = _cells(row, layout, boxes)
                if cells is not None:
                    rid, score, box = cells
                    ids.append(rid)
                    scores.append(score)
                    layout_of.append(at)
                    box_of.append(box)
                    continue
            whole[i] = dict(row)
            ids.append(0)
            # NaN is never at or above a line; count_above counts these rows from the side table.
            scores.append(float("nan"))
            layout_of.append(0)
            box_of.append(-1)
        self.ids = np.array(ids, dtype=np.int64)
        self.scores = np.array(scores, dtype=np.float64)
        self.layouts = layouts
        self.layout_of = np.array(layout_of, dtype=np.uint8) if len(layouts) > 1 else None
        self.boxes = list(boxes)
        self.box_of = np.array(box_of, dtype=np.int32) if boxes else None
        self.whole = whole

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def nbytes(self) -> int:
        """Bytes held by the columns; the box table and the side table are not counted."""
        columns = (self.ids, self.scores, self.layout_of, self.box_of)
        return sum(column.nbytes for column in columns if column is not None)

    def rows(self, start: int, end: int) -> list[dict]:
        """The rows in ``[start, end)``, built back into fresh dicts."""
        ids = self.ids[start:end].tolist()
        scores = self.scores[start:end].tolist()
        layout_of = self.layout_of[start:end].tolist() if self.layout_of is not None else repeat(0)
        box_of = self.box_of[start:end].tolist() if self.box_of is not None else repeat(-1)
        layouts, boxes, whole = self.layouts, self.boxes, self.whole
        out: list[dict] = []
        for i, rid, score, at, box in zip(range(start, start + len(ids)), ids, scores, layout_of, box_of):
            kept = whole.get(i)
            if kept is not None:
                out.append(dict(kept))
                continue
            score_key, has_region = layouts[at]
            row: dict[str, Any] = {"id": rid}
            if score_key is not None:
                row[score_key] = score
            if has_region:
                row[_REGION_KEY] = list(boxes[box])
            out.append(row)
        return out

    def count_above(self, threshold: float | None) -> int:
        """:func:`count_above_threshold` over the stored rows."""
        if threshold is None:
            return len(self)
        columnar = int(np.count_nonzero(self.scores >= threshold))
        return columnar + count_above_threshold(list(self.whole.values()), threshold)


class SortResultsCache:
    """LRU cache of full sorted result lists, keyed by an opaque token.

    Bounded twice: at most *max_entries* rankings, and at most *max_rows* rows
    across them (see :meth:`store`).  Views kept by :meth:`derived` are not
    counted, but go with their ranking.

    Thread-safe: sort routes ``store`` from request threads (and background
    learned-sort workers), while ``/api/sort/page`` reads concurrently.
    """

    def __init__(self, max_entries: int = 8, max_rows: int = SORT_CACHE_MAX_ROWS) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max_entries
        self._max_rows = max_rows

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

        Evicts the least-recently-used entries beyond ``max_entries``, then
        until the rankings left hold at most ``max_rows`` rows between them.
        The new entry always stays, even when it alone is over ``max_rows``, so
        a fresh sort of a very large dataset can still be paged.  The rows are
        copied into columns (#5012), so the caller may change *results* after
        handing it off.  *line* is how to redraw *threshold* at another balance
        (see :meth:`redraw`); ``None`` for a sort whose line is not redrawn
        here.
        """
        # Outside the lock: a 1 M-row ranking takes a fraction of a second to copy.
        ranking = _Ranking(results)
        token = uuid.uuid4().hex
        with self._lock:
            self._entries[token] = {
                "ranking": ranking,
                "threshold": threshold,
                "dataset_id": dataset_id,
                "detector_id": detector_id,
                "line": line,
            }
            self._entries.move_to_end(token)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            rows = sum(len(entry["ranking"]) for entry in self._entries.values())
            # The new entry is last, so evicting from the front never reaches it.
            while rows > self._max_rows and len(self._entries) > 1:
                _, evicted = self._entries.popitem(last=False)
                rows -= len(evicted["ranking"])
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
        dataset's ranking with another dataset's active context.  The window's
        rows are fresh dicts, equal to the ones stored.
        """
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return None
            if dataset_id is not None and entry["dataset_id"] and entry["dataset_id"] != dataset_id:
                return None
            # Touch: paging keeps a still-in-use ranking warm against eviction.
            self._entries.move_to_end(token)
            ranking: _Ranking = entry["ranking"]
            threshold = entry["threshold"]

        total = len(ranking)
        start = max(0, offset)
        end = total if limit < 0 else min(total, start + limit)
        window = ranking.rows(start, end)
        return {
            "results": window,
            "offset": start,
            "limit": limit,
            "total": total,
            "threshold": threshold,
            "has_more": end < total,
        }

    def derived(
        self,
        token: str,
        key: Hashable,
        build: Callable[[list[dict]], Any],
        *,
        dataset_id: str | None = None,
    ) -> tuple[bool, Any]:
        """Return a view of a stored ranking, built once per token and *key*.

        *build* is called with the stored ranking, as a list of fresh row dicts
        equal to the ones stored, the first time *key* is asked for under
        *token*, outside the lock.  Its result is kept on the entry, so it is
        evicted with the ranking and a re-sort (a new token) builds afresh.
        The list itself is not kept (#5012).  Collapse to photos keeps its
        grouping here (#4751).  If *build* raises, nothing is kept and the
        error propagates.

        Returns ``(found, view)``: ``(False, None)`` when the token is unknown
        or belongs to another dataset (as in :meth:`page`).
        """
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return False, None
            if dataset_id is not None and entry["dataset_id"] and entry["dataset_id"] != dataset_id:
                return False, None
            self._entries.move_to_end(token)
            views: dict[Hashable, Any] = entry.setdefault("derived", {})
            if key in views:
                return True, views[key]
            ranking: _Ranking = entry["ranking"]

        view = build(ranking.rows(0, len(ranking)))
        with self._lock:
            # A concurrent first ask may have built it too; either copy is the same view.
            views.setdefault(key, view)
            return True, views[key]

    def forget_derived(self, token: str, key: Hashable) -> None:
        """Drop the view :meth:`derived` kept under *token* and *key*, if any."""
        with self._lock:
            entry = self._entries.get(token)
            if entry is not None:
                entry.get("derived", {}).pop(key, None)

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
            ranking: _Ranking = entry["ranking"]

        # Outside the lock: a text line refits the score mixture.
        threshold = line.threshold_at(beta)
        with self._lock:
            entry["threshold"] = threshold
        return {
            "threshold": threshold,
            "above_threshold": ranking.count_above(threshold),
            "total": len(ranking),
        }

    def reset_for_tests(self) -> None:
        """Drop all cached lists (called from the autouse test-reset fixtures)."""
        with self._lock:
            self._entries.clear()


#: Application-wide singleton used by the sort routes and ``/api/sort/page``.
sort_results_cache = SortResultsCache()
