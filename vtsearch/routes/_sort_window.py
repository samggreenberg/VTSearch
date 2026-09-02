"""Windowing a full ranking into a sort response.

Sits between the sort routes and :mod:`vtscore.state.sort_results_cache`: the
whole ranking is registered with the cache so ``GET /api/sort/page`` can serve
any window, while the response itself carries only the head window once the
ranking is large enough to be worth paging.
"""

from __future__ import annotations

from typing import Any


def _windowed_sort_extras(results: list[dict], threshold: float | None) -> dict[str, Any]:
    """Register a full sorted ``results`` list and return the windowing extras.

    Stores *results* in the process-global :data:`sort_results_cache` keyed to
    the active (dataset, detector) pair and returns the extra fields a sort
    response carries so a client can page deeper without holding the whole list:

    - ``sort_token`` — opaque handle for ``GET /api/sort/page``; also the
      sort-generation token (a re-sort mints a new one).
    - ``total`` — full ranking length.
    - ``above_threshold`` — rows scoring at or above *threshold*.

    Additive: the caller still returns the full ``results`` today (the frontend
    windowed model lands in a later slice, see ``docs/plans/scalability.md``
    S3/S17/S19), so wiring this in never changes existing behaviour.
    """
    from vtscore.state.core import get_active_context, get_active_detector_context  # noqa: PLC0415
    from vtscore.state.sort_results_cache import count_above_threshold, sort_results_cache  # noqa: PLC0415

    dataset_id = getattr(get_active_context(), "dataset_id", "") or ""
    detector_id = getattr(get_active_detector_context(), "detector_id", "") or ""
    token = sort_results_cache.store(results, threshold, dataset_id=dataset_id, detector_id=detector_id)
    return {
        "sort_token": token,
        "total": len(results),
        "above_threshold": count_above_threshold(results, threshold),
    }


def windowed_sort_response(
    results: list[dict],
    threshold: float | None,
    acq_threshold: float | None = None,
) -> dict[str, Any]:
    """Build a sort-response body, windowing the transmitted ``results``.

    *acq_threshold* is the **acquisition** cut for a detector sort - the rank
    position Autopilot's Hard / New picks sample around, which since #2876 sits
    above the reporting ``threshold`` (see
    :func:`vtscore.state.core.detector_acquisition_threshold`).  Only the learned
    sort carries one; the text / example / label-file sorts have no detector
    behind them, so they leave it ``None`` and the client falls back to
    ``threshold``.  It is deliberately *not* fed to ``_windowed_sort_extras``:
    ``above_threshold`` counts what the user is told matched, which is the
    reporting cut's job.

    Stores the full ranking (so ``/api/sort/page`` can serve any window) and
    returns ``{results, threshold, acq_threshold, sort_token, total,
    above_threshold, has_more_below}``.  Below :data:`SORT_WINDOW_THRESHOLD` the full list is
    transmitted unchanged and ``has_more_below`` is ``False`` — small / medium
    sorts behave exactly as before.  At or above it, only the initial head window
    rides the response and the client pages the rest.

    The threshold is read off the cache module at call time so tests can lower it
    via ``monkeypatch`` without generating tens of thousands of rows.
    """
    from vtscore.state import sort_results_cache as _cache_mod  # noqa: PLC0415

    extras = _windowed_sort_extras(results, threshold)
    total = extras["total"]
    if total < _cache_mod.SORT_WINDOW_THRESHOLD:
        window = results
        has_more = False
    else:
        end = _cache_mod.initial_window_end(total, extras["above_threshold"])
        window = results[:end]
        has_more = end < total
    return {
        "results": window,
        "threshold": threshold,
        "acq_threshold": acq_threshold,
        "has_more_below": has_more,
        **extras,
    }
