"""Helpers for serving the saved labelset of a detector as right-pane content.

The right pane in label/train mode is *labelset-driven* (not cid-driven):
each entry is a :class:`~vtscore.datasets.labelset.LabeledElement` from the
detector JSON on disk.  This module turns those elements into the shape the
frontend needs (stable element id, display metadata, optional current-dataset
cid for click-time / learned-score correlation) and provides the lookup
helpers used by the preview and vote routes.
"""

from __future__ import annotations

import hashlib
import json as _json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from vtscore.datasets.labelset import LabeledElement, LabelSet, element_key


def stable_element_id(elem: LabeledElement) -> str:
    """Return a stable, URL-safe id for *elem* derived from its identity.

    Built from :func:`~vtscore.datasets.labelset.element_key` when possible,
    so the id is stable across label flips and unrelated edits to the
    labelset.  Elements with neither an origin nor an md5 fall back to a
    hash of the full serialised dict.
    """
    key = element_key(elem)
    if key is None:
        payload = _json.dumps(elem.to_dict(), sort_keys=True)
    else:
        payload = _json.dumps(key, default=str, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def find_element_by_id(elements: list[LabeledElement], target_id: str) -> tuple[int, LabeledElement] | None:
    """Linear-scan for the labelset element with stable id *target_id*."""
    for idx, el in enumerate(elements):
        if stable_element_id(el) == target_id:
            return idx, el
    return None


def resolve_current_dataset_cid(
    elem: LabeledElement,
    lookups: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None = None,
) -> int | None:
    """Return the cid in the active dataset that matches *elem*, or ``None``.

    Matches by origin+name first, then by MD5.  Only consults the dataset
    snapshot - does not trigger origin-file resolution.

    When *lookups* is provided (the ``(origin_lookup, md5_lookup, name_lookup)``
    triple from :func:`~vtscore.state.media_lookup.build_media_lookup`), it is
    used directly and no snapshot is taken.  Loop callers that resolve many
    elements against the same dataset build the tables **once** and thread them
    through, turning an O(labels × N) poll into O(N + labels).  When *lookups*
    is ``None`` the tables are built from a fresh :func:`snapshot_medias`, the
    single-element convenience path.

    Returning a single cid (``cids[0]``) from the union returned by
    :func:`~vtscore.state.media_lookup.resolve_media_ids` is safe because
    every Flask-driven dataset load runs
    :func:`~vtscore.state.media_lookup.collapse_duplicates`, which guarantees
    each MD5 maps to at most one cid in the active medias dict - so the union
    never yields multiple md5-matched cids.  If a future code path inserts
    media post-dedup without re-running it, this contract breaks; the
    invariant is covered by ``test_collapse_duplicates_yields_unique_md5_lookup``.
    """
    from vtscore.state import (
        cached_media_lookups,
        resolve_media_ids,
        snapshot_medias,
    )

    if lookups is None:
        snap = snapshot_medias()
        if not snap:
            return None
        lookups = cached_media_lookups()
    origin_lookup, md5_lookup, name_lookup = lookups
    cids = resolve_media_ids(elem.to_dict(), origin_lookup, md5_lookup, name_lookup)
    return cids[0] if cids else None


@contextmanager
def resolve_element_to_path(elem: LabeledElement) -> Iterator[Path | None]:
    """Resolve the underlying media file for *elem* via its origin.

    Yields the path on disk if the importer's
    :meth:`~vtscore.datasets.importers.base.DatasetImporter.resolve_file`
    (or the corresponding :class:`~vtscore.datasets.sources.base.MediaSource`)
    can locate it (e.g. demo, folder, synthetic, server-folder importers
    with a stable cache directory), otherwise ``None``.

    Must be used as a ``with`` block: some media sources materialise the
    file inside a temp dir they own, and the dir is finalised on exit.
    Callers must read bytes (or otherwise finish with the file) inside
    the block.
    """
    from vtscore.detectors.resolver import resolve_file_context

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as p:
        yield p


def build_element_view(
    elem: LabeledElement,
    *,
    media_type: str,
    click_times: Mapping[int, float | int],
    learned_scores: Mapping[int, float],
    lookups: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None = None,
) -> dict[str, Any]:
    """Build the JSON-serialisable dict the frontend consumes for *elem*.

    Includes a current-dataset ``cid`` (and its click-time / learned-score)
    when the element resolves into the active dataset; otherwise those
    fields are ``null``.

    *lookups* is threaded straight into :func:`resolve_current_dataset_cid`;
    pass the pre-built ``build_media_lookup`` triple when calling this in a
    loop so the lookup tables aren't rebuilt per element.
    """
    cid = resolve_current_dataset_cid(elem, lookups)
    name = elem.origin_name or elem.filename or (elem.md5[:12] if elem.md5 else "")

    score = -1.0
    time_ = -1.0
    if cid is not None:
        score = float(learned_scores.get(cid, -1.0))
        time_ = float(click_times.get(cid, -1.0))

    return {
        "id": stable_element_id(elem),
        "label": elem.label,
        "media_type": media_type,
        "name": name,
        "filename": elem.filename,
        "origin_name": elem.origin_name,
        "md5": elem.md5,
        "cid": cid,
        "time": time_,
        "score": score,
        # Present only for region votes; lets the Good pile bust its thumbnail
        # cache when the voted box changes (the thumbnail route crops to the
        # element's stored box server-side).
        "region_box": list(elem.region_box) if elem.region_box is not None else None,
    }


def build_labels_detail(detector_data: dict[str, Any]) -> dict[str, Any]:
    """Build the response body for ``GET /api/detectors/<name>/labels-detail``.

    Returns ``{"good": [...], "bad": [...], "media_type": "..."}``.
    """
    from vtscore.state import cached_media_lookups, get_active_detector_context, snapshot_medias

    media_type = detector_data.get("media_type", "") or ""
    labelset = LabelSet.from_dict(detector_data.get("labelset") or {})

    det_ctx = get_active_detector_context()
    click_times = dict(det_ctx.vote_click_times)
    learned_scores = dict(det_ctx.last_learned_scores)

    # Reuse the active dataset's cached origin/md5/name lookup tables (S14) and
    # thread them through every element's view, so the labels-detail poll is
    # O(N + labels) - or O(labels) on a cache hit - instead of O(labels × N)
    # (each per-element rebuild would json.dumps every origin).
    snap = snapshot_medias()
    lookups = cached_media_lookups() if snap else None

    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for el in labelset.elements:
        view = build_element_view(
            el,
            media_type=media_type,
            click_times=click_times,
            learned_scores=learned_scores,
            lookups=lookups,
        )
        if el.label == "good":
            good.append(view)
        elif el.label == "bad":
            bad.append(view)
    return {"good": good, "bad": bad, "media_type": media_type}


def apply_element_vote_in_data(
    detector_data: dict[str, Any],
    target_id: str,
    target: str,
) -> tuple[bool, LabeledElement | None, str]:
    """Set the label on the element with stable id *target_id* to *target*.

    *target* is an **absolute** end state, one of ``"good"`` / ``"bad"``
    (set the element's label) or ``"remove"`` (drop the element from the
    labelset).

    Returns ``(changed, updated_element, action)`` where:

    * ``changed`` is ``True`` if the on-disk labelset must be re-written.
    * ``updated_element`` is the element after the operation, or ``None``
      when the element was removed or the target is invalid / not found.
    * ``action`` is one of ``"removed"``, ``"flipped"``, ``"unchanged"``,
      or ``"not_found"``.

    Absolute-target semantics mirror :func:`~vtsearch.state.set_vote`:
    behaviour is **idempotent**, so re-sending ``"good"`` on an
    already-good element is an ``"unchanged"`` no-op rather than a removal.
    A stale-view tab can no longer flip an element off the labelset by
    re-asserting its current label (logical-bug-audit H1).
    """
    if target not in ("good", "bad", "remove"):
        return False, None, "unchanged"

    labelset = LabelSet.from_dict(detector_data.get("labelset") or {})
    found = find_element_by_id(labelset.elements, target_id)
    if found is None:
        return False, None, "not_found"

    idx, el = found
    if target == "remove":
        labelset.elements.pop(idx)
        detector_data["labelset"] = labelset.to_dict()
        return True, None, "removed"

    if el.label == target:
        return False, el, "unchanged"

    el.label = target
    detector_data["labelset"] = labelset.to_dict()
    return True, el, "flipped"
