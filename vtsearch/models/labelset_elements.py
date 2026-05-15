"""Helpers for serving the saved labelset of a detector as right-pane content.

The right pane in label/train mode is *labelset-driven* (not cid-driven):
each entry is a :class:`~vtsearch.datasets.labelset.LabeledElement` from the
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
from typing import Any, Iterator

from vtsearch.datasets.labelset import LabeledElement, LabelSet, element_key


def stable_element_id(elem: LabeledElement) -> str:
    """Return a stable, URL-safe id for *elem* derived from its identity.

    Built from :func:`~vtsearch.datasets.labelset.element_key` when possible,
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


def resolve_current_dataset_cid(elem: LabeledElement) -> int | None:
    """Return the cid in the active dataset that matches *elem*, or ``None``.

    Matches by origin+name first, then by MD5.  Only consults the dataset
    snapshot — does not trigger origin-file resolution.
    """
    from vtsearch.state import (
    build_media_lookup,
    resolve_media_ids,
    snapshot_medias,
)

    snap = snapshot_medias()
    if not snap:
        return None
    origin_lookup, md5_lookup, name_lookup = build_media_lookup(snap)
    cids = resolve_media_ids(elem.to_dict(), origin_lookup, md5_lookup, name_lookup)
    return cids[0] if cids else None


@contextmanager
def resolve_element_to_path(elem: LabeledElement) -> Iterator[Path | None]:
    """Resolve the underlying media file for *elem* via its origin.

    Yields the path on disk if the importer's
    :meth:`~vtsearch.datasets.importers.base.DatasetImporter.resolve_file`
    (or the corresponding :class:`~vtsearch.datasets.sources.base.MediaSource`)
    can locate it (e.g. demo, folder, synthetic, server-folder importers
    with a stable cache directory), otherwise ``None``.

    Must be used as a ``with`` block: some media sources materialise the
    file inside a temp dir they own, and the dir is finalised on exit.
    Callers must read bytes (or otherwise finish with the file) inside
    the block.
    """
    from vtsearch.models.resolver import resolve_file_context

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as p:
        yield p


def build_element_view(
    elem: LabeledElement,
    *,
    media_type: str,
    click_times: dict[int, float],
    learned_scores: dict[int, float],
) -> dict[str, Any]:
    """Build the JSON-serialisable dict the frontend consumes for *elem*.

    Includes a current-dataset ``cid`` (and its click-time / learned-score)
    when the element resolves into the active dataset; otherwise those
    fields are ``null``.
    """
    cid = resolve_current_dataset_cid(elem)
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
    }


def build_labels_detail(detector_data: dict[str, Any]) -> dict[str, Any]:
    """Build the response body for ``GET /api/detectors/<name>/labels-detail``.

    Returns ``{"good": [...], "bad": [...], "media_type": "..."}``.
    """
    from vtsearch.state import get_active_detector_context

    media_type = detector_data.get("media_type", "") or ""
    labelset = LabelSet.from_dict(detector_data.get("labelset") or {})

    det_ctx = get_active_detector_context()
    click_times = dict(det_ctx.vote_click_times)
    learned_scores = dict(det_ctx.last_learned_scores)

    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for el in labelset.elements:
        view = build_element_view(
            el,
            media_type=media_type,
            click_times=click_times,
            learned_scores=learned_scores,
        )
        if el.label == "good":
            good.append(view)
        elif el.label == "bad":
            bad.append(view)
    return {"good": good, "bad": bad, "media_type": media_type}


def apply_element_vote_in_data(
    detector_data: dict[str, Any],
    target_id: str,
    vote: str,
) -> tuple[bool, LabeledElement | None, str]:
    """Toggle the label on the element with stable id *target_id*.

    Returns ``(changed, updated_element, action)`` where:

    * ``changed`` is ``True`` if the on-disk labelset must be re-written.
    * ``updated_element`` is the element after the operation, or ``None``
      when the element was removed.
    * ``action`` is one of ``"removed"``, ``"flipped"``, ``"unchanged"``,
      or ``"not_found"``.

    Toggle semantics mirror :func:`~vtsearch.state.toggle_vote`:

    * Same vote on the same element → remove the element from the labelset.
    * Opposite vote → flip the element's label.
    * Vote that's neither ``"good"`` nor ``"bad"`` → unchanged.
    """
    if vote not in ("good", "bad"):
        return False, None, "unchanged"

    labelset = LabelSet.from_dict(detector_data.get("labelset") or {})
    found = find_element_by_id(labelset.elements, target_id)
    if found is None:
        return False, None, "not_found"

    idx, el = found
    if el.label == vote:
        labelset.elements.pop(idx)
        detector_data["labelset"] = labelset.to_dict()
        return True, None, "removed"

    el.label = vote
    detector_data["labelset"] = labelset.to_dict()
    return True, el, "flipped"
