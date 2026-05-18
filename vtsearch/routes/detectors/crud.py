"""Blueprint for detector CRUD routes.

Detectors are a persistent reference to a labelset file plus a text-sort
query.  They live on disk in ``data/detectors/<slug>.json`` and can
accumulate labels over repeated training sessions.

Endpoints
---------
GET  /api/detectors
    List all detectors.

POST /api/detectors
    Create a new detector (requires ``name`` and ``text_query``).

GET  /api/detectors/<name>
    Retrieve a single detector with its labelset.

DELETE /api/detectors/<name>
    Delete a detector.

PUT  /api/detectors/<name>/rename
    Rename a detector.

PUT  /api/detectors/<name>/examples
    Set/replace the examples list for a detector.

POST /api/detectors/combine
    Combine the labelsets of two or more detectors into a new detector.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.
"""

from __future__ import annotations

import logging
import time

from flask_smorest import Blueprint, abort

from vtsearch.detectors.store import (
    _detector_path,
    _read_detector,
    _write_detector,
    get_detectors_dir,
)
from vtsearch.schemas.detectors import (
    DetectorCombineRequestSchema,
    DetectorCombineResponseSchema,
    DetectorCreateRequestSchema,
    DetectorCreateResponseSchema,
    DetectorDeleteResponseSchema,
    DetectorDetailSchema,
    DetectorExamplesRequestSchema,
    DetectorExamplesResponseSchema,
    DetectorRenameRequestSchema,
    DetectorRenameResponseSchema,
    DetectorsListResponseSchema,
)

logger = logging.getLogger(__name__)

detectors_crud_bp = Blueprint(
    "detectors_crud",
    __name__,
    description="Create, list, rename, delete, and combine detectors.",
)


def _list_all() -> list[dict]:
    """Return summary info for every detector on disk."""
    det_dir = get_detectors_dir()
    if not det_dir.is_dir():
        return []
    detectors = []
    for p in sorted(det_dir.iterdir()):
        if p.suffix != ".json":
            continue
        data = _read_detector(p)
        if data is None:
            continue
        labels = data.get("labelset", {}).get("labels", [])
        detectors.append(
            {
                "name": data["name"],
                "text_query": data.get("text_query", ""),
                "media_example": data.get("media_example", ""),
                "media_type": data.get("media_type", ""),
                "examples": data.get("examples", []),
                "num_labels": len(labels),
                "created_at": data.get("created_at", 0),
            }
        )
    return detectors


# ---------------------------------------------------------------------------
# GET /api/detectors
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors", methods=["GET"])
@detectors_crud_bp.response(200, DetectorsListResponseSchema)
def list_detectors():
    """Return all detectors (summary only, no full labelset)."""
    return {"detectors": _list_all()}


# ---------------------------------------------------------------------------
# POST /api/detectors
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors", methods=["POST"])
@detectors_crud_bp.arguments(DetectorCreateRequestSchema)
@detectors_crud_bp.response(201, DetectorCreateResponseSchema)
@detectors_crud_bp.alt_response(400, description="Missing example/query, or media_type is 'any'.")
@detectors_crud_bp.alt_response(409, description="A detector with this name already exists.")
def create_detector(body: dict):
    """Create a new detector.

    Requires ``name`` and ``media_type``; at least one of ``text_query``,
    ``media_example``, or ``examples`` must be provided.
    """
    name = body["name"].strip()
    media_type = body["media_type"].strip()
    text_query = body["text_query"].strip()
    media_example = body["media_example"].strip()
    examples = body.get("examples")

    if not name:
        abort(400, message="name is required")
    if not text_query and not media_example and not examples:
        abort(400, message="text_query, media_example, or examples is required")
    if not media_type or media_type == "any":
        abort(400, message="media_type is required (must be a specific type, not 'any')")

    path = _detector_path(name)
    if path.exists():
        abort(409, message=f"A detector named '{name}' already exists")

    # Build examples list; if text_query/media_example provided without
    # explicit examples, create a single example from it for backward compat.
    if examples is None and text_query:
        examples = [{"type": "text", "value": text_query}]
    elif examples is None and media_example:
        examples = [{"type": "media", "value": media_example}]

    detector_data = {
        "name": name,
        "text_query": text_query,
        "media_example": media_example,
        "media_type": media_type,
        "examples": examples or [],
        "created_at": time.time(),
        "labelset": {"labels": []},
    }
    _write_detector(path, detector_data)

    return {
        "success": True,
        "name": name,
        "text_query": text_query,
        "media_example": media_example,
        "media_type": media_type,
        "examples": examples or [],
        "num_labels": 0,
    }


# ---------------------------------------------------------------------------
# GET /api/detectors/<name>
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors/<name>", methods=["GET"])
@detectors_crud_bp.response(200, DetectorDetailSchema)
@detectors_crud_bp.alt_response(404, description="Detector not found.")
def get_detector(name: str):
    """Retrieve a single detector with its full labelset."""
    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        abort(404, message=f"Detector '{name}' not found")
    return data


# ---------------------------------------------------------------------------
# DELETE /api/detectors/<name>
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors/<name>", methods=["DELETE"])
@detectors_crud_bp.response(200, DetectorDeleteResponseSchema)
@detectors_crud_bp.alt_response(404, description="Detector not found.")
def delete_detector(name: str):
    """Delete a detector."""
    path = _detector_path(name)
    if not path.exists():
        abort(404, message=f"Detector '{name}' not found")
    path.unlink()
    return {"success": True, "name": name}


# ---------------------------------------------------------------------------
# PUT /api/detectors/<name>/rename
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors/<name>/rename", methods=["PUT"])
@detectors_crud_bp.arguments(DetectorRenameRequestSchema)
@detectors_crud_bp.response(200, DetectorRenameResponseSchema)
@detectors_crud_bp.alt_response(404, description="Detector not found.")
@detectors_crud_bp.alt_response(409, description="A detector with the new name already exists.")
def rename_detector(body: dict, name: str):
    """Rename a detector."""
    old_path = _detector_path(name)
    data = _read_detector(old_path)
    if data is None:
        abort(404, message=f"Detector '{name}' not found")

    new_name = body["new_name"].strip()
    if not new_name:
        abort(400, message="new_name is required")

    new_path = _detector_path(new_name)
    if new_path.exists() and new_path != old_path:
        abort(409, message=f"A detector named '{new_name}' already exists")

    data["name"] = new_name
    _write_detector(new_path, data)
    if new_path != old_path:
        old_path.unlink(missing_ok=True)

    # Update the detector registry entry that references this detector
    from vtsearch.detectors.labelset_rename import detect_pending_labelset_move
    from vtsearch.detectors.registry import find_by_name, rename_detector as _rename_in_registry
    from vtsearch.state.core import get_detector_context

    reg_entry = find_by_name(name)
    pending_move: dict[str, str] | None = None
    if reg_entry:
        registry_id = reg_entry["id"]
        ctx = get_detector_context(registry_id)
        if ctx is not None:
            pending_move = detect_pending_labelset_move(
                ctx.labelset_source,
                detector_id=registry_id,
                old_name=name,
                new_name=new_name,
            )
            ctx.name = new_name
        _rename_in_registry(registry_id, new_name)

    # Rename autorun flag if present.
    try:
        from vtsearch.settings import get_autorun_detectors, set_autorun_detectors

        current = get_autorun_detectors()
        if name in current:
            current = [new_name if n == name else n for n in current]
            set_autorun_detectors(current)
    except Exception:
        logger.exception("Failed to rename autorun entry for %s", name)

    return {
        "success": True,
        "old_name": name,
        "new_name": new_name,
        "pending_labelset_move": pending_move,
    }


# ---------------------------------------------------------------------------
# PUT /api/detectors/<name>/examples
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors/<name>/examples", methods=["PUT"])
@detectors_crud_bp.arguments(DetectorExamplesRequestSchema)
@detectors_crud_bp.response(200, DetectorExamplesResponseSchema)
@detectors_crud_bp.alt_response(404, description="Detector not found.")
def set_detector_examples(body: dict, name: str):
    """Set/replace the examples for a detector."""
    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        abort(404, message=f"Detector '{name}' not found")

    examples = body["examples"]

    data["examples"] = examples
    # Update text_query from first text example for backward compat
    text_examples = [e for e in examples if e.get("type") == "text" and e.get("value")]
    if text_examples:
        data["text_query"] = text_examples[0]["value"]
    _write_detector(path, data)

    return {"success": True, "name": name, "examples": examples}


# ---------------------------------------------------------------------------
# POST /api/detectors/combine
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors/combine", methods=["POST"])
@detectors_crud_bp.arguments(DetectorCombineRequestSchema)
@detectors_crud_bp.response(201, DetectorCombineResponseSchema)
@detectors_crud_bp.alt_response(400, description="Validation error (unsupported policy, mixed media types, etc.).")
@detectors_crud_bp.alt_response(404, description="A source detector was not found.")
@detectors_crud_bp.alt_response(409, description="A detector with the new name already exists.")
@detectors_crud_bp.alt_response(422, description="Combined labelset is empty after applying the conflict policy.")
def combine_detectors(body: dict):  # noqa: C901
    """Combine the labelsets of two or more detectors into a new detector.

    All source detectors must share the same ``media_type``.  The new
    detector's labelset is the merge of all source labelsets, keyed by
    ``Origin`` (importer + params + origin_name) and falling back to
    ``md5`` for legacy entries.  Per ``conflict_policy="drop"`` (the only
    supported policy today), any element key that appears with disagreeing
    labels across the sources is removed entirely.

    The combined detector is *purely a labelset entry* — no
    labelset-source is inherited from the sources, and the threshold/MLP
    are computed later when the detector is activated against a dataset.
    """
    names = body["names"]
    new_name = body["new_name"].strip()
    conflict_policy = body["conflict_policy"].strip()

    if not new_name:
        abort(400, message="new_name is required")
    if conflict_policy != "drop":
        abort(400, message=f"Unsupported conflict_policy: {conflict_policy!r}")

    new_path = _detector_path(new_name)
    if new_path.exists():
        abort(409, message=f"A detector named '{new_name}' already exists")

    from vtsearch.datasets.labelset import LabelSet

    sources: list[dict] = []
    for src_name in names:
        src_path = _detector_path(src_name)
        src_data = _read_detector(src_path)
        if src_data is None:
            abort(404, message=f"Detector '{src_name}' not found")
        sources.append(src_data)

    media_types = {s.get("media_type", "") for s in sources}
    if len(media_types) > 1:
        abort(400, message=f"All source detectors must share the same media_type; got {sorted(media_types)}")
    media_type = next(iter(media_types))
    if not media_type or media_type == "any":
        abort(400, message="Source detectors must have a specific media_type (not empty or 'any')")

    labelsets = [LabelSet.from_dict(s.get("labelset") or {}) for s in sources]
    merged = labelsets[0].merge(*labelsets[1:], conflict_policy=conflict_policy)

    if len(merged) == 0:
        abort(
            422,
            message=(
                f"Combined labelset is empty after applying conflict policy {conflict_policy!r}; nothing to save."
            ),
        )

    # Dedupe examples across sources by (type, value)
    merged_examples: list[dict] = []
    seen_ex: set[tuple[str, str]] = set()
    for s in sources:
        for ex in s.get("examples") or []:
            key = (ex.get("type", ""), ex.get("value", ""))
            if key in seen_ex:
                continue
            seen_ex.add(key)
            merged_examples.append(ex)

    text_query = ""
    for ex in merged_examples:
        if ex.get("type") == "text" and ex.get("value"):
            text_query = ex["value"]
            break
    if not text_query:
        for s in sources:
            if s.get("text_query"):
                text_query = s["text_query"]
                break

    new_data = {
        "name": new_name,
        "text_query": text_query,
        "media_example": "",
        "media_type": media_type,
        "examples": merged_examples,
        "created_at": time.time(),
        "labelset": merged.to_dict(),
        "combined_from": list(names),
    }
    _write_detector(new_path, new_data)

    return {
        "success": True,
        "name": new_name,
        "media_type": media_type,
        "num_labels": len(merged),
        "combined_from": list(names),
        "source_label_counts": [len(ls) for ls in labelsets],
        "examples": merged_examples,
    }
