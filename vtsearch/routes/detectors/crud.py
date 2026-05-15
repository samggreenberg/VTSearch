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
"""

from __future__ import annotations

import logging
import time

from flask import Blueprint, jsonify, request

from vtsearch.detectors.store import (
    _detector_path,
    _read_detector,
    _write_detector,
    get_detectors_dir,
)

logger = logging.getLogger(__name__)

detectors_crud_bp = Blueprint("detectors_crud", __name__)


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
def list_detectors():
    """Return all detectors (summary only, no full labelset)."""
    return jsonify({"detectors": _list_all()})


# ---------------------------------------------------------------------------
# POST /api/detectors
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors", methods=["POST"])
def create_detector():
    """Create a new detector.

    Expects JSON::

        {"name": "Dog Barks", "text_query": "dog barking sounds"}
    """
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "").strip()
    text_query = data.get("text_query", "").strip()
    media_example = data.get("media_example", "").strip()
    media_type = data.get("media_type", "").strip()
    examples = data.get("examples")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not text_query and not media_example and not examples:
        return jsonify({"error": "text_query, media_example, or examples is required"}), 400
    if not media_type or media_type == "any":
        return jsonify({"error": "media_type is required (must be a specific type, not 'any')"}), 400

    path = _detector_path(name)
    if path.exists():
        return jsonify({"error": f"A detector named '{name}' already exists"}), 409

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

    return jsonify(
        {
            "success": True,
            "name": name,
            "text_query": text_query,
            "media_example": media_example,
            "media_type": media_type,
            "examples": examples or [],
            "num_labels": 0,
        }
    ), 201


# ---------------------------------------------------------------------------
# GET /api/detectors/<name>
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors/<name>", methods=["GET"])
def get_detector(name: str):
    """Retrieve a single detector with its full labelset."""
    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        return jsonify({"error": f"Detector '{name}' not found"}), 404
    return jsonify(data)


# ---------------------------------------------------------------------------
# DELETE /api/detectors/<name>
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors/<name>", methods=["DELETE"])
def delete_detector(name: str):
    """Delete a detector."""
    path = _detector_path(name)
    if not path.exists():
        return jsonify({"error": f"Detector '{name}' not found"}), 404
    path.unlink()
    return jsonify({"success": True, "name": name})


# ---------------------------------------------------------------------------
# PUT /api/detectors/<name>/rename
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors/<name>/rename", methods=["PUT"])
def rename_detector(name: str):
    """Rename a detector.

    Expects JSON::

        {"new_name": "Cat Meows"}
    """
    old_path = _detector_path(name)
    data = _read_detector(old_path)
    if data is None:
        return jsonify({"error": f"Detector '{name}' not found"}), 404

    body = request.get_json(force=True, silent=True) or {}
    new_name = body.get("new_name", "").strip()
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400

    new_path = _detector_path(new_name)
    if new_path.exists() and new_path != old_path:
        return jsonify({"error": f"A detector named '{new_name}' already exists"}), 409

    data["name"] = new_name
    _write_detector(new_path, data)
    if new_path != old_path:
        old_path.unlink(missing_ok=True)

    # Update the detector registry entry that references this detector
    from vtsearch.detectors.registry import find_by_name, rename_detector as _rename_in_registry

    reg_entry = find_by_name(name)
    if reg_entry:
        _rename_in_registry(reg_entry["id"], new_name)

    # Rename autorun flag if present.
    try:
        from vtsearch.settings import get_autorun_detectors, set_autorun_detectors

        current = get_autorun_detectors()
        if name in current:
            current = [new_name if n == name else n for n in current]
            set_autorun_detectors(current)
    except Exception:
        logger.exception("Failed to rename autorun entry for %s", name)

    return jsonify({"success": True, "old_name": name, "new_name": new_name})


# ---------------------------------------------------------------------------
# PUT /api/detectors/<name>/examples
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors/<name>/examples", methods=["PUT"])
def set_detector_examples(name: str):
    """Set/replace the examples for a detector.

    Expects JSON::

        {"examples": [{"type": "text", "value": "dog barking"}]}
    """
    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        return jsonify({"error": f"Detector '{name}' not found"}), 404

    body = request.get_json(force=True, silent=True) or {}
    examples = body.get("examples")
    if examples is None:
        return jsonify({"error": "examples is required"}), 400

    data["examples"] = examples
    # Update text_query from first text example for backward compat
    text_examples = [e for e in examples if e.get("type") == "text" and e.get("value")]
    if text_examples:
        data["text_query"] = text_examples[0]["value"]
    _write_detector(path, data)

    return jsonify({"success": True, "name": name, "examples": examples})


# ---------------------------------------------------------------------------
# POST /api/detectors/combine
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/detectors/combine", methods=["POST"])
def combine_detectors():
    """Combine the labelsets of two or more detectors into a new detector.

    Expects JSON::

        {
            "names": ["Dog Barks", "More Dog Barks"],
            "new_name": "All Dog Barks",
            "conflict_policy": "drop"        # optional; default "drop"
        }

    All source detectors must share the same ``media_type``.  The new
    detector's labelset is the merge of all source labelsets, keyed by
    ``Origin`` (importer + params + origin_name) and falling back to ``md5``
    for legacy entries.  Per ``conflict_policy="drop"`` (the only supported
    policy today), any element key that appears with disagreeing labels
    across the sources is removed entirely.

    The combined detector is *purely a labelset entry* — no labelset-source
    is inherited from the sources, and the threshold/MLP are computed later
    when the detector is activated against a dataset.

    Returns ``201`` with a summary on success, or ``4xx`` on validation
    errors (missing names, unknown source, media-type mismatch, name
    collision, empty merged result).
    """
    body = request.get_json(force=True, silent=True) or {}
    names = body.get("names") or []
    new_name = (body.get("new_name") or "").strip()
    conflict_policy = (body.get("conflict_policy") or "drop").strip()

    if not isinstance(names, list) or len(names) < 2:
        return jsonify({"error": "names must be a list of at least 2 detector names"}), 400
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400
    if conflict_policy != "drop":
        return jsonify({"error": f"Unsupported conflict_policy: {conflict_policy!r}"}), 400

    new_path = _detector_path(new_name)
    if new_path.exists():
        return jsonify({"error": f"A detector named '{new_name}' already exists"}), 409

    from vtsearch.datasets.labelset import LabelSet

    sources: list[dict] = []
    for src_name in names:
        src_path = _detector_path(src_name)
        src_data = _read_detector(src_path)
        if src_data is None:
            return jsonify({"error": f"Detector '{src_name}' not found"}), 404
        sources.append(src_data)

    media_types = {s.get("media_type", "") for s in sources}
    if len(media_types) > 1:
        return jsonify(
            {"error": f"All source detectors must share the same media_type; got {sorted(media_types)}"}
        ), 400
    media_type = next(iter(media_types))
    if not media_type or media_type == "any":
        return jsonify({"error": "Source detectors must have a specific media_type (not empty or 'any')"}), 400

    labelsets = [LabelSet.from_dict(s.get("labelset") or {}) for s in sources]
    merged = labelsets[0].merge(*labelsets[1:], conflict_policy=conflict_policy)

    if len(merged) == 0:
        return jsonify(
            {
                "error": (
                    f"Combined labelset is empty after applying conflict policy {conflict_policy!r}; nothing to save."
                )
            }
        ), 422

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

    return jsonify(
        {
            "success": True,
            "name": new_name,
            "media_type": media_type,
            "num_labels": len(merged),
            "combined_from": list(names),
            "source_label_counts": [len(ls) for ls in labelsets],
            "examples": merged_examples,
        }
    ), 201
