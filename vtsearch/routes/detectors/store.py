"""Blueprint for detector routes.

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

POST /api/detectors/<name>/labels
    Save the current votes as the detector's labelset.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from vtsearch.detectors.store import (
    _detector_path,
    _read_detector,
    _write_detector,
    get_detectors_dir,
)

logger = logging.getLogger(__name__)

detectors_bp = Blueprint("detectors", __name__)


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


@detectors_bp.route("/api/detectors", methods=["GET"])
def list_detectors():
    """Return all detectors (summary only, no full labelset)."""
    return jsonify({"detectors": _list_all()})


# ---------------------------------------------------------------------------
# POST /api/detectors
# ---------------------------------------------------------------------------


@detectors_bp.route("/api/detectors", methods=["POST"])
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


@detectors_bp.route("/api/detectors/<name>", methods=["GET"])
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


@detectors_bp.route("/api/detectors/<name>", methods=["DELETE"])
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


@detectors_bp.route("/api/detectors/<name>/rename", methods=["PUT"])
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


@detectors_bp.route("/api/detectors/<name>/examples", methods=["PUT"])
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
# POST /api/detectors/<name>/labels
# ---------------------------------------------------------------------------


@detectors_bp.route("/api/detectors/<name>/labels", methods=["POST"])
def save_detector_labels(name: str):
    """Save the current votes as the detector's labelset.

    Reads good_votes/bad_votes from global state and the current medias
    to build a fresh LabelSet, then persists it into the detector's JSON file.
    """
    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        return jsonify({"error": f"Detector '{name}' not found"}), 404

    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.state import (
    bad_votes,
    good_votes,
    snapshot_medias,
    vote_region_boxes,
)

    labelset = LabelSet.from_clips_and_votes(
        snapshot_medias(),
        good_votes,
        bad_votes,
        expand_dupes=False,
        vote_region_boxes=dict(vote_region_boxes),
    )
    data["labelset"] = labelset.to_dict()
    _write_detector(path, data)

    # Also update the detector registry entry if one exists
    from vtsearch.detectors.registry import find_by_name, update_detector

    import time as _time

    reg_entry = find_by_name(name)
    if reg_entry:
        update_detector(reg_entry["id"], num_training=len(labelset), last_trained_at=_time.time())

    return jsonify(
        {
            "success": True,
            "name": name,
            "num_labels": len(labelset),
        }
    )


# ---------------------------------------------------------------------------
# POST /api/detectors/<name>/import-labels/<importer_name>
# ---------------------------------------------------------------------------


@detectors_bp.route(
    "/api/detectors/<name>/import-labels/<importer_name>",
    methods=["POST"],
)
def import_labels_into_detector(name: str, importer_name: str):
    """Run a label importer and merge results into this detector's labelset.

    Unlike the regular ``/api/label-importers/import/`` route, this does
    **not** require a dataset to be loaded.  The imported label entries are
    merged directly into the detector's persisted labelset, and the
    detector-registry entry is updated so the dashboard reflects the new
    count.

    When the detector is loaded into memory, the new labels are also resolved
    against the loaded dataset's medias, applied to the detector's votes, and
    a fresh MLP is trained with a cross-validated threshold — all inside the
    loaded detector context.

    Returns JSON with ``applied``, ``skipped``, ``num_labels``, and
    ``message`` keys.
    """
    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        return jsonify({"error": f"Detector '{name}' not found"}), 404

    from vtsearch.labels.importers import get_label_importer, list_label_importers
    from vtsearch.routes._shared import (
        extract_plugin_fields,
        get_plugin_or_404,
        run_plugin_or_error,
        validate_filepath_field,
        validate_required_fields,
    )

    importer, err = get_plugin_or_404(get_label_importer, list_label_importers, importer_name, "label importer")
    if err:
        return err

    field_values = extract_plugin_fields(importer)
    err = validate_required_fields(importer, field_values)
    if err:
        return err
    err = validate_filepath_field(field_values)
    if err:
        return err

    label_entries, err = run_plugin_or_error(importer, "run", field_values)
    if err:
        return err
    if not isinstance(label_entries, list):
        return jsonify({"error": "Importer did not return a list of label dicts."}), 500

    # ------------------------------------------------------------------
    # 1) Merge into the persisted labelset (always, whether loaded or not)
    # ------------------------------------------------------------------
    from vtsearch.datasets.labelset import LabeledElement, LabelSet

    existing_ls = LabelSet.from_dict(data.get("labelset") or {})

    # Build a set of existing (md5, label) pairs for dedup
    existing_keys: set[tuple[str, str]] = set()
    for el in existing_ls.elements:
        if el.md5:
            existing_keys.add((el.md5, el.label))

    applied = 0
    skipped = 0
    new_entries: list[dict] = []
    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        md5 = entry.get("md5", "")
        if md5 and (md5, label) in existing_keys:
            skipped += 1
            continue
        elem = LabeledElement.from_dict(entry)
        existing_ls.elements.append(elem)
        new_entries.append(entry)
        if md5:
            existing_keys.add((md5, label))
        applied += 1

    data["labelset"] = existing_ls.to_dict()
    _write_detector(path, data)

    # Update the detector registry entry
    from vtsearch.detectors.registry import find_by_name, update_detector

    reg_entry = find_by_name(name)
    if reg_entry:
        update_detector(reg_entry["id"], num_training=len(existing_ls), last_trained_at=time.time())

    # ------------------------------------------------------------------
    # 2) If the detector is loaded, resolve + apply + retrain in context
    # ------------------------------------------------------------------
    resolved = 0
    trained = False
    if applied > 0 and reg_entry:
        from vtsearch.state.core import get_detector_context

        det_ctx = get_detector_context(reg_entry["id"])
        if det_ctx is not None:
            resolved, trained = _apply_and_retrain(
                reg_entry["id"],
                det_ctx,
                new_entries,
                name,
            )

    msg = f"Added {applied} label(s) to detector '{name}', skipped {skipped}."
    if resolved > 0:
        msg += f" Resolved {resolved} into the loaded detector."
    if trained:
        msg += " Retrained MLP."
    return jsonify(
        {
            "applied": applied,
            "skipped": skipped,
            "resolved": resolved,
            "trained": trained,
            "num_labels": len(existing_ls),
            "message": msg,
        }
    )


# ---------------------------------------------------------------------------
# POST /api/detectors/combine
# ---------------------------------------------------------------------------


@detectors_bp.route("/api/detectors/combine", methods=["POST"])
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
        return jsonify({"error": f"All source detectors must share the same media_type; got {sorted(media_types)}"}), 400
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


# ---------------------------------------------------------------------------
# GET /api/detectors/<name>/labels-detail
# ---------------------------------------------------------------------------


@detectors_bp.route("/api/detectors/<name>/labels-detail", methods=["GET"])
def get_detector_labels_detail(name: str):
    """Return the detector's saved labelset elements with right-pane render data.

    Each element gets a stable ``id`` (derived from its origin/md5 identity)
    plus ``label``, ``media_type``, display ``name``, and — when the
    element resolves into the active dataset — its current ``cid``,
    ``time`` (click time), and ``score`` (last learned-sort score).

    This is the right pane's data source in label/train mode.  Unlike
    ``/api/votes`` it is *not* gated on the loaded dataset, so detector
    labels survive across dataset switches.
    """
    from vtsearch.detectors.labelset_elements import build_labels_detail

    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        return jsonify({"error": f"Detector '{name}' not found"}), 404
    return jsonify(build_labels_detail(data))


# ---------------------------------------------------------------------------
# GET /api/detectors/<name>/labels/<element_id>/preview
# ---------------------------------------------------------------------------


@detectors_bp.route(
    "/api/detectors/<name>/labels/<element_id>/preview",
    methods=["GET"],
)
def preview_detector_label(name: str, element_id: str):
    """Stream the underlying media file for a saved labelset element.

    Resolves the element via its origin (using the importer's
    ``resolve_file()`` hook) and serves the raw bytes with a mimetype
    chosen by the detector's ``media_type``.  Returns 404 if the element
    is unknown or its file cannot be located.
    """
    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.detectors.labelset_elements import find_element_by_id, resolve_element_to_path

    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        return jsonify({"error": f"Detector '{name}' not found"}), 404

    labelset = LabelSet.from_dict(data.get("labelset") or {})
    found = find_element_by_id(labelset.elements, element_id)
    if found is None:
        return jsonify({"error": "Label element not found"}), 404

    _, elem = found
    media_type = data.get("media_type", "")

    with resolve_element_to_path(elem) as file_path:
        if file_path is None or not file_path.is_file():
            return jsonify({"error": "Element media file unavailable"}), 404

        if media_type == "text":
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                return jsonify({"error": "Element media file unreadable"}), 404
            return jsonify(
                {
                    "content": content,
                    "word_count": len(content.split()),
                    "character_count": len(content),
                }
            )

        try:
            file_bytes = file_path.read_bytes()
        except OSError:
            return jsonify({"error": "Element media file unreadable"}), 404

        suffix = file_path.suffix.lower()

    mimetype = _MIMETYPE_BY_SUFFIX.get(suffix) or _DEFAULT_MIMETYPE_BY_TYPE.get(media_type, "application/octet-stream")
    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mimetype,
        download_name=elem.origin_name or elem.filename or f"label{suffix or ''}",
    )


_MIMETYPE_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

_DEFAULT_MIMETYPE_BY_TYPE = {
    "audio": "audio/wav",
    "image": "image/jpeg",
    "video": "video/mp4",
    "document": "application/pdf",
    "text": "text/plain",
}


# ---------------------------------------------------------------------------
# GET /api/detectors/<name>/labels/<element_id>/thumbnail
# ---------------------------------------------------------------------------


def _in_memory_thumbnail_response(media: dict, media_type: str):
    """Build a small thumbnail send_file from an in-memory media dict.

    Images: serves the cached ``media_bytes``. Audio/video/document: defers
    to the media-type's ``image_response`` (cached waveform / midframe PNG).
    Returns ``None`` if no thumbnail can be produced from memory.
    """
    if media_type == "image":
        media_bytes = media.get("media_bytes")
        if not media_bytes:
            return None
        filename = media.get("filename", "") or ""
        suffix = Path(filename).suffix.lower()
        mimetype = _MIMETYPE_BY_SUFFIX.get(suffix) or "image/jpeg"
        return send_file(
            io.BytesIO(media_bytes),
            mimetype=mimetype,
            download_name=f"media_{media.get('id', 0)}{suffix or '.jpg'}",
        )

    from vtsearch.media import get as get_media_type  # noqa: PLC0415

    try:
        mt = get_media_type(media_type)
    except KeyError:
        return None
    if not hasattr(mt, "image_response"):
        return None
    resp = mt.image_response(media)
    if resp is None:
        return None
    return send_file(
        io.BytesIO(resp.data),
        mimetype=resp.mimetype,
        download_name=resp.download_name,
    )


def _origin_thumbnail_response(file_path: Path, media_type: str, elem):
    """Build a thumbnail response from an on-disk file resolved via origin."""
    if media_type == "image":
        suffix = file_path.suffix.lower()
        mimetype = _MIMETYPE_BY_SUFFIX.get(suffix) or "image/jpeg"
        try:
            return send_file(
                io.BytesIO(file_path.read_bytes()),
                mimetype=mimetype,
                download_name=elem.origin_name or elem.filename or f"label{suffix}",
            )
        except OSError:
            return jsonify({"error": "Element media file unreadable"}), 404

    if media_type == "audio":
        from vtsearch.media.audio.media_type import generate_waveform_thumbnail_from_file  # noqa: PLC0415

        thumb = generate_waveform_thumbnail_from_file(file_path)
        if thumb is None:
            return jsonify({"error": "Could not generate audio thumbnail"}), 500
        return send_file(
            io.BytesIO(thumb),
            mimetype="image/png",
            download_name=f"{file_path.stem}_waveform.png",
        )

    if media_type == "video":
        from vtsearch.media.video.media_type import generate_video_thumbnail_from_file  # noqa: PLC0415

        thumb = generate_video_thumbnail_from_file(file_path)
        if thumb is None:
            return jsonify({"error": "Could not generate video thumbnail"}), 500
        return send_file(
            io.BytesIO(thumb),
            mimetype="image/png",
            download_name=f"{file_path.stem}_frame.png",
        )

    return jsonify({"error": f"No thumbnail for media type '{media_type}'"}), 404


@detectors_bp.route(
    "/api/detectors/<name>/labels/<element_id>/thumbnail",
    methods=["GET"],
)
def thumbnail_detector_label(name: str, element_id: str):
    """Stream a small thumbnail image for a saved labelset element.

    Mirrors :func:`vtsearch.routes.media.list.media_image` for the right pane:
    audio elements get a waveform PNG, video elements a mid-frame PNG, image
    elements get the file bytes. When the element resolves into the active
    dataset we serve the cached in-memory ``thumbnail_bytes`` (fast path);
    otherwise we resolve the underlying file via the importer and generate
    on the fly. Much smaller than ``/preview`` (which serves the full file).
    """
    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.detectors.labelset_elements import (
        find_element_by_id,
        resolve_current_dataset_cid,
        resolve_element_to_path,
    )
    from vtsearch.state import get_media

    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        return jsonify({"error": f"Detector '{name}' not found"}), 404

    labelset = LabelSet.from_dict(data.get("labelset") or {})
    found = find_element_by_id(labelset.elements, element_id)
    if found is None:
        return jsonify({"error": "Label element not found"}), 404

    _, elem = found
    media_type = data.get("media_type", "") or ""

    cid = resolve_current_dataset_cid(elem)
    if cid is not None:
        media = get_media(cid)
        if media:
            resp = _in_memory_thumbnail_response(media, media_type)
            if resp is not None:
                return resp

    with resolve_element_to_path(elem) as file_path:
        if file_path is None or not file_path.is_file():
            return jsonify({"error": "Element media file unavailable"}), 404

        return _origin_thumbnail_response(file_path, media_type, elem)


# ---------------------------------------------------------------------------
# POST /api/detectors/<name>/labels/<element_id>/vote
# ---------------------------------------------------------------------------


@detectors_bp.route(
    "/api/detectors/<name>/labels/<element_id>/vote",
    methods=["POST"],
)
def vote_detector_label(name: str, element_id: str):
    """Toggle the label on a saved labelset element.

    Body: ``{"vote": "good"}`` or ``{"vote": "bad"}``.

    Toggle semantics mirror :func:`~vtsearch.state.toggle_vote`: the same
    vote on an element with that label removes the element; the opposite
    vote flips it.  When the element resolves into the active dataset, the
    detector's in-memory ``good_votes`` / ``bad_votes`` are kept in sync so
    MLP retraining and learned-sort see the change.
    """
    from vtsearch.detectors.labelset_elements import (
        apply_element_vote_in_data,
        resolve_current_dataset_cid,
    )

    body = request.get_json(force=True, silent=True) or {}
    vote = body.get("vote", "")
    if vote not in ("good", "bad"):
        return jsonify({"error": "vote must be 'good' or 'bad'"}), 400

    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        return jsonify({"error": f"Detector '{name}' not found"}), 404

    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.detectors.labelset_elements import find_element_by_id

    pre_labelset = LabelSet.from_dict(data.get("labelset") or {})
    pre_found = find_element_by_id(pre_labelset.elements, element_id)
    if pre_found is None:
        return jsonify({"error": "Label element not found"}), 404
    _, pre_elem = pre_found

    cid_before = resolve_current_dataset_cid(pre_elem)

    changed, _updated, action = apply_element_vote_in_data(data, element_id, vote)
    if not changed:
        return jsonify({"ok": True, "action": action})

    _write_detector(path, data)

    # Mirror into in-memory votes when the element resolves into the active
    # dataset, so the MLP and learned-sort stay aligned with the labelset.
    if cid_before is not None:
        from vtsearch.state import toggle_vote

        toggle_vote(cid_before, vote)

    from vtsearch.detectors.registry import find_by_name, update_detector

    reg_entry = find_by_name(name)
    if reg_entry:
        new_count = len(LabelSet.from_dict(data.get("labelset") or {}))
        update_detector(reg_entry["id"], num_training=new_count, last_trained_at=time.time())

    return jsonify({"ok": True, "action": action})


# Canonical location: vtsearch.models.training_workflow
from vtsearch.models.training_workflow import apply_and_retrain as _apply_and_retrain  # noqa: E402
