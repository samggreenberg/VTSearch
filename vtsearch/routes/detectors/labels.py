"""Blueprint for detector label routes.

Each detector has an associated labelset persisted on disk alongside its
JSON file.  These routes manage that labelset: saving votes, importing
external labels, listing the labels for the right pane, and serving
per-label previews / thumbnails / toggle-vote actions.

Endpoints
---------
POST /api/detectors/<name>/labels
    Save the current votes as the detector's labelset.

POST /api/detectors/<name>/import-labels/<importer_name>
    Run a label importer and merge results into this detector's labelset.

GET  /api/detectors/<name>/labels-detail
    Return the detector's saved labelset elements with right-pane render data.

GET  /api/detectors/<name>/labels/<element_id>/preview
    Stream the underlying media file for a saved labelset element.

GET  /api/detectors/<name>/labels/<element_id>/thumbnail
    Stream a small thumbnail image for a saved labelset element.

POST /api/detectors/<name>/labels/<element_id>/vote
    Toggle the label on a saved labelset element.

Migrated to ``flask_smorest`` for the JSON-shaped routes (save, labels-detail,
vote). ``import-labels`` keeps its plain-Flask route on the same smorest
blueprint; its body shape depends on the importer plugin and isn't
described in the OpenAPI spec, but runtime validation goes through
:func:`validate_plugin_args` so the per-plugin field types are enforced
and schema-level failures surface as 422.  See *Resolved questions /
Plugin field endpoints* in ``docs/plans/openapi-schema.md``.  The
``preview`` and ``thumbnail`` routes serve binary bodies (or a tiny
content-only JSON for text media); they declare their non-default JSON
error responses via ``alt_response`` but no success schema, so OpenAPI
describes the error shape without lying about the success body.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

from flask import jsonify, send_file
from flask_smorest import Blueprint, abort

from vtscore.detectors.store import (
    _detector_path,
    _read_detector,
    _write_detector,
)
from vtscore.detectors.workflow import apply_and_retrain as _apply_and_retrain
from vtsearch.routes._shared import image_thumbnail_response
from vtsearch.schemas.detectors import (
    DetectorLabelsDetailResponseSchema,
    DetectorLabelVoteRequestSchema,
    DetectorLabelVoteResponseSchema,
    DetectorSaveLabelsResponseSchema,
)

logger = logging.getLogger(__name__)

detectors_labels_bp = Blueprint(
    "detectors_labels",
    __name__,
    description="Persist, list, preview, and vote on a detector's labelset.",
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
# POST /api/detectors/<name>/labels
# ---------------------------------------------------------------------------


@detectors_labels_bp.route("/api/detectors/<name>/labels", methods=["POST"])
@detectors_labels_bp.response(200, DetectorSaveLabelsResponseSchema)
@detectors_labels_bp.alt_response(404, description="Detector not found.")
@detectors_labels_bp.alt_response(409, description="Detector vote state is not aligned with the active dataset.")
def save_detector_labels(name: str):
    """Save the current votes as the detector's labelset.

    Reads good_votes/bad_votes from global state and the current medias
    to build a fresh LabelSet, then persists it into the detector's JSON file.
    """
    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        abort(404, message=f"Detector '{name}' not found")

    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.dataset_sync import validated_vote_snapshot
    from vtscore.detectors.input_spec import extract_input_spec_from_medias

    snap = validated_vote_snapshot()
    if not snap.safe:
        # Refuse to overwrite the on-disk labelset with an empty composition
        # when the (dataset, detector) state can't be proved consistent.
        # Surfacing 409 lets the frontend retry on a stable request instead
        # of silently destroying labels.
        abort(409, message="Cannot save labels: detector vote state is not aligned with the active dataset")
    medias_snap = snap.medias
    labelset = LabelSet.from_clips_and_votes(
        medias_snap,
        snap.good_votes,
        snap.bad_votes,
        expand_dupes=False,
        vote_region_boxes=snap.vote_region_boxes,
    )
    data["labelset"] = labelset.to_dict()

    # Capture the active dataset's clipper into the detector's input_spec
    # so downstream consumers (CLI autodetect, labelset-source sync) can
    # tell what input format this detector was trained on.  ``None`` means
    # "no clipper / default clipper"; we drop any previously-stored
    # input_spec in that case so the field stays in sync with reality.
    captured_spec = extract_input_spec_from_medias(medias_snap)
    if captured_spec is not None:
        data["input_spec"] = captured_spec
    elif "input_spec" in data:
        data.pop("input_spec", None)
    _write_detector(path, data)

    # Also update the detector registry entry if one exists
    from vtscore.detectors.registry import find_by_name, update_detector

    import time as _time

    reg_entry = find_by_name(name)
    if reg_entry:
        update_detector(reg_entry["id"], num_training=len(labelset), last_trained_at=_time.time())

    return {
        "success": True,
        "name": name,
        "num_labels": len(labelset),
    }


# ---------------------------------------------------------------------------
# POST /api/detectors/<name>/import-labels/<importer_name>
#
# Plugin-field route: body shape depends on the importer plugin and isn't
# described in the OpenAPI spec.  Runtime validation goes through
# :func:`validate_plugin_args` (per-plugin schema built from the importer's
# :attr:`fields`), so missing required fields / invalid select values
# raise 422.  See ``docs/plans/openapi-schema.md`` (Resolved questions /
# Plugin field endpoints).
# ---------------------------------------------------------------------------


@detectors_labels_bp.route(
    "/api/detectors/<name>/import-labels/<importer_name>",
    methods=["POST"],
)
def import_labels_into_detector(name: str, importer_name: str):  # noqa: C901
    """Run a label importer and merge results into this detector's labelset.

    Unlike the regular ``/api/label-importers/import/`` route, this does
    **not** require a dataset to be loaded.  The imported label entries are
    merged directly into the detector's persisted labelset, and the
    detector-registry entry is updated so the dashboard reflects the new
    count.

    When the detector is loaded into memory, the new labels are also resolved
    against the loaded dataset's medias, applied to the detector's votes, and
    a fresh MLP is trained with a cross-validated threshold, all inside the
    loaded detector context.

    Plugin-dependent body shape: not described in the OpenAPI spec.
    """
    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        return jsonify({"error": f"Detector '{name}' not found"}), 404

    from vtscore.labels.importers import get_label_importer, list_label_importers
    from vtsearch.routes._shared import (
        get_plugin_or_404,
        run_plugin_or_error,
        validate_plugin_args,
    )

    importer, err = get_plugin_or_404(get_label_importer, list_label_importers, importer_name, "label importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_values = validate_plugin_args(importer)

    label_entries, err = run_plugin_or_error(importer, "run", field_values)
    if err:
        return err
    if not isinstance(label_entries, list):
        return jsonify({"error": "Importer did not return a list of label dicts."}), 500

    # ------------------------------------------------------------------
    # 1) Merge into the persisted labelset (always, whether loaded or not)
    # ------------------------------------------------------------------
    from vtscore.datasets.labelset import LabeledElement, LabelSet

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
    from vtscore.detectors.registry import find_by_name, update_detector

    reg_entry = find_by_name(name)
    if reg_entry:
        update_detector(reg_entry["id"], num_training=len(existing_ls), last_trained_at=time.time())

    # ------------------------------------------------------------------
    # 2) If the detector is loaded, resolve + apply + retrain in context
    # ------------------------------------------------------------------
    resolved = 0
    trained = False
    if applied > 0 and reg_entry:
        from vtscore.state.core import get_detector_context

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
# GET /api/detectors/<name>/labels-detail
# ---------------------------------------------------------------------------


@detectors_labels_bp.route("/api/detectors/<name>/labels-detail", methods=["GET"])
@detectors_labels_bp.response(200, DetectorLabelsDetailResponseSchema)
@detectors_labels_bp.alt_response(404, description="Detector not found.")
def get_detector_labels_detail(name: str):
    """Return the detector's saved labelset elements with right-pane render data.

    Each element gets a stable ``id`` (derived from its origin/md5 identity)
    plus ``label``, ``media_type``, display ``name``, and (when the
    element resolves into the active dataset) its current ``cid``,
    ``time`` (click time), and ``score`` (last learned-sort score).

    This is the right pane's data source in label/train mode.  Unlike
    ``/api/votes`` it is *not* gated on the loaded dataset, so detector
    labels survive across dataset switches.
    """
    from vtscore.detectors.labelset_elements import build_labels_detail

    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        abort(404, message=f"Detector '{name}' not found")
    return build_labels_detail(data)


# ---------------------------------------------------------------------------
# GET /api/detectors/<name>/labels/<element_id>/preview
#
# Binary response (or a small content-only JSON for text). flask-smorest only
# describes the alt error responses; the success body is not modeled.
# ---------------------------------------------------------------------------


@detectors_labels_bp.route(
    "/api/detectors/<name>/labels/<element_id>/preview",
    methods=["GET"],
)
@detectors_labels_bp.alt_response(404, description="Detector, element, or media file not found.")
def preview_detector_label(name: str, element_id: str):
    """Stream the underlying media file for a saved labelset element.

    Resolves the element via its origin (using the importer's
    ``resolve_file()`` hook) and serves the raw bytes with a mimetype
    chosen by the detector's ``media_type``.  Returns 404 if the element
    is unknown or its file cannot be located.
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.labelset_elements import find_element_by_id, resolve_element_to_path

    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        abort(404, message=f"Detector '{name}' not found")

    labelset = LabelSet.from_dict(data.get("labelset") or {})
    found = find_element_by_id(labelset.elements, element_id)
    if found is None:
        abort(404, message="Label element not found")

    _, elem = found
    media_type = data.get("media_type", "")

    with resolve_element_to_path(elem) as file_path:
        if file_path is None or not file_path.is_file():
            abort(404, message="Element media file unavailable")

        if media_type == "text":
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                abort(404, message="Element media file unreadable")
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
            abort(404, message="Element media file unreadable")

        suffix = file_path.suffix.lower()

    mimetype = _MIMETYPE_BY_SUFFIX.get(suffix) or _DEFAULT_MIMETYPE_BY_TYPE.get(media_type, "application/octet-stream")
    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mimetype,
        download_name=elem.origin_name or elem.filename or f"label{suffix or ''}",
    )


# ---------------------------------------------------------------------------
# GET /api/detectors/<name>/labels/<element_id>/thumbnail
#
# Binary response. See note above ``preview``.
# ---------------------------------------------------------------------------


def _in_memory_thumbnail_response(media: dict, media_type: str):
    """Build a small thumbnail send_file from an in-memory media dict.

    Images: serve the resolved media bytes via the media type's
    ``media_response``, which lazily reads from ``media_path`` / ``media_url``
    when ``media_bytes`` is not held in memory (thin-loaded datasets, e.g. a
    local folder import). This mirrors the center viewer's
    ``/api/medias/<id>/image`` byte resolution so a thumbnail never 404s for an
    item the center can display. Audio/video/document: defer to the media
    type's ``image_response`` (cached waveform / midframe PNG). Returns
    ``None`` if no thumbnail can be produced.
    """
    from vtscore.media import get as get_media_type  # noqa: PLC0415

    try:
        mt = get_media_type(media_type)
    except KeyError:
        return None

    if media_type == "image":
        resp = mt.media_response(media)
        if not isinstance(resp.data, (bytes, bytearray)) or not resp.data:
            return None
        return image_thumbnail_response(bytes(resp.data), resp.mimetype, resp.download_name)

    image_response_fn = getattr(mt, "image_response", None)
    if image_response_fn is None:
        return None
    resp = image_response_fn(media)
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
            image_bytes = file_path.read_bytes()
        except OSError:
            abort(404, message="Element media file unreadable")
        download_name = elem.origin_name or elem.filename or f"label{suffix}"
        return image_thumbnail_response(image_bytes, mimetype, download_name)

    if media_type == "audio":
        from vtscore.media.audio.media_type import generate_waveform_thumbnail_from_file  # noqa: PLC0415

        thumb = generate_waveform_thumbnail_from_file(file_path)
        if thumb is None:
            abort(500, message="Could not generate audio thumbnail")
        return send_file(
            io.BytesIO(thumb),
            mimetype="image/png",
            download_name=f"{file_path.stem}_waveform.png",
        )

    if media_type == "video":
        from vtscore.media.video.media_type import generate_video_thumbnail_from_file  # noqa: PLC0415

        thumb = generate_video_thumbnail_from_file(file_path)
        if thumb is None:
            abort(500, message="Could not generate video thumbnail")
        return send_file(
            io.BytesIO(thumb),
            mimetype="image/png",
            download_name=f"{file_path.stem}_frame.png",
        )

    abort(404, message=f"No thumbnail for media type '{media_type}'")


@detectors_labels_bp.route(
    "/api/detectors/<name>/labels/<element_id>/thumbnail",
    methods=["GET"],
)
@detectors_labels_bp.alt_response(404, description="Detector, element, or media file not found.")
@detectors_labels_bp.alt_response(500, description="Thumbnail could not be generated.")
def thumbnail_detector_label(name: str, element_id: str):
    """Stream a small thumbnail image for a saved labelset element.

    Mirrors :func:`vtsearch.routes.media.list.media_image` for the right pane:
    audio elements get a waveform PNG, video elements a mid-frame PNG, image
    elements get the file bytes. When the element resolves into the active
    dataset we serve the cached in-memory ``thumbnail_bytes`` (fast path);
    otherwise we resolve the underlying file via the importer and generate
    on the fly. Much smaller than ``/preview`` (which serves the full file).
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.labelset_elements import (
        find_element_by_id,
        resolve_current_dataset_cid,
        resolve_element_to_path,
    )
    from vtsearch.state import get_media

    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        abort(404, message=f"Detector '{name}' not found")

    labelset = LabelSet.from_dict(data.get("labelset") or {})
    found = find_element_by_id(labelset.elements, element_id)
    if found is None:
        abort(404, message="Label element not found")

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
            abort(404, message="Element media file unavailable")

        return _origin_thumbnail_response(file_path, media_type, elem)


# ---------------------------------------------------------------------------
# POST /api/detectors/<name>/labels/<element_id>/vote
# ---------------------------------------------------------------------------


@detectors_labels_bp.route(
    "/api/detectors/<name>/labels/<element_id>/vote",
    methods=["POST"],
)
@detectors_labels_bp.arguments(DetectorLabelVoteRequestSchema)
@detectors_labels_bp.response(200, DetectorLabelVoteResponseSchema)
@detectors_labels_bp.alt_response(404, description="Detector or label element not found.")
def vote_detector_label(body: dict, name: str, element_id: str):
    """Toggle the label on a saved labelset element.

    Body: ``{"vote": "good"}`` or ``{"vote": "bad"}``.

    Toggle semantics mirror :func:`~vtsearch.state.toggle_vote`: the same
    vote on an element with that label removes the element; the opposite
    vote flips it.  When the element resolves into the active dataset, the
    detector's in-memory ``good_votes`` / ``bad_votes`` are kept in sync so
    MLP retraining and learned-sort see the change.
    """
    from vtscore.detectors.labelset_elements import (
        apply_element_vote_in_data,
        resolve_current_dataset_cid,
    )

    vote = body["vote"]

    path = _detector_path(name)
    data = _read_detector(path)
    if data is None:
        abort(404, message=f"Detector '{name}' not found")

    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.labelset_elements import find_element_by_id

    pre_labelset = LabelSet.from_dict(data.get("labelset") or {})
    pre_found = find_element_by_id(pre_labelset.elements, element_id)
    if pre_found is None:
        abort(404, message="Label element not found")
    _, pre_elem = pre_found

    cid_before = resolve_current_dataset_cid(pre_elem)

    changed, _updated, action = apply_element_vote_in_data(data, element_id, vote)
    if not changed:
        return {"ok": True, "action": action}

    _write_detector(path, data)

    # Mirror into in-memory votes when the element resolves into the active
    # dataset, so the MLP and learned-sort stay aligned with the labelset.
    if cid_before is not None:
        from vtsearch.state import toggle_vote

        toggle_vote(cid_before, vote)

    from vtscore.detectors.registry import find_by_name, update_detector

    reg_entry = find_by_name(name)
    if reg_entry:
        new_count = len(LabelSet.from_dict(data.get("labelset") or {}))
        update_detector(reg_entry["id"], num_training=new_count, last_trained_at=time.time())

    return {"ok": True, "action": action}


# ---------------------------------------------------------------------------
# Per-plugin typed routes for /api/detectors/<name>/import-labels/<importer>.
# The detector ``<name>`` stays dynamic; only the importer segment is
# specialized per plugin so its body schema appears in /api/openapi.json
# with real per-field types.  Unknown importer names fall through to the
# parameterized route above (preserving the legacy 404 message).
# Plugins with file fields stay on the parameterized fallback.
# ---------------------------------------------------------------------------

from vtscore.labels.importers import list_label_importers as _list_label_importers  # noqa: E402
from vtsearch.routes._shared import register_plugin_typed_routes as _register_plugin_typed_routes  # noqa: E402

_register_plugin_typed_routes(
    detectors_labels_bp,
    list_plugins=_list_label_importers,
    path_template="/api/detectors/<name>/import-labels/{plugin_name}",
    endpoint_prefix="import_labels_into_detector",
    delegate=import_labels_into_detector,
)
