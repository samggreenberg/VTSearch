"""Flask routes for the Label Importer API.

Endpoints
---------
GET  /api/label-importers
    List all registered label importers with their metadata and field definitions.

POST /api/label-importers/import/<importer_name>
    Run the named label importer.  Accepts ``multipart/form-data`` (when the
    importer has a ``"file"`` field) or JSON (for text-only importers).

    Returns::

        {
          "applied": <int>,
          "skipped": <int>,
          "missing_count": <int>,
          "missing": [<entry>, ...],
          "message": "<str>"
        }

    When ``missing_count`` is non-zero the response contains the label entries
    that could not be matched to any media in the current dataset (neither by
    ``origin`` + ``origin_name`` nor by ``md5``).  The frontend can prompt the
    user and then call ``POST /api/label-importers/ingest-missing`` to pull
    those medias from their origins.

POST /api/label-importers/ingest-missing
    Accept a list of missing label entries, re-ingest them from their origins,
    and apply the labels.
"""

from __future__ import annotations

from flask import Blueprint, jsonify

from vtsearch.labels.importers import get_label_importer, list_label_importers
from vtsearch.routes._shared import (
    extract_plugin_fields,
    get_json_safe,
    get_plugin_or_404,
    run_plugin_or_error,
    validate_filepath_field,
    validate_required_fields,
)
from vtsearch.state import (
    apply_label,
    build_media_lookup,
    medias,
    find_missing_entries,
    resolve_media_ids,
    snapshot_medias,
)

label_importers_bp = Blueprint("label_importers", __name__)


def _apply_labels(
    label_entries: list[dict],
    origin_lookup: dict[str, list[int]],
    md5_lookup: dict[str, list[int]],
    name_lookup: dict[str, list[int]] | None = None,
) -> tuple[int, int]:
    """Apply label entries to the global vote state.

    Returns ``(applied, skipped)`` counts.
    """
    applied = 0
    skipped = 0

    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
        if not cids:
            skipped += 1
            continue

        for cid in cids:
            apply_label(cid, label)
        applied += 1

    return applied, skipped


# ---------------------------------------------------------------------------
# GET /api/label-importers
# ---------------------------------------------------------------------------


@label_importers_bp.route("/api/label-importers", methods=["GET"])
def get_label_importers():
    """Return a list of all registered label importers."""
    return jsonify([imp.to_dict() for imp in list_label_importers()])


# ---------------------------------------------------------------------------
# POST /api/label-importers/import/<importer_name>
# ---------------------------------------------------------------------------


@label_importers_bp.route("/api/label-importers/import/<importer_name>", methods=["POST"])
def run_label_import(importer_name: str):
    """Run the named label importer and apply the resulting labels.

    Accepts ``multipart/form-data`` when the importer has a ``"file"`` field,
    or ``application/json`` for text-only importers.  In both cases the route
    builds a ``field_values`` dict and passes it to
    :meth:`~vtsearch.labels.importers.base.LabelImporter.run`.

    Returns JSON with ``applied``, ``skipped``, ``missing_count``,
    ``missing``, and ``message`` keys.  When ``missing_count > 0`` the
    client should prompt the user and optionally call
    ``POST /api/label-importers/ingest-missing`` with the ``missing`` list.
    """
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

    # Apply labels to global vote state
    origin_lookup, md5_lookup, name_lookup = build_media_lookup(snapshot_medias())
    applied, skipped = _apply_labels(label_entries, origin_lookup, md5_lookup, name_lookup)

    # Detect entries that could not be matched at all
    missing = find_missing_entries(label_entries, origin_lookup, md5_lookup, name_lookup)
    # Adjust skipped count: missing entries were already counted as skipped
    # by _apply_labels, but we report them separately now.
    skipped -= len(missing)

    # Auto-resolve: try to ingest missing medias from their origins
    ingested = 0
    resolved_applied = 0
    unresolved: list[dict] = []
    if missing:
        from vtsearch.datasets.ingest import ingest_missing_medias

        ingested = ingest_missing_medias(missing, medias)

        if ingested > 0:
            # Re-apply labels now that new medias are available.  These entries
            # were already removed from `skipped` above when we subtracted
            # `len(missing)`, so we only need to bump `applied` here.
            origin_lookup, md5_lookup, name_lookup = build_media_lookup(snapshot_medias())
            resolved_applied, _ = _apply_labels(missing, origin_lookup, md5_lookup, name_lookup)
            applied += resolved_applied

        # Check which entries still couldn't be resolved
        if ingested < len(missing):
            origin_lookup, md5_lookup, name_lookup = build_media_lookup(snapshot_medias())
            unresolved = find_missing_entries(missing, origin_lookup, md5_lookup, name_lookup)

    # Sync updated votes into the loaded model so the dashboard reflects
    # the new label count (num_training) immediately.
    if applied > 0:
        from vtsearch.models.label_sync import sync_labels_to_loaded_detector

        sync_labels_to_loaded_detector()

        from vtsearch.labels.sync import sync_to_labelset_source

        sync_to_labelset_source()

        from vtsearch.achievements import record_detector_import
        from vtsearch.state.core import get_active_detector_context

        record_detector_import(get_active_detector_context().detector_id)

    msg = f"Applied {applied} label(s), skipped {skipped}."
    if ingested > 0:
        msg += f" Auto-resolved {ingested} missing element(s) from their sources."
    if unresolved:
        msg += f" {len(unresolved)} element(s) could not be resolved."

    return jsonify(
        {
            "applied": applied,
            "skipped": skipped,
            "missing_count": len(unresolved),
            "missing": unresolved,
            "ingested": ingested,
            "message": msg,
        }
    )


# ---------------------------------------------------------------------------
# POST /api/label-importers/ingest-missing
# ---------------------------------------------------------------------------


@label_importers_bp.route("/api/label-importers/ingest-missing", methods=["POST"])
def ingest_missing():
    """Re-ingest missing medias from their origins, then apply their labels.

    Expects a JSON body::

        {"entries": [<label-entry>, ...]}

    Groups the entries by origin, runs each origin's dataset importer to
    recover the full media data (media bytes + embedding), appends the
    matched medias to the live dataset, and applies the labels.

    Returns::

        {"ingested": <int>, "applied": <int>, "message": "<str>"}
    """
    body = get_json_safe()
    entries = body.get("entries", [])

    if not isinstance(entries, list) or not entries:
        return jsonify({"error": "Request must contain a non-empty 'entries' list."}), 400

    from vtsearch.datasets.ingest import ingest_missing_medias

    ingested = ingest_missing_medias(entries, medias)

    # Now apply labels to the newly ingested medias
    origin_lookup, md5_lookup, name_lookup = build_media_lookup(snapshot_medias())
    applied, _ = _apply_labels(entries, origin_lookup, md5_lookup, name_lookup)

    # Sync updated votes into the loaded model so the dashboard reflects
    # the new label count immediately.
    if applied > 0:
        from vtsearch.models.label_sync import sync_labels_to_loaded_detector

        sync_labels_to_loaded_detector()

        from vtsearch.labels.sync import sync_to_labelset_source

        sync_to_labelset_source()

    return jsonify(
        {
            "ingested": ingested,
            "applied": applied,
            "message": f"Ingested {ingested} media(s), applied {applied} label(s).",
        }
    )
