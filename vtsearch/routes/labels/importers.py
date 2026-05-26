"""Flask routes for the Label Importer API.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.

Endpoints
---------
GET  /api/label-importers
    List all registered label importers with their metadata and field
    definitions.

POST /api/label-importers/import/<importer_name>
    Run the named label importer. Accepts ``multipart/form-data`` (when
    the importer has a ``"file"`` field) or JSON (for text-only
    importers). The request body shape is plugin-dependent and not
    described in the OpenAPI spec - instead, the body is validated at
    request time via :func:`validate_plugin_args`, which builds a
    marshmallow schema from the named importer's :attr:`fields`
    declaration. Schema-level rejects (missing required field, invalid
    select value) surface as ``422`` with the standard ``errors``
    envelope; handler-level rejects (path traversal, plugin error) keep
    their original HTTP codes (400 / 500) with the standard ``message``
    envelope. See *Resolved questions / Plugin field endpoints* in
    ``docs/plans/openapi-schema.md``.

POST /api/label-importers/ingest-missing
    Accept a list of missing label entries, re-ingest them from their
    origins, and apply the labels. JSON-only - fully spec'd.

Partial-failure semantics
-------------------------
:func:`_apply_labels` isolates each entry with a per-entry try/except.
If ``apply_label`` raises mid-loop, that entry is recorded in the
``failed`` return list and the loop continues with the remaining
entries.  This addresses logical-bug-audit H31 - a single bad entry no
longer aborts the whole import and silently strands the already-applied
labels behind a 500 response.  The handler still runs the downstream
sync block (``sync_labels_to_loaded_detector``,
``sync_to_labelset_source``, ``record_detector_import``) whenever
``applied > 0`` so the in-memory detector, the labelset source, and the
achievement counters all reflect what actually landed.
"""

from __future__ import annotations

import logging

from flask import jsonify
from flask_smorest import Blueprint

logger = logging.getLogger(__name__)

from vtscore.labels.importers import get_label_importer, list_label_importers
from vtsearch.routes._shared import (
    get_plugin_or_404,
    require_dataset_header,
    require_detector_header,
    run_plugin_or_error,
    validate_plugin_args,
)
from vtsearch.schemas.labels import (
    IngestMissingRequestSchema,
    IngestMissingResponseSchema,
    LabelImporterEntrySchema,
)
from vtsearch.state import (
    apply_label,
    build_media_lookup,
    medias,
    find_missing_entries,
    resolve_media_ids,
    snapshot_medias,
)

label_importers_bp = Blueprint(
    "label_importers",
    __name__,
    description="List and run label importers; resolve missing-media entries.",
)


def _apply_labels(
    label_entries: list[dict],
    origin_lookup: dict[str, list[int]],
    md5_lookup: dict[str, list[int]],
    name_lookup: dict[str, list[int]] | None = None,
) -> tuple[int, int, list[dict]]:
    """Apply label entries to the global vote state.

    Returns ``(applied, skipped, failed)``.  ``failed`` is a list of
    ``{"entry": <original entry>, "error": <message>}`` dicts - one per
    entry whose ``apply_label`` call raised.  A single bad entry never
    aborts the loop (logical-bug-audit H31); the caller is expected to
    surface the ``failed`` list to the user so they can retry just those
    entries.
    """
    applied = 0
    skipped = 0
    failed: list[dict] = []

    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
        if not cids:
            skipped += 1
            continue

        try:
            for cid in cids:
                apply_label(cid, label)
        except Exception as exc:
            logger.exception("Failed to apply label for entry %r", entry)
            failed.append({"entry": entry, "error": str(exc) or exc.__class__.__name__})
            continue
        applied += 1

    return applied, skipped, failed


# ---------------------------------------------------------------------------
# GET /api/label-importers
# ---------------------------------------------------------------------------


@label_importers_bp.route("/api/label-importers", methods=["GET"])
@label_importers_bp.response(200, LabelImporterEntrySchema(many=True))
def get_label_importers():
    """Return a list of all registered label importers."""
    from vtsearch.settings import filter_visible_plugins

    return [imp.to_dict() for imp in filter_visible_plugins("label_importers", list_label_importers())]


# ---------------------------------------------------------------------------
# POST /api/label-importers/import/<importer_name>
#
# Plugin-field route - the request body is the importer's declared ``fields``
# (file uploads, free-form params).  Static marshmallow schemas can't
# describe the shape because each importer has its own field list, so the
# OpenAPI spec doesn't list a request body for this route.  Runtime
# validation goes through :func:`validate_plugin_args`, which builds a
# per-plugin schema from the importer's :attr:`fields` declaration and
# raises 422 on schema-level failures (matching the rest of the API).
# ---------------------------------------------------------------------------


@label_importers_bp.route("/api/label-importers/import/<importer_name>", methods=["POST"])
@require_dataset_header
@require_detector_header
def run_label_import(importer_name: str):  # noqa: C901
    """Run the named label importer and apply the resulting labels.

    Plugin-dependent body shape: not described in the OpenAPI spec.
    """
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

    # Apply labels to global vote state
    origin_lookup, md5_lookup, name_lookup = build_media_lookup(snapshot_medias())
    applied, skipped, failed = _apply_labels(label_entries, origin_lookup, md5_lookup, name_lookup)

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
        from vtscore.datasets.ingest import ingest_missing_medias

        ingested = ingest_missing_medias(missing, medias)

        if ingested > 0:
            # Re-apply labels now that new medias are available.  These entries
            # were already removed from `skipped` above when we subtracted
            # `len(missing)`, so we only need to bump `applied` here.  Any
            # per-entry mutation errors from the auto-resolve pass are
            # appended to the same `failed` list as the first pass.
            origin_lookup, md5_lookup, name_lookup = build_media_lookup(snapshot_medias())
            resolved_applied, _, resolved_failed = _apply_labels(missing, origin_lookup, md5_lookup, name_lookup)
            applied += resolved_applied
            failed.extend(resolved_failed)

        # Check which entries still couldn't be resolved
        if ingested < len(missing):
            origin_lookup, md5_lookup, name_lookup = build_media_lookup(snapshot_medias())
            unresolved = find_missing_entries(missing, origin_lookup, md5_lookup, name_lookup)

    # Sync updated votes into the loaded model so the dashboard reflects
    # the new label count (num_training) immediately.
    if applied > 0:
        from vtscore.detectors.label_sync import sync_labels_to_loaded_detector

        sync_labels_to_loaded_detector()

        from vtscore.labels.sync import sync_to_labelset_source

        sync_to_labelset_source()

        from vtsearch.achievements import record_detector_import
        from vtscore.state.core import get_active_detector_context

        record_detector_import(get_active_detector_context().detector_id)

    msg = f"Applied {applied} label(s), skipped {skipped}."
    if ingested > 0:
        msg += f" Auto-resolved {ingested} missing element(s) from their sources."
    if unresolved:
        msg += f" {len(unresolved)} element(s) could not be resolved."
    if failed:
        msg += f" {len(failed)} element(s) failed to apply (see 'failed' for details)."

    return jsonify(
        {
            "applied": applied,
            "skipped": skipped,
            "missing_count": len(unresolved),
            "missing": unresolved,
            "ingested": ingested,
            "failed_count": len(failed),
            "failed": failed,
            "message": msg,
        }
    )


# ---------------------------------------------------------------------------
# POST /api/label-importers/ingest-missing
# ---------------------------------------------------------------------------


@label_importers_bp.route("/api/label-importers/ingest-missing", methods=["POST"])
@label_importers_bp.arguments(IngestMissingRequestSchema)
@label_importers_bp.response(200, IngestMissingResponseSchema)
@require_dataset_header
@require_detector_header
def ingest_missing(body: dict):
    """Re-ingest missing medias from their origins, then apply their labels."""
    entries = body["entries"]

    from vtscore.datasets.ingest import ingest_missing_medias

    ingested = ingest_missing_medias(entries, medias)

    # Now apply labels to the newly ingested medias
    origin_lookup, md5_lookup, name_lookup = build_media_lookup(snapshot_medias())
    applied, _, failed = _apply_labels(entries, origin_lookup, md5_lookup, name_lookup)

    # Sync updated votes into the loaded model so the dashboard reflects
    # the new label count immediately.
    if applied > 0:
        from vtscore.detectors.label_sync import sync_labels_to_loaded_detector

        sync_labels_to_loaded_detector()

        from vtscore.labels.sync import sync_to_labelset_source

        sync_to_labelset_source()

    message = f"Ingested {ingested} media(s), applied {applied} label(s)."
    if failed:
        message += f" {len(failed)} element(s) failed to apply."

    return {
        "ingested": ingested,
        "applied": applied,
        "failed_count": len(failed),
        "failed": failed,
        "message": message,
    }
