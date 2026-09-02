"""Flask routes for the Label Importer API.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``.

Endpoints
---------
GET  /api/label-importers
    List all registered label importers with their metadata and field
    definitions.

POST /api/label-importers/field-options/<importer_name>
    Return dropdown options for a dynamic-options field.  Delegates to
    ``get_field_options(field_key, current_values)`` on the importer.

POST /api/label-importers/import/<importer_name>
    Run the named label importer. Accepts ``multipart/form-data`` (when
    the importer has a ``"file"`` field) or JSON (for text-only
    importers). The request body shape is plugin-dependent and not
    described in the OpenAPI spec; instead, the body is validated at
    request time via :func:`validate_plugin_args`, which builds a
    marshmallow schema from the named importer's :attr:`fields`
    declaration. Schema-level rejects (missing required field, invalid
    select value) surface as ``422`` with the standard ``errors``
    envelope; handler-level rejects (path traversal, plugin error) keep
    their original HTTP codes (400 / 500) with the standard ``message``
    envelope. See "Routes absent from the spec" in ``docs/API.md``.

POST /api/label-importers/ingest-missing
    Accept a list of missing label entries, re-ingest them from their
    origins, and apply the labels. JSON-only and fully spec'd.

Partial-failure semantics
-------------------------
:func:`_apply_labels` isolates each entry with a per-entry try/except.
If ``apply_label`` raises mid-loop, that entry is recorded in the
``failed`` return list and the loop continues with the remaining
entries.  This addresses logical-bug-audit H31: a single bad entry no
longer aborts the whole import and silently strands the already-applied
labels behind a 500 response.  The handler still runs the downstream
sync block (``sync_labels_to_loaded_detector``,
``sync_to_labelset_source``, ``record_detector_import``) whenever
``applied > 0`` so the in-memory detector, the labelset source, and the
achievement counters all reflect what actually landed.

Background auto-resolve
-----------------------
Entries whose media aren't in the active dataset are pulled in from their
origins, which means one fetch + embed per entry - far too slow to run
inside the request (issue #2703).  :func:`_start_auto_resolve` moves that
onto the ``detector_loading_tasks`` tracker and the import response carries
``ingest_task_id`` / ``ingest_pending_count`` instead of the final numbers.
The task re-applies those labels, re-syncs, and publishes
``{"ingested", "applied", "unresolved", "failed"}`` as its terminal
``ingest_result``, which the client reads off the ``detector-loading-tasks``
SSE channel to report the outcome.
"""

from __future__ import annotations

import logging

from flask import jsonify
from flask_smorest import Blueprint

logger = logging.getLogger(__name__)

from vtscore.labels.importers import get_label_importer, list_label_importers
from vtsearch.errors import error_response
from vtsearch.routes._shared import (
    get_plugin_or_404,
    plugin_field_options,
    register_plugin_typed_routes,
    require_dataset_header,
    require_detector_header,
    run_plugin_or_error,
    validate_plugin_args,
)
from vtsearch.schemas.datasets import (
    ImporterFieldOptionsRequestSchema,
    ImporterFieldOptionsResponseSchema,
)
from vtsearch.schemas.labels import (
    IngestMissingRequestSchema,
    IngestMissingResponseSchema,
    LabelImporterEntrySchema,
)
from vtsearch.state import (
    apply_label,
    cached_media_lookups,
    medias,
    find_missing_entries,
    resolve_media_ids,
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
    ``{"entry": <original entry>, "error": <message>}`` dicts, one per
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


def _sync_after_import() -> None:
    """Publish freshly-applied labels: detector, labelset source, achievements.

    Shared by the in-request pass and the background auto-resolve task so the
    dashboard's ``num_training``, the labelset file, and the import
    achievement all reflect whatever actually landed, whichever pass landed
    it.  ``record_detector_import`` dedupes by detector id, so running this
    from both passes counts the import once.
    """
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector
    from vtscore.labels.sync import sync_to_labelset_source
    from vtsearch.achievements import record_detector_import
    from vtscore.state.core import get_active_detector_context

    sync_labels_to_loaded_detector()
    sync_to_labelset_source()
    record_detector_import(get_active_detector_context().detector_id)


def _apply_ingested_labels(missing: list[dict], ingested: int) -> dict:
    """Apply *missing*'s labels once their media have been ingested.

    Runs on the ingest task's worker thread (see
    :func:`~vtscore.datasets.ingest_task.start_ingest_task`), which replays
    the request's dataset / detector context, so the lookups and vote writes
    land where the request's first pass did.  The returned dict is published
    as the task's terminal ``ingest_result`` so the client can report the
    same numbers the synchronous response used to carry.
    """
    applied = 0
    failed: list[dict] = []
    if ingested > 0:
        origin_lookup, md5_lookup, name_lookup = cached_media_lookups()
        applied, _, failed = _apply_labels(missing, origin_lookup, md5_lookup, name_lookup)

    unresolved: list[dict] = []
    if ingested < len(missing):
        origin_lookup, md5_lookup, name_lookup = cached_media_lookups()
        unresolved = find_missing_entries(missing, origin_lookup, md5_lookup, name_lookup)

    if applied > 0:
        _sync_after_import()

    return {"applied": applied, "unresolved": len(unresolved), "failed": len(failed)}


def _start_auto_resolve(missing: list[dict]) -> str:
    """Start the background fetch+embed of *missing*, then re-apply their labels.

    Fetching and embedding one file per missing label is far too slow to run
    inside the request (issue #2703), so it goes onto the detector task
    tracker: the caller gets a task id back immediately and watches the
    existing SSE feed for progress and for the terminal ``ingest_result``.
    """
    from vtscore.datasets.ingest_task import start_ingest_task
    from vtscore.state.core import get_active_detector_context
    from vtsearch.threading import spawn

    det_ctx = get_active_detector_context()
    detector_id = det_ctx.detector_id or ""
    return start_ingest_task(
        missing,
        medias,
        task_id=f"_labelingest_{detector_id}",
        name=det_ctx.name or "Label import",
        spawn=spawn,
        detector_id=detector_id,
        after_ingest=lambda ingested: _apply_ingested_labels(missing, ingested),
    )


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
# POST /api/label-importers/field-options/<importer_name>
# ---------------------------------------------------------------------------


@label_importers_bp.route("/api/label-importers/field-options/<importer_name>", methods=["POST"])
@label_importers_bp.arguments(ImporterFieldOptionsRequestSchema)
@label_importers_bp.response(200, ImporterFieldOptionsResponseSchema)
@label_importers_bp.alt_response(400, description="Unknown or non-dynamic field key.")
@label_importers_bp.alt_response(404, description="Unknown importer name.")
@label_importers_bp.alt_response(500, description="get_field_options did not return a list.")
@label_importers_bp.alt_response(501, description="Importer does not implement get_field_options.")
@label_importers_bp.alt_response(502, description="Remote service backing dynamic options raised an error.")
def label_importer_field_options(body: dict, importer_name: str):
    """Return dropdown options for a dynamic-options field on a label importer."""
    importer, err = get_plugin_or_404(get_label_importer, list_label_importers, importer_name, "label importer")
    if err:
        return err
    assert importer is not None

    return plugin_field_options(importer, body)


# ---------------------------------------------------------------------------
# POST /api/label-importers/import/<importer_name>
#
# Plugin-field route: the request body is the importer's declared ``fields``
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
        return error_response("Importer did not return a list of label dicts.", 500)

    # Apply labels to global vote state
    origin_lookup, md5_lookup, name_lookup = cached_media_lookups()
    applied, skipped, failed = _apply_labels(label_entries, origin_lookup, md5_lookup, name_lookup)

    # Detect entries that could not be matched at all
    missing = find_missing_entries(label_entries, origin_lookup, md5_lookup, name_lookup)
    # Adjust skipped count: missing entries were already counted as skipped
    # by _apply_labels, but we report them separately now.
    skipped -= len(missing)

    # Auto-resolve: pull the missing medias in from their origins.  The fetch
    # + embed runs on a background task (issue #2703) and re-applies those
    # labels when it lands, so the counts below describe only the in-request
    # pass; the task publishes its own numbers as ``ingest_result``.
    ingest_task_id = _start_auto_resolve(missing) if missing else ""

    # Sync updated votes into the loaded model so the dashboard reflects
    # the new label count (num_training) immediately.
    if applied > 0:
        _sync_after_import()

    msg = f"Applied {applied} label(s), skipped {skipped}."
    if ingest_task_id:
        msg += f" Resolving {len(missing)} missing element(s) from their sources in the background…"
    if failed:
        msg += f" {len(failed)} element(s) failed to apply (see 'failed' for details)."

    return jsonify(
        {
            "applied": applied,
            "skipped": skipped,
            # Resolution of the missing entries is still in flight, so nothing
            # is known to be unresolvable yet; the ingest task's terminal
            # ``ingest_result`` reports what it couldn't reach.
            "missing_count": 0,
            "missing": [],
            "ingest_task_id": ingest_task_id,
            "ingest_pending_count": len(missing),
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
    origin_lookup, md5_lookup, name_lookup = cached_media_lookups()
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


# ---------------------------------------------------------------------------
# Per-plugin typed routes for /api/label-importers/import/<name>.
# Registered at module-import time by iterating the label-importer
# registry, so each known importer gets a static URL whose body schema
# is described in /api/openapi.json with real per-field types.  Unknown
# importer names fall through to the parameterized route above.
# Plugins with file fields stay on the parameterized fallback.
# ---------------------------------------------------------------------------

register_plugin_typed_routes(
    label_importers_bp,
    list_plugins=list_label_importers,
    path_template="/api/label-importers/import/{plugin_name}",
    endpoint_prefix="run_label_import",
    delegate=run_label_import,
    extra_decorators=(require_detector_header, require_dataset_header),
)
