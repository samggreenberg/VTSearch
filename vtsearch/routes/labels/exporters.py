"""Flask routes for the Labelset Exporter API.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``.

Endpoints
---------
GET  /api/exporters
    List all registered exporters with their metadata and field definitions.

POST /api/exporters/export
    Run a specific exporter on the payload supplied in the request body.
    ``payload_kind`` says whether that payload is a scored run
    (``find_results``) or a detector's labels (``labelset``); omitting it
    falls back to inferring from the dict shape, for API clients written
    before the kinds were named. ``field_values`` is permissive at the
    schema layer because its inner keys depend on the named exporter; the
    handler validates it against the selected plugin's :attr:`fields`.
    Schema-level failures (missing ``exporter_name``, unknown
    ``payload_kind``) surface as 422; handler-level rejects (unknown
    exporter, an exporter that doesn't implement the requested kind,
    missing plugin field, invalid filepath, plugin error) keep their
    original HTTP codes (404 / 400 / 500) with the standard ``message``
    envelope. An ``open_url`` in the exporter's outcome is re-validated
    against the browser-URL scheme allowlist before it reaches the
    frontend; a URL that fails is a 500.
"""

from __future__ import annotations

import logging

from flask_smorest import Blueprint, abort

from vtscore.exporters import get_exporter, list_exporters
from vtscore.exporters.base import ResultsExporter
from vtscore.security.url_validation import validate_browser_url
from vtsearch.routes._shared import validate_exporter_field_values
from vtsearch.schemas.labels import (
    ExporterEntrySchema,
    RunExportRequestSchema,
    RunExportResponseSchema,
)

logger = logging.getLogger(__name__)

exporters_bp = Blueprint(
    "exporters",
    __name__,
    description="List and run labelset exporters.",
)


def _infer_payload_kind(payload: dict) -> str:
    """Guess the payload kind for a client that didn't send one.

    The pre-payload-kinds fallback, and the same test every exporter used to
    run for itself: a serialised LabelSet has a top-level ``labels`` key and a
    scored run doesn't. Kept only so an API client written against the old body
    keeps working; the frontend always sends ``payload_kind`` explicitly, which
    is what lets the capability check below reject a bad pairing instead of
    guessing at one.
    """
    return "labelset" if "labels" in payload else "find_results"


def _run(exporter: ResultsExporter, payload_kind: str, payload: dict, field_values: dict) -> dict:
    """Dispatch to the exporter method for *payload_kind*."""
    if payload_kind == "labelset":
        return exporter.export_labelset(payload, field_values)
    return exporter.export_find_results(payload, field_values)


@exporters_bp.route("/api/exporters", methods=["GET"])
@exporters_bp.response(200, ExporterEntrySchema(many=True))
def get_exporters():
    """Return a list of all registered labelset exporters."""
    from vtsearch.settings import filter_visible_plugins

    return [exp.to_dict() for exp in filter_visible_plugins("exporters", list_exporters())]


@exporters_bp.route("/api/exporters/export", methods=["POST"])
@exporters_bp.arguments(RunExportRequestSchema)
@exporters_bp.response(200, RunExportResponseSchema)
@exporters_bp.alt_response(400, description="Missing plugin field or invalid filepath.")
@exporters_bp.alt_response(404, description="Unknown exporter name.")
@exporters_bp.alt_response(500, description="Exporter raised an unexpected error.")
def run_export(body: dict):
    """Run the named exporter on the supplied auto-detect results."""
    exporter_name = body["exporter_name"].strip()
    if not exporter_name:
        abort(422, message="exporter_name is required")

    exporter = get_exporter(exporter_name)
    if exporter is None:
        known = [p.name for p in list_exporters()]
        abort(404, message=f"Unknown exporter '{exporter_name}'. Available: {known}")

    field_values = validate_exporter_field_values(exporter, dict(body.get("field_values") or {}))
    results: dict = dict(body.get("results") or {})
    payload_kind = body.get("payload_kind") or _infer_payload_kind(results)

    if payload_kind not in exporter.supported_payloads:
        supported = ", ".join(sorted(exporter.supported_payloads)) or "nothing"
        abort(
            400,
            message=(
                f"Exporter '{exporter_name}' does not export a {payload_kind} payload "
                f"(it supports: {supported})."
            ),
        )

    try:
        outcome = dict(_run(exporter, payload_kind, results, field_values) or {})
    except ValueError as exc:
        abort(400, message=str(exc))
    except Exception as exc:
        logger.exception("%s export of a %s payload failed: %s", type(exporter).__name__, payload_kind, exc)
        abort(500, message=str(exc))

    if outcome.get("open_url") is not None:
        # The frontend hands this straight to ``window.open``, so re-validate
        # here regardless of what the plugin already did: a scheme allowlist is
        # what stops an exporter (in-tree, third-party entry point, or simply
        # buggy) from pushing a ``javascript:`` URL into the browser. Not the
        # SSRF guard — the *browser* fetches this, so a localhost viewer is a
        # legitimate target and resolving the host would buy nothing.
        try:
            outcome["open_url"] = validate_browser_url(str(outcome["open_url"]))
        except ValueError as exc:
            logger.error("%s.export() returned an unusable open_url: %s", type(exporter).__name__, exc)
            abort(500, message=f"Exporter '{exporter_name}' returned an unusable open_url: {exc}")

    return {"success": True, **outcome}
