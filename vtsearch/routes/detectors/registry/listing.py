"""Detector-registry routes that list detectors and create them.

Endpoints
---------
GET    /api/detectors/registry                        List registered detectors.
POST   /api/detectors/registry                        Register a new detector.
POST   /api/detectors/registry/from-labelset/<imp>    Seed a new detector from a label importer.

``POST /from-labelset/<importer>`` takes plugin-dependent fields and declares
no schema (see "Routes with no typed schema" in ``docs/API.md``).
"""

from __future__ import annotations

import logging
import time

from flask import jsonify
from flask_smorest import abort

from vtsearch.errors import error_response
from vtsearch.auth import get_current_user
from vtscore.detectors.embedder_type import detector_embedder_type_from_data
from vtscore.detectors.store import (
    _detector_path,
    _read_detector,
    _write_detector,
)
from vtsearch.routes.detectors.registry._common import (
    _abort_if_name_taken,
    _creation_beta,
    _keep_creation_beta,
    detectors_registry_bp,
)
from vtsearch.schemas.detectors import (
    DetectorRegistryCreateRequestSchema,
    DetectorRegistryCreateResponseSchema,
    DetectorRegistryListResponseSchema,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GET /api/detectors/registry
# ---------------------------------------------------------------------------


def _fill_type_and_examples(entry: dict) -> None:
    """Set *entry*'s ``embedder_type`` and ``examples``, reading the detector JSON at most once.

    The locked embedder type drives the frontend's type-based compatibility
    gate, and the seed-example list is what the label session reads so
    Autopilot can sort by every media example.  Both normally come from the
    registry entry.  An entry missing either (registered before the field
    existed, or created with no dataset loaded, which leaves the type ``""``)
    falls back to one parse of the detector JSON, and whatever that parse
    learns is written back to the registry so later listings never open the
    file for it again (#5001).  A type the JSON cannot supply either stays
    ``""`` unpersisted (it resolves at first train), and a missing file has
    nothing to persist.
    """
    needs_type = not entry.get("embedder_type")
    needs_examples = "examples" not in entry
    if not (needs_type or needs_examples):
        return
    data = _read_detector(_detector_path(entry.get("name", "")))
    learned: dict = {}
    if needs_type:
        entry["embedder_type"] = detector_embedder_type_from_data(data or {})
        if entry["embedder_type"]:
            learned["embedder_type"] = entry["embedder_type"]
    if needs_examples:
        entry["examples"] = (data or {}).get("examples", [])
        if data is not None:
            learned["examples"] = entry["examples"]
    if not learned:
        return
    from vtscore.detectors.registry import update_detector

    try:
        update_detector(entry["id"], **learned)
    except Exception as exc:
        # A listing must not fail over a cache write; the next one retries.
        logger.warning("Failed to persist %s for detector %s: %s", sorted(learned), entry["id"], exc)


@detectors_registry_bp.route("/api/detectors/registry")
@detectors_registry_bp.response(200, DetectorRegistryListResponseSchema)
def list_registered_detectors():
    """Return detectors visible to the current user, with loaded/AutoFind flags.

    Detectors are user-shared like datasets: each entry is the creator's plus
    anyone the creator added to ``readers`` (or everyone via ``"*"``). The
    response also carries ``created_by``, ``readers``, and ``is_owner`` so the
    dashboard can render the access column and gate the security button.
    An AutoFind detector also carries ``test_verdict``, the newest test
    verdict it keeps (#4526), for the AutoFind tab to show beside it; drafts
    leave it out, so the listing reads no detector file it does not need.
    """
    from vtscore.detectors.line_verdicts import verdict_summaries
    from vtscore.detectors.registry import get_loaded_detector_ids, list_detectors_for_user
    from vtsearch.settings import get_autofind_detectors

    from vtscore.state.core import get_detector_context

    from vtscore.labels.binding import deployment_default_labelset_source, labelset_source_status

    current_user = get_current_user()
    entries = list_detectors_for_user(current_user)
    loaded_ids = get_loaded_detector_ids()
    autofind_names = set(get_autofind_detectors())
    default_source = deployment_default_labelset_source()
    for entry in entries:
        # Where the detector's labels sync to (#4913) - status only, never the
        # binding's field values, which the entry's ``labelset_source`` holds.
        entry["labelset_source_status"] = labelset_source_status(entry, default=default_source)
        did = entry["id"]
        entry["loaded"] = did in loaded_ids
        entry["autofind"] = entry.get("name", "") in autofind_names
        entry.setdefault("last_trained_at", None)
        entry.setdefault("trained_on", [])
        entry.setdefault("created_by", "default")
        entry.setdefault("readers", [])
        entry["is_owner"] = entry.get("created_by", "default") == current_user
        entry["detector_loaded"] = did in loaded_ids
        # Expose the loaded detector's recorded embedder so the frontend
        # can detect a cross-embedder switch and trigger a label re-embed
        # via /api/detectors/registry/load.  Loaded contexts always win
        # because they reflect the live cache state; for unloaded
        # detectors fall back to the registry's persisted embedder (which
        # is stamped the first time training runs against a dataset).
        if entry["detector_loaded"]:
            ctx = get_detector_context(did)
            ctx_emb = ctx.embedder if ctx is not None else ""
            entry["embedder"] = ctx_emb or entry.get("embedder", "") or ""
        else:
            entry["embedder"] = entry.get("embedder", "") or ""
        _fill_type_and_examples(entry)
        if entry["autofind"]:
            latest = verdict_summaries(_read_detector(_detector_path(entry.get("name", ""))), latest_only=True)
            entry["test_verdict"] = latest[0] if latest else None
    return {"detectors": entries}


# ---------------------------------------------------------------------------
# POST /api/detectors/registry
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry", methods=["POST"])
@detectors_registry_bp.arguments(DetectorRegistryCreateRequestSchema)
@detectors_registry_bp.response(201, DetectorRegistryCreateResponseSchema)
@detectors_registry_bp.alt_response(400, description="Empty name after stripping, or media_type is 'any'.")
@detectors_registry_bp.alt_response(409, description="A detector with this name already exists.")
def register_detector_route(body: dict):
    """Register a new detector in the detector registry."""
    from vtscore.detectors.registry import register_detector

    name = body["name"].strip()
    media_type = body["media_type"].strip()
    text_query = body["text_query"]
    media_example = body["media_example"]

    if not name:
        abort(400, message="name is required")
    if not media_type or media_type == "any":
        abort(400, message="media_type is required (must be a specific type, not 'any')")

    from vtscore.detectors.embedder_type import resolve_detector_embedder_type
    from vtsearch.routes._policy import abort_if_semantic_only_type

    embedder_type, type_err = resolve_detector_embedder_type(body.get("embedder_type", ""))
    if type_err:
        abort(400, message=type_err)
    abort_if_semantic_only_type(embedder_type)

    # Two registry entries sharing one labelset file silently overwrite each
    # other's labels, so a name that is already taken is a hard 409 (matching
    # the sibling create routes) rather than a reuse of the existing file.
    _abort_if_name_taken(name)

    beta = _creation_beta(body.get("beta"))

    examples = body.get("examples") or []
    if not examples and text_query:
        examples = [{"type": "text", "value": text_query}]
    if not examples and media_example:
        examples = [{"type": "media", "value": media_example}]
    if not media_example:
        # Keep the scalar in sync with the list so legacy readers (dashboard
        # display, Autopilot fallback) see the first media example.
        media_example = next((ex.get("value", "") for ex in examples if ex.get("type") == "media"), "")

    # Media exemplars are good votes the user already cast, so they land in
    # the labelset immediately - origin-keyed and dataset-agnostic, which is
    # what keeps an https:// exemplar alive against an all-local dataset
    # (issue #3045).  Text examples are queries, not labels.
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.media_seeding import labeled_elements_from_examples

    example_labels = labeled_elements_from_examples(examples)

    detector_data = {
        "name": name,
        "text_query": text_query,
        "media_example": media_example,
        "media_type": media_type,
        "examples": examples,
        "created_at": time.time(),
        "embedder_type": embedder_type,
        "labelset": LabelSet(example_labels).to_dict(),
    }
    _keep_creation_beta(detector_data, beta)
    _write_detector(_detector_path(name), detector_data)

    entry = register_detector(
        name=name,
        media_type=media_type,
        num_training=len(example_labels),
        text_query=text_query,
        media_example=media_example,
        examples=examples,
        embedder_type=embedder_type,
        created_by=get_current_user(),
    )
    return {"ok": True, "detector": entry}


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/from-labelset/<importer_name>
#
# Plugin-field route: body shape depends on the importer plugin and isn't
# described in the OpenAPI spec.  Runtime validation goes through
# :func:`validate_plugin_args` (per-plugin schema built from the importer's
# :attr:`fields`), so missing required fields / invalid select values
# raise 422.  See "Routes with no typed schema" in ``docs/API.md``.
# ---------------------------------------------------------------------------


def _start_imported_labelset_ingest(label_dicts: list[dict], entry: dict) -> str:
    """Start the background pull of an imported labelset's media into the active dataset.

    A labelset is origin-keyed and dataset-agnostic, but everything
    downstream of it - vote state, the labeling grid, and above all
    ``GET /api/labels/export``, which composes its payload from
    ``LabelSet.from_clips_and_votes(active_medias, goods, bads)`` - can only
    see labels whose media resolve to the *active* dataset's medias.  Without
    this step, importing a detector whose labels came from somewhere else
    leaves the right pane showing the imported labels (it reads the labelset)
    while export returns nothing (it reads votes), and the user has to Browse
    Positives / Find / Train first to materialise the media before the labels
    become exportable at all (issue #2690).

    Mirrors what :func:`~vtsearch.routes.labels.importers.run_label_import`
    already does for the in-session label importer, so the two import paths
    land the same media in the same place.  The subsequent detector load then
    restores those labels into votes normally - which is why the caller must
    let this task *finish* before loading the detector, or label restore runs
    against medias that aren't there yet.

    The fetch+embed itself is far too slow to run inline (issue #2703), so it
    goes onto the detector task tracker and reports through the SSE feed.
    Returns the task id, or ``""`` when there is nothing to ingest (no active
    dataset, or every label already resolves).
    """
    from vtscore.state.core import get_active_context, is_request_missing_dataset_context

    try:
        ds_ctx = get_active_context()
        if not ds_ctx.dataset_id or is_request_missing_dataset_context(ds_ctx):
            # No dataset to ingest into; the labels stay origin-only until a
            # dataset is loaded and the detector is (re)loaded against it.
            return ""

        from vtscore.datasets.ingest_task import start_ingest_task
        from vtsearch.state import cached_media_lookups, find_missing_entries, medias
        from vtsearch.runtime.context_threads import spawn

        origin_lookup, md5_lookup, name_lookup = cached_media_lookups()
        missing = find_missing_entries(label_dicts, origin_lookup, md5_lookup, name_lookup)
        if not missing:
            return ""
        detector_id = entry.get("id", "")
        return start_ingest_task(
            missing,
            medias,
            task_id=f"_detingest_{detector_id}",
            name=entry.get("name", detector_id),
            spawn=spawn,
            detector_id=detector_id,
            media_type=entry.get("media_type", "") or "",
        )
    except Exception:
        logger.exception("Ingesting imported labelset media failed")
        return ""


@detectors_registry_bp.route(
    "/api/detectors/registry/from-labelset/<importer_name>",
    methods=["POST"],
)
@detectors_registry_bp.alt_response(
    400,
    description=(
        "Invalid request body, field value, ``beta`` or ``embedder_type``; an embedder type the Semantic-only "
        "lock forbids; the importer rejected the run; or the labels carry no media type, or more than one."
    ),
)
@detectors_registry_bp.alt_response(404, description="Unknown label importer name.")
@detectors_registry_bp.alt_response(409, description="A detector with this name already exists.")
@detectors_registry_bp.alt_response(
    422, description="``name`` or a required field is missing, or a field value is invalid."
)
@detectors_registry_bp.alt_response(500, description="The importer failed, or did not return a list of label dicts.")
def register_detector_from_labelset(importer_name: str):  # noqa: C901
    """Create a detector seeded with labels from a label importer.

    Plugin-dependent body shape: the accepted keys are the named
    plugin's declared ``fields``, so they cannot be enumerated in a
    static OpenAPI schema.  Fetch them at runtime from
    ``GET /api/label-importers``, whose ``fields`` array carries each key's
    ``field_type``, ``required`` flag, and any ``options`` / bounds.
    """
    from vtscore.datasets.ingest import _media_type_from_origin
    from vtscore.datasets.labelset import LabeledElement, LabelSet
    from vtscore.labels.importers import get_label_importer, list_label_importers
    from vtscore.detectors.import_record import build_import_record
    from vtscore.detectors.registry import register_detector, update_detector
    from vtsearch.routes._plugins import get_plugin_or_404, run_plugin_or_error, validate_plugin_args
    from vtsearch.routes._policy import abort_if_semantic_only_type

    importer, err = get_plugin_or_404(get_label_importer, list_label_importers, importer_name, "label importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_values = validate_plugin_args(importer, extra_keys=("name", "embedder_type", "beta"))

    # ``name``, ``embedder_type`` and ``beta`` are pass-through keys (not
    # declared plugin fields) but are owned by this route.
    # ``validate_plugin_args`` only keeps the keys we list in ``extra_keys``,
    # so the route enforces presence.
    name = str(field_values.pop("name", "") or "").strip()
    if not name:
        abort(422, message="Validation error", errors={"json": {"name": ["Missing data for required field."]}})
    beta = _creation_beta(field_values.pop("beta", None))

    from vtscore.detectors.embedder_type import resolve_detector_embedder_type

    requested_type = str(field_values.pop("embedder_type", "") or "").strip()
    embedder_type_val, type_err = resolve_detector_embedder_type(requested_type)
    if type_err:
        abort(400, message=type_err)
    abort_if_semantic_only_type(embedder_type_val)

    _abort_if_name_taken(name)
    det_path = _detector_path(name)

    label_entries, err = run_plugin_or_error(importer, "run", field_values)
    if err:
        return err
    if not isinstance(label_entries, list):
        return error_response("Importer did not return a list of label dicts.", 500)

    elements: list[LabeledElement] = []
    detected_types: set[str] = set()
    applied = 0
    skipped = 0
    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        origin = entry.get("origin")
        if isinstance(origin, dict):
            mt = _media_type_from_origin(origin)
            if mt:
                detected_types.add(mt)
        elements.append(LabeledElement.from_dict(entry))
        applied += 1

    if not detected_types:
        return error_response(
            "Could not infer media type from the imported labels; none of "
            "the entries carry origin information with a detectable type. "
            "Re-export the labels with origin metadata, or use a different "
            "importer.",
            400,
        )
    if len(detected_types) > 1:
        return error_response(
            f"Imported labels span multiple media types: {sorted(detected_types)}. "
            "A detector must be for a single media type.",
            400,
        )

    media_type = next(iter(detected_types))
    labelset = LabelSet(elements)

    detector_data = {
        "name": name,
        "text_query": "",
        "media_example": "",
        "media_type": media_type,
        "examples": [],
        "created_at": time.time(),
        "embedder_type": embedder_type_val,
        "labelset": labelset.to_dict(),
    }
    _keep_creation_beta(detector_data, beta)
    _write_detector(det_path, detector_data)

    entry = register_detector(
        name=name,
        media_type=media_type,
        num_training=len(labelset),
        embedder_type=embedder_type_val,
        created_by=get_current_user(),
        # The importer, its fields and the time, for the Stats window (#5334).
        imported=build_import_record(importer, field_values, imported_at=detector_data["created_at"]),
    )
    update_detector(entry["id"], last_trained_at=time.time())
    entry["num_training"] = len(labelset)
    entry["last_trained_at"] = time.time()

    # Materialise any imported label whose media isn't in the active dataset,
    # so the labels are exportable / visible immediately instead of only after
    # a Browse Positives / Find / Train pass (issue #2690).  Runs in the
    # background (issue #2703): the caller polls ``ingest_task_id`` on the
    # detector-task SSE channel and must wait for it before loading the
    # detector, since label restore resolves against the ingested medias.
    ingest_task_id = _start_imported_labelset_ingest([el.to_dict() for el in elements], entry)

    return jsonify(
        {
            "ok": True,
            "detector": entry,
            "applied": applied,
            "skipped": skipped,
            "num_labels": len(labelset),
            "ingest_task_id": ingest_task_id,
        }
    ), 201
