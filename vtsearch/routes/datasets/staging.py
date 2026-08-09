"""Dataset staging, importer dispatch, and the combine-datasets endpoint.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.

JSON-shaped routes (available-files, combine, stage-demo, clear-staging,
importer-field-options, importer-suggested-name) use the standard
``@arguments`` + ``@response`` decorators; schema-level validation
failures surface as 422.
Multipart-upload routes (``stage-file``) and plugin-field routes
(``stage-import/<importer>``, ``import/<importer>``) keep ``@arguments``
omitted; the latter pair's body shape depends on the importer plugin
and isn't described in the OpenAPI spec.  Runtime validation goes
through :func:`validate_plugin_args` (per-plugin schema built from the
importer's :attr:`fields`), so missing required fields / invalid select
values raise 422.  See *Resolved questions / Plugin field endpoints*
in the plan doc.
"""

import gc
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from flask import jsonify, request
from flask_smorest import Blueprint, abort

import vtscore.security.path_validation as _paths
from vtscore.concurrency.progress import CancelledError, loading_tasks
from vtscore.config import EMBEDDINGS_DIR
from vtscore.datasets import DEMO_DATASETS, export_dataset_to_file, get_importer, list_importers
from vtscore.datasets.file_types import count_file_types
from vtscore.datasets.importers.base import DATASET_NAME_FIELD_KEY
from vtscore.datasets.load_pipeline import (
    STAGING_DIR,
    _run_importer_in_background,
    _stage_importer_in_background,
)
from vtscore.datasets.registry import (
    find_by_pkl_path as _reg_find_by_pkl,
    get_dataset as _reg_get,
    get_saved_datasets_dir,
    register_dataset as _reg_register,
)
from vtsearch.auth import get_current_user
from vtsearch.routes._shared import (
    _normalise_option,
    abort_if_semantic_only_embedders,
    get_plugin_or_404,
    register_plugin_typed_routes,
    validate_plugin_args,
)
from vtsearch.routes.datasets._helpers import _extract_clipper_params
from vtsearch.schemas.datasets import (
    ClearStagingResponseSchema,
    DatasetAvailableFilesResponseSchema,
    DatasetCombineRequestSchema,
    DatasetLoadStartedResponseSchema,
    DatasetPromoteRequestSchema,
    DatasetStageDemoRequestSchema,
    DatasetStageFileResponseSchema,
    DatasetStagingStartedResponseSchema,
    ImporterFieldOptionsRequestSchema,
    ImporterFieldOptionsResponseSchema,
    ImporterSuggestedNameRequestSchema,
    ImporterSuggestedNameResponseSchema,
)
from vtsearch.state import get_active_context, snapshot_medias
from vtscore.security.pickle import peek_pickle_dataset_summary

datasets_staging_bp = Blueprint(
    "datasets_staging",
    __name__,
    description="Stage, combine, and import datasets.",
)


@datasets_staging_bp.route("/api/dataset/available-files")
@datasets_staging_bp.response(200, DatasetAvailableFilesResponseSchema)
def available_dataset_files():
    """List ``.pkl`` files in the embeddings directory."""
    files = []
    if EMBEDDINGS_DIR.exists():
        for pkl in sorted(EMBEDDINGS_DIR.glob("*.pkl")):
            files.append(
                {
                    "name": pkl.stem,
                    "path": str(pkl),
                    "size_mb": round(pkl.stat().st_size / (1024 * 1024), 1),
                }
            )
    return {"files": files}


def _bound_embedders_for_path(path: str) -> list[str]:
    """The concrete embedders a to-be-combined dataset binds, for conflict checks.

    Reads the persisted ``bound_embedders`` off the registry entry whose pkl this
    path names (falling back to its single legacy ``embedder``).  A path with no
    registry entry (a raw staged / demo pkl) returns ``[]`` — its embedders are
    unknown without loading it, so it contributes nothing to conflict detection.
    """
    entry = _reg_find_by_pkl(path)
    if entry is None:
        return []
    bound = entry.get("bound_embedders")
    if bound:
        return [str(n) for n in bound if n]
    emb = entry.get("embedder")
    return [emb] if emb else []


def _abort_if_semantic_only(field_values: dict) -> None:
    """400 when *field_values* names a patch/structural embedder under the lock.

    Covers both slots the importer body can carry: the scalar ``embedder``
    (primary) and the ``embedders`` trio, which arrives as a JSON array string
    on multipart bodies and a real list on JSON ones.
    """
    from vtscore.datasets.load_pipeline import _parse_embedder_list  # noqa: PLC0415

    names = [str(field_values.get("embedder") or "")]
    names.extend(_parse_embedder_list(field_values.get("embedders")) or [])
    abort_if_semantic_only_embedders(names)


@datasets_staging_bp.route("/api/dataset/combine", methods=["POST"])
@datasets_staging_bp.arguments(DatasetCombineRequestSchema)
@datasets_staging_bp.response(200, DatasetLoadStartedResponseSchema)
@datasets_staging_bp.alt_response(400, description="Invalid path, or an unresolved embedder conflict.")
@datasets_staging_bp.alt_response(500, description="The combine_datasets importer is unavailable.")
def combine_datasets_route(body: dict):
    """Combine multiple pickle datasets in a background thread.

    When the sources bind conflicting embedders of the same type (e.g. one
    ``siglip`` dataset and one ``clip`` dataset, both semantic), the caller must
    settle each conflict via ``resolutions`` — re-embed every source to one
    winner, or drop that embedder type from the result.  An unresolved conflict
    is refused (400) rather than silently producing a mixed vector space; the
    Combine-Datasets modal detects the conflicts client-side and collects the
    resolutions before posting here.
    """
    dataset_paths = body["datasets"]
    name = str(body.get("name", "") or "").strip()
    resolutions = body.get("resolutions") or {}

    # Carry the *approved* paths forward rather than the raw strings: under
    # multi-user confinement a relative path is checked against the user's
    # data dir but would be opened relative to the process CWD.
    _base = _paths.get_file_access_base_dir()
    confined_paths: list[str] = []
    for p in dataset_paths:
        try:
            confined = _paths.confine_server_filepath(str(p), _base)
        except ValueError as exc:
            abort(400, message=str(exc))
        if not Path(confined).exists():
            abort(400, message=f"File not found: {p}")
        confined_paths.append(confined)
    dataset_paths = confined_paths

    importer = get_importer("combine_datasets")
    if importer is None:
        abort(500, message="combine_datasets importer not available")

    field_values: dict = {"datasets": dataset_paths, "name": name}

    # Detect per-embedder-type conflicts across the sources and, when present,
    # bake the caller's resolution into the load: pass the kept embedder set so
    # the combine importer prunes to it and the pipeline's embed stage re-embeds
    # the winners.  No conflict → the pre-conflict-UI fast path (no re-embed).
    from vtscore.embedding.binding import (
        combine_type_state,
        derive_binding_from_names,
        resolve_keep_embedders,
    )

    # Only datasets whose embedders are *known* (registry-backed) inform conflict
    # detection; a raw staged pkl with unknown embedders is skipped rather than
    # counted as "missing" (which would flag a spurious partial-coverage conflict).
    per_dataset = [be for p in dataset_paths if (be := _bound_embedders_for_path(str(p)))]
    type_state = combine_type_state(per_dataset)
    if any(st["conflict"] for st in type_state.values()):
        keep, err = resolve_keep_embedders(type_state, resolutions)
        if err:
            abort(400, message=err)
        text, patch, structural = derive_binding_from_names(keep)
        field_values["keep_embedders"] = keep
        field_values["embedders"] = keep
        field_values["embedder"] = structural or patch or text or (keep[0] if keep else "")

    task_id = _run_importer_in_background(importer, field_values)
    return {"ok": True, "message": "Combining datasets...", "task_id": str(task_id) if task_id else ""}


def _coverage_atlas_pickle_keys(
    subset: dict[int, dict],
    on_progress: Callable[[int, int], None] | None = None,
) -> dict | None:
    """Return ``{"coverage_atlas": <payload>}`` to cache in a promoted pickle.

    Builds the atlas over *subset* at creation, exactly like a fresh import
    does, so reopening a promoted dataset restores the atlas instead of paying
    the hierarchical-k-means rebuild on every reload (the promote save used to
    omit it, and the subset's renumbered IDs make the source atlas unusable — so
    a promoted dataset rebuilt from scratch each time). Returns ``None`` past
    the auto-build threshold (matching the load pipeline's deferral) or when the
    subset carries no usable vectors.

    *on_progress* is forwarded to the atlas build (called as
    ``on_progress(current, total)`` per completed k-means fit) so the
    background promote task can report fine-grained progress.
    """
    from vtscore.state import build_coverage_atlas_serializable, should_auto_build_coverage_atlas

    if not should_auto_build_coverage_atlas(len(subset)):
        return None
    payload = build_coverage_atlas_serializable(subset, on_progress=on_progress)
    return {"coverage_atlas": payload} if payload is not None else None


#: Promote runs as a 3-step background task: coverage-atlas build,
#: pickle serialization + disk write, registry insert.
_PROMOTE_TOTAL_STEPS = 3

#: Timing-profile task name; its step names and shipped fallback weights live in
#: :data:`vtscore.timing.tasks.TASKS`. An admin ``VTSEARCH_TIMING_PROFILE``
#: replaces those with measured seconds, which matters most here: whether the
#: atlas k-means or the pickle write dominates depends entirely on whether the
#: host has cuML and how fast its disk is.
_PROMOTE_TASK = "dataset_promote"


def _promote_in_background(
    subset: dict[int, dict],
    *,
    name: str,
    media_type: str,
    embedder: str,
    clipper: str,
    expires_at: float | None,
    source: dict,
    file_type_counts: dict[str, int],
    created_by: str,
) -> str:
    """Build, write, and register the promoted dataset in a daemon thread.

    Mirrors :func:`vtscore.datasets.load_pipeline._stage_importer_in_background`:
    a dedicated per-task :class:`ProgressTracker` (via ``loading_tasks``) carries
    progress — including the coverage-atlas build, previously a long silent hang
    in the request thread — to the ``loading-tasks`` SSE channel, and the
    dashboard's Cancel button works through the standard task-cancel route.

    Concurrency: *subset* is a private snapshot built in the request thread
    (shallow clones of a ``snapshot_medias()`` copy), so concurrent mutation of
    the source dataset — items added/removed, votes, even ``clear_medias()`` —
    cannot affect this job and no lock is held while the atlas builds.  The
    clones share embedding arrays with the live medias, but embeddings are
    replaced (never mutated in place) throughout the codebase, so the shared
    references stay valid.

    On success the task's ``dataset_id`` association is set to the new registry
    entry's id so the frontend's completion callback can identify the dataset.
    Failures (and cancellation) surface as the task's ``error`` field; a
    partially written pkl is removed.

    Returns the ``task_id`` for progress polling / cancellation.
    """
    from vtsearch.auth import thread_user

    from vtscore import timing

    task_id = f"_promote_{uuid4().hex[:8]}"
    tracker = loading_tasks.create_task(
        task_id,
        name,
        media_type=media_type,
        embedder=embedder,
        step_weights=timing.step_weights(_PROMOTE_TASK, media_type=media_type, embedder=embedder, n=len(subset)),
    )
    timing_recorder = timing.record_task(tracker, _PROMOTE_TASK, media_type=media_type, embedder=embedder)
    timing_recorder.start()
    timing_recorder.set_scale(n=len(subset))
    tracker.update("loading", "Preparing promoted dataset…", 0, 0, step=1, total_steps=_PROMOTE_TOTAL_STEPS)

    def task():
        pkl_path: str | None = None
        try:
            with thread_user(created_by):

                def atlas_progress(current: int, total: int) -> None:
                    tracker.check_cancelled()
                    tracker.update(
                        "loading",
                        "Building coverage atlas…",
                        current,
                        total,
                        step=1,
                        total_steps=_PROMOTE_TOTAL_STEPS,
                    )

                extra_pickle_keys = _coverage_atlas_pickle_keys(subset, on_progress=atlas_progress)
                tracker.check_cancelled()

                tracker.update("loading", "Writing dataset file…", 0, 0, step=2, total_steps=_PROMOTE_TOTAL_STEPS)
                ds_dir = get_saved_datasets_dir()
                ds_dir.mkdir(parents=True, exist_ok=True)
                pkl_path = str(ds_dir / f"ds_{uuid4().hex}.pkl")
                data_bytes = export_dataset_to_file(
                    subset,
                    embedder=embedder,
                    clipper=clipper,
                    media_type=media_type,
                    name=name,
                    created_at=time.time(),
                    expires_at=expires_at,
                    extra_pickle_keys=extra_pickle_keys,
                )
                Path(pkl_path).write_bytes(data_bytes)
                del data_bytes
                tracker.check_cancelled()

                tracker.update("loading", "Registering dataset…", 0, 0, step=3, total_steps=_PROMOTE_TOTAL_STEPS)
                entry = _reg_register(
                    name=name,
                    media_type=media_type,
                    num_items=len(subset),
                    pkl_path=pkl_path,
                    origin="promote",
                    source=source,
                    clipper=clipper,
                    embedder=embedder,
                    created_by=created_by,
                    file_type_counts=file_type_counts,
                    expires_at=expires_at,
                )
                loading_tasks.set_dataset_id(task_id, entry["id"])
                tracker.update(
                    "idle",
                    f'Promoted {len(subset)} items to "{name}"',
                    100,
                    100,
                    step=_PROMOTE_TOTAL_STEPS,
                    total_steps=_PROMOTE_TOTAL_STEPS,
                )
        except CancelledError:
            if pkl_path:
                Path(pkl_path).unlink(missing_ok=True)
            tracker.update("idle", "", 0, 0, error="Cancelled")
        except MemoryError:
            if pkl_path:
                Path(pkl_path).unlink(missing_ok=True)
            tracker.update(
                "idle",
                "",
                0,
                0,
                error="Out of memory: this selection is too large. Try promoting fewer items or free up system RAM.",
            )
        except Exception as exc:  # noqa: BLE001 (surface the failure on the task tracker)
            traceback.print_exc()
            if pkl_path:
                Path(pkl_path).unlink(missing_ok=True)
            tracker.update("idle", "", 0, 0, error=str(exc) or repr(exc) or "Unknown error during promote")
        finally:
            # Every branch above parks the tracker at "idle", setting ``error``
            # when it failed or was cancelled — which is what says whether these
            # phase timings describe a real promote.
            timing_recorder.finish(ok=not tracker.get().get("error"))
            gc.collect()
            loading_tasks.mark_finished(task_id)

    threading.Thread(target=task, daemon=True).start()
    return task_id


@datasets_staging_bp.route("/api/dataset/promote", methods=["POST"])
@datasets_staging_bp.arguments(DatasetPromoteRequestSchema)
@datasets_staging_bp.response(200, DatasetLoadStartedResponseSchema)
@datasets_staging_bp.alt_response(400, description="No dataset loaded or none of the items resolved.")
def promote_to_dataset(body: dict):
    """Promote a set of media items into a brand-new saved dataset.

    Used by the Find interface's "To Dataset" button to turn the Goods
    pile into its own dataset. The promoted items keep their original
    origins and in-memory embeddings (so preprocessing is preserved for
    free; the new pickle is a self-contained snapshot). The new dataset
    gets a fresh ``created_at`` but inherits the source dataset's
    ``expires_at`` (death date).

    The subset snapshot and metadata derivation run synchronously (cheap:
    shallow copies); the expensive part — coverage-atlas build, pickle
    write, registry insert — runs in a background task.  The response
    carries the ``task_id``; the caller polls the ``loading-tasks`` SSE
    channel for progress and reads the finished task's ``dataset_id``
    association for the new dataset's id.
    """
    name = body["name"].strip()
    media_ids = body["media_ids"]

    snap = snapshot_medias()
    if not snap:
        abort(400, message="No dataset loaded")

    # Build the subset, renumbering IDs from 1 and preserving each item's
    # origin/embedding (a shallow copy is enough; the background task
    # serialises it and never mutates the embedding array).  Snapshotting
    # here, in the request thread, is what isolates the background job
    # from concurrent mutation of the source dataset — see
    # :func:`_promote_in_background`.
    subset: dict[int, dict] = {}
    new_id = 1
    for mid in media_ids:
        media = snap.get(mid)
        if media is None:
            continue
        clone = dict(media)
        clone["id"] = new_id
        subset[new_id] = clone
        new_id += 1

    if not subset:
        abort(400, message="None of the selected items are in the current dataset")

    # Inherit embedder / clipper / media_type / death-date from the source
    # dataset's registry entry when available; otherwise derive from the
    # promoted items themselves.
    ctx = get_active_context()
    src_id = ctx.dataset_id or None
    src_entry = _reg_get(src_id) if src_id else None

    first = next(iter(subset.values()))
    media_type = (src_entry or {}).get("media_type") or first.get("media_type", "audio")
    embedder = (src_entry or {}).get("embedder") or first.get("embedder", "")
    clipper = (src_entry or {}).get("clipper", "") if src_entry else ""
    expires_at = (src_entry or {}).get("expires_at") if src_entry else None

    source = {
        "importer": "promote",
        "params": {
            "source_dataset_id": src_id or "",
            "source_name": (src_entry or {}).get("name", "") if src_entry else "",
        },
    }
    task_id = _promote_in_background(
        subset,
        name=name,
        media_type=media_type,
        embedder=embedder,
        clipper=clipper,
        expires_at=expires_at,
        source=source,
        file_type_counts=count_file_types(subset.values()),
        created_by=get_current_user(),
    )
    return {"ok": True, "message": "Promoting to dataset...", "task_id": task_id}


@datasets_staging_bp.route("/api/dataset/stage-file", methods=["POST"])
@datasets_staging_bp.response(200, DatasetStageFileResponseSchema)
@datasets_staging_bp.alt_response(400, description="Multipart body has no file or empty filename.")
def stage_file():
    """Upload a ``.pkl`` file and save it to the staging directory."""
    if "file" not in request.files:
        abort(400, message="No file provided")

    file = request.files["file"]
    if not file.filename:
        abort(400, message="No file selected")

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staging_path = STAGING_DIR / f"stage_{uuid4().hex}.pkl"
    file.save(staging_path)

    # Peek the pkl's dict structure cheaply; embeddings and inline media
    # bytes are skipped, so this stays light even on multi-GB uploads. The
    # response always returns 200 (so the client keeps the staged path and
    # can clean it up); peek failures are surfaced via the ``error`` field
    # instead of swallowed silently.
    error = ""
    try:
        import zipfile

        # Stream the member into the peeker instead of zf.read(): a
        # highly-compressed medias.pkl would otherwise materialise its whole
        # decompressed body in RAM before the peek even starts.  ZipExtFile
        # is a BufferedIOBase at runtime; typeshed types zf.open as
        # IO[bytes], hence the cast.
        import io
        from typing import cast

        with zipfile.ZipFile(str(staging_path), "r") as zf, zf.open("medias.pkl") as member:
            peeked = peek_pickle_dataset_summary(cast(io.BufferedIOBase, member))
        if isinstance(peeked, dict) and "medias" in peeked:
            media_dict = peeked["medias"]
        elif isinstance(peeked, dict):
            media_dict = peeked
        else:
            media_dict = {}
        count = len(media_dict) if isinstance(media_dict, dict) else 0
        media_type = "unknown"
        if isinstance(media_dict, dict) and media_dict:
            first = next(iter(media_dict.values()))
            if isinstance(first, dict):
                media_type = first.get("media_type", "audio") or "unknown"
        del peeked, media_dict
    except Exception as e:
        count = 0
        media_type = "unknown"
        error = f"{type(e).__name__}: {e}"

    name = file.filename or "Uploaded dataset"
    return {
        "path": str(staging_path),
        "name": name,
        "count": count,
        "media_type": media_type,
        "error": error,
    }


# ---------------------------------------------------------------------------
# POST /api/dataset/stage-import/<importer_name>
#
# Plugin-field route: body shape depends on the importer plugin.  Not
# described in the OpenAPI spec; runtime validation goes through
# :func:`validate_plugin_args` (per-plugin schema built from the
# importer's :attr:`fields`), so missing required fields / invalid
# select values raise 422.  File fields are read into ``BytesIO`` so the
# background staging thread can consume them after the request context
# tears down.  Pass-through keys (``source_specs``, ``dataset_name``)
# ride along on the request body and are preserved via
# ``Meta.unknown = "include"`` in the per-plugin schema.
# ---------------------------------------------------------------------------


@datasets_staging_bp.route("/api/dataset/stage-import/<importer_name>", methods=["POST"])
def stage_import(importer_name: str):
    """Run a registered importer in staging mode.

    Plugin-dependent body shape: not described in the OpenAPI spec.
    """
    importer, err = get_plugin_or_404(get_importer, list_importers, importer_name, "importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_values = validate_plugin_args(
        importer,
        file_mode="bytesio",
        extra_keys=("source_specs", "dataset_name"),
    )

    task_id = _stage_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Staging started", "task_id": str(task_id) if task_id else ""})


@datasets_staging_bp.route("/api/dataset/stage-demo/<name>", methods=["POST"])
@datasets_staging_bp.arguments(DatasetStageDemoRequestSchema)
@datasets_staging_bp.response(200, DatasetStagingStartedResponseSchema)
@datasets_staging_bp.alt_response(400, description="Unknown demo dataset name.")
@datasets_staging_bp.alt_response(500, description="The demo importer is unavailable.")
def stage_demo(body: dict, name: str):
    """Stage a demo dataset as a temporary ``.pkl`` file."""
    if name not in DEMO_DATASETS:
        abort(400, message="Invalid dataset name")

    importer = get_importer("demo")
    if importer is None:
        abort(500, message="demo importer not available")

    converter_name = body.get("converter", "")
    dataset_name = str(body.get("dataset_name") or "").strip()

    field_values: dict = {"name": name}
    if converter_name:
        field_values["converter"] = converter_name
    if dataset_name:
        field_values["dataset_name"] = dataset_name

    label = dataset_name or DEMO_DATASETS[name].get("label", name)
    task_id = _stage_importer_in_background(importer, field_values, label=label)
    return {"ok": True, "message": "Staging demo dataset...", "task_id": str(task_id) if task_id else ""}


@datasets_staging_bp.route("/api/dataset/staging", methods=["DELETE"])
@datasets_staging_bp.response(200, ClearStagingResponseSchema)
def clear_staging():
    """Remove all files from the staging directory."""
    if STAGING_DIR.exists():
        for f in STAGING_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
    return {"ok": True}


@datasets_staging_bp.route("/api/dataset/import/<importer_name>/options", methods=["POST"])
@datasets_staging_bp.arguments(ImporterFieldOptionsRequestSchema)
@datasets_staging_bp.response(200, ImporterFieldOptionsResponseSchema)
@datasets_staging_bp.alt_response(400, description="Unknown or non-dynamic field key.")
@datasets_staging_bp.alt_response(404, description="Unknown importer name.")
@datasets_staging_bp.alt_response(500, description="get_field_options did not return a list.")
@datasets_staging_bp.alt_response(501, description="Importer does not implement get_field_options.")
@datasets_staging_bp.alt_response(502, description="Remote service backing dynamic options raised an error.")
def importer_field_options(body: dict, importer_name: str):
    """Return dropdown options for a dynamic-options field.

    The importer's ``get_field_options(field_key, current_values)`` is
    called with the supplied snapshot of current form values. Errors
    from the plugin (network failure, auth error, etc.) are surfaced as
    a 502 with the original message so the frontend can display them
    inline.
    """
    importer, err = get_plugin_or_404(get_importer, list_importers, importer_name, "importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_key = body["field_key"].strip()
    values = body.get("values") or {}

    field = next((f for f in importer.fields if f.key == field_key), None)
    if field is None:
        abort(400, message=f"Unknown field: {field_key!r}")
    if not getattr(field, "dynamic_options", False):
        abort(400, message=f"Field {field_key!r} is not dynamic")

    try:
        options = importer.get_field_options(field_key, values)
    except NotImplementedError as exc:
        abort(501, message=str(exc) or "Importer does not implement get_field_options")
    except Exception as exc:  # noqa: BLE001 (surface remote-service errors verbatim)
        abort(502, message=str(exc) or type(exc).__name__)

    if not isinstance(options, list):
        abort(500, message="get_field_options must return a list")
    return {"options": [_normalise_option(o) for o in options]}


@datasets_staging_bp.route("/api/dataset/import/<importer_name>/suggested-name", methods=["POST"])
@datasets_staging_bp.arguments(ImporterSuggestedNameRequestSchema)
@datasets_staging_bp.response(200, ImporterSuggestedNameResponseSchema)
@datasets_staging_bp.alt_response(404, description="Unknown importer name.")
@datasets_staging_bp.alt_response(502, description="default_display_name raised an error.")
def importer_suggested_name(body: dict, importer_name: str):
    """Return the name the importer would give a dataset built from these values.

    Calls the importer's ``default_display_name(field_values)`` with a
    snapshot of the current form, so the Add-Dataset form can prefill its
    Dataset Name box with something human-readable -- including a label a
    plugin resolved from an opaque internal selection, which no
    client-side derivation could produce.  The user's own ``dataset_name``
    is deliberately ignored here: the caller only asks for the suggestion,
    and decides for itself whether to overwrite what the user typed.

    Errors from the plugin are surfaced as a 502 with the original
    message; the frontend treats any failure as "no suggestion" and leaves
    the box alone, since a name hint is never worth blocking an import.
    """
    importer, err = get_plugin_or_404(get_importer, list_importers, importer_name, "importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    values = {k: v for k, v in (body.get("values") or {}).items() if k != DATASET_NAME_FIELD_KEY}

    try:
        suggested = importer.default_display_name(values)
    except Exception as exc:  # noqa: BLE001 (surface remote-service errors verbatim)
        abort(502, message=str(exc) or type(exc).__name__)

    return {"dataset_name": str(suggested or "").strip()}


# ---------------------------------------------------------------------------
# POST /api/dataset/import/<importer_name>
#
# Plugin-field route: same per-plugin validation pattern as
# ``stage-import``.  Body shape isn't in the OpenAPI spec, but
# :func:`validate_plugin_args` enforces the per-plugin field types at
# request time; pass-through keys (``source_specs``, ``clipper``,
# ``cleaners``, ``embedder``, ``embedders``, ``clipper_params``,
# ``dataset_name``) ride along on the body and are preserved via
# ``Meta.unknown = "include"``.
# ---------------------------------------------------------------------------


@datasets_staging_bp.route("/api/dataset/import/<importer_name>", methods=["POST"])
def import_dataset(importer_name: str):
    """Run a registered importer by name in a background thread.

    Plugin-dependent body shape: not described in the OpenAPI spec.
    """
    importer, err = get_plugin_or_404(get_importer, list_importers, importer_name, "importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_values = validate_plugin_args(
        importer,
        file_mode="bytesio",
        extra_keys=(
            "source_specs",
            "clipper",
            "cleaners",
            "embedder",
            "embedders",
            "dataset_name",
            "build_projection",
            "merge_near_duplicates",
        ),
    )

    # A Semantic-locked instance never offers a patch/structural embedder in a
    # picker, so an import that names one is stale or hand-rolled: reject it
    # here rather than binding a type the rest of the UI hides.
    _abort_if_semantic_only(field_values)

    # ``clipper_params`` is multipart-encoded as a JSON string when the
    # importer has file fields; the per-plugin schema treats it as an
    # opaque pass-through (string); decode it here before handing off.
    file_keys = {f.key for f in importer.fields if f.field_type == "file"}
    clipper_params, params_err = _extract_clipper_params(bool(file_keys))
    if params_err:
        return params_err
    if clipper_params is not None:
        field_values["clipper_params"] = clipper_params

    task_id = _run_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""})


# ---------------------------------------------------------------------------
# Per-plugin typed routes for /api/dataset/import/<name> and /api/dataset/
# stage-import/<name>.  Registered at module-import time by iterating the
# importer registry, so each known importer gets a static URL whose body
# schema is described in /api/openapi.json with real per-field types.
# Unknown importer names fall through to the parameterized routes above
# (preserving the legacy 404 message that names the unknown importer).
# Plugins with file fields stay on the parameterized fallback (multipart
# bodies aren't usefully described by the generic plugin schema).
# ---------------------------------------------------------------------------

register_plugin_typed_routes(
    datasets_staging_bp,
    list_plugins=list_importers,
    path_template="/api/dataset/import/{plugin_name}",
    endpoint_prefix="import_dataset",
    delegate=import_dataset,
    extra_keys=(
        "source_specs",
        "clipper",
        "cleaners",
        "embedder",
        "embedders",
        "dataset_name",
        "clipper_params",
        "build_projection",
        "merge_near_duplicates",
    ),
)
register_plugin_typed_routes(
    datasets_staging_bp,
    list_plugins=list_importers,
    path_template="/api/dataset/stage-import/{plugin_name}",
    endpoint_prefix="stage_import",
    delegate=stage_import,
    extra_keys=("source_specs", "dataset_name"),
)
