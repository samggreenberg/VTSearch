"""Dataset loading entry points: an importer's form values in, a background load out.

:func:`_run_importer_in_background` is where an importer run starts (the Add
Dataset routes, the CLI, reload-from-origin): it pops the pipeline's own
fields (clipper, cleaners, embedders, the reference / projection / near-dup
toggles), picks whole or chunked loading, and hands the load to
:func:`~vtscore.datasets.load_origin._run_origin_load_in_background`.  The
rest of the load is split by concern:

* :mod:`vtscore.datasets.load_workers` - the download / embed concurrency
  gates and their controller, the per-task harness (task registration, the
  worker thread), the request-field parsers, the failure and terminal-state
  handling, and the app's last-embedder persistence hook.
* :mod:`vtscore.datasets.load_origin` - the single-dataset load itself: the
  importer under the gates, then every post-import stage of
  :mod:`vtscore.datasets.stages`.
* :mod:`vtscore.datasets.load_staging` - the combine flow's staging import,
  saved to a temporary pkl rather than registered.
* :mod:`vtscore.datasets.load_multi` - one importer run, several datasets.

The public names keep this path: :func:`register_last_embedder_persistence_hook`,
:func:`clear_dataset`, :func:`reset_load_gates_for_tests` and
:data:`STAGING_DIR` are re-exported from their owners.  The gates and the hook
themselves are rebound at run time, so they are not re-exported: read them
off :mod:`~vtscore.datasets.load_workers`.  A test that stubs a helper patches
the module that calls it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from vtscore.datasets.clipper_chain import append_cleaner_steps
from vtscore.datasets.import_event import DatasetImported
from vtscore.datasets.load_origin import _run_origin_load_in_background
from vtscore.datasets.load_staging import STAGING_DIR  # noqa: F401  - re-exported for consumers
from vtscore.datasets.load_workers import (  # noqa: F401  - re-exported for consumers
    clear_dataset,
    register_last_embedder_persistence_hook,
    reset_load_gates_for_tests,
)
from vtscore.datasets.load_workers import (
    _normalize_media_type,
    _parse_bool,
    _parse_chain_field,
    _parse_embedder_list,
)
from vtscore.state import DatasetContext


def consume_chunks_into(
    target: dict[int, dict[str, Any]],
    chunks: Iterable[dict[int, dict[str, Any]]],
) -> None:
    """Drain *chunks* into *target* with sequential IDs.

    Each chunk yielded by an importer's ``run_chunked()`` re-uses IDs
    starting at 1, so naive ``target.update(chunk)`` would overwrite
    earlier chunks.  Renumber every media to a unique ID continuing from
    whatever IDs are already present in *target*.
    """
    next_id = max(target.keys(), default=0) + 1
    for chunk in chunks:
        for media in chunk.values():
            media["id"] = next_id
            target[next_id] = media
            next_id += 1


_CHUNK_SIZE_BY_MEDIA_TYPE: dict[str, int] = {
    "text": 5000,
    "image": 500,
    "audio": 100,
    "video": 25,
    "document": 50,
}


def auto_chunk_size(media_type: str) -> int:
    """Pick a chunk size for *media_type* that bounds peak memory.

    Tuned roughly so a single in-flight chunk's raw bytes + embeddings stay
    below ~1 GB on typical inputs.  Returns a positive int.  Importers that
    do not support chunked loading silently ignore the value.
    """
    return _CHUNK_SIZE_BY_MEDIA_TYPE.get(_normalize_media_type(media_type), 100)


def _run_importer_in_background(
    importer,
    field_values: dict,
    *,
    post_load: Callable[[DatasetContext], None] | None = None,
    on_finished: Callable[[DatasetImported], None] | None = None,
) -> str:
    """Start *importer*.run() in a daemon thread.

    When the importer reports ``supports_chunked``, the loader streams
    medias in via ``run_chunked`` to bound peak memory during the
    import/embedding phase.  The chunk size is auto-selected from the
    field's ``media_type`` (see :func:`auto_chunk_size`); there is no
    user-facing knob.

    *post_load* and *on_finished* are forwarded to
    :func:`_run_origin_load_in_background`, which documents when they run.

    Returns the task_id for progress tracking.
    """
    from vtscore.plugins.uploads import wrap_cli_file_fields  # noqa: PLC0415

    # Normalize ``field_type="file"`` values to UploadedFile.  The
    # request path supplies a FileStorage / BytesIOUploadedFile already;
    # the reload-from-origin path supplies a server path string that
    # needs CliUploadedFile wrapping so ``run()`` doesn't have to
    # branch on the input shape.
    from vtscore.state.current_user import get_current_user  # noqa: PLC0415

    field_values = wrap_cli_file_fields(importer.fields, field_values)
    created_by = get_current_user()
    origin = importer.build_origin(field_values)
    # Reference mode (server importers): store path references instead of
    # copying media bytes.  Maps onto the loader's ``thin`` flag.  Popped so
    # it isn't forwarded into the importer's field_values (run() takes ``thin``
    # as a parameter, not a field).
    reference_files = _parse_bool(field_values.pop("reference_files", None))
    clipper_name = field_values.pop("clipper", "") or ""
    clipper_params = field_values.pop("clipper_params", None)
    chain_steps = _parse_chain_field(field_values.pop("clipper_chain", None))
    # Enabled cleanup gates always run last, on the finished units, so they are
    # appended to the chain here rather than positioned by the client.
    chain_steps = append_cleaner_steps(chain_steps, field_values.pop("cleaners", None))
    # Keep clipper in field_values for importers that need it (e.g. demo
    # importer stores it in the container metadata for readiness tracking).
    field_values["clipper"] = clipper_name
    # Importers that clip themselves (``handles_own_clipping``) own the full
    # clip config: hand the params/chain back so their ``run`` clips with the
    # user's real settings, and suppress the pipeline-level clipper so the
    # media isn't clipped a second time on top of the importer's own pass.
    if getattr(importer, "handles_own_clipping", False):
        if clipper_params is not None:
            field_values["clipper_params"] = clipper_params
        if chain_steps is not None:
            field_values["clipper_chain"] = chain_steps
        pipeline_clipper, pipeline_clipper_params, pipeline_chain_steps = "", None, None
    else:
        pipeline_clipper, pipeline_clipper_params, pipeline_chain_steps = clipper_name, clipper_params, chain_steps
    embedder_name = field_values.get("embedder", "")
    embedders = _parse_embedder_list(field_values.pop("embedders", None))
    # The primary picker's choice always leads the embed order (it becomes each
    # media's recorded primary); the trio's patch/structural picks ride behind
    # it.  Defensive: include the primary even if the client omitted it.
    if embedders and embedder_name and embedder_name not in embedders:
        embedders = [embedder_name, *embedders]
    build_projection = _parse_bool(field_values.pop("build_projection", None))
    merge_near_duplicates = _parse_bool(field_values.pop("merge_near_duplicates", None))

    # Extract media_type from field_values so in-progress tasks can expose it
    # to the frontend (used for guessing the type in subsequent add dialogs).
    media_type_hint = _normalize_media_type(field_values.get("media_type", ""))

    use_chunked = getattr(importer, "supports_chunked", False)
    chunk_size = auto_chunk_size(media_type_hint) if use_chunked else 0

    def _load(target_medias):
        if use_chunked:
            consume_chunks_into(target_medias, importer.run_chunked(field_values, chunk_size, thin=reference_files))
        else:
            importer.run(field_values, target_medias, thin=reference_files)

    # For demo datasets we know the expected item count + archive size up front,
    # which lets load_step_weights pace by the measured n-aware cost model rather
    # than the static asymptote. Unknown for streaming folder imports -> None.
    demo_id, n_hint, download_size_mb_hint = _demo_load_hints(importer, field_values)

    return _run_origin_load_in_background(
        _load,
        origin,
        name=importer.resolve_display_name(field_values),
        clipper=pipeline_clipper,
        clipper_params=pipeline_clipper_params,
        chain_steps=pipeline_chain_steps,
        embedder=embedder_name,
        embedders=embedders,
        created_by=created_by,
        media_type=media_type_hint,
        build_projection=build_projection,
        merge_near_duplicates=merge_near_duplicates,
        dataset_id=demo_id,
        n_hint=n_hint,
        download_size_mb_hint=download_size_mb_hint,
        post_load=post_load,
        on_finished=on_finished,
    )


def _demo_load_hints(importer, field_values: dict) -> tuple[str, int | None, float | None]:
    """Demo dataset id, expected item count, and archive size for a demo load,
    else ``("", None, None)``.

    The ``n`` / size hints enable the ``n``-aware cost-model weights (see
    :func:`vtscore.datasets.stages._common.load_step_weights`); the id is also
    threaded to the load profiler so its rows carry ``dataset_id`` (and can
    resolve the archive size) rather than landing empty (see #2614). Only the
    demo importer knows these up front; folder importers stream, so they fall
    back to the static weight profile.
    """
    if getattr(importer, "name", "") != "demo":
        return "", None, None
    dataset_id = field_values.get("name", "")
    if not dataset_id:
        return "", None, None
    from vtscore.datasets.config import DEMO_DATASETS  # noqa: PLC0415
    from vtscore.datasets.demo_counts import exact_demo_count  # noqa: PLC0415

    n = exact_demo_count(dataset_id)
    info = DEMO_DATASETS.get(dataset_id) or {}
    dl = info.get("download_size_mb")
    return dataset_id, n, (float(dl) if dl is not None else None)
