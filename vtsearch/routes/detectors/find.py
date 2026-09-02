"""Multi-dataset, multi-detector Find routes.

Run selected detectors against selected datasets and return merged hit/miss
results.  Each detector's MLP is sourced from its in-memory
:class:`~vtsearch.state.DetectorContext` (when loaded) or trained on demand
from its on-disk labelset.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import numpy as np
from flask_smorest import Blueprint, abort

from vtscore.concurrency.progress import CancelledError, find_progress, update_find_progress
from vtscore.detectors.training import (
    ScoringRows,
    score_rows_with_model,
    scoring_rows_for_snap,
    train_and_threshold,
)
from vtscore.embedding.media_vectors import media_embedding
from vtsearch.routes._shared import (
    find_idle,
    find_idle_on_crash,
    require_dataset_header,
    require_detector_header,
)
from vtsearch.schemas.detectors import (
    FindBoundaryNextQuerySchema,
    FindBoundaryNextResponseSchema,
    FindCancelResponseSchema,
    FindCheckLabelsRequestSchema,
    FindCheckLabelsResponseSchema,
    FindEndSessionResponseSchema,
    FindQueueIdsQuerySchema,
    FindQueueIdsResponseSchema,
    FindRequestSchema,
    FindResponseSchema,
)

detector_find_bp = Blueprint(
    "detector_find",
    __name__,
    description="Run detectors against datasets and return merged hits / "
    "negative hits, plus a pre-flight label-resolution check.",
)


# Number of high-level Find steps: prepare detectors, load data, score.
_FIND_STEPS = 3
#: Timing-profile task name; its step names and shipped fallback weights live in
#: :data:`vtscore.timing.tasks.TASKS`. An admin ``VTSEARCH_TIMING_PROFILE``
#: replaces those with seconds measured on this deployment's own storage, which
#: is what the "load datasets from pkl" step actually varies with.
_FIND_TASK = "find"


def _load_pkl_for_check(pkl_path: str) -> dict | None:
    """Best-effort load of a pkl file for label resolution.

    Returns the medias-ids-stringified-to-int snapshot, or ``None`` on
    any read / parse error (the check-labels endpoint silently skips
    unreadable datasets).
    """
    from vtscore.datasets.container import read_container  # noqa: PLC0415

    try:
        pkl_data, _meta = read_container(pkl_path)
    except Exception:
        return None
    raw_medias = pkl_data.get("medias", pkl_data)
    temp_medias: dict[int, dict] = {}
    for cid, mdata in raw_medias.items():
        mid = int(cid) if not isinstance(cid, int) else cid
        temp_medias[mid] = {**mdata, "id": mid}
    return temp_medias


def _any_label_resolves_in_pkl(pkl_path: str, labels: list[dict]) -> bool:
    """Return True if at least one label resolves to a media in *pkl_path*."""
    from vtsearch.state import build_media_lookup, resolve_media_ids  # noqa: PLC0415

    temp_medias = _load_pkl_for_check(pkl_path)
    if temp_medias is None:
        return False
    origin_lookup, md5_lookup, _ = build_media_lookup(temp_medias)
    return any(resolve_media_ids(lbl, origin_lookup, md5_lookup) for lbl in labels)


def _detector_has_direct_match(labels: list[dict], dataset_ids: list[str]) -> bool:
    """True if any selected dataset can resolve at least one label."""
    from vtscore.datasets.registry import get_dataset as reg_get_ds  # noqa: PLC0415

    for ds_id in dataset_ids:
        ds = reg_get_ds(ds_id)
        if ds is None:
            continue
        pkl_path = ds.get("pkl_path", "")
        if not pkl_path or not Path(pkl_path).is_file():
            continue
        if _any_label_resolves_in_pkl(pkl_path, labels):
            return True
    return False


def _check_detector_warning(d: dict, dataset_ids: list[str]) -> dict | None:
    """Compute the resolver-fallback warning for one detector, or ``None``.

    Returns ``None`` when the detector is unknown, has no name, has no
    on-disk labelset, has no labels, or already resolves directly via one
    of the selected datasets.
    """
    from vtscore.detectors.resolver import resolve_label_embeddings  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415

    name = d.get("name", "")
    if not name:
        return None
    det_data = _read_detector(_detector_path(name))
    if not det_data:
        return None
    labels = det_data.get("labelset", {}).get("labels", [])
    if not labels:
        return None
    if _detector_has_direct_match(labels, dataset_ids):
        return None

    resolved = resolve_label_embeddings(labels, det_data.get("media_type", "audio"))
    failed = resolved.total_count - resolved.resolved_count
    if failed <= 0:
        return None
    return {
        "detector_name": name,
        "total_labels": resolved.total_count,
        "resolved_labels": resolved.resolved_count,
        "failed_labels": failed,
    }


@detector_find_bp.route("/api/find/check-labels", methods=["POST"])
@detector_find_bp.arguments(FindCheckLabelsRequestSchema)
@detector_find_bp.response(200, FindCheckLabelsResponseSchema)
def find_check_labels(body: dict):
    """Pre-flight check: report how many detector labels can be resolved.

    Takes the same ``detector_ids`` / ``dataset_ids`` payload as ``/api/find``
    and returns per-detector resolution statistics so the frontend can warn
    the user before starting the (potentially expensive) Find operation.
    """
    from vtscore.detectors.registry import get_detector as reg_get_detector  # noqa: PLC0415

    dataset_ids = body["dataset_ids"]
    detector_ids = body["detector_ids"]

    if not dataset_ids or not detector_ids:
        return {"warnings": []}

    warnings: list[dict] = []
    for d_id in detector_ids:
        d = reg_get_detector(d_id)
        if d is None:
            continue
        warning = _check_detector_warning(d, dataset_ids)
        if warning is not None:
            warnings.append(warning)
    return {"warnings": warnings}


def _abort_find(code: int, message: str) -> NoReturn:
    """Reset find_progress to idle and abort with *code* / *message*."""
    find_idle()
    abort(code, message=message)
    raise RuntimeError("unreachable (abort() raises)")  # for type narrowing


def _resolve_find_datasets(dataset_ids: list[str]) -> list[dict]:
    """Resolve dataset IDs to registry entries, asserting each pkl file exists."""
    from vtscore.datasets.registry import get_dataset as reg_get_ds  # noqa: PLC0415

    datasets: list[dict] = []
    for ds_id in dataset_ids:
        ds = reg_get_ds(ds_id)
        if ds is None:
            _abort_find(404, f"Dataset '{ds_id}' not found")
        pkl_path = ds.get("pkl_path", "")
        if not pkl_path or not Path(pkl_path).is_file():
            _abort_find(404, f"Dataset file missing for '{ds.get('name', ds_id)}'")
        datasets.append(ds)
    return datasets


def _resolve_find_detectors(detector_ids: list[str]) -> list[dict]:
    """Resolve detector IDs to registry entries."""
    from vtscore.detectors.registry import get_detector as reg_get_detector  # noqa: PLC0415

    detectors: list[dict] = []
    for d_id in detector_ids:
        d = reg_get_detector(d_id)
        if d is None:
            _abort_find(404, f"Detector '{d_id}' not found")
        detectors.append(d)
    return detectors


def _build_detector_config(d: dict) -> dict:
    """Build a per-detector config carrying both score paths.

    Multi-dataset Find loads each dataset's medias on demand, so the
    embedder check that decides live vs cold has to happen per (detector,
    dataset) pair, not once at config-build time.  We therefore carry
    *both* the live MLP (when one is cached on the detector context) and
    the on-disk labelset (when present) so the per-dataset dispatcher can
    pick safely (see :func:`_select_scorer`).

    Aborts with 400 when the detector has no usable labels in either form.
    """
    from vtscore.detectors.embedder_type import detector_embedder_type_from_data  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415
    from vtscore.state.core import get_detector_context  # noqa: PLC0415

    config: dict = {"name": d["name"], "detector_id": d["id"]}

    det_ctx = get_detector_context(d["id"])
    if det_ctx is not None and det_ctx.model is not None:
        config["live_mlp"] = det_ctx.model
        config["threshold"] = det_ctx.threshold
        config["live_embedder"] = det_ctx.embedder or ""

    det_data = _read_detector(_detector_path(d["name"]))
    if det_data and det_data.get("labelset", {}).get("labels"):
        config["detector_data"] = det_data

    # The detector's locked embedder *type* drives per-(detector, dataset) score
    # routing: for each dataset loaded by Find, the matrix (and any cold retrain)
    # is built in the concrete embedder of this type the dataset supplies, not
    # the dataset's primary vector.  Prefer the loaded context's type, else the
    # on-disk record's (legacy records migration-read their primary name to a
    # type).  Empty for a legacy/typeless detector, which routes via the dataset
    # score precedence - byte-for-byte the pre-routing behaviour on any
    # single-embedder dataset.
    config["embedder_type"] = (det_ctx.embedder_type if det_ctx is not None else "") or (
        detector_embedder_type_from_data(det_data or {})
    )

    if "live_mlp" not in config and "detector_data" not in config:
        _abort_find(400, f"Detector '{d['name']}' has no labels for detection")
    return config


def _find_score_embedder(dc: dict, temp_medias: dict[int, dict]) -> str:
    """The concrete embedder *temp_medias* should be scored in for detector *dc*.

    Routes through the detector's locked embedder type: the concrete embedder of
    that type *temp_medias* binds, else the dataset score precedence (structural
    ▸ patch ▸ text ▸ primary) for a legacy/typeless detector.  This is the
    cross-dataset counterpart of the active-context ``keying_embedder_for_snap``
    call in the find-label / auto-detect routes, so a trio dataset is scored
    against the same role-bound vector the active-context paths use rather than
    each media's primary vector.  Empty only when *temp_medias* is empty.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    from vtscore.embedding.binding import keying_embedder_for_snap  # noqa: PLC0415

    return keying_embedder_for_snap(SimpleNamespace(embedder_type=dc.get("embedder_type", "")), temp_medias)


def _select_scorer(dc: dict, temp_medias: dict[int, dict]) -> str:
    """Pick ``"live"`` / ``"cold"`` / ``"na"`` for *dc* against *temp_medias*.

    The cached MLP can only be reused when its embedder matches the space the
    dataset about to be scored resolves to; otherwise the cross-space output is
    silent garbage at best and a ``nn.Linear`` size-mismatch crash at worst
    (H5).  The comparison is against the dataset's *score* embedder for this
    detector's type (:func:`_find_score_embedder`), not its primary vector, so a
    trio dataset whose primary differs from the detector's keying embedder is
    routed to the live MLP when (and only when) that keying embedder matches.
    When they don't match, fall back to the cold path (which retrains from the
    labelset in that same score embedder), or ``"na"`` if no cold data exists.
    """
    temp_embedder = _find_score_embedder(dc, temp_medias) if temp_medias else ""
    live_embedder = dc.get("live_embedder", "") or ""
    has_live = "live_mlp" in dc
    has_cold = "detector_data" in dc
    if has_live and (not live_embedder or not temp_embedder or live_embedder == temp_embedder):
        return "live"
    if has_cold:
        return "cold"
    return "na"


def _build_detector_configs(detectors: list[dict]) -> list[dict]:
    """Walk *detectors* and emit one config per detector, with progress updates."""
    configs: list[dict] = []
    for di, d in enumerate(detectors):
        update_find_progress(
            "running",
            f'Preparing detector "{d["name"]}"…',
            current=di + 1,
            total=len(detectors),
            step=1,
            total_steps=_FIND_STEPS,
        )
        configs.append(_build_detector_config(d))
    return configs


def _load_find_dataset_medias(ds: dict) -> dict[int, dict]:
    """Load *ds*'s pkl into a per-find ``temp_medias`` snapshot.

    Aborts with 500 on any load error.  The snapshot is owned by the
    caller and freed after scoring completes.
    """
    from vtscore.datasets.container import read_container  # noqa: PLC0415
    from vtscore.embedding.media_vectors import ensure_embeddings_dict  # noqa: PLC0415

    try:
        pkl_data, _meta = read_container(ds["pkl_path"])
        raw_medias = pkl_data.get("medias", pkl_data)

        temp_medias: dict[int, dict] = {}
        for cid, mdata in raw_medias.items():
            mid = int(cid) if not isinstance(cid, int) else cid
            media = {**mdata, "id": mid}
            # Re-key a legacy single-vector pickle into media["embeddings"] and
            # drop the singular key, then coerce every vector to float32 (the
            # snap is paired against origin-resolved vectors during scoring).
            ensure_embeddings_dict(media)
            embs = media.get("embeddings")
            if isinstance(embs, dict) and embs:
                media["embeddings"] = {n: np.array(v, dtype=np.float32) for n, v in embs.items()}
            temp_medias[mid] = media
        return temp_medias
    except Exception as e:
        _abort_find(500, f"Failed to load dataset '{ds['name']}': {e}")


def _seed_media_results(all_ids: list[int], temp_medias: dict[int, dict], dataset_name: str) -> dict[int, dict]:
    """Build the per-media result skeleton that scorers fill in."""
    media_results: dict[int, dict] = {}
    for cid in all_ids:
        clip = temp_medias[cid]
        media_results[cid] = {
            "id": cid,
            "filename": clip.get("filename", ""),
            "md5": clip.get("md5", ""),
            "origin_name": clip.get("origin_name", clip.get("filename", "")),
            "origin": clip.get("origin"),
            "dataset_name": dataset_name,
            "detector_verdicts": {},
        }
    return media_results


def _record_verdicts(
    media_results: dict[int, dict],
    dc_name: str,
    all_ids: list[int],
    scores: list[float] | None,
    threshold: float,
    fallback_verdict: str | None,
) -> None:
    """Write each detector's verdict for every media id into *media_results*.

    When *scores* is provided, verdicts come from ``score >= threshold``.
    Otherwise *fallback_verdict* (``"Error"`` / ``"N/A"``) is written with
    ``score=0`` for every id.
    """
    if scores is not None:
        for cid, score in zip(all_ids, scores, strict=True):
            verdict = "Good" if score >= threshold else "Bad"
            media_results[cid]["detector_verdicts"][dc_name] = {
                "verdict": verdict,
                "score": round(score, 4),
            }
        return
    for cid in all_ids:
        media_results[cid]["detector_verdicts"][dc_name] = {
            "verdict": fallback_verdict,
            "score": 0,
        }


def _score_with_live_mlp(dc: dict, rows: ScoringRows, all_ids: list[int], media_results: dict[int, dict]) -> None:
    """Score with an already-trained MLP from a :class:`DetectorContext`.

    The detector keeps its **own** calibrated cut (``det_ctx.threshold``) rather
    than being re-cut against this dataset: the head is fixed, so its threshold
    is a property of that head and the population it was calibrated on, and
    re-fitting per Find corpus would make one media's verdict depend on which
    *other* datasets the run happened to select.  The cold path re-fits because
    it retrains the head per dataset (see :func:`_score_with_cold_detector`);
    this one does not retrain, so it does not re-cut.

    That cut is only meaningful in the geometry it was fitted in, which is why
    the scores here come from :func:`score_rows_with_model` - the region
    max-pool every other scoring path in the app uses - and not from a matmul
    against the image-level matrix.
    """
    try:
        scores, _best = score_rows_with_model(dc["live_mlp"], rows)
        _record_verdicts(media_results, dc["name"], rows.ids, scores, dc.get("threshold", 0.5), None)
    except Exception:
        _record_verdicts(media_results, dc["name"], all_ids, None, 0.0, "Error")


def _collect_cold_training_data(
    det_data: dict,
    temp_medias: dict[int, dict],
    score_emb: str,
) -> tuple[list, list, set[int]]:
    """Assemble training X/y/voted-ids for a cold detector, in *score_emb*'s space.

    Prefers labels that resolve directly into the dataset (good_ids/bad_ids
    via origin/md5 lookup); falls back to :func:`resolve_label_embeddings`
    when the direct path doesn't yield both classes.

    The third return value is the set of *temp_medias* ids this detector's
    labels name - the media whose labels the training set carries.  Both
    branches report it (the fallback re-embeds the same labels from their
    origins, so a label that also sits in this dataset is still a voted media),
    and the threshold estimator drops them from the haystack it fits on: their
    scores under a model trained on them are optimistically shifted (issue
    #3308).

    Both paths read/embed in *score_emb* - the concrete embedder of the
    detector's type this dataset supplies - so the cold-trained MLP and the
    matrix it is scored against share one vector space rather than mixing the
    dataset's primary vector (which differs from the keying embedder on a trio
    dataset) with the score-space matrix.  Empty *score_emb* falls back to each
    media's primary vector / the media type's default embedder, the legacy
    single-embedder behaviour.
    """
    from vtsearch.state import build_media_lookup, resolve_media_ids  # noqa: PLC0415

    labels = det_data.get("labelset", {}).get("labels", [])
    origin_lookup, md5_lookup, _ = build_media_lookup(temp_medias)
    good_ids: list[int] = []
    bad_ids: list[int] = []
    for lbl in labels:
        matched = resolve_media_ids(lbl, origin_lookup, md5_lookup)
        label_val = lbl.get("label", "")
        for mid in matched:
            if label_val == "good":
                good_ids.append(mid)
            elif label_val == "bad":
                bad_ids.append(mid)

    voted_ids = {mid for mid in (*good_ids, *bad_ids) if mid in temp_medias}

    if good_ids and bad_ids:
        good_embs = [media_embedding(temp_medias[i], score_emb or None) for i in good_ids if i in temp_medias]
        bad_embs = [media_embedding(temp_medias[i], score_emb or None) for i in bad_ids if i in temp_medias]
        return good_embs + bad_embs, [1.0] * len(good_embs) + [0.0] * len(bad_embs), voted_ids

    from vtscore.detectors.resolver import resolve_label_embeddings  # noqa: PLC0415

    # Resolve+embed using the dataset's score embedder for this detector's type
    # so the resulting training vectors share one space with the score-space
    # matrix the MLP will be scored against.  Empty when the dataset is somehow
    # embedder-less, which falls back to the media type's default embedder.
    resolved = resolve_label_embeddings(
        labels,
        det_data.get("media_type", "audio"),
        embedder_name=score_emb,
    )
    if resolved.has_good_and_bad:
        return resolved.embeddings, resolved.labels, voted_ids
    return [], [], set()


def _score_with_cold_detector(
    dc: dict,
    temp_medias: dict[int, dict],
    rows: ScoringRows,
    all_ids: list[int],
    media_results: dict[int, dict],
    score_emb: str,
) -> None:
    """Train an MLP on-the-fly from the cold detector's labelset and score.

    *rows* must already be built in *score_emb*'s space (the same embedder the
    cold labelset is resolved in), so the freshly-trained MLP scores the rows
    in the space it was trained against - and **image-level**
    (``region_pooling=False``), because :func:`_collect_cold_training_data`
    builds a whole-image head: it reads each label's image-level vector and
    floods nothing, so max-pooling its scores over patch rows it was never
    shown as negatives would over-fire (``docs/ML.md``, region flooding).

    **The cut is fitted on the corpus it decides.**  *temp_medias* is handed to
    :func:`~vtscore.detectors.training.train_and_threshold` as the haystack, so
    the threshold is the fold-anchored population cut rather than the pooled
    cross-calibration one that ships when there is no snap (issue #3516).  The
    pooled rule is the weakest of the three the #3115 study measured, and it is
    biased in a direction that matters here: ``CONFORMAL_BASE_BUDGET`` caps
    false negatives and only floors false positives, so the cut sits
    systematically below the oracle - more media returned Good than the
    detector supports, which is exactly the complaint a cross-dataset Find
    draws.  There was never a reason to ship it here: the haystack is the very
    snapshot this function is scoring.

    Below 4 votes (or a single class) there are no usable folds and the
    schedule blend answers, exactly as everywhere else.

    *rows* is passed through as ``haystack_rows`` for two reasons.  It is what
    fits the cut on **the distribution it will actually cut** - the same
    train/score geometry rule #2797 established, which here means the
    image-level rows above rather than the max-pool
    :func:`~vtscore.detectors.training.scoring_rows_for_snap` would otherwise
    pick for a patch dataset.  And it is the cost bound: the estimator's fold
    passes read the matrix Find already built rather than restacking the corpus
    once per fold, so its marginal cost is ``calibrate_count + 1`` matmuls
    against a matrix already in memory (measured: bounded at ~2 s per
    (detector, dataset) pair, flat beyond ~50k media because the fit samples
    the haystack).
    """
    try:
        X_list, y_list, voted_ids = _collect_cold_training_data(dc["detector_data"], temp_medias, score_emb)
        has_both_classes = X_list and any(v == 1.0 for v in y_list) and any(v == 0.0 for v in y_list)
        if not has_both_classes:
            _record_verdicts(media_results, dc["name"], all_ids, None, 0.0, "N/A")
            return

        # ``score_emb`` is the space the labelset was resolved and *rows* built
        # in; naming it lets the unset calibration_fraction resolve to that
        # space's production split rather than the unknown-embedder fallback.
        mlp, threshold = train_and_threshold(
            X_list,
            y_list,
            snap=temp_medias,
            embedder_name=score_emb,
            voted_ids=voted_ids,
            haystack_rows=rows,
        )
        scores, _best = score_rows_with_model(mlp, rows)
        _record_verdicts(media_results, dc["name"], rows.ids, scores, threshold, None)
    except Exception:
        _record_verdicts(media_results, dc["name"], all_ids, None, 0.0, "Error")


def _record_unscored(media_results: dict[int, dict], dc_name: str, all_ids: list[int]) -> None:
    """Write ``"N/A"`` for every media *dc_name* produced no verdict for.

    :func:`~vtscore.detectors.training.scoring_rows_for_snap` skips a media with
    no usable vector in the scored space instead of failing the detector
    (issue #3179), so the scorers report on ``rows.ids``, which can be shorter
    than *all_ids*.  Filling the difference keeps every media in exactly one of
    the hit / miss tables; :func:`_partition_find_results` reads only the
    verdicts that exist, so a silently missing one would drop the media from
    both.
    """
    for cid in all_ids:
        media_results[cid]["detector_verdicts"].setdefault(dc_name, {"verdict": "N/A", "score": 0})


def _partition_find_results(media_results: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    """Split media results into positives (any Good verdict) and negatives."""
    positives: list[dict] = []
    negatives: list[dict] = []
    for mr in media_results.values():
        verdicts = mr["detector_verdicts"]
        if any(v["verdict"] == "Good" for v in verdicts.values()):
            positives.append(mr)
        elif any(v["verdict"] in ("Bad", "Error", "N/A") for v in verdicts.values()):
            negatives.append(mr)
    return positives, negatives


def _score_dataset(
    ds: dict,
    detector_configs: list[dict],
    scored_units: int,
    total_scoring_units: int,
) -> tuple[list[dict], list[dict], int, int, str]:
    """Score one dataset against every detector config.

    Returns ``(positives, negatives, new_scored_units, added_scoring_units,
    media_type)``.  ``media_type`` is the type id of the first media in the
    dataset (used to populate the response's top-level ``media_type``).
    """
    import gc  # noqa: PLC0415

    temp_medias = _load_find_dataset_medias(ds)
    if not temp_medias:
        return [], [], scored_units, 0, ""

    detected_media_type = next(iter(temp_medias.values()), {}).get("media_type", "")

    # The media order is the sorted media ids and is embedder-independent; the
    # score rows themselves are built per detector in *its* score-embedder space
    # (see below), so a trio dataset scores each detector against the role-bound
    # vector instead of every media's primary vector.
    all_ids = sorted(temp_medias.keys())

    # One row stack per distinct score embedder the detectors call for, built
    # lazily and shared across detectors that resolve to the same space (the
    # common single-embedder dataset collapses every detector to one build).
    #
    # These are `scoring_rows_for_snap` rows - the app's single definition of
    # what a detector scores a media at - not an image-level matrix stacked
    # here.  Each head is scored at the geometry it was *trained and calibrated*
    # in, which differs between the two paths and is why the cache is keyed on
    # the pooling flag as well as the embedder:
    #
    # * **live** - the head came from the app's own training, which floods a Bad
    #   vote's patch rows as negatives and cuts its threshold on the max-pooled
    #   distribution.  So it max-pools here too; scoring its image-level row
    #   alone reads a systematically *lower* quantity than the cut it is about
    #   to be compared against, and the detector under-returns.
    # * **cold** - the head is re-derived here from image-level label vectors
    #   with no flooding, so it is a whole-image head and stays on image-level
    #   rows (`docs/ML.md`, region flooding).  Its threshold is fitted on that
    #   same distribution, so the two agree.
    rows_cache: dict[tuple[str | None, bool | None], ScoringRows] = {}

    def _rows_for(embedder_name: str | None, region_pooling: bool | None = None) -> ScoringRows:
        key = (embedder_name or None, region_pooling)
        rows = rows_cache.get(key)
        if rows is None:
            rows = scoring_rows_for_snap(temp_medias, key[0], region_pooling=region_pooling)
            rows_cache[key] = rows
        return rows

    added_units = len(all_ids) * len(detector_configs)
    new_total = total_scoring_units + added_units
    media_results = _seed_media_results(all_ids, temp_medias, ds["name"])

    for dc in detector_configs:
        find_progress.check_cancelled()
        score_label = f'Scoring with "{dc["name"]}" on "{ds["name"]}"…'
        update_find_progress(
            "running",
            score_label,
            current=scored_units,
            total=new_total,
            step=3,
            total_steps=_FIND_STEPS,
        )

        choice = _select_scorer(dc, temp_medias)
        # Building the score-space rows can raise when this dataset lacks the
        # detector's embedder vector (a media missing that role's vector); record
        # an Error verdict for the detector rather than sinking the whole run,
        # matching the scorers' own failure handling.
        try:
            if choice == "live":
                _score_with_live_mlp(dc, _rows_for(dc.get("live_embedder") or None), all_ids, media_results)
            elif choice == "cold":
                score_emb = _find_score_embedder(dc, temp_medias)
                _score_with_cold_detector(
                    dc,
                    temp_medias,
                    _rows_for(score_emb or None, region_pooling=False),
                    all_ids,
                    media_results,
                    score_emb,
                )
            else:
                _record_verdicts(media_results, dc["name"], all_ids, None, 0.0, "N/A")
        except Exception:
            _record_verdicts(media_results, dc["name"], all_ids, None, 0.0, "Error")
        # The row builder drops media with no usable vector in this space rather
        # than failing the whole detector, so a scorer can return fewer verdicts
        # than there are media.  Name the remainder explicitly - an unmentioned
        # media falls out of both the hit and the miss table.
        _record_unscored(media_results, dc["name"], all_ids)

        scored_units += len(all_ids)
        update_find_progress(
            "running",
            f'Scored "{dc["name"]}" on "{ds["name"]}"',
            current=scored_units,
            total=new_total,
            step=3,
            total_steps=_FIND_STEPS,
        )

    positives, negatives = _partition_find_results(media_results)
    # Drop the score-space row stacks eagerly (temp_medias is freed when this
    # frame returns).  They are captured by the _rows_for closure, so they are
    # cleared in place rather than ``del``'d (deleting a closed-over name is a
    # SyntaxError).
    rows_cache.clear()
    gc.collect()
    return positives, negatives, scored_units, added_units, detected_media_type


@detector_find_bp.route("/api/find", methods=["POST"])
@detector_find_bp.arguments(FindRequestSchema)
@detector_find_bp.response(200, FindResponseSchema)
@detector_find_bp.alt_response(
    400,
    description="Empty datasets/detectors list, or a selected detector has no labels.",
)
@detector_find_bp.alt_response(404, description="A selected dataset or detector ID is unknown / missing.")
@detector_find_bp.alt_response(409, description="Find was cancelled via /api/find/cancel.")
@detector_find_bp.alt_response(500, description="A dataset pkl file could not be loaded.")
def multi_find(body: dict):
    """Run selected detectors on selected datasets and return merged results.

    For each dataset: loads it from its saved pkl, then for each detector runs
    detection.  Returns a merged results table.
    """
    dataset_ids = body["dataset_ids"]
    detector_ids = body["detector_ids"]

    if not dataset_ids:
        _abort_find(400, "No datasets selected")
    if not detector_ids:
        _abort_find(400, "No detectors selected")

    # Clear a leftover cancel flag from a previously-cancelled run so
    # the new operation doesn't trip on it immediately.
    find_progress.reset_cancel()

    update_find_progress(
        "running",
        "Preparing detectors…",
        current=0,
        total=len(detector_ids),
        step=1,
        total_steps=_FIND_STEPS,
    )

    datasets = _resolve_find_datasets(dataset_ids)

    # Pace against the actual size of this Find: the scoring step scales with
    # every media in every selected dataset, and the pkl-load step with the same
    # count on this host's storage. Both are known now that the registry entries
    # are resolved, so the weights need not fall back to a size-blind guess.
    from vtscore import timing  # noqa: PLC0415

    n_scored = sum(int(ds.get("num_items") or 0) for ds in datasets)
    find_media_type = next((ds.get("media_type", "") for ds in datasets if ds.get("media_type")), "")
    find_embedder = next((ds.get("embedder", "") for ds in datasets if ds.get("embedder")), "")
    find_progress.set_step_weights(
        timing.step_weights(_FIND_TASK, media_type=find_media_type, embedder=find_embedder, n=n_scored)
    )
    # Every exit below — success, cancel, abort, and unexpected crash alike —
    # parks the tracker at "idle", which is what closes the recorder.  The
    # crash case is the guard's job (see :func:`find_idle_on_crash`).
    recorder = timing.record_task(
        find_progress, _FIND_TASK, media_type=find_media_type, embedder=find_embedder, auto_finish=True
    )
    recorder.start()
    recorder.set_scale(n=n_scored)

    with find_idle_on_crash(recorder):
        detectors = _resolve_find_detectors(detector_ids)
        detector_configs = _build_detector_configs(detectors)
        detector_names = [dc["name"] for dc in detector_configs]

        all_results: list[dict] = []
        all_negative_results: list[dict] = []
        detected_media_type = ""
        total_scoring_units = 0
        scored_units = 0

        try:
            for di, ds in enumerate(datasets):
                find_progress.check_cancelled()
                ds_label = f'Loading dataset "{ds["name"]}"…'
                update_find_progress(
                    "running",
                    ds_label,
                    current=di,
                    total=len(datasets),
                    step=2,
                    total_steps=_FIND_STEPS,
                )

                positives, negatives, scored_units, added_units, ds_media_type = _score_dataset(
                    ds,
                    detector_configs,
                    scored_units,
                    total_scoring_units,
                )
                all_results.extend(positives)
                all_negative_results.extend(negatives)
                total_scoring_units += added_units
                if not detected_media_type and ds_media_type:
                    detected_media_type = ds_media_type
        except CancelledError:
            # A cancelled run's phase timings describe a partial job, so they are
            # recorded as not-ok and the fit drops them.
            recorder.finish(ok=False)
            _abort_find(409, "Find cancelled")

        find_idle()

        return {
            "results": all_results,
            "negative_results": all_negative_results,
            "datasets": [ds["name"] for ds in datasets],
            "detectors": detector_names,
            "media_type": detected_media_type,
            "multiple_datasets": len(datasets) > 1,
            "multiple_detectors": len(detector_configs) > 1,
            "total_hits": len(all_results),
        }


@detector_find_bp.route("/api/find/cancel", methods=["POST"])
@detector_find_bp.response(200, FindCancelResponseSchema)
def cancel_find():
    """Cancel any in-flight find-style scoring.

    Sets the cancel flag on the shared ``find_progress`` tracker, which
    every scoring path (``/api/find``, ``/api/find-label``, and
    ``/api/auto-detect``) reports progress through. Long-running loops
    poll the flag between iterations and bail out by raising
    :class:`CancelledError`. Always returns 200; calling cancel when
    nothing is running is a no-op.
    """
    find_progress.cancel()
    return {"ok": True}


@detector_find_bp.route("/api/find/end-session", methods=["POST"])
@detector_find_bp.response(200, FindEndSessionResponseSchema)
@require_dataset_header
@require_detector_header
def end_find_session_route():
    """Discard the active detector's live Find session, if it has one.

    ``/api/find-label`` leaves the detector's in-memory votes holding its own
    call for *every* item in the dataset, flagged ``find_mode`` so they are kept
    out of the labelset.  Those presumptions are not training labels, and the
    Train window reads the same vote dicts - so it must say "I am training now"
    before it reads them, or it shows the whole collection as voted and
    Autopilot lands in a terminal phase on arrival (issue #3212).

    Idempotent: with no session to end this is a no-op that reports
    ``ended = false``.
    """
    from vtscore.detectors.dataset_sync import end_find_session  # noqa: PLC0415

    return {"ok": True, "ended": end_find_session()}


@detector_find_bp.route("/api/find/queue-ids", methods=["GET"])
@detector_find_bp.arguments(FindQueueIdsQuerySchema, location="query")
@detector_find_bp.response(200, FindQueueIdsResponseSchema)
@require_detector_header
def find_queue_ids_route(query: dict):
    """Return the Find work-queue media ids the bulk actions operate over.

    The left work queue (Browse / To Dataset / Export) and the right-panel
    actions scope over the *entire* positive set, which the frontend used to
    derive by scanning the whole client-side ranking.  This endpoint computes it
    server-side from the frozen Find scores + current cutoff + verified set, so a
    windowed client that no longer holds the full order can still act on every
    matching item (``docs/plans/scalability.md`` S3/S17/S19).

    **No frontend caller yet, deliberately.** It is built and tested ahead of
    the ``find-view`` windowing slice that switches ``unverifiedGoodIds`` /
    ``goodIds`` onto it; the two must land atomically, so this half shipped
    first. Not dead code — see the plan above before proposing its removal.

    Empty outside Find mode or before a scoring pass has run.
    """
    from vtsearch.state import find_queue_ids  # noqa: PLC0415

    ids = find_queue_ids(query["filter"])
    return {"ids": ids, "count": len(ids)}


@detector_find_bp.route("/api/find/boundary-next", methods=["GET"])
@detector_find_bp.arguments(FindBoundaryNextQuerySchema, location="query")
@detector_find_bp.response(200, FindBoundaryNextResponseSchema)
@require_detector_header
def find_boundary_next_route(query: dict):
    """Return the next unverified item on the Find boundary walk.

    Walks outward from the cutoff, alternating faces, so "just sit and vote"
    samples the marginal positives and marginal negatives rather than only
    marching up the positive pile.  Computed server-side from the frozen scores +
    cutoff + verified set so it stays correct once the client holds only a
    window of the ranking.  Returns ``{"id": null, "side": null}`` when both
    sides are exhausted (the done state).

    **No frontend caller yet, deliberately** — the same story as
    ``/api/find/queue-ids`` above: it awaits the ``find-view`` windowing slice
    that moves ``advanceToBoundary`` onto it (``docs/plans/scalability.md``
    S3/S17/S19). Not dead code.
    """
    from vtsearch.state import find_boundary_next  # noqa: PLC0415

    return find_boundary_next(query["side"], exclude=query.get("exclude_id"))
