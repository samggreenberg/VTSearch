"""Command-line interface utilities for VTSearch.

The only CLI workflow is autodetect: load a dataset (from pickle or via an
importer), score it against the detectors flagged for Auto-Find in the settings
file, and export the results.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from collections.abc import Iterator
from typing import Any


from vtscore import cli_progress

from vtscore.datasets.loader import apply_custom_metadata_md5, load_dataset_from_pickle
from vtscore.utils.hits import build_media_hit
from vtscore.utils.scores import sigmoid_to_finite_scores

logger = logging.getLogger(__name__)


def _list_importer_names() -> list[str]:
    """Return the names of all registered importers."""
    from vtscore.datasets.importers import list_importers

    return [imp.name for imp in list_importers()]


def _summarize_autofind_detectors(detector_names: list[str]) -> list[dict[str, Any]]:
    """Read each named detector's on-disk JSON and return a small summary.

    Used by ``--dry-run`` to describe which detectors would be trained and
    scored without actually loading models or embedding any media.
    """
    from vtscore.detectors.store import _detector_path, _read_detector

    summaries: list[dict[str, Any]] = []
    for name in detector_names:
        path = _detector_path(name)
        data = _read_detector(path)
        if data is None:
            summaries.append({"name": name, "path": str(path), "missing": True})
            continue
        labelset = data.get("labelset") or {}
        labels = labelset.get("labels") if isinstance(labelset, dict) else None
        n_labels = len(labels) if isinstance(labels, list) else 0
        summaries.append(
            {
                "name": name,
                "path": str(path),
                "media_type": data.get("media_type", "") or "",
                "labels": n_labels,
                "missing": False,
            }
        )
    return summaries


def _print_dry_run_source(source_description: dict[str, Any]) -> None:
    """Print the ``Source:`` block of the dry-run plan (kind, params, chunking)."""
    print("Source:", flush=True)
    kind = source_description.get("kind", "")
    if kind == "pickle":
        print(f"  Dataset pickle: {source_description.get('dataset', '')}", flush=True)
    elif kind == "importer":
        print(f"  Importer: {source_description.get('importer', '')}", flush=True)
        params = source_description.get("params") or {}
        if params:
            print("  Params:", flush=True)
            for k, v in params.items():
                print(f"    {k}: {v if v != '' else '(empty)'}", flush=True)
        else:
            print("  Params: (none)", flush=True)
    chunk_size = source_description.get("chunk_size")
    print(f"  Chunk size: {chunk_size if chunk_size else 'whole dataset'}", flush=True)
    if source_description.get("stream_results"):
        neg = "included" if source_description.get("keep_negatives") else "dropped"
        print(f"  Streaming: yes (hits written to the exporter per chunk; negatives {neg})", flush=True)


def _print_dry_run_plan(
    *,
    source_description: dict[str, Any],
    settings_path: str | None,
    autofind_detectors: list[str],
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
) -> None:
    """Print the autodetect plan that ``--dry-run`` would otherwise execute."""
    print("DRY RUN - no media will be loaded, embedded, scored, or exported.", flush=True)
    print("", flush=True)

    _print_dry_run_source(source_description)
    print("", flush=True)

    print(f"Settings: {settings_path or '(default: data/settings.json)'}", flush=True)
    if not autofind_detectors:
        print("Auto-Find detectors: (none - pipeline would abort with an error)", flush=True)
    else:
        summaries = _summarize_autofind_detectors(autofind_detectors)
        print(f"Auto-Find detectors ({len(summaries)}):", flush=True)
        for s in summaries:
            if s.get("missing"):
                print(f"  - {s['name']}  [MISSING - {s['path']}]", flush=True)
            else:
                print(
                    f"  - {s['name']}  [media_type={s['media_type'] or '?'}, labels={s['labels']}, file={s['path']}]",
                    flush=True,
                )
    print("", flush=True)

    print(f"Exporter: {exporter_name or 'gui (default - print to console)'}", flush=True)
    if exporter_field_values:
        for k, v in exporter_field_values.items():
            print(f"  {k}: {v if v != '' else '(empty)'}", flush=True)


def _list_exporter_names() -> list[str]:
    """Return the names of all registered exporters."""
    from vtscore.exporters import list_exporters

    return [exp.name for exp in list_exporters()]


def _load_and_train_detectors(
    detector_names: list[str],
    media_type: str,
    snap: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve, re-embed, and train an MLP for each named detector.

    For every name in *detector_names* the on-disk JSON is read, the
    labelset's origins are resolved via their importer's ``resolve_file()``,
    the files are embedded with the dataset's embedder, and an MLP is trained
    from the resulting vectors.  Detectors whose ``media_type`` doesn't match
    the dataset are skipped with a warning.  Detectors that declare an
    ``input_spec`` (a clipper the detector was trained on) are also skipped
    when the loaded dataset wasn't clipped to match - the resulting
    embeddings would be from a different granularity and the scores would
    be meaningless.

    Returns a ``{name: {"mlp": nn.Sequential, "threshold": float}}`` map.
    Raises :class:`ValueError` if a detector cannot be trained - for example
    when none of its labels' origin files are resolvable from the CLI
    environment.
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.converter_routing import detector_can_score
    from vtscore.detectors.input_spec import (
        clipper_matches,
        extract_input_spec_from_medias,
    )
    from vtscore.detectors.store import _detector_path, _read_detector
    from vtscore.detectors.labelset_training import train_from_labelset
    from vtscore.state.core import DetectorContext

    dataset_spec = extract_input_spec_from_medias(snap)
    # The dataset can hold mixed source types (a folder of videos + PDFs);
    # match each detector against the whole set, not just the first media's
    # type, so a converter-reachable detector isn't skipped on iteration order.
    source_types = {m.get("media_type") or "" for m in snap.values()}

    out: dict[str, dict[str, Any]] = {}
    for det_name in detector_names:
        det = _read_detector(_detector_path(det_name))
        if det is None:
            raise ValueError(f"Detector '{det_name}' not found in the detectors dir.")

        det_media_type = det.get("media_type", "") or ""
        # A detector matches when its target type is present directly or is
        # reachable from some source type via a one-hop converter route (so one
        # image detector scores native images, ``video2image``, and
        # ``document2image`` in the same run). Legacy detectors with no
        # media_type match anything, as before.
        if det_media_type and not detector_can_score(det_media_type, source_types):
            cli_progress.emit(
                "detector_skipped",
                text=(
                    f"Skipping detector '{det_name}': media_type "
                    f"{det_media_type!r} has no direct or converter route from "
                    f"dataset types {sorted(source_types)!r}."
                ),
                detector=det_name,
                detector_media_type=det_media_type,
                dataset_media_type=media_type,
            )
            continue

        # When the detector was trained on a clipper granularity the loaded
        # dataset doesn't already match, re-clip the dataset to that granularity
        # at scoring time (auto-clip + re-embed) instead of skipping. The clipper
        # spec rides on the detector's scoring info and is applied by
        # ``route_and_embed``. A matching (or absent) clipper needs no re-clip.
        det_input_spec = det.get("input_spec") if isinstance(det.get("input_spec"), dict) else None
        reclip_clipper = ""
        reclip_params: dict[str, Any] = {}
        if det_input_spec and not clipper_matches(det_input_spec, dataset_spec):
            reclip_clipper = det_input_spec.get("clipper") or ""
            reclip_params = dict(det_input_spec.get("clipper_params") or {})
            if reclip_clipper:
                dataset_clipper = (dataset_spec or {}).get("clipper") or "(none)"
                cli_progress.emit(
                    "detector_reclip",
                    text=(
                        f"Re-clipping dataset for detector '{det_name}' with "
                        f"clipper {reclip_clipper!r} to match its input_spec "
                        f"(dataset clipper: {dataset_clipper!r})."
                    ),
                    detector=det_name,
                    detector_input_spec=det_input_spec,
                    dataset_input_spec=dataset_spec or {},
                )

        labelset = LabelSet.from_dict(det.get("labelset") or {})
        if not labelset.elements:
            raise ValueError(f"Detector '{det_name}' has no labels.")

        det_ctx = DetectorContext(det_name, media_type=det_media_type or media_type)
        trained = train_from_labelset(
            det_ctx,
            labelset,
            media_type=det_media_type or media_type,
            snap=snap,
        )
        if not trained:
            cached = len(det_ctx.label_embeddings)
            total = len(labelset.elements)
            raise ValueError(
                f"Detector '{det_name}': could not train MLP "
                f"(resolved {cached} of {total} label origins, need ≥1 good and ≥1 bad). "
                "The original media may not be reachable from the CLI - for example, "
                "labels collected through the local_folder importer have no resolve_file() path."
            )
        # Alongside the scoring artifacts (mlp/threshold), carry the metadata a
        # portable-detector export needs so the exporter never re-reads the
        # detector file: the concrete embedder space it trained in, its locked
        # type, and the labelset good/bad tallies.
        from vtscore.detectors.embedder_type import detector_embedder_type_from_data  # noqa: PLC0415

        out[det_name] = {
            "mlp": det_ctx.model,
            "threshold": det_ctx.threshold,
            "embedder": det_ctx.embedder or "",
            "media_type": det_media_type or media_type,
            "embedder_type": detector_embedder_type_from_data(det),
            "good_count": sum(1 for el in labelset.elements if el.label == "good"),
            "bad_count": sum(1 for el in labelset.elements if el.label == "bad"),
            # Clipper to re-apply at scoring time (empty when the dataset already
            # matches the detector's granularity, or the detector has no clipper).
            "clipper": reclip_clipper,
            "clipper_params": reclip_params,
        }
    return out


def _score_medias_with_detectors(
    medias: dict[int, dict[str, Any]],
    detector_mlps: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Score *medias* against pre-trained detector MLPs, routing across types.

    *medias* may arrive unembedded and may mix source types.  For each group of
    detectors sharing a target ``media_type`` + embedder + re-clip granularity,
    the medias are routed to that target (native match scored directly, other
    types converted via a one-hop converter such as ``video2image``), optionally
    re-clipped to the detector's ``input_spec.clipper``, and embedded in the
    detector's space by
    :func:`~vtscore.detectors.converter_routing.route_and_embed`.  A converter or
    clipper that fans one source media out into several (a video into frames, a
    recording into tiles) produces several scores, which are aggregated back to
    the source media by ``max`` - a source is a positive hit when *any* of its
    sub-items clears the threshold ("find the needle").  Homogeneous single-type
    datasets with no re-clip take the identity route (one hit per media),
    byte-for-byte the pre-routing behaviour.
    """
    if not medias or not detector_mlps:
        return {}

    from collections import defaultdict  # noqa: PLC0415

    from vtscore.detectors.converter_routing import route_and_embed  # noqa: PLC0415

    # Detectors sharing a (target type, embedder, re-clip spec) share one
    # routed+clipped+embedded snapshot, so a dataset scored by two image/siglip
    # detectors with the same granularity is prepared once, not per detector.
    groups: dict[tuple[str, str, str, tuple], list[str]] = defaultdict(list)
    for det_name, info in detector_mlps.items():
        clipper_params = info.get("clipper_params") or {}
        key = (
            info.get("media_type") or "",
            info.get("embedder") or "",
            info.get("clipper") or "",
            tuple(sorted((str(k), str(v)) for k, v in clipper_params.items())),
        )
        groups[key].append(det_name)

    results: dict[str, dict[str, Any]] = {}
    for (target_type, embedder_name, clipper, clipper_params_items), det_names in groups.items():
        if not target_type:
            # Legacy detector with no declared media_type: score every media
            # directly, one hit per media (raises on a missing embedding). The
            # matrix is built in the detectors' shared score embedder
            # (``embedder_name`` is part of this group's key, so every detector
            # here trained in it), not each media's primary vector - on a trio
            # dataset those differ. Typed detectors take the routing path below.
            results.update(_score_direct_all(det_names, detector_mlps, medias, embedder_name))
            continue
        scoring_medias, scoring_to_source = route_and_embed(
            medias,
            target_type,
            embedder_name,
            clipper=clipper,
            clipper_params=dict(clipper_params_items),
        )
        if not scoring_medias:
            continue
        for det_name in det_names:
            results[det_name] = _score_one_detector(
                det_name,
                detector_mlps[det_name],
                medias,
                scoring_medias,
                scoring_to_source,
                embedder_name,
            )

    if results:
        from vtscore.achievements_hooks import record_achievement

        record_achievement("find", len(medias) * len(results))

    return results


def _score_direct_all(
    det_names: list[str],
    detector_mlps: dict[str, dict[str, Any]],
    medias: dict[int, dict[str, Any]],
    embedder_name: str = "",
) -> dict[str, dict[str, Any]]:
    """Score *det_names* directly against every media's *embedder_name* vector.

    The legacy path for detectors that declare no ``media_type``: they score
    whatever embeddings the dataset already holds, one hit per media.  The
    matrix is built in *embedder_name* - the concrete space these detectors
    trained in, shared across the group - so a trio dataset whose primary vector
    differs from that space is scored correctly rather than against each media's
    primary vector; empty *embedder_name* falls back to the primary vector, the
    single-embedder behaviour.  A media that lacks that vector raises (via
    ``get_embedding_matrix_for_snap``) rather than silently scoring NaN, and the
    ``strict=True`` zip guards against an id/score length mismatch (audit M11).
    """
    import torch  # noqa: PLC0415

    from vtscore.embedding.matrix import get_embedding_matrix_for_snap  # noqa: PLC0415

    all_ids, all_embs = get_embedding_matrix_for_snap(medias, embedder_name or None)
    X_all = torch.from_numpy(all_embs)

    out: dict[str, dict[str, Any]] = {}
    for det_name in det_names:
        info = detector_mlps[det_name]
        mlp = info["mlp"]
        threshold = info["threshold"]
        with torch.no_grad():
            X_in = X_all.to(next(mlp.parameters()).device)
            scores = sigmoid_to_finite_scores(mlp(X_in))

        positive_hits: list[dict[str, Any]] = []
        negative_hits: list[dict[str, Any]] = []
        for cid, score in zip(all_ids, scores, strict=True):
            hit = build_media_hit(cid, medias[cid], score)
            if score >= threshold:
                positive_hits.append(hit)
            else:
                negative_hits.append(hit)
        positive_hits.sort(key=lambda x: x["score"], reverse=True)
        negative_hits.sort(key=lambda x: x["score"], reverse=True)

        out[det_name] = {
            "detector_name": det_name,
            "threshold": round(threshold, 4),
            "total_hits": len(positive_hits),
            "hits": positive_hits,
            "negative_hits": negative_hits,
        }
    return out


def _score_one_detector(
    det_name: str,
    info: dict[str, Any],
    source_medias: dict[int, dict[str, Any]],
    scoring_medias: dict[int, dict[str, Any]],
    scoring_to_source: dict[int, int],
    embedder_name: str,
) -> dict[str, Any]:
    """Score one detector over a routed snapshot and fold scores to source media.

    Runs the MLP over *scoring_medias* (already embedded in the detector's
    space), then reduces per-clip scores to one score per source media via
    ``max`` and builds the hit from the *source* media so a video routed through
    ``video2image`` surfaces as a single hit on the video, not one per frame.
    """
    import torch  # noqa: PLC0415

    from vtscore.embedding.matrix import get_embedding_matrix_for_snap  # noqa: PLC0415

    mlp = info["mlp"]
    threshold = info["threshold"]
    ids, embs = get_embedding_matrix_for_snap(scoring_medias, embedder_name or None)
    with torch.no_grad():
        X_in = torch.from_numpy(embs).to(next(mlp.parameters()).device)
        scores = sigmoid_to_finite_scores(mlp(X_in))

    # Aggregate clip-level scores back to the source media, keeping the best
    # (max) score per source.
    best_by_source: dict[int, float] = {}
    for scoring_id, score in zip(ids, scores, strict=True):
        src_id = scoring_to_source[scoring_id]
        prev = best_by_source.get(src_id)
        if prev is None or score > prev:
            best_by_source[src_id] = float(score)

    positive_hits: list[dict[str, Any]] = []
    negative_hits: list[dict[str, Any]] = []
    for src_id, score in best_by_source.items():
        hit = build_media_hit(src_id, source_medias[src_id], score)
        if score >= threshold:
            positive_hits.append(hit)
        else:
            negative_hits.append(hit)
    positive_hits.sort(key=lambda x: x["score"], reverse=True)
    negative_hits.sort(key=lambda x: x["score"], reverse=True)

    return {
        "detector_name": det_name,
        "threshold": round(threshold, 4),
        "total_hits": len(positive_hits),
        "hits": positive_hits,
        "negative_hits": negative_hits,
    }


def _build_multi_results_dict(
    detector_results: dict[str, dict[str, Any]],
    media_type: str = "unknown",
) -> dict[str, Any]:
    """Build the full results dict from multi-detector scoring."""
    return {
        "media_type": media_type,
        "detectors_run": len(detector_results),
        "results": detector_results,
    }


def _detect_media_type(medias: dict[int, dict[str, Any]]) -> str:
    """Return the media type from the first media, or ``"unknown"``."""
    for media in medias.values():
        return media.get("media_type", "unknown")
    return "unknown"


def _run_exporter(
    exporter_name: str,
    field_values: dict[str, Any],
    results: dict[str, Any],
    detector_mlps: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Validate and run a named exporter, printing its confirmation message.

    Most exporters consume the scored *results*.  An exporter that instead
    exports the trained classifiers themselves (``needs_trained_detectors``,
    the portable-detector bundle) is handed the *detector_mlps* the pipeline
    trained, via :meth:`LabelsetExporter.export_cli_detectors`.

    An exporter that returns an ``open_url`` gets it surfaced here rather than
    dropped: there is no browser to open it on the command line, so the URL is
    printed under the confirmation message (text mode) and carried as a field
    on the ``export_complete`` event (JSON mode), which is what lets a wrapping
    script open it itself.
    """
    from vtscore.exporters import get_exporter

    exporter = get_exporter(exporter_name)
    if exporter is None:
        available = _list_exporter_names()
        raise ValueError(f"Unknown exporter: {exporter_name}. Available: {', '.join(available)}")

    exporter.validate_cli_field_values(field_values)
    if getattr(exporter, "needs_trained_detectors", False):
        descriptors = _portable_detector_descriptors(detector_mlps or {})
        result = exporter.export_cli_detectors(descriptors, field_values)
    else:
        result = exporter.export_cli(results, field_values)
    message = result.get("message", "Export complete.")

    open_url = _validated_open_url(result, exporter_name)
    if open_url is None:
        cli_progress.emit("export_complete", text=message, message=message)
    else:
        cli_progress.emit(
            "export_complete",
            text=f"{message}\n\n  {open_url}",
            message=message,
            open_url=open_url,
        )


def _validated_open_url(result: dict[str, Any], exporter_name: str) -> str | None:
    """Return *result*'s ``open_url`` if it is one, else ``None``.

    The same scheme allowlist the HTTP route applies, for the same reason: a
    plugin should never be able to put a ``javascript:`` URL in front of the
    user.  A bad one is dropped with a warning rather than raised, because the
    export itself already succeeded — sinking a completed delivery over a
    cosmetic field would lose the run.
    """
    raw = result.get("open_url")
    if raw is None:
        return None
    from vtscore.security.url_validation import validate_browser_url

    try:
        return validate_browser_url(str(raw))
    except ValueError as exc:
        logger.warning("Exporter %r returned an unusable open_url, ignoring it: %s", exporter_name, exc)
        return None


def _portable_detector_descriptors(detector_mlps: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialise each trained detector into a portable-export descriptor.

    Turns the pipeline's ``{name: {"mlp", "threshold", ...}}`` map into the
    list of plain-data dicts :meth:`LabelsetExporter.export_cli_detectors`
    consumes - the live torch model is reduced to nested-list weights here so
    the exporter itself stays torch-free.
    """
    from vtscore.detectors.training import serialize_weights

    descriptors: list[dict[str, Any]] = []
    for name, info in detector_mlps.items():
        descriptors.append(
            {
                "detector_name": name,
                "media_type": info.get("media_type", "") or "",
                "weights": serialize_weights(info["mlp"]),
                "threshold": info["threshold"],
                "embedder": info.get("embedder", "") or "",
                "embedder_type": info.get("embedder_type", "") or "",
                "good_count": int(info.get("good_count", 0)),
                "bad_count": int(info.get("bad_count", 0)),
            }
        )
    return descriptors


def import_labels_into_detector_from_file(
    det_name: str,
    importer_name: str,
    filepath: str,
) -> tuple[int, int]:
    """Run a label importer against a single file and merge into a detector."""
    from vtscore.datasets.labelset import LabeledElement, LabelSet
    from vtscore.labels.importers import get_label_importer
    from vtscore.detectors.store import _detector_path, _read_detector, _write_detector

    path = _detector_path(det_name)
    data = _read_detector(path)
    if data is None:
        raise ValueError(f"Detector '{det_name}' not found.")

    importer = get_label_importer(importer_name)
    if importer is None:
        raise ValueError(f"Unknown label importer: {importer_name!r}.")

    label_entries = importer.run_cli({"filepath": filepath})
    if not isinstance(label_entries, list):
        raise ValueError(f"Label importer {importer_name!r} returned {type(label_entries).__name__}, expected list.")

    existing = LabelSet.from_dict(data.get("labelset") or {})
    existing_keys: set[tuple[str, str]] = {(el.md5, el.label) for el in existing.elements if el.md5}

    applied = 0
    skipped = 0
    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        md5 = entry.get("md5", "")
        if md5 and (md5, label) in existing_keys:
            skipped += 1
            continue
        existing.elements.append(LabeledElement.from_dict(entry))
        if md5:
            existing_keys.add((md5, label))
        applied += 1

    data["labelset"] = existing.to_dict()
    _write_detector(path, data)
    return applied, skipped


def _merge_detector_results(
    accumulated: dict[str, dict[str, Any]],
    new_chunk: dict[str, dict[str, Any]],
) -> None:
    """Merge detector results from a new chunk into *accumulated* in-place.

    Hits are appended, not sorted: this path holds every hit in RAM by
    design, so it defers ordering to a single final :func:`_sort_detector_results`
    pass rather than re-sorting the growing list on every chunk.
    """
    for det_name, det_result in new_chunk.items():
        if det_name not in accumulated:
            accumulated[det_name] = det_result
        else:
            accumulated[det_name]["hits"].extend(det_result["hits"])
            accumulated[det_name]["total_hits"] += det_result["total_hits"]
            if "negative_hits" in det_result:
                accumulated[det_name].setdefault("negative_hits", []).extend(det_result["negative_hits"])


def _sort_detector_results(accumulated: dict[str, dict[str, Any]]) -> None:
    """Sort every detector's hit lists by score descending, in place.

    Run once after all chunks have been merged, replacing the per-chunk sort
    that :func:`_merge_detector_results` used to do.
    """
    for det_result in accumulated.values():
        det_result["hits"].sort(key=lambda x: x["score"], reverse=True)
        if "negative_hits" in det_result:
            det_result["negative_hits"].sort(key=lambda x: x["score"], reverse=True)


def _load_pickle_whole(dataset_path: str) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield a single medias dict loaded from a pickle file."""
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    medias: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(dataset_file, medias, thin=True)
    if not medias:
        raise ValueError(f"No medias loaded from dataset: {dataset_path}")
    yield medias


def _renumber_chunks(
    chunks: Iterator[dict[int, dict[str, Any]]],
) -> Iterator[dict[int, dict[str, Any]]]:
    """Re-issue media IDs across a chunk stream so they are globally unique.

    Every chunked importer (and every chunked pickle/folder loader) emits
    chunks whose IDs restart at 1 - the in-process consumer
    :func:`vtscore.datasets.load_pipeline.consume_chunks_into` renumbers
    them as it drains. The CLI pipeline scores each chunk independently
    and merges the per-chunk hit lists, so without renumbering the hits
    in the merged export carry colliding ``id`` values across chunks.
    Wrap the source generator with this helper at the CLI boundary to
    give every media a unique id.
    """
    next_id = 1
    for chunk in chunks:
        renumbered: dict[int, dict[str, Any]] = {}
        for media in chunk.values():
            media["id"] = next_id
            renumbered[next_id] = media
            next_id += 1
        yield renumbered


def _load_pickle_chunked(dataset_path: str, chunk_size: int) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias loaded from a pickle file."""
    from vtscore.datasets.loader import load_dataset_from_pickle_chunked

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    yield from _renumber_chunks(load_dataset_from_pickle_chunked(dataset_file, chunk_size, thin=True))


def _load_importer_whole(importer_name: str, field_values: dict[str, Any]) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield a single medias dict loaded via a named importer."""
    from vtscore.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")

    importer.validate_cli_field_values(field_values)

    medias: dict[int, dict[str, Any]] = {}
    importer.run_cli(field_values, medias, thin=True)
    apply_custom_metadata_md5(medias)
    if not medias:
        raise ValueError(f"No medias loaded by importer '{importer_name}'")
    yield medias


def _load_importer_chunked(
    importer_name: str, field_values: dict[str, Any], chunk_size: int
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias loaded via a named importer."""
    from vtscore.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")

    importer.validate_cli_field_values(field_values)
    yield from _renumber_chunks(importer.run_chunked_cli(field_values, chunk_size, thin=True))


def _validate_dry_run_source(sd: dict[str, Any]) -> None:
    """Validate the source description block passed to a dry run."""
    kind = sd.get("kind")
    if kind == "pickle":
        dataset = sd.get("dataset", "")
        if dataset and not Path(dataset).exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset}")
    elif kind == "importer":
        from vtscore.datasets.importers import get_importer

        importer_name = sd.get("importer", "")
        importer = get_importer(importer_name)
        if importer is None:
            available = _list_importer_names()
            raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")
        importer.validate_cli_field_values(sd.get("params") or {})


def _validate_dry_run_exporter(exporter_name: str, exporter_field_values: dict[str, Any] | None) -> None:
    """Resolve *exporter_name* in the registry and validate its CLI field values."""
    from vtscore.exporters import get_exporter

    exporter = get_exporter(exporter_name)
    if exporter is None:
        available = _list_exporter_names()
        raise ValueError(f"Unknown exporter: {exporter_name}. Available: {', '.join(available)}")
    exporter.validate_cli_field_values(exporter_field_values or {})


def _emit_dry_run_plan(
    source_description: dict[str, Any],
    settings_path: str | None,
    autofind_detectors: list[str],
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
) -> None:
    """Emit the JSON ``dry_run_plan`` event or the human-readable plan text."""
    if cli_progress.get_format() == "json":
        cli_progress.emit(
            "dry_run_plan",
            source=source_description,
            settings_path=settings_path,
            autofind_detectors=_summarize_autofind_detectors(autofind_detectors),
            exporter=exporter_name,
            exporter_field_values=exporter_field_values or {},
        )
    else:
        _print_dry_run_plan(
            source_description=source_description,
            settings_path=settings_path,
            autofind_detectors=autofind_detectors,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
        )


def _run_dry_run(
    source_description: dict[str, Any] | None,
    settings_path: str | None,
    autofind_detectors: list[str],
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
) -> None:
    """Validate the planned pipeline + exporter and emit the dry-run plan."""
    sd = source_description or {}
    _validate_dry_run_source(sd)
    if exporter_name:
        _validate_dry_run_exporter(exporter_name, exporter_field_values)
    _emit_dry_run_plan(sd, settings_path, autofind_detectors, exporter_name, exporter_field_values)


def _train_detectors_for_first_chunk(
    chunk_medias: dict[int, dict[str, Any]],
    media_type: str,
    override_detectors: list[str] | None,
    autofind_detectors: list[str],
) -> dict[str, dict[str, Any]]:
    """Train each Auto-Find (or override) detector once against the first chunk.

    Raises :class:`ValueError` when no detector applies to *media_type* -
    that's almost always a settings-file misconfiguration the caller wants
    surfaced immediately.
    """
    detector_names = list(override_detectors) if override_detectors is not None else list(autofind_detectors)
    detector_mlps: dict[str, dict[str, Any]] = (
        _load_and_train_detectors(detector_names, media_type, chunk_medias) if detector_names else {}
    )
    if not detector_mlps:
        raise ValueError(
            f"No Auto-Find detectors found for media type: {media_type}. "
            "Add detectors to the settings file's autofind_detectors list."
        )
    return detector_mlps


def _score_chunk(
    chunk_medias: dict[int, dict[str, Any]],
    chunk_num: int,
    total_medias: int,
    detector_mlps: dict[str, dict[str, Any]],
    merged_results: dict[str, dict[str, Any]],
) -> None:
    """Emit chunk-start progress when relevant, score this chunk, and merge in place."""
    if chunk_num > 1 or total_medias != len(chunk_medias):
        cli_progress.emit(
            "chunk_start",
            text=f"Processing chunk {chunk_num} ({len(chunk_medias)} medias)...",
            chunk_num=chunk_num,
            chunk_size=len(chunk_medias),
        )
    chunk_results = _score_medias_with_detectors(chunk_medias, detector_mlps)
    _merge_detector_results(merged_results, chunk_results)


def _run_live_pipeline(
    media_source: Iterator[dict[int, dict[str, Any]]],
    *,
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
    override_detectors: list[str] | None,
    autofind_detectors: list[str],
    empty_error: str,
) -> None:
    """Iterate *media_source*, score each chunk, and run the exporter on the merged results."""
    merged_results: dict[str, dict[str, Any]] = {}
    media_type: str | None = None
    detector_mlps: dict[str, dict[str, Any]] | None = None
    total_medias = 0
    chunk_num = 0

    for chunk_num, chunk_medias in enumerate(media_source, 1):
        if not chunk_medias:
            continue

        apply_custom_metadata_md5(chunk_medias)
        total_medias += len(chunk_medias)

        if media_type is None:
            media_type = _detect_media_type(chunk_medias)
            detector_mlps = _train_detectors_for_first_chunk(
                chunk_medias, media_type, override_detectors, autofind_detectors
            )

        if detector_mlps:
            _score_chunk(chunk_medias, chunk_num, total_medias, detector_mlps, merged_results)

    if not merged_results:
        raise ValueError(empty_error)

    _sort_detector_results(merged_results)

    if chunk_num > 1:
        cli_progress.emit(
            "chunks_done",
            text=f"Finished processing {total_medias} medias across {chunk_num} chunk(s).",
            total_medias=total_medias,
            chunks=chunk_num,
        )

    results = _build_multi_results_dict(merged_results, media_type or "unknown")
    _run_exporter(exporter_name or "gui", exporter_field_values or {}, results, detector_mlps)


def _list_streaming_exporter_names() -> list[str]:
    """Return the names of exporters that support ``--stream-results``."""
    from vtscore.exporters import list_exporters

    return [exp.name for exp in list_exporters() if getattr(exp, "supports_streaming", False)]


def _stream_hit_records(
    first_chunk: dict[int, dict[str, Any]],
    rest: Iterator[dict[int, dict[str, Any]]],
    detector_mlps: dict[str, dict[str, Any]],
    keep_negatives: bool,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Score each chunk and yield ``(detector_name, hit)`` pairs in chunk order.

    No global accumulation and no global sort: each chunk is scored, its hits
    are yielded, and the chunk is dropped before the next one is pulled, so
    peak memory stays bounded by the chunk size regardless of how many hits
    the whole run produces.  Above-threshold hits carry ``label="good"``;
    below-threshold hits are emitted (with ``label="bad"``) only when
    *keep_negatives* is set.
    """
    import itertools  # noqa: PLC0415

    for chunk_num, chunk in enumerate(itertools.chain([first_chunk], rest), 1):
        if not chunk:
            continue
        if chunk is not first_chunk:
            # The first chunk was already normalised by the caller (it had to
            # be, to train the detectors and build the header).
            apply_custom_metadata_md5(chunk)
            cli_progress.emit(
                "chunk_start",
                text=f"Processing chunk {chunk_num} ({len(chunk)} medias)...",
                chunk_num=chunk_num,
                chunk_size=len(chunk),
            )
        chunk_results = _score_medias_with_detectors(chunk, detector_mlps)
        for det_name, det_result in chunk_results.items():
            for hit in det_result.get("hits", []):
                yield det_name, {**hit, "label": "good"}
            if keep_negatives:
                for hit in det_result.get("negative_hits", []):
                    yield det_name, {**hit, "label": "bad"}


def _run_streaming_pipeline(
    media_source: Iterator[dict[int, dict[str, Any]]],
    *,
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
    override_detectors: list[str] | None,
    autofind_detectors: list[str],
    keep_negatives: bool,
    empty_error: str,
) -> None:
    """Stream scored hits straight to a streaming-capable exporter.

    Trains detectors on the first chunk (so the exporter gets its header
    before any hit), then hands the exporter a lazy record iterator.  Nothing
    accumulates across chunks, so this is the path that scales to a media
    source with more items (and more hits) than fit in RAM.
    """
    from vtscore.exporters import get_exporter

    exporter = get_exporter(exporter_name or "gui")
    if exporter is None:
        available = _list_exporter_names()
        raise ValueError(f"Unknown exporter: {exporter_name}. Available: {', '.join(available)}")
    if not getattr(exporter, "supports_streaming", False):
        streaming = ", ".join(_list_streaming_exporter_names())
        raise ValueError(
            f"Exporter '{exporter.name}' does not support --stream-results. Streaming-capable exporters: {streaming}."
        )
    exporter.validate_cli_field_values(exporter_field_values or {})

    # Pull the first non-empty chunk so we can detect the media type and train
    # the detectors before any hit streams out.
    iterator = iter(media_source)
    first_chunk: dict[int, dict[str, Any]] | None = None
    for chunk in iterator:
        if chunk:
            first_chunk = chunk
            break
    if first_chunk is None:
        raise ValueError(empty_error)

    apply_custom_metadata_md5(first_chunk)
    media_type = _detect_media_type(first_chunk)
    detector_mlps = _train_detectors_for_first_chunk(first_chunk, media_type, override_detectors, autofind_detectors)

    header = {
        "media_type": media_type,
        "detectors": [
            {"detector_name": name, "threshold": round(info["threshold"], 4)} for name, info in detector_mlps.items()
        ],
        "keep_negatives": bool(keep_negatives),
    }

    records = _stream_hit_records(first_chunk, iterator, detector_mlps, keep_negatives)
    result = exporter.export_cli_streaming(header, records, exporter_field_values or {})
    message = result.get("message", "Export complete.")
    cli_progress.emit("export_complete", text=message, message=message)


def _run_pipeline(
    media_source: Iterator[dict[int, dict[str, Any]]],
    *,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    override_detectors: list[str] | None = None,
    empty_error: str = "No medias loaded",
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    source_description: dict[str, Any] | None = None,
) -> None:
    """Shared pipeline: read settings, iterate media chunks, score, export.

    All four CLI entry points (pickle / importer, whole / chunked) delegate
    to this single function, differing only in the *media_source* iterator
    they supply.

    When *dry_run* is True the function prints the plan derived from
    *source_description* + the settings file and returns without consuming
    the iterator, so no importer runs and no embedding or scoring occurs.

    When *override_detectors* is supplied, that list of detector names is
    used in place of the settings file's ``autofind_detectors``.  The pipeline
    YAML loader uses this to declare detectors inline without mutating the
    settings file on disk.
    """
    from vtscore.config import CoreConfig

    # Build the runtime config once (routing the optional settings_path
    # redirect through the same call) so this function - and the helpers
    # below - never import ``vtsearch.settings`` directly.
    config = CoreConfig.from_settings(settings_path=settings_path) if settings_path else CoreConfig.from_settings()
    autofind_detectors = list(config.autofind_detectors)

    # When no explicit ``--exporter`` was given, fall back to the Auto-Find
    # results exporter configured in settings (its per-exporter field values
    # come along too). An explicit ``--exporter`` always wins; if neither is
    # set the downstream default (``gui``) applies.
    if exporter_name is None and config.autofind_exporter:
        exporter_name = config.autofind_exporter
        if exporter_field_values is None:
            exporter_field_values = dict(config.autofind_exporter_field_values.get(config.autofind_exporter, {}))

    if dry_run:
        _run_dry_run(
            source_description,
            settings_path,
            autofind_detectors,
            exporter_name,
            exporter_field_values,
        )
        return

    if stream_results:
        _run_streaming_pipeline(
            media_source,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            override_detectors=override_detectors,
            autofind_detectors=autofind_detectors,
            keep_negatives=keep_negatives,
            empty_error=empty_error,
        )
        return

    _run_live_pipeline(
        media_source,
        exporter_name=exporter_name,
        exporter_field_values=exporter_field_values,
        override_detectors=override_detectors,
        autofind_detectors=autofind_detectors,
        empty_error=empty_error,
    )


def autodetect_main(
    dataset_path: str,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> None:
    """CLI entry point: run autodetect with all Auto-Find detectors."""
    try:
        _run_pipeline(
            _load_pickle_whole(dataset_path) if not dry_run else iter(()),
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            empty_error=f"No medias loaded from dataset: {dataset_path}",
            dry_run=dry_run,
            source_description={"kind": "pickle", "dataset": dataset_path, "chunk_size": None},
        )
    except Exception as e:
        cli_progress.emit_error(str(e))
        sys.exit(1)


def autodetect_importer_main(
    importer_name: str,
    field_values: dict[str, Any],
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> None:
    """CLI entry point: run autodetect with a named importer and output results."""
    try:
        _run_pipeline(
            _load_importer_whole(importer_name, field_values) if not dry_run else iter(()),
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            empty_error=f"No medias loaded by importer '{importer_name}'",
            dry_run=dry_run,
            source_description={
                "kind": "importer",
                "importer": importer_name,
                "params": field_values,
                "chunk_size": None,
            },
        )
    except Exception as e:
        cli_progress.emit_error(str(e))
        sys.exit(1)


def autodetect_main_chunked(
    dataset_path: str,
    chunk_size: int,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
) -> None:
    """CLI entry point: chunked autodetect on a pickle dataset."""
    try:
        _run_pipeline(
            _load_pickle_chunked(dataset_path, chunk_size) if not dry_run else iter(()),
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            empty_error=f"No medias loaded from dataset: {dataset_path}",
            dry_run=dry_run,
            stream_results=stream_results,
            keep_negatives=keep_negatives,
            source_description={
                "kind": "pickle",
                "dataset": dataset_path,
                "chunk_size": chunk_size,
                "stream_results": stream_results,
                "keep_negatives": keep_negatives,
            },
        )
    except Exception as e:
        cli_progress.emit_error(str(e))
        sys.exit(1)


def autodetect_importer_main_chunked(
    importer_name: str,
    field_values: dict[str, Any],
    chunk_size: int,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
) -> None:
    """CLI entry point: chunked autodetect with a named importer."""
    try:
        _run_pipeline(
            _load_importer_chunked(importer_name, field_values, chunk_size) if not dry_run else iter(()),
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            empty_error=f"No medias loaded by importer '{importer_name}'",
            dry_run=dry_run,
            stream_results=stream_results,
            keep_negatives=keep_negatives,
            source_description={
                "kind": "importer",
                "importer": importer_name,
                "params": field_values,
                "chunk_size": chunk_size,
                "stream_results": stream_results,
                "keep_negatives": keep_negatives,
            },
        )
    except Exception as e:
        cli_progress.emit_error(str(e))
        sys.exit(1)
