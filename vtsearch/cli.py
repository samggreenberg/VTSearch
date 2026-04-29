"""Command-line interface utilities for VTSearch.

The only CLI workflow is autodetect: load a dataset (from pickle or via an
importer), score it against autorun processors from a settings file, and
export the results.  Datasets, detectors, and labelsets are loaded
indirectly as part of this workflow — there are no standalone CLI commands
for importing them individually.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import numpy as np

from vtsearch.datasets.loader import apply_custom_metadata_md5, load_dataset_from_pickle
from vtsearch.utils.hits import build_media_hit


def _score_clips_with_detector(
    medias: dict[int, dict[str, Any]],
    detector_path: str,
) -> list[dict[str, Any]]:
    """Score all *medias* using the detector at *detector_path*.

    Returns a list of dicts for medias predicted as "Good", sorted descending
    by score.  Each dict contains ``id``, ``filename``, ``category``, and
    ``score``.

    Raises:
        FileNotFoundError: If the detector file does not exist.
        ValueError: If the medias dict is empty or the detector is invalid.
    """
    detector_file = Path(detector_path)

    if not detector_file.exists():
        raise FileNotFoundError(f"Detector file not found: {detector_path}")

    if not medias:
        raise ValueError("No medias loaded from dataset")

    # Load detector
    try:
        with open(detector_file, "r") as f:
            detector_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in detector file: {detector_path}") from exc

    from vtsearch.models.weights_compat import normalize_detector_weights

    nw = normalize_detector_weights(detector_data)
    weights = nw.weights
    threshold = nw.threshold

    # Reconstruct the MLP model from weights
    import torch  # noqa: PLC0415

    from vtsearch.models.training import build_model_from_weights

    model = build_model_from_weights(weights)

    # Score all medias
    all_ids = sorted(medias.keys())
    all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    with torch.no_grad():
        scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

    # Collect positive hits (score >= threshold)
    positive_hits = []
    for cid, score in zip(all_ids, scores):
        if score >= threshold:
            positive_hits.append(build_media_hit(cid, medias[cid], score))

    # Sort by score descending
    positive_hits.sort(key=lambda x: x["score"], reverse=True)

    return positive_hits


def run_autodetect(dataset_path: str, detector_path: str) -> list[dict[str, Any]]:
    """Load a dataset and detector, run the detector, and return positive hits.

    Args:
        dataset_path: Path to a pickle file containing the dataset.
        detector_path: Path to a JSON file containing detector weights and threshold.

    Returns:
        A list of dicts for medias predicted as "Good", each containing
        the media's ``id``, ``filename``, ``category``, and ``score``.

    Raises:
        FileNotFoundError: If the dataset or detector file does not exist.
        ValueError: If the dataset is empty or the detector file is invalid.
    """
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load dataset in thin mode — CLI only needs embeddings, not media bytes
    medias: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(dataset_file, medias, thin=True)

    if not medias:
        raise ValueError(f"No medias loaded from dataset: {dataset_path}")

    return _score_clips_with_detector(medias, detector_path)


def run_autodetect_with_importer(
    importer_name: str,
    field_values: dict[str, Any],
    detector_path: str,
) -> list[dict[str, Any]]:
    """Load a dataset via a named importer, then run a detector on it.

    This is the importer-aware counterpart of :func:`run_autodetect`.

    Args:
        importer_name: Registered name of the importer (e.g. ``"server_folder"``).
        field_values: Mapping of importer field keys to their CLI values.
        detector_path: Path to a JSON file containing detector weights and threshold.

    Returns:
        A list of dicts for medias predicted as "Good", each containing
        the media's ``id``, ``filename``, ``category``, and ``score``.

    Raises:
        ValueError: If the importer is unknown, required fields are missing,
            the loaded dataset is empty, or the detector file is invalid.
        FileNotFoundError: If the detector file does not exist.
    """
    from vtsearch.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")

    importer.validate_cli_field_values(field_values)

    # Use thin mode — CLI only needs embeddings for scoring, not media bytes
    medias: dict[int, dict[str, Any]] = {}
    importer.run_cli(field_values, medias, thin=True)
    apply_custom_metadata_md5(medias)

    if not medias:
        raise ValueError(f"No medias loaded by importer '{importer_name}'")

    return _score_clips_with_detector(medias, detector_path)


def _list_importer_names() -> list[str]:
    """Return the names of all registered importers."""
    from vtsearch.datasets.importers import list_importers

    return [imp.name for imp in list_importers()]


def _list_exporter_names() -> list[str]:
    """Return the names of all registered exporters."""
    from vtsearch.exporters import list_exporters

    return [exp.name for exp in list_exporters()]


def _score_medias_with_detectors(
    medias: dict[int, dict[str, Any]],
    detectors: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Score all *medias* against every detector in *detectors*.

    Embeddings are extracted once and reused across all detectors.

    Args:
        medias: The medias dict (must contain ``"embedding"`` per media).
        detectors: Mapping of detector name to detector data dict (with
            ``"weights"`` and ``"threshold"`` keys).

    Returns:
        A dict mapping detector name to its results sub-dict (with
        ``"detector_name"``, ``"threshold"``, ``"total_hits"``, ``"hits"``).

    Raises:
        ValueError: If *medias* or *detectors* is empty.
    """
    if not medias:
        raise ValueError("No medias loaded from dataset")
    if not detectors:
        raise ValueError("No autorun processors found for the dataset's media type")

    import torch  # noqa: PLC0415

    all_ids = sorted(medias.keys())
    all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    results: dict[str, dict[str, Any]] = {}
    for detector_name, detector_data in detectors.items():
        weights = detector_data["weights"]
        threshold = detector_data["threshold"]

        from vtsearch.models.training import build_model_from_weights

        model = build_model_from_weights(weights)

        with torch.no_grad():
            scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

        positive_hits: list[dict[str, Any]] = []
        negative_hits: list[dict[str, Any]] = []
        for cid, score in zip(all_ids, scores):
            hit = build_media_hit(cid, medias[cid], score)
            if score >= threshold:
                positive_hits.append(hit)
            else:
                negative_hits.append(hit)

        positive_hits.sort(key=lambda x: x["score"], reverse=True)
        negative_hits.sort(key=lambda x: x["score"], reverse=True)

        results[detector_name] = {
            "detector_name": detector_name,
            "threshold": round(threshold, 4),
            "total_hits": len(positive_hits),
            "hits": positive_hits,
            "negative_hits": negative_hits,
        }

    return results


def _build_multi_results_dict(
    detector_results: dict[str, dict[str, Any]],
    media_type: str = "unknown",
) -> dict[str, Any]:
    """Build the full results dict from multi-detector scoring.

    Args:
        detector_results: Per-detector results from
            :func:`_score_medias_with_detectors`.
        media_type: The media type string for the dataset.

    Returns:
        A dict matching the shape expected by exporters.
    """
    return {
        "media_type": media_type,
        "detectors_run": len(detector_results),
        "results": detector_results,
    }


def _detect_media_type(medias: dict[int, dict[str, Any]]) -> str:
    """Return the media type from the first media, or ``"unknown"``."""
    for media in medias.values():
        return media.get("type", "unknown")
    return "unknown"


def _run_exporter(
    exporter_name: str,
    field_values: dict[str, Any],
    results: dict[str, Any],
) -> None:
    """Validate and run a named exporter, printing its confirmation message.

    Raises:
        ValueError: If the exporter is unknown or required fields are missing.
    """
    from vtsearch.exporters import get_exporter

    exporter = get_exporter(exporter_name)
    if exporter is None:
        available = _list_exporter_names()
        raise ValueError(f"Unknown exporter: {exporter_name}. Available: {', '.join(available)}")

    exporter.validate_cli_field_values(field_values)
    result = exporter.export_cli(results, field_values)
    print(result.get("message", "Export complete."))


def _import_autorun_processors(settings_path: str | None = None) -> None:
    """Import autorun processors from settings (if any).

    When *settings_path* is provided the settings module is pointed at that
    file before importing; otherwise the default ``data/settings.json`` is
    used.

    Errors are logged as warnings but do not halt the autodetect run.
    """
    try:
        if settings_path:
            from vtsearch.settings import set_settings_path

            set_settings_path(settings_path)

        from vtsearch.settings import ensure_autorun_processors_imported

        imported = ensure_autorun_processors_imported()
        if imported:
            print(f"Imported {len(imported)} autorun processor(s) from settings: {', '.join(imported)}")
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"Warning: could not load autorun processors from settings: {exc}", file=sys.stderr)


def _merge_detector_results(
    accumulated: dict[str, dict[str, Any]],
    new_chunk: dict[str, dict[str, Any]],
) -> None:
    """Merge detector results from a new chunk into *accumulated* in-place.

    For each detector, the hit lists are concatenated and re-sorted by
    score descending, and ``total_hits`` is updated.

    Args:
        accumulated: The running results dict (mutated in-place).
        new_chunk: Results from the latest chunk (same shape as
            :func:`_score_medias_with_detectors` output).
    """
    for det_name, det_result in new_chunk.items():
        if det_name not in accumulated:
            accumulated[det_name] = det_result
        else:
            accumulated[det_name]["hits"].extend(det_result["hits"])
            accumulated[det_name]["total_hits"] += det_result["total_hits"]
            accumulated[det_name]["hits"].sort(key=lambda x: x["score"], reverse=True)
            if "negative_hits" in det_result:
                accumulated[det_name].setdefault("negative_hits", []).extend(det_result["negative_hits"])
                accumulated[det_name]["negative_hits"].sort(key=lambda x: x["score"], reverse=True)


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


def _load_pickle_chunked(dataset_path: str, chunk_size: int) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias loaded from a pickle file."""
    from vtsearch.datasets.loader import load_dataset_from_pickle_chunked

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    yield from load_dataset_from_pickle_chunked(dataset_file, chunk_size, thin=True)


def _load_importer_whole(importer_name: str, field_values: dict[str, Any]) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield a single medias dict loaded via a named importer."""
    from vtsearch.datasets.importers import get_importer

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
    from vtsearch.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")

    importer.validate_cli_field_values(field_values)
    yield from importer.run_chunked_cli(field_values, chunk_size, thin=True)


def _run_pipeline(
    media_source: Iterator[dict[int, dict[str, Any]]],
    *,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    empty_error: str = "No medias loaded",
) -> None:
    """Shared pipeline: import processors, iterate media chunks, score, export.

    All four CLI entry points (pickle / importer, whole / chunked) delegate
    to this single function, differing only in the *media_source* iterator
    they supply.

    Args:
        media_source: An iterator that yields one or more ``dict[int, dict]``
            chunks of medias.  A non-chunked source yields exactly one chunk.
        settings_path: Optional path to a settings JSON file.
        exporter_name: Optional registered exporter name.
        exporter_field_values: Optional exporter field values.
        empty_error: Error message used when the source yields no medias.
    """
    _import_autorun_processors(settings_path)

    merged_results: dict[str, dict[str, Any]] = {}
    media_type: str | None = None
    detectors: dict[str, dict[str, Any]] | None = None
    total_medias = 0
    chunk_num = 0

    for chunk_num, chunk_medias in enumerate(media_source, 1):
        if not chunk_medias:
            continue

        apply_custom_metadata_md5(chunk_medias)
        total_medias += len(chunk_medias)

        if media_type is None:
            media_type = _detect_media_type(chunk_medias)

            from vtsearch.utils import get_autodetect_detectors_by_media

            detectors = get_autodetect_detectors_by_media(media_type)
            if not detectors:
                raise ValueError(
                    f"No autorun processors found for media type: {media_type}. "
                    "Add processors to the settings file or use --settings to specify one."
                )

        if chunk_num > 1 or total_medias != len(chunk_medias):
            # Only print chunk progress when there are multiple chunks
            print(f"Processing chunk {chunk_num} ({len(chunk_medias)} medias)...", flush=True)

        chunk_results = _score_medias_with_detectors(chunk_medias, detectors)
        _merge_detector_results(merged_results, chunk_results)

    if not merged_results:
        raise ValueError(empty_error)

    if chunk_num > 1:
        print(f"Finished processing {total_medias} medias across {chunk_num} chunk(s).", flush=True)

    results = _build_multi_results_dict(merged_results, media_type or "unknown")
    _run_exporter(exporter_name or "gui", exporter_field_values or {}, results)


def autodetect_main(
    dataset_path: str,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
) -> None:
    """CLI entry point: run autodetect with all autorun processors and output results.

    Loads autorun processors from the settings file (defaulting to the
    normal ``data/settings.json``), scores the dataset against every
    processor matching the dataset's media type, and exports a combined
    result set with one column per processor.

    When *exporter_name* is ``None`` the hits are printed to stdout.
    Otherwise the named exporter is used to deliver the results.

    Exits with code 0 on success, 1 on error.

    Args:
        dataset_path: Path to the dataset pickle file.
        settings_path: Optional path to a settings JSON file.  When ``None``
            the default ``data/settings.json`` is used.
        exporter_name: Optional registered exporter name.
        exporter_field_values: Optional exporter field values.
    """
    try:
        _run_pipeline(
            _load_pickle_whole(dataset_path),
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            empty_error=f"No medias loaded from dataset: {dataset_path}",
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def autodetect_importer_main(
    importer_name: str,
    field_values: dict[str, Any],
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
) -> None:
    """CLI entry point: run autodetect with a named importer and output results.

    Loads autorun processors from the settings file (defaulting to the
    normal ``data/settings.json``), scores the imported dataset against
    every processor matching the dataset's media type, and exports a
    combined result set with one column per processor.

    When *exporter_name* is ``None`` the hits are printed to stdout.
    Otherwise the named exporter is used to deliver the results.

    Exits with code 0 on success, 1 on error.

    Args:
        importer_name: Registered name of the importer.
        field_values: Mapping of importer field keys to their CLI values.
        settings_path: Optional path to a settings JSON file.  When ``None``
            the default ``data/settings.json`` is used.
        exporter_name: Optional registered exporter name.
        exporter_field_values: Optional exporter field values.
    """
    try:
        _run_pipeline(
            _load_importer_whole(importer_name, field_values),
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            empty_error=f"No medias loaded by importer '{importer_name}'",
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def autodetect_main_chunked(
    dataset_path: str,
    chunk_size: int,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
) -> None:
    """CLI entry point: chunked autodetect on a pickle dataset.

    Identical to :func:`autodetect_main` but processes the dataset in
    chunks of *chunk_size* medias at a time, merging results across
    chunks.  This bounds peak memory usage regardless of dataset size.

    Args:
        dataset_path: Path to the dataset pickle file.
        chunk_size: Maximum number of medias to load at once.
        settings_path: Optional path to a settings JSON file.
        exporter_name: Optional registered exporter name.
        exporter_field_values: Optional exporter field values.
    """
    try:
        _run_pipeline(
            _load_pickle_chunked(dataset_path, chunk_size),
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            empty_error=f"No medias loaded from dataset: {dataset_path}",
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def autodetect_importer_main_chunked(
    importer_name: str,
    field_values: dict[str, Any],
    chunk_size: int,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
) -> None:
    """CLI entry point: chunked autodetect with a named importer.

    Identical to :func:`autodetect_importer_main` but uses the importer's
    :meth:`~DatasetImporter.run_chunked_cli` to load the dataset in chunks
    of *chunk_size* medias, processing each chunk independently and merging
    results.  This bounds peak memory usage regardless of dataset size.

    Args:
        importer_name: Registered name of the importer.
        field_values: Mapping of importer field keys to their CLI values.
        chunk_size: Maximum number of medias to load at once.
        settings_path: Optional path to a settings JSON file.
        exporter_name: Optional registered exporter name.
        exporter_field_values: Optional exporter field values.
    """
    try:
        _run_pipeline(
            _load_importer_chunked(importer_name, field_values, chunk_size),
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            empty_error=f"No medias loaded by importer '{importer_name}'",
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
