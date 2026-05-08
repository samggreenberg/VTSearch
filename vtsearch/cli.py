"""Command-line interface utilities for VTSearch.

The only CLI workflow is autodetect: load a dataset (from pickle or via an
importer), score it against the trainable models flagged for autorun in the
settings file, and export the results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import numpy as np

from vtsearch.datasets.loader import apply_custom_metadata_md5, load_dataset_from_pickle
from vtsearch.utils.hits import build_media_hit


def _list_importer_names() -> list[str]:
    """Return the names of all registered importers."""
    from vtsearch.datasets.importers import list_importers

    return [imp.name for imp in list_importers()]


def _list_exporter_names() -> list[str]:
    """Return the names of all registered exporters."""
    from vtsearch.exporters import list_exporters

    return [exp.name for exp in list_exporters()]


def _load_and_train_trainable_models(
    tm_names: list[str],
    media_type: str,
    snap: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve, re-embed, and train an MLP for each named trainable model.

    For every name in *tm_names* the on-disk JSON is read, the labelset's
    origins are resolved via their importer's ``resolve_file()``, the files
    are embedded with the dataset's embedder, and an MLP is trained from
    the resulting vectors.  Models whose ``media_type`` doesn't match the
    dataset are skipped with a warning.

    Returns a ``{name: {"model": nn.Sequential, "threshold": float}}``
    map.  Raises :class:`ValueError` if a model cannot be trained — for
    example when none of its labels' origin files are resolvable from the
    CLI environment.
    """
    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.models.labelset_training import train_from_labelset
    from vtsearch.models.trainable_model_store import _model_path, _read_model
    from vtsearch.utils.state_core import DetectorContext

    out: dict[str, dict[str, Any]] = {}
    for tm_name in tm_names:
        tm = _read_model(_model_path(tm_name))
        if tm is None:
            raise ValueError(f"Trainable model '{tm_name}' not found in the trainable_models dir.")

        tm_media_type = tm.get("media_type", "") or ""
        if media_type and tm_media_type and tm_media_type != media_type:
            print(
                f"Skipping trainable model '{tm_name}': media_type "
                f"{tm_media_type!r} doesn't match dataset {media_type!r}.",
                flush=True,
            )
            continue

        labelset = LabelSet.from_dict(tm.get("labelset") or {})
        if not labelset.elements:
            raise ValueError(f"Trainable model '{tm_name}' has no labels.")

        det_ctx = DetectorContext(tm_name, media_type=tm_media_type or media_type)
        trained = train_from_labelset(
            det_ctx,
            labelset,
            media_type=tm_media_type or media_type,
            snap=snap,
        )
        if not trained:
            cached = len(det_ctx.label_embeddings)
            total = len(labelset.elements)
            raise ValueError(
                f"Trainable model '{tm_name}': could not train MLP "
                f"(resolved {cached} of {total} label origins, need ≥1 good and ≥1 bad). "
                "The original media may not be reachable from the CLI — for example, "
                "labels collected through the local_folder importer have no resolve_file() path."
            )
        out[tm_name] = {"model": det_ctx.model, "threshold": det_ctx.threshold}
    return out


def _score_medias_with_trainable_models(
    medias: dict[int, dict[str, Any]],
    tm_models: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Score *medias* against pre-trained trainable-model MLPs."""
    if not medias or not tm_models:
        return {}

    import torch  # noqa: PLC0415

    all_ids = sorted(medias.keys())
    all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    results: dict[str, dict[str, Any]] = {}
    for tm_name, info in tm_models.items():
        model = info["model"]
        threshold = info["threshold"]
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

        results[tm_name] = {
            "detector_name": tm_name,
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
    """Build the full results dict from multi-detector scoring."""
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
    """Validate and run a named exporter, printing its confirmation message."""
    from vtsearch.exporters import get_exporter

    exporter = get_exporter(exporter_name)
    if exporter is None:
        available = _list_exporter_names()
        raise ValueError(f"Unknown exporter: {exporter_name}. Available: {', '.join(available)}")

    exporter.validate_cli_field_values(field_values)
    result = exporter.export_cli(results, field_values)
    print(result.get("message", "Export complete."))


def import_labels_into_trainable_model_from_file(
    tm_name: str,
    importer_name: str,
    filepath: str,
) -> tuple[int, int]:
    """Run a label importer against a single file and merge into a trainable model."""
    from vtsearch.datasets.labelset import LabeledElement, LabelSet
    from vtsearch.labels.importers import get_label_importer
    from vtsearch.models.trainable_model_store import _model_path, _read_model, _write_model

    path = _model_path(tm_name)
    data = _read_model(path)
    if data is None:
        raise ValueError(f"Trainable model '{tm_name}' not found.")

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
    _write_model(path, data)
    return applied, skipped


def _merge_detector_results(
    accumulated: dict[str, dict[str, Any]],
    new_chunk: dict[str, dict[str, Any]],
) -> None:
    """Merge detector results from a new chunk into *accumulated* in-place."""
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
    """Shared pipeline: read settings, iterate media chunks, score, export.

    All four CLI entry points (pickle / importer, whole / chunked) delegate
    to this single function, differing only in the *media_source* iterator
    they supply.
    """
    if settings_path:
        from vtsearch.settings import set_settings_path

        set_settings_path(settings_path)

    from vtsearch.settings import get_autorun_trainable_models

    merged_results: dict[str, dict[str, Any]] = {}
    media_type: str | None = None
    tm_models: dict[str, dict[str, Any]] | None = None
    total_medias = 0
    chunk_num = 0

    for chunk_num, chunk_medias in enumerate(media_source, 1):
        if not chunk_medias:
            continue

        apply_custom_metadata_md5(chunk_medias)
        total_medias += len(chunk_medias)

        if media_type is None:
            media_type = _detect_media_type(chunk_medias)

            tm_names = get_autorun_trainable_models()
            if tm_names:
                # Train each trainable model exactly once, using the first
                # chunk as the fast-path snap; subsequent chunks reuse the
                # cached MLPs.
                tm_models = _load_and_train_trainable_models(tm_names, media_type, chunk_medias)
            else:
                tm_models = {}

            if not tm_models:
                raise ValueError(
                    f"No autorun trainable models found for media type: {media_type}. "
                    "Add trainable models to the settings file's autorun_trainable_models list."
                )

        if chunk_num > 1 or total_medias != len(chunk_medias):
            print(f"Processing chunk {chunk_num} ({len(chunk_medias)} medias)...", flush=True)

        if tm_models:
            tm_results = _score_medias_with_trainable_models(chunk_medias, tm_models)
            _merge_detector_results(merged_results, tm_results)

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
    """CLI entry point: run autodetect with all autorun trainable models."""
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
    """CLI entry point: run autodetect with a named importer and output results."""
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
    """CLI entry point: chunked autodetect on a pickle dataset."""
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
    """CLI entry point: chunked autodetect with a named importer."""
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
