"""Detector input-spec helpers.

A detector's ``input_spec`` describes the input format it was trained on
- specifically the clipper that split source media into the clips its
MLP saw.  At inference time the CLI uses this to decide whether the
loaded dataset matches the detector's expected granularity.

The module also assembles the ``detector_meta`` block that travels with
a labelset when synced through a :class:`LabelsetSource`, so a downstream
consumer can reproduce the detector's input format and decision boundary
without reading the detector JSON directly.

``input_spec`` and ``detector_meta`` are both **optional**: detectors
that never set ``input_spec`` behave exactly as before, and labelsets
exported without a ``detector_meta`` block are still valid.
"""

from __future__ import annotations

from typing import Any


def extract_input_spec_from_medias(
    medias: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the clipper config stamped onto *medias* by the loader.

    The dataset-load pipeline (:func:`vtscore.datasets.load_pipeline._apply_clipper`)
    writes the active clipper's name and effective parameter values into
    every clip's ``origin.params`` (as ``clipper`` plus ``clipper_<key>``
    entries).  This function reads those back from the first non-empty
    origin so the calling detector can record the input format it was
    trained on.

    Returns ``None`` when no clipper info is present - including the case
    where the dataset was loaded without any clipper, or with a ``*_default``
    clipper (which is a no-op pass-through and not worth persisting).
    """
    for media in medias.values():
        origin = media.get("origin")
        if not isinstance(origin, dict):
            continue
        params = origin.get("params")
        if not isinstance(params, dict):
            continue
        clipper_name = params.get("clipper", "")
        if not clipper_name or clipper_name.endswith("_default"):
            # A media with no clipper (or the default pass-through) is
            # uninformative - keep scanning in case a clipped media
            # appears later in iteration order.
            continue
        clipper_params: dict[str, str] = {}
        for key, value in params.items():
            if not isinstance(key, str) or not key.startswith("clipper_"):
                continue
            # Skip the bare ``clipper`` key itself (handled above) and
            # per-clip boundary fields that are not clipper parameters.
            inner = key[len("clipper_") :]
            if inner in ("", "start", "end", "box", "index"):
                continue
            clipper_params[inner] = str(value)
        spec: dict[str, Any] = {"clipper": clipper_name}
        if clipper_params:
            spec["clipper_params"] = clipper_params
        return spec
    return None


def build_detector_meta(
    detector_data: dict[str, Any],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Assemble a ``detector_meta`` block for labelset export.

    Pulls ``media_type`` and ``input_spec`` straight from the detector
    JSON dict and folds in the supplied *threshold* (typically the active
    in-memory MLP's calibrated threshold).  Keys with empty/None values
    are omitted so the block stays minimal - a detector with neither an
    ``input_spec`` nor a current threshold produces a block containing
    only ``media_type``.
    """
    meta: dict[str, Any] = {}
    media_type = detector_data.get("media_type") or ""
    if media_type:
        meta["media_type"] = media_type
    spec = detector_data.get("input_spec")
    if isinstance(spec, dict) and spec:
        meta["input_spec"] = dict(spec)
    if threshold is not None:
        meta["threshold"] = float(threshold)
    return meta


def apply_detector_meta(
    detector_data: dict[str, Any],
    detector_meta: dict[str, Any] | None,
) -> bool:
    """Apply an inbound ``detector_meta`` block to a detector JSON dict.

    Writes ``input_spec`` (and ``media_type`` when the receiver doesn't
    already have one) so the receiving detector picks up the originating
    detector's expected input format.  Does **not** persist ``threshold``
    - the receiver retrains its own MLP from the imported labels, so the
    threshold is rederived on the next load.

    Returns ``True`` when *detector_data* was modified.
    """
    if not detector_meta:
        return False
    changed = False
    spec = detector_meta.get("input_spec")
    if isinstance(spec, dict) and spec:
        existing = detector_data.get("input_spec")
        if existing != spec:
            detector_data["input_spec"] = dict(spec)
            changed = True
    incoming_type = detector_meta.get("media_type") or ""
    if incoming_type and not detector_data.get("media_type"):
        detector_data["media_type"] = incoming_type
        changed = True
    return changed


def clipper_matches(
    detector_spec: dict[str, Any] | None,
    dataset_spec: dict[str, Any] | None,
) -> bool:
    """Return True when *detector_spec* and *dataset_spec* describe the same clipper.

    Both arguments use the format returned by
    :func:`extract_input_spec_from_medias` (``{"clipper": ..., "clipper_params": {...}}``)
    and accept ``None`` to mean "no clipper / default clipper".  A detector
    without an ``input_spec`` accepts any dataset, since the legacy
    behaviour is to score whatever embeddings the dataset already
    contains.
    """
    if not detector_spec:
        return True
    detector_clipper = detector_spec.get("clipper") or ""
    dataset_clipper = (dataset_spec or {}).get("clipper") or ""
    if detector_clipper != dataset_clipper:
        return False
    detector_params = detector_spec.get("clipper_params") or {}
    dataset_params = (dataset_spec or {}).get("clipper_params") or {}
    # Compare as string maps - load_pipeline stores effective values as
    # strings, so a detector saved from a sound_tiling(duration=2.0) load
    # records ``{"duration": "2.0"}`` and we want a downstream dataset
    # loaded with the same params to match.
    return {str(k): str(v) for k, v in detector_params.items()} == {str(k): str(v) for k, v in dataset_params.items()}
