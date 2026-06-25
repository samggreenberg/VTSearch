"""Shared constants and helpers for the load-pipeline stages.

This is the leaf module of the stages package: it imports nothing from
:mod:`vtscore.datasets.load_pipeline`, so both the orchestrator and the
individual stage modules can depend on it without forming an import cycle.
"""

from __future__ import annotations

# Maps the status strings emitted by inner functions to step numbers.
# "downloading" covers both download and extraction.
# "loading"/"converting" cover model loading, pickle loading, and source→media
# conversion (document→image, video→frames): all pre-embed work that produces
# the medias to embed, so they share the loading slice.
# "embedding" covers per-file embedding (and clip+embed for clipped datasets;
# see clipper.py — clipping IS the embed phase there, so it reports this step).
#
# Every status that can fire during a load MUST appear here. A status missing
# from this map resolves to ``step=None``, which nulls the whole-job ``overall``
# fraction for that update and makes the bar fall back to the raw within-phase
# ``current``/``total`` — a different scale that visibly knocks the unified bar
# off its track. Keep the map exhaustive instead.
_STATUS_TO_STEP = {
    "downloading": 1,
    "loading": 2,
    "converting": 2,
    "embedding": 3,
}
_TOTAL_LOAD_STEPS = 4  # download, load model, embed, finalize

# Rough typical wall-clock split across the four load phases, used to pace the
# unified whole-job progress bar (see ProgressTracker.set_step_weights).
# Embedding dominates almost any real dataset; the model load is a roughly
# fixed one-time cost; finalize (dedup + diversity tree + registry) is short.
#
# The model-load slice is kept deliberately small: it is the one phase that
# cannot report fine-grained progress, so the bar sits at its floor for the
# whole load and then fills the slice in one step the moment embedding starts.
# A smaller weight there means a smaller between-stage jump; the freed weight
# goes to embedding, the phase that *does* report per-item progress and so
# advances the bar smoothly. These weights only shape pacing — the overall ETA
# self-corrects from the real rate — so they need only be in the right ballpark.
#               download  model  embed  finalize
_LOAD_STEP_WEIGHTS = [0.25, 0.10, 0.55, 0.10]


def _origin_to_str(origin: dict | None) -> str:
    """Convert an origin dict to a human-readable string."""
    if not origin:
        return "unknown"
    importer_name = origin.get("importer", "")
    if not importer_name:
        return "unknown"

    from vtscore.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is not None:
        return importer.origin_display(origin)

    params = origin.get("params", {})
    if params:
        first_val = next(iter(params.values()))
        return f"{importer_name}:{first_val}"
    return importer_name
