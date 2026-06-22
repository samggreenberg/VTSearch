"""Shared constants and helpers for the load-pipeline stages.

This is the leaf module of the stages package: it imports nothing from
:mod:`vtscore.datasets.load_pipeline`, so both the orchestrator and the
individual stage modules can depend on it without forming an import cycle.
"""

from __future__ import annotations

# Maps the status strings emitted by inner functions to step numbers.
# "downloading" covers both download and extraction.
# "loading" covers model loading and pickle loading.
# "embedding" covers per-file embedding.
_STATUS_TO_STEP = {
    "downloading": 1,
    "loading": 2,
    "embedding": 3,
}
_TOTAL_LOAD_STEPS = 4  # download, load model, embed, finalize

# Rough typical wall-clock split across the four load phases, used to pace the
# unified whole-job progress bar (see ProgressTracker.set_step_weights).
# Embedding dominates almost any real dataset; the model load is a roughly
# fixed one-time cost; finalize (dedup + diversity tree + registry) is short.
# These only shape the bar's pacing — the overall ETA self-corrects from the
# real rate — so they need only be in the right ballpark.
#               download  model  embed  finalize
_LOAD_STEP_WEIGHTS = [0.25, 0.15, 0.50, 0.10]


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
