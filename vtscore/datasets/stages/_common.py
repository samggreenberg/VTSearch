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


class FinalizeProgress:
    """Tracker proxy that maps each finalize sub-stage into an ordered slice
    of step 4 (the finalize phase).

    Step 4 bundles several sub-stages — drop-failed-embeds, dedup, the
    diversity-tree build, the registry save, and the optional Browse
    projection. Each of them used to report its *own* ``current``/``total``
    against the single step-4 slice, so whichever finished first (usually
    dedup) drove the within-step fraction to ``1.0`` and pinned the unified
    bar at 100% while the slower sub-stages (serialize + zip + disk write,
    diversity k-means) were still grinding — the bar looked done, the ETA
    froze at its last embed-phase estimate, and the user sat there.

    This proxy fixes that by assigning each sub-stage an ordered, non-
    overlapping sub-range of the finalize phase. The sub-stage code is
    unchanged: it still calls ``tracker.update(..., step=_TOTAL_LOAD_STEPS)``
    and ``tracker.check_cancelled()``. The load pipeline wraps the real
    tracker in one of these for the finalize block and calls :meth:`begin`
    before each sub-stage; the proxy rewrites that stage's within-step
    fraction into the active slot before forwarding, so the bar advances once,
    monotonically, across the whole phase and the overall ETA self-corrects.

    A skipped sub-stage (opt-in near-dup merge, deferred diversity tree above
    the size threshold, opt-out projection) simply leaves its slice unfilled;
    the next :meth:`begin` jumps the bar forward, which is monotonic and fine.
    """

    #: Resolution of the synthetic within-step counter forwarded to the real
    #: tracker. The finalize fraction (0..1) is reported as ``current`` out of
    #: this ``total`` so the tracker's overall math sees a normal sub-step.
    _SCALE = 1000

    #: Ordered finalize sub-stages with a rough wall-clock share each. The
    #: registry save (serialize + zip + write) dominates, with the diversity
    #: tree second; the rest are quick. Shares need only be in the right
    #: ballpark — they shape pacing within the finalize slice, and the overall
    #: ETA self-corrects from the real rate (see ProgressTracker._compute_overall).
    _SLOTS: tuple[tuple[str, float], ...] = (
        ("cleanup", 0.05),
        ("dedup", 0.15),
        ("diversity", 0.30),
        ("registry", 0.45),
        ("projection", 0.05),
    )

    def __init__(self, tracker) -> None:
        self._tracker = tracker
        total = sum(weight for _, weight in self._SLOTS) or 1.0
        self._ranges: dict[str, tuple[float, float]] = {}
        acc = 0.0
        for name, weight in self._SLOTS:
            self._ranges[name] = (acc / total, weight / total)
            acc += weight
        # Default to the first slot so an update before any begin() still maps
        # somewhere sane rather than raising.
        self._base, self._span = self._ranges[self._SLOTS[0][0]]

    def begin(self, slot: str) -> None:
        """Activate *slot*; subsequent :meth:`update` calls map into its range."""
        self._base, self._span = self._ranges[slot]

    def check_cancelled(self) -> None:
        self._tracker.check_cancelled()

    def update(self, status: str, message: str = "", current: int = 0, total: int = 0, **kwargs) -> None:
        within = current / total if total and total > 0 else 0.0
        within = min(max(within, 0.0), 1.0)
        frac = self._base + self._span * within
        # The active slot already encodes the finalize step; the sub-stage's own
        # step/total_steps are redundant, so drop them and stamp step 4 here.
        kwargs.pop("step", None)
        kwargs.pop("total_steps", None)
        self._tracker.update(
            status,
            message,
            current=int(frac * self._SCALE),
            total=self._SCALE,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
            **kwargs,
        )


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
