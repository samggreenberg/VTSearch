"""Shared constants and helpers for the load-pipeline stages.

This is the leaf module of the stages package: it imports nothing from
:mod:`vtscore.datasets.load_pipeline`, so both the orchestrator and the
individual stage modules can depend on it without forming an import cycle.
"""

from __future__ import annotations

import time
from typing import Optional

# Maps the status strings emitted by inner functions to step numbers.
# "downloading" and "extracting" are the two sub-phases of the acquire step:
# they share step 1 (the bar shows one unified acquire slice) but report on
# different scales (transfer bytes vs archive members), so the pacer maps each
# into its own ordered sub-range of the slice — see AdaptiveLoadPacer.
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
    "extracting": 1,
    "loading": 2,
    "converting": 2,
    "embedding": 3,
}
_TOTAL_LOAD_STEPS = 4  # download, load model, embed, finalize

# Rough typical wall-clock split across the four load phases, used to pace the
# unified whole-job progress bar (see ProgressTracker.set_step_weights).
# Embedding dominates almost any real dataset on a CPU host; the model load is a
# roughly fixed one-time cost; finalize (dedup + coverage atlas + registry) is
# short *relative to a slow CPU embed*.
#
# The model-load slice is kept deliberately small: it is the one phase that
# cannot report fine-grained progress, so the bar sits at its floor for the
# whole load and then fills the slice in one step the moment embedding starts.
# A smaller weight there means a smaller between-stage jump; the freed weight
# goes to embedding, the phase that *does* report per-item progress and so
# advances the bar smoothly. These weights only shape pacing — the overall ETA
# self-corrects from the real rate — so they need only be in the right ballpark.
#                   download  model  embed  finalize
_LOAD_STEP_WEIGHTS_CPU = [0.25, 0.10, 0.55, 0.10]

# Image CPU profile. The generic CPU profile above assumes embedding dominates,
# which holds for audio/video where each item is *seconds* of decode + a long
# encoder pass. An image embed is a single ViT forward over one frame — far
# cheaper per item — so on a CPU image import the embed phase no longer
# dominates: archive download/extraction (many image demo sources are tens of MB
# pulled over the network) and the un-accelerated finalize (dedup + coverage-
# k-means + registry serialize/zip/write) take a proportionally larger share.
# With the generic 55%-embed profile the bar raced through embedding and then
# crawled download/finalize. Shifting weight off embed and onto download +
# finalize paces the image-on-CPU bar against its real phase split. (On a GPU
# host the embed phase is fast for every media type, so the device profile below
# already covers images — this override is CPU-only.)
#                         download  model  embed  finalize
_LOAD_STEP_WEIGHTS_CPU_IMAGE = [0.35, 0.10, 0.35, 0.20]

# GPU profile. When embedding runs on a CUDA device it is several times faster,
# so the embed phase shrinks while the *finalize* phase does not: the registry
# save (serialize → zip → disk write) is never GPU-accelerated, and the
# atlas k-means only moves to the GPU when cuML is installed (a
# best-effort RAPIDS dependency that is frequently absent — see
# ``vtscore/gpu_backends.py``), so it usually still runs on CPU ``sklearn``.
# The net effect on a GPU host is that finalize dominates wall-clock instead of
# embedding, so it gets a much larger slice of the bar. Without this the bar
# raced to ~90% during the fast embed phase and then crawled through the last
# 10% for many seconds while reporting "< 5 sec left" (the rate-based ETA was
# dominated by the fast embed phase). See ``load_step_weights``.
#                   download  model  embed  finalize
_LOAD_STEP_WEIGHTS_GPU = [0.20, 0.10, 0.30, 0.40]

# Back-compat / CPU default alias. The detector-load weights and the progress
# tests reference this; the dataset loader picks the right profile at task
# creation via :func:`load_step_weights`.
_LOAD_STEP_WEIGHTS = _LOAD_STEP_WEIGHTS_CPU


def _resolve_device_name() -> str:
    """Return "cuda"/"cpu" for the active device, defaulting to "cpu" when torch
    can't be resolved (library-only callers must never crash here)."""
    try:
        from vtscore.config import resolve_device  # noqa: PLC0415

        return "cuda" if resolve_device().startswith("cuda") else "cpu"
    except Exception:
        return "cpu"


def _default_embedder_name(media_type: str) -> str:
    """Name of the default embedder for *media_type* (empty if none/unknown)."""
    try:
        from vtscore.media import embedders_for_type  # noqa: PLC0415

        embs = embedders_for_type(media_type)
        return embs[0].name if embs else ""
    except Exception:
        return ""


def load_step_weights(
    media_type: str = "",
    *,
    n: Optional[int] = None,
    download_size_mb: Optional[float] = None,
    embedder: Optional[str] = None,
) -> list[float]:
    """Return the dataset-load step weights for the active device and media type.

    When the item count *n* is known (demo datasets know it up front) and a
    measured affine cost-model row exists for (device, media_type, embedder), the
    weights are computed from that model (see ``_load_cost_model``), so they adapt
    to dataset size — model-load weighs heavier for small ``n``, embed dominates
    for large ``n``. Otherwise it returns the static per-(device, media_type)
    profile below — the large-``n`` asymptote — so callers without a known ``n``
    (folder importers that stream) keep today's behaviour unchanged.

    Static fallback rationale: GPU hosts embed several times faster, so the
    un-accelerated finalize phase earns a larger slice; on a CPU host an image
    embed is a single cheap ViT forward per item, so ``image`` shifts weight off
    embed onto download + finalize. Falls back to the CPU profile whenever the
    device can't be resolved, so a library-only caller never crashes here.
    """
    device = _resolve_device_name()

    if n is not None and n > 0:
        from vtscore.datasets.stages._load_cost_model import cost_model_weights  # noqa: PLC0415

        emb = embedder or _default_embedder_name(media_type)
        weights = cost_model_weights(device, media_type, emb, n, download_size_mb)
        if weights is not None:
            return weights

    if device == "cuda":
        return list(_LOAD_STEP_WEIGHTS_GPU)
    if media_type == "image":
        return list(_LOAD_STEP_WEIGHTS_CPU_IMAGE)
    return list(_LOAD_STEP_WEIGHTS_CPU)


#: Share of a static profile's acquire slice assigned to the download sub-phase
#: when the archive size is unknown (the rest is extraction). Only shapes the
#: initial pacing; the AdaptiveLoadPacer rebases from observed durations.
_STATIC_DOWNLOAD_SHARE = 0.75


def load_cost_terms(
    media_type: str = "",
    *,
    n: Optional[int] = None,
    download_size_mb: Optional[float] = None,
    embedder: Optional[str] = None,
) -> dict[str, float]:
    """Predicted per-phase cost terms for a dataset load, in phase order
    ``download``, ``extract``, ``load``, ``embed``, ``finalize``.

    When ``n`` is known and a measured cost-model row exists, the terms are
    seconds from the affine model. Otherwise they are pseudo-times derived from
    the static per-(device, media) weight profile — only their ratios matter,
    and the acquire slice is split between download and extraction by archive
    size when known, else by :data:`_STATIC_DOWNLOAD_SHARE`.

    Always returns a usable dict (never raises), so the pacing layer can rely
    on it for every import path, including streaming folder importers.
    """
    device = _resolve_device_name()

    if n is not None and n > 0:
        from vtscore.datasets.stages._load_cost_model import cost_model_terms  # noqa: PLC0415

        emb = embedder or _default_embedder_name(media_type)
        terms = cost_model_terms(device, media_type, emb, n, download_size_mb)
        if terms is not None:
            return terms

    if device == "cuda":
        w = _LOAD_STEP_WEIGHTS_GPU
    elif media_type == "image":
        w = _LOAD_STEP_WEIGHTS_CPU_IMAGE
    else:
        w = _LOAD_STEP_WEIGHTS_CPU

    dl_share = _STATIC_DOWNLOAD_SHARE
    if download_size_mb:
        from vtscore.datasets.stages._load_cost_model import DOWNLOAD_MB_PER_S, EXTRACT_MB_PER_S  # noqa: PLC0415

        if DOWNLOAD_MB_PER_S > 0 and EXTRACT_MB_PER_S > 0:
            t_dl = download_size_mb / DOWNLOAD_MB_PER_S
            t_ex = download_size_mb / EXTRACT_MB_PER_S
            dl_share = t_dl / (t_dl + t_ex)
    return {
        "download": w[0] * dl_share,
        "extract": w[0] * (1.0 - dl_share),
        "load": w[1],
        "embed": w[2],
        "finalize": w[3],
    }


class FinalizeProgress:
    """Tracker proxy that maps each finalize sub-stage into an ordered slice
    of step 4 (the finalize phase).

    Step 4 bundles several sub-stages — drop-failed-embeds, dedup, the
    coverage-atlas build, the registry save, and the optional Browse
    projection. Each of them used to report its *own* ``current``/``total``
    against the single step-4 slice, so whichever finished first (usually
    dedup) drove the within-step fraction to ``1.0`` and pinned the unified
    bar at 100% while the slower sub-stages (serialize + zip + disk write,
    coverage k-means) were still grinding — the bar looked done, the ETA
    froze at its last embed-phase estimate, and the user sat there.

    This proxy fixes that by assigning each sub-stage an ordered, non-
    overlapping sub-range of the finalize phase. The sub-stage code is
    unchanged: it still calls ``tracker.update(..., step=_TOTAL_LOAD_STEPS)``
    and ``tracker.check_cancelled()``. The load pipeline wraps the real
    tracker in one of these for the finalize block and calls :meth:`begin`
    before each sub-stage; the proxy rewrites that stage's within-step
    fraction into the active slot before forwarding, so the bar advances once,
    monotonically, across the whole phase and the overall ETA self-corrects.

    A skipped sub-stage (opt-in near-dup merge, deferred coverage atlas above
    the size threshold, opt-out projection) simply leaves its slice unfilled;
    the next :meth:`begin` jumps the bar forward, which is monotonic and fine.
    """

    #: Resolution of the synthetic within-step counter forwarded to the real
    #: tracker. The finalize fraction (0..1) is reported as ``current`` out of
    #: this ``total`` so the tracker's overall math sees a normal sub-step.
    _SCALE = 1000

    #: Ordered finalize sub-stages with a rough wall-clock share each. The
    #: registry save (serialize + zip + write) dominates, with the coverage
    #: tree second; the rest are quick. Shares need only be in the right
    #: ballpark — they shape pacing within the finalize slice, and the overall
    #: ETA self-corrects from the real rate (see ProgressTracker._compute_overall).
    _SLOTS: tuple[tuple[str, float], ...] = (
        ("cleanup", 0.05),
        ("dedup", 0.15),
        ("coverage", 0.30),
        ("signpost_texts", 0.05),
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
        # Stamp the sub-slot boundary for the env-gated load profiler (no-op when
        # profiling is off). See docs/plans/progress-weight-calibration.md.
        from vtscore.datasets.stages._load_profiler import note_finalize_slot  # noqa: PLC0415

        note_finalize_slot(slot)

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


#: Canonical phase order for a dataset load. ``download`` and ``extract`` are
#: the two sub-phases of step 1 (acquire); the rest map 1:1 onto steps 2–4.
_PHASE_ORDER: tuple[str, ...] = ("download", "extract", "load", "embed", "finalize")
_PHASE_STEP: dict[str, int] = {"download": 1, "extract": 1, "load": 2, "embed": 3, "finalize": 4}


class AdaptiveLoadPacer:
    """Tracker proxy that paces the unified load bar from a per-phase cost model
    and rebases it on what actually happens.

    The static/n-aware step weights fix two of the bar's failure modes but two
    remained (issue #2556):

    - **Frozen bar during extraction.** Download and extraction both reported
      status ``"downloading"`` against step 1 on different scales (transfer
      bytes vs archive members), so once the download filled the acquire slice
      the monotonic clamp pinned the bar for the whole extraction while the
      ETA climbed. Extraction now reports ``"extracting"``, and this proxy maps
      each sub-phase into its own ordered sub-range of the step-1 slice.
    - **Weights that predict work that never happens (or lies).** A cached
      archive skips download+extraction entirely, a cached embedded pkl skips
      nearly everything, and the real network can be several times faster or
      slower than the calibrated bandwidth. Static weights then make the bar
      leap or crawl and the ETA start absurdly low or high. This proxy
      re-estimates the *current phase's* term from its observed pace while
      counts flow (a 2026-07-18 GTZAN run showed why download-only observation
      is not enough: decode ran 11× its calibrated term, the bar span stayed
      tiny, and the tracker's rate-extrapolated ETA ballooned to ~55 min for a
      3-min load), and at every phase boundary re-divides the *remaining* span
      of the bar over the phases still to come (proportional to their model
      terms). What has been consumed stays consumed — the mapping is monotonic
      by construction — but the future is always paced against the latest
      evidence.

    The proxy exposes the tracker surface the pipeline stages use (``update`` /
    ``check_cancelled``), accepts the same ``step``/``total_steps`` kwargs, and
    forwards to the real tracker with re-derived step weights and, for step-1
    sub-phases, a composite within-step fraction (via the tracker's ``within``
    override). The caller's ``current``/``total`` always pass through, so byte
    and item counts stay visible in the UI; pacing is controlled purely through
    the weight vector and the override.

    Known limitation: a multi-archive source that downloads several archives
    back-to-back under one ``"downloading"`` phase still freezes within that
    phase after the first archive (the fractions restart on a new scale); the
    alternating download→extract→download pattern the demos actually use is
    handled by the re-entry rebase.
    """

    #: Observe a phase at least this long / this far before trusting its
    #: projected duration over the calibrated prior.
    _RATE_MIN_ELAPSED = 2.0
    _RATE_MIN_FRACTION = 0.02
    #: Re-derive the weights from the observed rate at most this often.
    _RATE_REWEIGHT_INTERVAL = 1.0

    def __init__(self, tracker, terms: dict[str, float]) -> None:
        self._tracker = tracker
        self._terms = {p: max(0.0, float(terms.get(p, 0.0))) for p in _PHASE_ORDER}
        self._phase: str | None = None
        self._phase_t0 = 0.0
        self._frac = 0.0  # monotone within-phase fraction
        self._consumed = 0.0  # overall fraction consumed when this phase began
        self._span = 0.0  # overall span assigned to the current phase
        self._raw = 0.0  # last overall fraction targeted (monotone)
        self._last_rate_reweight = 0.0
        self._weights = self._weights_vector()
        tracker.set_step_weights(self._weights)

    # -- tracker surface ----------------------------------------------------
    def check_cancelled(self) -> None:
        self._tracker.check_cancelled()

    def update(self, status: str, message: str = "", current: int = 0, total: int = 0, **kwargs) -> None:
        step = kwargs.pop("step", None)
        kwargs.pop("total_steps", None)
        phase = self._resolve_phase(status, step)
        if phase is None:
            # Terminal/idle (or unmapped) updates pass through untouched.
            self._tracker.update(status, message, current, total, step=step, total_steps=None, **kwargs)
            return

        now = time.monotonic()
        if phase != self._phase:
            self._begin_phase(phase, now)
        if total and total > 0:
            self._frac = min(max(self._frac, current / total), 1.0)
        self._observe_phase_rate(now)
        self._raw = max(self._raw, self._consumed + self._span * self._frac)

        s = _PHASE_STEP[phase]
        if s == 1:
            # Composite within-step fraction against step 1's actual slice
            # (consumed so far + every step-1 share still pending), so the
            # tracker's weight mapping lands exactly on the targeted fraction.
            # Passed via the ``within`` override so the displayed
            # ``current``/``total`` keep the caller's real byte/member counts.
            w1 = self._weights[0]
            kwargs["within"] = self._raw / w1 if w1 > 0 else 1.0
        self._tracker.update(status, message, current, total, step=s, total_steps=_TOTAL_LOAD_STEPS, **kwargs)

    # -- pacing internals ---------------------------------------------------
    @staticmethod
    def _resolve_phase(status: str, step) -> str | None:
        if step == 4:
            return "finalize"  # FinalizeProgress stamps step 4 explicitly
        if status == "extracting":
            return "extract"
        if step == 1:
            return "download"
        if step == 2:
            return "load"
        if step == 3:
            return "embed"
        return None

    def _begin_phase(self, phase: str, now: float) -> None:
        """Rebase the remaining bar span over *phase* and the phases after it.

        Whatever fraction the bar has actually consumed stays consumed; the
        remainder is divided over the current-and-later model terms. Re-entering
        an earlier phase (multi-archive download after an extraction) is the
        same operation — the phase gets a fresh, proportionally smaller slice
        of what is left, keeping the bar monotone without archive-count
        knowledge.
        """
        self._consumed = self._raw
        self._phase = phase
        self._phase_t0 = now
        self._frac = 0.0
        self._span = (1.0 - self._consumed) * self._phase_share(phase)
        self._weights = self._weights_vector()
        self._tracker.set_step_weights(self._weights)

    def _phase_share(self, phase: str) -> float:
        idx = _PHASE_ORDER.index(phase)
        rem = [self._terms[p] for p in _PHASE_ORDER[idx:]]
        total = sum(rem)
        return rem[0] / total if total > 0 else 1.0 / len(rem)

    def _observe_phase_rate(self, now: float) -> None:
        """Replace the current phase's calibrated term with one projected from
        its observed pace (``elapsed / fraction-done``), and re-derive the
        weights.

        A phase running *slower* than its term grows its span, so the bar
        target jumps forward to where reality says it should be and the
        rate-extrapolated overall ETA converges instead of ballooning. A phase
        running *faster* shrinks its span, which can momentarily target a
        smaller overall fraction than already shown; the tracker's monotonic
        clamp holds the bar flat until the target catches up, which reads as a
        brief pause rather than a rewind.
        """
        phase = self._phase
        if phase is None or self._frac >= 1.0:
            # A completed phase needs no re-projection — the next phase
            # boundary rebases anyway, and shrinking a finished phase's span
            # would only clamp the bar below the span it just earned.
            return
        elapsed = now - self._phase_t0
        if elapsed < self._RATE_MIN_ELAPSED or self._frac < self._RATE_MIN_FRACTION:
            return
        if now - self._last_rate_reweight < self._RATE_REWEIGHT_INTERVAL:
            return
        self._last_rate_reweight = now
        self._terms[phase] = elapsed / self._frac
        self._span = (1.0 - self._consumed) * self._phase_share(phase)
        self._weights = self._weights_vector()
        self._tracker.set_step_weights(self._weights)

    def _weights_vector(self) -> list[float]:
        """Express the current pacing state as a 4-step weight vector.

        The vector always sums to 1: everything consumed before the current
        step, the current phase's span, then the remaining phases' shares —
        so the tracker's ``(Σ w[:s-1] + w[s-1]·within)/Σ w`` mapping lands
        exactly on ``consumed + span·frac``.
        """
        w = [0.0] * _TOTAL_LOAD_STEPS
        phase = self._phase if self._phase is not None else "download"
        idx = _PHASE_ORDER.index(phase)
        rem_phases = _PHASE_ORDER[idx:]
        rem_terms = [self._terms[p] for p in rem_phases]
        total_rem = sum(rem_terms)
        remaining = 1.0 - self._consumed
        if total_rem > 0:
            shares = {p: remaining * t / total_rem for p, t in zip(rem_phases, rem_terms)}
        else:
            shares = {p: remaining / len(rem_phases) for p in rem_phases}
        for p, share in shares.items():
            w[_PHASE_STEP[p] - 1] += share
        # Fold the consumed share into the first step. While step 1 is active
        # its synthetic within-step fraction is computed against this same
        # slice (see update); for later steps the tracker only sums the
        # weights *before* the current step, so w[0] is as good a home as any.
        w[0] += self._consumed
        return w


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
