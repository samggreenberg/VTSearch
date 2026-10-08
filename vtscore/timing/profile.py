"""Turn the shipped per-step timing defaults into progress-bar weights.

Every long-running task paces its unified bar with a per-step weight vector
(see :meth:`vtscore.concurrency.progress.ProgressTracker.set_step_weights`).
:func:`step_weights` builds that vector from the default terms each task
declares in :mod:`vtscore.timing.tasks`.

**The per-environment profile is retired** (#4667). This module used to read an
admin-measured profile JSON named by ``VTSEARCH_TIMING_PROFILE`` (written by a
tuning script that drove each task family on the serving hardware) and let its
per-``(device, media_type, embedder)`` coefficients override the defaults. Its
main job was steadying the remaining-time estimate on dataset imports, and that
estimate is gone: an import is set by the network, the source's disks and the
files themselves, so no table predicted it well enough to show. The profile,
its reader, the tuning script and the fitter went with it; every bar now paces
from the shipped defaults.

The public names that only made sense with a profile — :class:`TimingProfile`,
:data:`EMPTY_PROFILE`, :func:`parse_profile`, :func:`active_profile`,
:func:`reload_profile`, :func:`profile_covers`, :func:`slot_shares`, and the
``PROFILE_ENV_VAR`` / ``SCHEMA_*`` constants — are kept as deprecated no-ops
that always answer "no profile", so out-of-tree callers keep importing them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from vtscore.timing.tasks import TASKS, task_spec

#: Deprecated (#4667): nothing reads this environment variable any more.
PROFILE_ENV_VAR = "VTSEARCH_TIMING_PROFILE"

#: Deprecated (#4667): the schema marker of the retired profile JSON.
SCHEMA_NAME = "vtsearch-timing-profile"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def normalize_device(device: str) -> str:
    """Collapse a resolved device string to the coarse timing device key.

    ``resolve_device()`` returns things like ``"cuda:0"``, ``"cuda"``, ``"cpu"``,
    or ``"mps"``; timing only distinguishes "has a GPU doing the tensor work"
    from "does not", because that is the split that moves phase costs by an
    order of magnitude.
    """
    return "cuda" if device.startswith("cuda") else "cpu"


def resolve_device_name() -> str:
    """Return ``"cuda"``/``"cpu"`` for the active device.

    Defaults to ``"cpu"`` when torch can't be resolved, so a library-only caller
    (or a machine mid-driver-upgrade) never crashes inside progress pacing.
    """
    try:
        from vtscore.config import resolve_device  # noqa: PLC0415

        return normalize_device(resolve_device())
    except Exception:
        return "cpu"


def cuml_active() -> bool:
    """Whether cuML will serve this process's clustering (never raises).

    cuML moves the coverage-atlas k-means and the UMAP projection onto the GPU,
    which materially changes the cost of any step that clusters — enough that
    CUDA cells are measured as two variants, ``"cuda+cuml"`` and ``"cuda"``.
    """
    try:
        from vtscore.gpu_backends import cuml_enabled  # noqa: PLC0415

        return cuml_enabled()
    except Exception:
        return False


def device_candidates(device: Optional[str] = None) -> tuple[str, ...]:
    """Device keys to try, best match first.

    A CPU host has exactly one key. A CUDA host has two — with and without cuML
    — and the live cuML state decides which is tried first; the other still
    beats falling back to a generic row, because a same-device measurement with
    a different clustering backend is much closer than no measurement at all.
    """
    dev = normalize_device(device) if device else resolve_device_name()
    if dev != "cuda":
        return (dev,)
    return ("cuda+cuml", "cuda") if cuml_active() else ("cuda", "cuda+cuml")


def cell_keys(device: Optional[str], media_type: str = "", embedder: str = "") -> tuple[str, ...]:
    """Cell keys to look up, most specific first.

    Specificity of ``(media_type, embedder)`` outranks the CUDA cuML variant: a
    row measured for exactly this media type and encoder on the other clustering
    backend predicts far better than a media-agnostic row on the right one.
    """
    devices = device_candidates(device)
    specificities = ((media_type, embedder), (media_type, ""), ("", ""))
    keys: list[str] = []
    for media, emb in specificities:
        for dev in devices:
            key = f"{dev}|{media}|{emb}"
            if key not in keys:
                keys.append(key)
    return tuple(keys)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepCoeffs:
    """Affine cost of one step: ``a + b · n + per_mb · archive_mb`` seconds."""

    a: float = 0.0
    b: float = 0.0
    per_mb: float = 0.0
    #: Goodness of the affine fit these coefficients came from, or NaN when the
    #: step was not fitted that way (#3329).
    r2: float = float("nan")

    def seconds(self, n: float = 0.0, size_mb: float = 0.0) -> float:
        """Predicted wall-clock seconds for this step, never negative.

        A least-squares fit over noisy timings can land a negative intercept
        (a steep slope overshooting at small ``n``); clamping here keeps a
        pathological row from handing a step a negative slice of the bar.
        """
        return max(0.0, self.a + self.b * max(0.0, n) + self.per_mb * max(0.0, size_mb))

    @classmethod
    def from_json(cls, raw: Any) -> Optional["StepCoeffs"]:
        """Parse one step's coefficients, or ``None`` if unusable.

        A bare number is accepted as shorthand for a fixed cost.
        """
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return cls(a=float(raw))
        if not isinstance(raw, dict):
            return None
        try:
            return cls(
                r2=float(raw.get("r2", float("nan"))),
                a=float(raw.get("a", 0.0)),
                b=float(raw.get("b", 0.0)),
                per_mb=float(raw.get("per_mb", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    def to_json(self) -> dict[str, float]:
        """Serialize, omitting zero terms."""
        out: dict[str, float] = {"a": round(self.a, 6)}
        if self.b:
            out["b"] = round(self.b, 9)
        if self.per_mb:
            out["per_mb"] = round(self.per_mb, 6)
        # NaN means "this step was not fitted as a line", which is a different
        # statement from a bad fit, so it is omitted rather than written as
        # null: `from_json` defaults it back to NaN either way.
        if not math.isnan(self.r2):
            out["r2"] = round(self.r2, 4)
        return out


@dataclass(frozen=True)
class TimingProfile:
    """Deprecated (#4667): the retired per-environment profile, now always empty.

    Kept so out-of-tree code that names the type keeps importing. The only
    instance this module hands out is :data:`EMPTY_PROFILE`, which is falsy.
    """

    steps: dict[str, dict[str, dict[str, StepCoeffs]]] = field(default_factory=dict)
    branches: dict[str, dict[str, dict[str, dict[str, StepCoeffs]]]] = field(default_factory=dict)
    slots: dict[str, dict[str, dict[str, dict[str, float]]]] = field(default_factory=dict)
    source: str = ""
    generated_at: str = ""
    host: str = ""
    notes: str = ""

    def __bool__(self) -> bool:
        return bool(self.steps or self.slots)

    def describe(self) -> str:
        """One-line human summary."""
        return "timing profile: none (built-in defaults)"


EMPTY_PROFILE = TimingProfile()


def parse_profile(raw: Any, source: str = "") -> TimingProfile:
    """Deprecated (#4667): always :data:`EMPTY_PROFILE`; profiles are no longer read."""
    return EMPTY_PROFILE


def active_profile() -> TimingProfile:
    """Deprecated (#4667): always :data:`EMPTY_PROFILE`; profiles are no longer read."""
    return EMPTY_PROFILE


def reload_profile(path: Optional[str] = None) -> TimingProfile:
    """Deprecated (#4667): always :data:`EMPTY_PROFILE`; *path* is ignored."""
    return EMPTY_PROFILE


def slot_shares(
    task: str,
    step: str,
    *,
    device: Optional[str] = None,
    media_type: str = "",
    embedder: str = "",
) -> Optional[dict[str, float]]:
    """Deprecated (#4667): always ``None``, the answer for an unprofiled step.

    It returned a profile's measured sub-stage shares within one step. The
    dataset load's finalize phase, its only consumer, paces from the shipped
    ``FINALIZE_SLOT_SHARES`` table in
    :mod:`vtscore.datasets.stages._load_cost_model` instead.
    """
    return None


def profile_covers(task: str) -> bool:
    """Deprecated (#4667): always ``False``; no profile covers anything."""
    return False


# ---------------------------------------------------------------------------
# Public lookup API
# ---------------------------------------------------------------------------


def step_terms(
    task: str,
    *,
    device: Optional[str] = None,
    media_type: str = "",
    embedder: str = "",
    n: float = 0.0,
    size_mb: float = 0.0,
    skip_steps: Iterable[str] = (),
    branch: Optional[Union[str, Mapping[str, str]]] = None,
) -> Optional[dict[str, float]]:
    """Shipped default pseudo-seconds per step for *task*, keyed by step name.

    Returns ``None`` when the task is unregistered or ships no default terms
    (``dataset_load``, whose model lives in
    :mod:`vtscore.datasets.stages._load_cost_model`), so the caller can keep its
    own fallback. The default is the media type's own vector when the task
    carries one (:meth:`~vtscore.timing.tasks.TaskSpec.defaults_for`), so
    *media_type* matters.

    *skip_steps* names steps **this invocation will not enter at all**, and
    prices them at zero. It is not a cost model and needs no measurement: a step
    that does not run costs nothing, and that is knowable in advance where a
    step's cost is not. See :func:`step_weights` for why a task reaches for it.

    *device*, *embedder*, *n*, *size_mb* and *branch* selected and scaled the
    retired profile's per-cell coefficients (#4667). The shipped defaults are
    flat ratios with no device, encoder, scale or branch axis, so they are
    accepted for compatibility and ignored.
    """
    spec = task_spec(task)
    if spec is None:
        return None
    shipped = spec.defaults_for(media_type)
    if not shipped:
        return None
    skipped = frozenset(skip_steps)
    terms = {step: 0.0 if step in skipped else max(0.0, term) for step, term in zip(spec.steps, shipped)}
    if sum(terms.values()) <= 0:
        # An all-zero prediction carries no pacing information and would make
        # the weight vector a division by zero; the caller's fallback is better.
        return None
    return terms


def step_weights(
    task: str,
    *,
    device: Optional[str] = None,
    media_type: str = "",
    embedder: str = "",
    n: float = 0.0,
    size_mb: float = 0.0,
    skip_steps: Iterable[str] = (),
    branch: Optional[Union[str, Mapping[str, str]]] = None,
    fallback: Optional[list[float]] = None,
) -> Optional[list[float]]:
    """Normalized per-tracker-step weights for *task*, or *fallback*.

    The returned vector has one entry per tracker step (``TaskSpec.tracker_steps``)
    and sums to 1, ready for
    :meth:`~vtscore.concurrency.progress.ProgressTracker.set_step_weights`.
    Phases that share a tracker step have their terms summed into that step's
    slot.

    **Steps this run will skip.** A default answers "how big a share of the job
    is this step"; it cannot answer "does this step happen at all", and for a
    step that forks on process state the second question decides the bar. A
    text sort's ``load_model`` is seconds on the first sort of a process and
    *exactly zero* on every later one (#3596). The caller usually *knows* — the
    sort route can ask whether the encoder is already resident before it starts
    — and naming the step in *skip_steps* prices it at zero for this run.

    The other keyword arguments are passed through to :func:`step_terms`, which
    ignores all but *media_type* and *skip_steps*.
    """
    spec = task_spec(task)
    terms = step_terms(
        task,
        device=device,
        media_type=media_type,
        embedder=embedder,
        n=n,
        size_mb=size_mb,
        skip_steps=skip_steps,
        branch=branch,
    )
    if spec is None or terms is None:
        return fallback
    weights = [0.0] * spec.tracker_steps
    for step, index in zip(spec.steps, spec.step_index):
        weights[index - 1] += terms[step]
    total = sum(weights)
    if total <= 0:
        return fallback
    return [w / total for w in weights]


def known_tasks() -> tuple[str, ...]:
    """Names of every registered task family, in registry order."""
    return tuple(TASKS)
