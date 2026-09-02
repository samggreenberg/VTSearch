"""Turn recorded step timings into a timing-profile document.

This is the writer for the format :mod:`vtscore.timing.profile` reads. It lives
here, next to the reader, so the two cannot drift: a change to the schema has to
be made in one directory or it will not round-trip.

Input is JSONL from either recorder — the generic one
(:mod:`vtscore.timing.recorder`) or the older dataset-load profiler
(:mod:`vtscore.datasets.stages._load_profiler`) — because an admin should be
able to fold an existing calibration sweep into a new profile rather than
re-measuring what has already been measured. :func:`normalize_row` flattens both
shapes into one.

The fit itself is deliberately plain. Per ``(task, cell, step)``:

- **Byte-scaled steps** (a dataset download and its unpack) get a per-MB rate:
  the median of ``seconds / archive_mb``. Regressing them against item count
  would ask ``n`` to explain something it cannot see — 500 videos and 500 text
  files are the same ``n`` and two orders of magnitude apart in bytes.
- **Everything else** gets ordinary least squares against ``n``, giving the
  intercept (what the step costs at all — loading an encoder, opening a file)
  and the slope (what each additional item adds).
- A fit with no spread in ``n``, or one that comes back with a *negative* slope
  (noise beating signal on a short step), collapses to ``median seconds`` with
  no slope. A confidently wrong slope extrapolates badly at sizes the sweep
  never visited, and "we only know the average" is the honest answer there.

Cells are emitted at three specificities — exact ``(device, media, embedder)``,
then ``(device, media, *)``, then ``(device, *, *)``. The rollups are what make a
small sweep worth running: an admin who measures three exemplar datasets still
improves the pacing of every task on every dataset that host will ever see,
because the least-specific cell always matches.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable, Optional

from vtscore.timing.profile import SCHEMA_NAME, SCHEMA_VERSION, StepCoeffs
from vtscore.timing.tasks import TASKS, task_spec

#: Phase names the older dataset-load profiler writes, mapped onto the task
#: registry's step names for ``dataset_load``.
_LEGACY_PHASE_TO_STEP = {
    "download": "download",
    "extract": "extract",
    "model_load": "load",
    "embed": "embed",
    "finalize": "finalize",
}

#: A byte-scaled step under this many seconds is dominated by setup overhead
#: rather than transfer, so it makes a poor per-MB rate sample.
_MIN_BYTE_STEP_SECONDS = 0.1

#: Distinct ``n`` values a fit needs before its r2 means anything. Two points
#: define a line exactly, so r2 is 1.0 whatever they are.
_MIN_R2_POINTS = 3


def load_rows(paths: Iterable[str]) -> list[dict]:
    """Read JSONL from every path, skipping blank and unparseable lines.

    A recorder appends from live worker threads, so a file can end mid-line if
    the process died; one truncated row must not cost the sweep every good row
    before it.
    """
    import json  # noqa: PLC0415 - only needed on the file path

    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def device_key(device: str, cuml: bool) -> str:
    """Collapse a recorded device onto the profile's device key.

    CUDA splits in two: cuML moves the clustering work onto the GPU, which
    changes what the finalize-shaped steps cost enough that pooling the two
    would average away the thing being measured.
    """
    if not str(device).startswith("cuda"):
        return "cpu"
    return "cuda+cuml" if cuml else "cuda"


def normalize_row(raw: dict) -> Optional[dict]:
    """Flatten one recorded row into ``{task, device, media_type, embedder, n,
    size_mb, step, slot, seconds}``, or ``None`` if it is not a usable sample.

    Handles both recorder shapes. Rejects rows that describe something other
    than a completed unit of work: a failed or partial run, or a legacy row with
    ``n <= 0`` (which is how the old profiler marks a load that died before
    producing any items).
    """
    seconds = raw.get("seconds")
    if seconds is None:
        return None
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return None

    task = raw.get("task")
    phase = raw.get("phase")
    slot = ""
    if task:
        step = raw.get("step")
        if not raw.get("ok", True) or not raw.get("complete", True):
            return None
        size_mb = float(raw.get("size_mb") or 0.0)
    elif phase:
        # Legacy dataset-load profiler row.
        task = "dataset_load"
        if float(raw.get("n") or 0) <= 0:
            return None
        if str(phase).startswith("finalize:"):
            step, slot = "finalize", str(phase).split(":", 1)[1]
        else:
            step = _LEGACY_PHASE_TO_STEP.get(str(phase))
        size_mb = float(raw.get("download_size_mb") or 0.0)
    else:
        return None

    spec = task_spec(str(task))
    if spec is None or not step or (not slot and step not in spec.steps):
        return None

    return {
        "task": str(task),
        "device": device_key(str(raw.get("device", "cpu")), bool(raw.get("cuml"))),
        "media_type": str(raw.get("media_type", "")),
        "embedder": str(raw.get("embedder", "")),
        "n": float(raw.get("n") or 0.0),
        "size_mb": size_mb,
        "step": str(step),
        "slot": slot,
        "seconds": max(0.0, seconds),
    }


def affine_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares ``y ≈ a + b·x``, returning ``(a, b, r2)``.

    Falls back to ``(mean, 0, 0)`` when ``x`` has no spread — one dataset size
    can tell you what a step costs, but nothing about how it scales.
    """
    count = len(xs)
    if count == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-9 or count < 2:
        return mean_y, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
    return intercept, slope, r2


def _r2_of(xs: list[float], ys: list[float], a: float, b: float) -> float:
    """Goodness of the *given* coefficients against the data, not of a best fit.

    :func:`affine_fit` scores the line it computed. When the coefficients that
    get stored differ from that line — the intercept is clamped at zero below —
    that score describes a model the profile does not contain, so it has to be
    recomputed against the one it does.
    """
    count = len(ys)
    if count == 0:
        return float("nan")
    mean_y = sum(ys) / count
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot <= 1e-9:
        return 1.0
    ss_res = sum((y - max(0.0, a + b * x)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_res / ss_tot


def fit_step(samples: list[dict], byte_scaled: bool) -> Optional[StepCoeffs]:
    """Fit one step's coefficients from its samples, or ``None`` if unusable."""
    if not samples:
        return None
    if byte_scaled:
        rates = [
            s["seconds"] / s["size_mb"] for s in samples if s["size_mb"] > 0 and s["seconds"] >= _MIN_BYTE_STEP_SECONDS
        ]
        if not rates:
            # Every sample was a cached/absent archive: this deployment's loads
            # genuinely pay nothing here, which is a real (zero) measurement.
            return StepCoeffs()
        return StepCoeffs(per_mb=statistics.median(rates))

    xs = [s["n"] for s in samples]
    ys = [s["seconds"] for s in samples]
    intercept, slope, r2 = affine_fit(xs, ys)
    if len({round(x, 6) for x in xs}) < _MIN_R2_POINTS:
        # A line through two points has ss_res == 0, so its r2 is 1.0 by
        # construction and says nothing (#3345). The coefficients are still the
        # best line available and are kept; only the goodness claim is withheld.
        # The clamp below deliberately overrides this: once the stored model is
        # no longer the interpolant, its residuals are real at any point count.
        r2 = float("nan")
    if slope <= 0:
        # No credible scaling signal (a flat step, or noise swamping a short
        # one). Report the typical cost and claim nothing about growth. The r2
        # is deliberately NOT carried here: it describes a line this branch has
        # just declined to use, so reporting it would attach a goodness score to
        # coefficients that are a median.
        return StepCoeffs(a=max(0.0, statistics.median(ys)))

    a = max(0.0, intercept)
    if a != intercept:
        # A steep slope through a short step lands a negative intercept, and
        # `seconds()` would hand the bar a negative slice, so it is clamped. The
        # r2 above then scores a line nobody stores: #3345 measured 58 of 195
        # affine cells in this state, one of them annotating coefficients 52%
        # out with an r2 of 0.98. Rescore against what is actually kept -- which
        # can come back negative, meaning "worse than a constant", and that is
        # a true statement worth persisting rather than hiding.
        r2 = _r2_of(xs, ys, a, slope)
    return StepCoeffs(a=a, b=slope, r2=r2)


def _cell_variants(row: dict) -> tuple[tuple[str, str, str], ...]:
    """The cell keys *row* contributes to: exact, then the two rollups."""
    device, media, embedder = row["device"], row["media_type"], row["embedder"]
    variants = [(device, media, embedder), (device, media, ""), (device, "", "")]
    seen: list[tuple[str, str, str]] = []
    for variant in variants:
        if variant not in seen:
            seen.append(variant)
    return tuple(seen)


def _run_count(step_samples: dict[str, list[dict]]) -> int:
    """How many distinct runs fed a cell (its best-covered step's sample count)."""
    return max((len(v) for v in step_samples.values()), default=0)


#: ``task -> cell -> step -> samples``, plus the same shape one level deeper for
#: sub-slot durations (``… -> step -> slot -> seconds``).
_CellSamples = dict[tuple[str, str, str], dict[str, list[dict]]]
_CellSlots = dict[tuple[str, str, str], dict[str, dict[str, list[float]]]]


def _bucket_rows(rows: Iterable[dict]) -> tuple[dict[str, _CellSamples], dict[str, _CellSlots]]:
    """Group normalized rows by task and cell, splitting out sub-slot durations.

    Each row lands in *every* cell it qualifies for — exact and both rollups —
    so the broader cells are fit from strictly more evidence than the narrow
    ones they back up.
    """
    by_cell: dict[str, _CellSamples] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    slot_secs: dict[str, _CellSlots] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for raw in rows:
        row = normalize_row(raw)
        if row is None:
            continue
        for cell in _cell_variants(row):
            if row["slot"]:
                slot_secs[row["task"]][cell][row["step"]][row["slot"]].append(row["seconds"])
            else:
                by_cell[row["task"]][cell][row["step"]].append(row)
    return by_cell, slot_secs


def _fit_task_cells(spec, cells: _CellSamples, slots: _CellSlots, min_samples: int) -> dict[str, Any]:
    """Fit every sufficiently-sampled cell of one task into its JSON entry."""
    cells_out: dict[str, Any] = {}
    for cell, step_samples in cells.items():
        runs = _run_count(step_samples)
        if runs < min_samples:
            continue
        steps_out: dict[str, Any] = {}
        for step, samples in step_samples.items():
            coeffs = fit_step(samples, step in spec.byte_scaled)
            if coeffs is not None:
                steps_out[step] = coeffs.to_json()
        if not steps_out:
            continue
        entry: dict[str, Any] = {"samples": runs, "steps": steps_out}
        slots_out = _fit_slots(slots.get(cell, {}))
        if slots_out:
            entry["slots"] = slots_out
        cells_out["|".join(cell)] = entry
    return cells_out


def fit_profile(
    rows: Iterable[dict],
    *,
    min_samples: int = 2,
    generated_at: str = "",
    host: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Fit every ``(task, cell, step)`` present in *rows* into a profile document.

    Cells with fewer than *min_samples* runs are dropped: a single timing is an
    anecdote, and shipping it as a coefficient would replace a considered
    default with one machine's one bad afternoon. Dropped cells simply fall
    through to the next-broader cell — or, if none survives, to the built-in
    defaults, which is the correct outcome for a thin sweep.

    Returns a JSON-serializable dict ready to write and hand to
    ``VTSEARCH_TIMING_PROFILE``.
    """
    by_cell, slot_secs = _bucket_rows(rows)

    tasks_out: dict[str, Any] = {}
    for task, cells in by_cell.items():
        cells_out = _fit_task_cells(TASKS[task], cells, slot_secs.get(task, {}), min_samples)
        if cells_out:
            tasks_out[task] = {"cells": cells_out}

    doc: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "tasks": tasks_out,
    }
    if generated_at:
        doc["generated_at"] = generated_at
    if host:
        doc["host"] = host
    if notes:
        doc["notes"] = notes
    return doc


def _fit_slots(step_slots: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    """Normalize per-step sub-slot durations into shares of that step.

    Median seconds per slot (robust to the one run where the disk stalled), then
    normalized across the step. A slot that measured no time at all is dropped
    rather than shipped as a zero weight — the consumer merges what is here over
    its own defaults, so an unmeasured slot keeps a sane share instead of being
    told it is free.
    """
    out: dict[str, dict[str, float]] = {}
    for step, slots in step_slots.items():
        medians = {slot: statistics.median(v) for slot, v in slots.items() if v}
        total = sum(medians.values())
        if total <= 0:
            continue
        shares = {slot: round(value / total, 4) for slot, value in medians.items() if value > 0}
        if shares:
            out[step] = shares
    return out


#: An affine fit at or above this describes its own samples well enough that a
#: reader need not look further. Below it, the line is worth a second look.
_GOOD_R2 = 0.90


def _fit_quality(spec, cells: dict[str, Any]) -> str:
    """One line describing *how* a task's cells were fitted, and how well.

    Three outcomes, and the difference between them is the thing readers get
    wrong. A step with **no r²** is not a badly fitted line — it is not a line:
    either the step is byte-scaled (a per-MB rate, never regressed against
    ``n``) or ``fit_step`` found no credible slope and reported a median. Only
    the affine count has a goodness attached, so only it gets one here.

    A count of the affine fits *below* :data:`_GOOD_R2` is what makes the line
    actionable: a median of 0.99 over a hundred cells hides the six that the
    line does not describe, and those six are where a progress bar drifts.
    """
    affine: list[float] = []
    fallback = 0
    byte_rate = 0
    for cell in cells.values():
        for step, coeffs in cell.get("steps", {}).items():
            if step in spec.byte_scaled:
                byte_rate += 1
            elif "r2" in coeffs:
                affine.append(float(coeffs["r2"]))
            else:
                fallback += 1
    parts: list[str] = []
    if affine:
        poor = sum(1 for r2 in affine if r2 < _GOOD_R2)
        detail = f"median r² {statistics.median(affine):.2f}"
        if poor:
            detail += f", {poor} below {_GOOD_R2:.2f}"
        parts.append(f"{len(affine)} affine ({detail})")
    if fallback:
        parts.append(f"{fallback} median-fallback (no credible slope)")
    if byte_rate:
        parts.append(f"{byte_rate} byte-rate")
    return ", ".join(parts)


def coverage_report(rows: Iterable[dict], profile: dict[str, Any]) -> list[str]:
    """Human-readable lines describing what the sweep did and did not cover.

    Printed by the tuning script so a thin sweep is *visible* rather than
    quietly shipping a profile that improves two cells and leaves the rest to
    the defaults. Silent partial coverage is how a tuning run gets mistaken for
    a tuned system.

    Cell counts alone cannot say that, though: a task can be measured in fifty
    cells and still have every one of them fall back to a median, which reads
    as full coverage and paces like none. So each measured task also reports
    :func:`_fit_quality` — the r² ``StepCoeffs`` has carried since #3334 and
    which, until #3345, nothing in the tree ever read.
    """
    normalized = [n for n in (normalize_row(r) for r in rows) if n is not None]
    seen_tasks = {n["task"] for n in normalized}
    lines: list[str] = []
    for task in TASKS:
        cells = profile.get("tasks", {}).get(task, {}).get("cells", {})
        if cells:
            samples = sum(int(c.get("samples", 0)) for c in cells.values())
            lines.append(f"  {task:<16} {len(cells)} cells, {samples} step-samples")
            quality = _fit_quality(TASKS[task], cells)
            if quality:
                lines.append(f"  {'':<16} {quality}")
        elif task in seen_tasks:
            lines.append(f"  {task:<16} measured but too few runs to fit — using built-in defaults")
        else:
            lines.append(f"  {task:<16} NOT MEASURED — using built-in defaults")
    return lines
