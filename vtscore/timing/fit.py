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

That guarantee has a price, and it is not small. #3345 measured the three levels
against the same rows with the same fitter: exact cells at median r² 1.00 and
3 % prediction error, ``(device, media, *)`` at 0.99 and 9 %, and
``(device, *, *)`` at **0.29 and 50 %** — 162 % on one arm. So two things temper
the rollups here:

- A rollup step whose pooled groups disagree by more than
  :data:`_MAX_ROLLUP_SPREAD` is **not emitted at all**, and falls through to the
  shipped default. A rollup is only ever *reached* for a combination the sweep
  never measured (see :func:`_rollup_is_contradicted`), so a cell built by
  averaging things measured to be unlike is extrapolating from a number it has
  already been told is wrong for both of them.
- :func:`coverage_report` breaks its fit-quality line down **by specificity**,
  so an admin reading "5 cells" sees how many are exact measurements and how
  many are the fallbacks that will actually pace an unmeasured dataset.
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


#: Ratio between the cheapest and dearest group a rollup pools, above which the
#: pooled fit is treated as contradicted by its own samples rather than merely
#: noisy. #3345 measured the harmful case at 7.3x — ``(cuda+cuml, *, *)`` fitting
#: one slope through a 0.014 s/item image import and a 0.102 s/item audio one,
#: for a median prediction error of 50% against 3% for the exact cells. The
#: threshold sits well above the spread a well-behaved rollup shows (that study's
#: media rollups ran at 0.09 error) so it fires on disagreement, not on scatter.
_MAX_ROLLUP_SPREAD = 3.0

#: A predicted step cost at or below this is "free" for pacing purposes. Ratios
#: between such numbers are arithmetic noise, so they are compared against this
#: floor instead: two negligible groups agree, and one negligible beside one
#: material group is maximal disagreement however the division lands.
_NEGLIGIBLE_SECONDS = 0.01


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


def _wildcard_axes(cell: tuple[str, str, str]) -> tuple[str, ...]:
    """Which of ``("media_type", "embedder")`` this cell key wildcards.

    An empty component in a stored key is a wildcard at lookup time, so it is
    also the axis along which that cell pools unlike rows. The exact cell
    wildcards nothing and is returned as an empty tuple.
    """
    return tuple(axis for axis, value in (("media_type", cell[1]), ("embedder", cell[2])) if not value)


def _rollup_is_contradicted(samples: list[dict], cell: tuple[str, str, str]) -> bool:
    """Whether *cell* pools groups whose own fits disagree about this step.

    **A rollup cell is only ever reached for a combination the sweep never
    measured.** :func:`vtscore.timing.profile.cell_keys` tries every more
    specific key first, and this fitter emits one for every combination it saw,
    so ``(device, media, *)`` serves only encoders that media type was never
    measured with, and ``(device, *, *)`` only media types the sweep never
    touched at all. The rollup's whole job is extrapolation — which is exactly
    why it must not be built by averaging things it has measured to be unlike.

    The test is the one the profile is actually used for: fit each pooled group
    on its own, ask each what the step costs at a size all of them cover, and
    compare. That works the same for a sloped step and a flat one, where the raw
    ``seconds / n`` ratio does not — two constant-cost groups sampled at
    different ``n`` look wildly divergent per item and are not.

    Groups closer than :data:`_MAX_ROLLUP_SPREAD` keep their pooled fit: a
    rollup that is merely imprecise still beats the shipped default, and dropping
    it would throw away the coverage the rollups exist to provide.
    """
    axes = _wildcard_axes(cell)
    if not axes:
        return False
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for sample in samples:
        groups[tuple(sample[axis] for axis in axes)].append(sample)
    if len(groups) < 2:
        # One group behind the rollup: it is a rename of the specific cell it
        # backs up, and repeats its claim rather than blending anything.
        return False

    probe_n = statistics.median([s["n"] for s in samples])
    predictions: list[float] = []
    for group in groups.values():
        coeffs = fit_step(group, byte_scaled=False)
        if coeffs is None:
            return False
        predictions.append(coeffs.seconds(probe_n))

    hi, lo = max(predictions), min(predictions)
    if hi <= _NEGLIGIBLE_SECONDS:
        # Every group says this step is free; there is nothing to be wrong about.
        return False
    if lo <= _NEGLIGIBLE_SECONDS:
        # One group free beside one that is not: the pooled number is wrong for
        # whichever of them the lookup lands on.
        return True
    return hi / lo > _MAX_ROLLUP_SPREAD


def _fit_task_cells(spec, cells: _CellSamples, slots: _CellSlots, min_samples: int) -> dict[str, Any]:
    """Fit every sufficiently-sampled cell of one task into its JSON entry."""
    cells_out: dict[str, Any] = {}
    for cell, step_samples in cells.items():
        runs = _run_count(step_samples)
        if runs < min_samples:
            continue
        steps_out: dict[str, Any] = {}
        for step, samples in step_samples.items():
            byte_scaled = step in spec.byte_scaled
            if not byte_scaled and _rollup_is_contradicted(samples, cell):
                # Omitting the step is what "no measurement here" looks like to
                # the reader: `step_terms` falls that one step through to its
                # shipped default while the rest of this cell still applies. A
                # confidently wrong number cannot be fallen through to (#3522).
                continue
            coeffs = fit_step(samples, byte_scaled)
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

#: How a stored cell key is described in the coverage report, keyed by which of
#: ``(media_type, embedder)`` it wildcards. Written in the key's own ``|`` syntax
#: so a reader can match a report line against the profile JSON by eye.
_SPECIFICITY_LABELS: tuple[tuple[tuple[str, ...], str], ...] = (
    ((), "exact  (device|media|embedder)"),
    (("embedder",), "rollup (device|media|*)"),
    (("media_type",), "rollup (device|*|embedder)"),
    (("media_type", "embedder"), "rollup (device|*|*)"),
)


def _specificity_label(cell_key: str) -> str:
    """Describe one stored cell key by how much of its identity is wildcarded."""
    parts = (cell_key.split("|") + ["", ""])[:3]
    axes = _wildcard_axes((parts[0], parts[1], parts[2]))
    for candidate, label in _SPECIFICITY_LABELS:
        if candidate == axes:
            return label
    return "rollup (device|*|*)"  # pragma: no cover - _wildcard_axes is exhaustive


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


def _specificity_lines(spec, cells: dict[str, Any], buckets: _CellSamples) -> list[str]:
    """One :func:`_fit_quality` line per specificity level, most specific first.

    Ordered to match :func:`vtscore.timing.profile.cell_keys`, so the lines read
    down in the order a lookup tries them: the first level with a cell for a
    given media type and encoder is the one that will actually pace that job.

    A level with no surviving cell still gets a line when steps were *withheld*
    there, since "this rollup was refused" is exactly the fact a bare cell count
    cannot carry.
    """
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, cell in cells.items():
        grouped[_specificity_label(key)][key] = cell
    withheld = _withheld_by_specificity(spec, buckets)

    width = max(len(label) for _, label in _SPECIFICITY_LABELS)
    lines: list[str] = []
    for _, label in _SPECIFICITY_LABELS:
        level = grouped.get(label, {})
        dropped = withheld.get(label, 0)
        if not level and not dropped:
            continue
        parts = [f"{len(level)} cell{'' if len(level) == 1 else 's'}"]
        quality = _fit_quality(spec, level)
        if quality:
            parts.append(quality)
        if dropped:
            parts.append(f"{dropped} step{'' if dropped == 1 else 's'} withheld (pooled groups disagree)")
        lines.append(f"  {'':<16} {label:<{width}}  {', '.join(parts)}")
    return lines


def _withheld_by_specificity(spec, cells: _CellSamples) -> dict[str, int]:
    """Count the steps :func:`_rollup_is_contradicted` kept out, per specificity.

    Recomputed from the rows rather than recorded in the profile: a withheld
    step is precisely one the document does *not* contain, and inventing a
    schema field to say so would make every reader parse a negative claim. The
    tuning script runs this once at the end of a sweep, so the second pass over
    the buckets costs nothing anyone waits on.
    """
    out: dict[str, int] = defaultdict(int)
    for cell, step_samples in cells.items():
        for step, samples in step_samples.items():
            if step not in spec.byte_scaled and _rollup_is_contradicted(samples, cell):
                out[_specificity_label("|".join(cell))] += 1
    return dict(out)


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

    That quality line is then **split by specificity**, because pooling the
    levels hides the one an admin most needs to see. #3345 measured, on one
    sweep's own rows, exact cells at r² 1.00 / 3 % error against
    ``(device, *, *)`` cells at 0.29 / 50 %; a single median over both reads
    like the exact number and is used like the rollup one, since the rollup is
    the cell guaranteed to match. Split, "5 cells" resolves into how many are
    measurements and how many are fallbacks (#3522).
    """
    rows = list(rows)
    normalized = [n for n in (normalize_row(r) for r in rows) if n is not None]
    seen_tasks = {n["task"] for n in normalized}
    by_cell, _ = _bucket_rows(rows)
    lines: list[str] = []
    for task in TASKS:
        cells = profile.get("tasks", {}).get(task, {}).get("cells", {})
        if cells:
            samples = sum(int(c.get("samples", 0)) for c in cells.values())
            lines.append(f"  {task:<16} {len(cells)} cells, {samples} step-samples")
            lines.extend(_specificity_lines(TASKS[task], cells, by_cell.get(task, {})))
        elif task in seen_tasks:
            lines.append(f"  {task:<16} measured but too few runs to fit — using built-in defaults")
        else:
            lines.append(f"  {task:<16} NOT MEASURED — using built-in defaults")
    return lines
