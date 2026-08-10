#!/usr/bin/env python3
"""Plotting for the directory-driven train/test sweep (``sweep_train_test.py``).

Adds two things on top of :mod:`plots`, which it reuses for all the actual drawing:

**Confusion-matrix metrics.** ``accuracy`` / ``balanced_accuracy`` / ``precision`` /
``recall`` are *derived* from fields already on every row rather than recomputed from
scores. ``weighted_error`` defines ``fpr = FP/N`` and ``fnr = FN/P``, and every row
carries ``n_test`` and ``n_test_pos``, so the whole confusion matrix comes back:

    P = n_test_pos, N = n_test - P
    TP = (1-fnr)*P,  FN = fnr*P,  FP = fpr*N,  TN = (1-fpr)*N

Those counts are recovered **exactly**: the rates are stored rounded to 6 decimals, so
the error in ``fpr*N`` is under ``5e-7 * N`` and rounds to the correct integer for any
``N`` below a million. Deriving rather than recomputing means these metrics appear on
**results.jsonl files that already exist**, with no re-run. ``auroc`` is the exception -
it needs the score vector, so it is computed in ``region_curve._oracle_extra`` and is
NaN on runs made before that landed.

**The ``all+std`` band.** Every other band draws one figure per metric. ``all+std``
draws two: the per-seed overlay (every iteration as its own line, so outliers are
visible) *and* a mean +/- stdev summary beside it. Reading both matters here because a
single wild seed and a genuinely wide distribution look identical in a summary alone.
"""

from __future__ import annotations

import csv
from pathlib import Path

import plots

#: Metrics plotted by default for a train/test run: the existing rate/threshold family,
#: the derived confusion-matrix family, and threshold-free AUROC. ``mean_iou``/``corloc``
#: are omitted - the train/test tree is boxless, so both are always NaN there.
TRAIN_TEST_METRICS = (
    "cost",
    "fpr",
    "fnr",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "auroc",
)

#: Metrics this module derives from the stored rates (i.e. everything but ``auroc``).
DERIVED_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall")

#: Band modes: the ones :mod:`plots` understands, plus the dual-output mode.
BANDS = ("minmax", "std", "none", "all", "all+std")

_NAN = float("nan")


def _confusion(row: dict) -> tuple[float, float, float, float] | None:
    """``(TP, FP, TN, FN)`` for one row, or ``None`` when it cannot be reconstructed."""
    fpr, fnr = row.get("fpr"), row.get("fnr")
    n_test, n_pos = row.get("n_test"), row.get("n_test_pos")
    if fpr is None or fnr is None or not n_test or n_pos is None:
        return None
    if fpr != fpr or fnr != fnr:  # NaN rates (empty test set)
        return None
    p = float(n_pos)
    n = float(n_test) - p
    # round(): the stored rates are 6-dp, so this recovers the exact integer counts.
    tp = round((1.0 - fnr) * p)
    fn = round(fnr * p)
    fp = round(fpr * n)
    tn = round((1.0 - fpr) * n)
    return tp, fp, tn, fn


def derived_metrics(row: dict) -> dict[str, float]:
    """The four confusion-matrix metrics for one row (NaN where undefined).

    ``precision`` is NaN when nothing was predicted positive, matching ``f1_at``'s
    convention; ``recall`` and ``balanced_accuracy`` follow directly from the rates and
    need no counts at all, so they survive even a degenerate test split.
    """
    conf = _confusion(row)
    if conf is None:
        return dict.fromkeys(DERIVED_METRICS, _NAN)
    tp, fp, tn, fn = conf
    total = tp + fp + tn + fn
    fpr, fnr = float(row["fpr"]), float(row["fnr"])
    return {
        "accuracy": (tp + tn) / total if total else _NAN,
        # (TPR + TNR) / 2, i.e. accuracy re-weighted so each class counts equally.
        # The honest headline when prevalence is far from 50%.
        "balanced_accuracy": 1.0 - (fpr + fnr) / 2.0,
        "precision": tp / (tp + fp) if (tp + fp) else _NAN,
        "recall": 1.0 - fnr,
    }


def enrich_rows(rows: list[dict]) -> list[dict]:
    """Return *rows* with the derived metrics added (originals are not mutated).

    Idempotent, and it never overwrites a field a row already carries, so a future
    core-computed ``precision`` would win over the derived one.
    """
    out = []
    for r in rows:
        extra = {k: v for k, v in derived_metrics(r).items() if k not in r}
        out.append({**r, **extra} if extra else r)
    return out


def _normalize_metric(name: str) -> str:
    """``"Balanced Accuracy"`` / ``"balanced-accuracy"`` -> ``"balanced_accuracy"``."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def load_reference_csv(path: Path) -> dict[str, float]:
    """Read a ``metric,value`` CSV of external reference numbers.

    Two columns, no header required: the metric name on the left and its float value on
    the right. Every metric is optional and the order is arbitrary, so the file may carry
    any subset - only the metrics that are actually plotted get a line. Blank lines and
    ``#`` comments are skipped, a leading header row is tolerated (detected by its value
    column not parsing as a float), and names are matched case/space/hyphen-insensitively.

    A value that is not a float IS an error: the caller stated the right column is always
    a valid float, so a bad one means a malformed file rather than an absent metric, and
    silently dropping it would quietly remove a reference line from a figure. Unknown
    metric names are warned about (likely typos) but kept, so a metric added to
    :data:`plots._METRICS` later starts working without touching the file.
    """
    out: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for lineno, fields in enumerate(csv.reader(fh), start=1):
            if not fields or not (fields[0] or "").strip() or fields[0].lstrip().startswith("#"):
                continue
            if len(fields) < 2:
                raise ValueError(f"{path}:{lineno}: expected 'metric,value', got {fields!r}")
            metric = _normalize_metric(fields[0])
            try:
                value = float(fields[1])
            except ValueError as exc:
                if lineno == 1 and not out:  # a header row: "metric,value"
                    continue
                raise ValueError(f"{path}:{lineno}: value for {metric!r} is not a float: {fields[1]!r}") from exc
            if metric in out:
                print(f"  [reference] {path}:{lineno}: duplicate {metric!r}, using the later value", flush=True)
            if metric not in plots._METRICS:
                print(f"  [reference] {path}:{lineno}: unknown metric {metric!r} (kept, but nothing plots it)")
            out[metric] = value
    return out


def render_all(
    rows: list[dict],
    out_dir: Path,
    *,
    metrics: list[str] | tuple[str, ...] = TRAIN_TEST_METRICS,
    band: str = "all+std",
    show_oracle: bool = False,
    x_label: str = "t (total annotations)",
    x_tag: str = "t",
    reference: dict[str, float] | None = None,
) -> None:
    """One figure per metric per (dataset, class), with the derived metrics available.

    ``band="all+std"`` emits two files per metric: ``…_<metric>_vs_<x_tag>.png`` (one
    line per seed) and ``…_<metric>_vs_<x_tag>_summary.png`` (mean +/- stdev). Any other
    band behaves exactly as :func:`plots.render_all`.

    ``show_oracle`` annotates every metric that has a non-degenerate best-over-τ value -
    ``cost``, ``f1``, ``accuracy``, ``balanced_accuracy`` - from the ``oracle_*`` columns
    the rows already carry. The other metrics draw nothing: ``fpr``/``fnr``/``precision``/
    ``recall`` have degenerate optima and ``auroc`` is threshold-free. Rows written before
    ``oracle_accuracy``/``oracle_balanced_accuracy`` existed carry NaN and are skipped, so
    an old ``results.jsonl`` silently gets the cost/F1 pair only.

    ``reference`` (see :func:`load_reference_csv`) draws a flat black dash-dot line on
    each metric it names, on both figures of an ``all+std`` pair. Metrics it omits are
    simply not annotated.
    """
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}; expected one of {BANDS}")
    enriched = enrich_rows(rows)
    if band == "all+std":
        passes = (
            ("all", x_tag, " (per seed)"),
            ("std", f"{x_tag}_summary", " (mean ± stdev across seeds)"),
        )
    else:
        passes = ((band, x_tag, ""),)
    for pass_band, pass_file_tag, note in passes:
        plots.render_all(
            enriched,
            out_dir,
            metrics=list(metrics),
            band=pass_band,
            show_oracle=show_oracle,
            x_label=x_label,
            x_tag=x_tag,
            file_tag=pass_file_tag,
            title_note=note,
            reference=reference,
        )
