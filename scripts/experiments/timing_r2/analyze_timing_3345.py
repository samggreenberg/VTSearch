#!/usr/bin/env python
"""Read the r² off a recorded timing profile (#3345).

`vtscore/timing/fit.py` computes an OLS r² per `(task, cell, step)` and, since
#3334, keeps it on `StepCoeffs`. Nothing had ever looked at one. This reads
them, and reads apart the two things a missing r² can mean:

- **byte-rate** — the step is fit as seconds/MB (`download`, `extract`), so no
  line against `n` was ever drawn;
- **median fallback** — `fit_step` saw no spread in `n`, or a negative slope,
  and reported the typical cost with no growth term.

Neither is a bad fit. Only a *present* r² is a fit quality at all, which is why
the count of each kind is reported before any r² is.

Three profiles are compared, and the comparison is the point:

1. `profile_generic.json` — this run's generic recorder (`VTSEARCH_TIMING_RECORD`)
   across five task families;
2. `profile_loadprof.json` — the older per-load profiler
   (`VTSEARCH_PROFILE_LOAD`) on the *same* loads, so recorder-vs-recorder is a
   controlled contrast rather than a machine one;
3. `profile_resweep.json` — the #3062 load-cost resweep's rows, refit through
   this fitter: 17 datasets across 5 media types, the widest `n` spread on disk.

Beyond r², each fitted step is scored on what a progress bar actually needs:
the median absolute percentage error of `StepCoeffs.seconds()` against the very
samples it was fit from. A high r² on a step that costs 0.2 s buys nothing; a
step whose predictions are 40 % out is mis-pacing the bar whatever its r².
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vtscore.timing import profile as profile_mod  # noqa: E402
from vtscore.timing.fit import load_rows, normalize_row  # noqa: E402
from vtscore.timing.profile import StepCoeffs  # noqa: E402
from vtscore.timing.tasks import TASKS, task_spec  # noqa: E402

#: r² at or above this is called an affine step without qualification.
GOOD_R2 = 0.90
#: Below this the line is not describing the data at all.
POOR_R2 = 0.50
#: Median APE at or below this paces a bar acceptably.
GOOD_PRED_ERROR = 0.20


def specificity(cell: str) -> str:
    """How specific a profile cell key is — the axis the rollup design turns on."""
    _device, media, embedder = (cell.split("|") + ["", ""])[:3]
    if embedder:
        return "exact"
    if media:
        return "media rollup"
    return "device rollup"


def fit_kind(step: str, coeffs: dict, spec) -> str:
    """Classify one serialized step entry into the three fit shapes."""
    if spec is not None and step in spec.byte_scaled:
        return "byte rate"
    if "r2" in coeffs:
        return "affine"
    return "median fallback"


def read_profile(path: Path) -> list[dict]:
    """Flatten a profile JSON into one record per ``(task, cell, step)``."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for task, entry in sorted(doc.get("tasks", {}).items()):
        spec = task_spec(task)
        for cell, cell_entry in sorted(entry.get("cells", {}).items()):
            for step, coeffs in sorted(cell_entry.get("steps", {}).items()):
                out.append(
                    {
                        "profile": path.stem,
                        "task": task,
                        "cell": cell,
                        "specificity": specificity(cell),
                        "step": step,
                        "samples": int(cell_entry.get("samples", 0)),
                        "a": float(coeffs.get("a", 0.0)),
                        "b": float(coeffs.get("b", 0.0)),
                        "per_mb": float(coeffs.get("per_mb", 0.0)),
                        "r2": float(coeffs["r2"]) if "r2" in coeffs else float("nan"),
                        "kind": fit_kind(step, coeffs, spec),
                    }
                )
    return out


def bucket_samples(rows: Iterable[dict]) -> dict[tuple[str, str, str], list[dict]]:
    """``(task, cell, step) -> samples``, matching what the fitter bucketed.

    Re-derived with the shipped ``normalize_row`` and the same exact/rollup
    fan-out, so a residual is measured against the rows the coefficient was
    actually fit from rather than against a subset that merely resembles them.
    """
    out: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for raw in rows:
        row = normalize_row(raw)
        if row is None or row["slot"]:
            continue
        device, media, embedder = row["device"], row["media_type"], row["embedder"]
        for cell in ((device, media, embedder), (device, media, ""), (device, "", "")):
            out[(row["task"], "|".join(cell), row["step"])].append(row)
    return out


def residuals(record: dict, samples: list[dict]) -> dict[str, float]:
    """Prediction error of these coefficients against their own samples.

    Median APE, plus the spread in ``n`` the fit had to work with — a cell whose
    ``n`` never moved cannot have been fit as a line no matter what it reports.
    """
    coeffs = StepCoeffs(a=record["a"], b=record["b"], per_mb=record["per_mb"])
    errors: list[float] = []
    for s in samples:
        pred = coeffs.seconds(n=s["n"], size_mb=s["size_mb"])
        if s["seconds"] > 1e-6:
            errors.append(abs(pred - s["seconds"]) / s["seconds"])
    ns = sorted({round(s["n"]) for s in samples})
    secs = sorted(s["seconds"] for s in samples)
    return {
        "n_samples": len(samples),
        "n_distinct": len(ns),
        "n_min": float(ns[0]) if ns else float("nan"),
        "n_max": float(ns[-1]) if ns else float("nan"),
        "median_seconds": float(secs[len(secs) // 2]) if secs else float("nan"),
        "pred_error": float(sorted(errors)[len(errors) // 2]) if errors else float("nan"),
    }


def enrich(records: list[dict], row_paths: list[Path]) -> list[dict]:
    """Attach per-step residual and coverage columns from the recorded rows."""
    if not row_paths:
        return records
    buckets = bucket_samples(load_rows([str(p) for p in row_paths]))
    for rec in records:
        samples = buckets.get((rec["task"], rec["cell"], rec["step"]), [])
        rec.update(residuals(rec, samples))
    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _finite(value) -> bool:
    """True for a real number — NaN and None both mean "not measured" here."""
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def _fmt(value: float, digits: int = 2) -> str:
    """Two significant digits, and a visible marker for a missing r²."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def coverage_table(records: list[dict]) -> list[str]:
    """How many steps were fit each way, per profile. Read this before any r²."""
    lines = [
        "| profile | cells | exact cells | steps | affine (has r²) | median fallback | byte rate |",
        "|---|---|---|---|---|---|---|",
    ]
    by_profile: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_profile[rec["profile"]].append(rec)
    for profile, recs in sorted(by_profile.items()):
        kinds = defaultdict(int)
        for rec in recs:
            kinds[rec["kind"]] += 1
        cells = {(r["task"], r["cell"]) for r in recs}
        # The exact-cell count is the column the #3345 embedder fix moves, and
        # it moves from zero: a run that records a blank embedder can only ever
        # populate the rollups.
        exact = {(r["task"], r["cell"]) for r in recs if r["specificity"] == "exact"}
        lines.append(
            f"| `{profile}` | {len(cells)} | {len(exact)} | {len(recs)} | {kinds['affine']} | "
            f"{kinds['median fallback']} | {kinds['byte rate']} |"
        )
    return lines


def step_table(records: list[dict], profile: str) -> list[str]:
    """Per (task, step) r² across cells of one profile, split by specificity."""
    lines = [
        "| task | step | cells | affine | median r² | worst r² | median prediction error |",
        "|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        if rec["profile"] == profile:
            groups[(rec["task"], rec["step"])].append(rec)
    for (task, step), recs in sorted(groups.items()):
        affine = [r for r in recs if r["kind"] == "affine"]
        r2s = sorted(r["r2"] for r in affine)
        errors = sorted(r["pred_error"] for r in recs if not math.isnan(r.get("pred_error", float("nan"))))
        lines.append(
            f"| `{task}` | `{step}` | {len(recs)} | {len(affine)} | "
            f"{_fmt(r2s[len(r2s) // 2]) if r2s else '—'} | "
            f"{_fmt(r2s[0]) if r2s else '—'} | "
            f"{_fmt(errors[len(errors) // 2]) if errors else '—'} |"
        )
    return lines


def specificity_table(records: list[dict]) -> list[str]:
    """r² by cell specificity — the price of the rollups the fitter ships."""
    lines = ["| profile | specificity | affine steps | median r² | median prediction error |", "|---|---|---|---|---|"]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        groups[(rec["profile"], rec["specificity"])].append(rec)
    order = {"exact": 0, "media rollup": 1, "device rollup": 2}
    for (profile, spec), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
        affine = sorted(r["r2"] for r in recs if r["kind"] == "affine")
        errors = sorted(r["pred_error"] for r in recs if not math.isnan(r.get("pred_error", float("nan"))))
        lines.append(
            f"| `{profile}` | {spec} | {len(affine)} | "
            f"{_fmt(affine[len(affine) // 2]) if affine else '—'} | "
            f"{_fmt(errors[len(errors) // 2]) if errors else '—'} |"
        )
    return lines


def verdict_table(records: list[dict], profile: str) -> list[str]:
    """The answer the issue asked for: which steps are affine in n, which are not."""
    lines = ["| task | step | verdict | evidence |", "|---|---|---|---|"]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        if rec["profile"] == profile and rec["specificity"] == "exact":
            groups[(rec["task"], rec["step"])].append(rec)
    for (task, step), recs in sorted(groups.items()):
        affine = sorted(r["r2"] for r in recs if r["kind"] == "affine")
        kinds = {r["kind"] for r in recs}
        if not affine and kinds == {"byte rate"}:
            verdict, evidence = "not fitted as a line", "byte-scaled by design (seconds per MB)"
        elif not affine:
            # Two different diagnoses hide behind one branch, and the report is
            # useless if it cannot tell them apart. `n_distinct == 1` means the
            # sweep never varied n and the fitter was never given a chance. Any
            # more than that means it *was* given the chance and found no slope
            # worth reporting — which, when the step's typical cost is ~0, means
            # the sweep measured the cheap occurrence of a once-per-process cost.
            spreads = sorted({r.get("n_distinct", 0) for r in recs})
            secs = sorted(r["median_seconds"] for r in recs if _finite(r.get("median_seconds")))
            typical = secs[len(secs) // 2] if secs else float("nan")
            if max(spreads or [0]) <= 1:
                verdict = "never varied in n"
                evidence = f"one n per cell in {len(recs)} cells — no line was attempted"
            elif _finite(typical) and typical < 0.05:
                verdict = "measured as free"
                evidence = (
                    f"n_distinct {spreads} but median cost {_fmt(typical, 3)} s — the sweep caught the warm occurrence"
                )
            else:
                verdict = "no credible slope"
                evidence = f"median fallback in {len(recs)} cells, n_distinct {spreads}, {_fmt(typical)} s typical"
        else:
            median = affine[len(affine) // 2]
            verdict = "AFFINE" if median >= GOOD_R2 else ("marginal" if median >= POOR_R2 else "NOT affine")
            evidence = f"r² median {_fmt(median)} over {len(affine)}/{len(recs)} fitted cells"
        lines.append(f"| `{task}` | `{step}` | **{verdict}** | {evidence} |")
    return lines


def pacing_table(profile_paths: dict[str, Path], task: str, cases: list[dict]) -> list[str]:
    """What each profile does to the progress bar it was fitted to pace.

    r² and APE describe the coefficients. This describes the *consequence*: the
    normalized weight vector ``step_weights`` hands to the tracker, beside the
    shipped default. A step whose fitted cost is zero gets a zero share, and the
    bar then sweeps through a phase that takes real seconds — which is a much
    more legible statement of a fit's quality than any statistic over it.
    """
    spec = TASKS[task]
    lines = [
        "| profile | case | " + " | ".join(f"step {i}" for i in range(1, spec.tracker_steps + 1)) + " |",
        "|---" * (spec.tracker_steps + 2) + "|",
    ]
    # The shipped baseline for `dataset_load` is NOT `step_weights`' default
    # terms — that task deliberately carries none — but the measured affine
    # table in `_load_cost_model`, reached through `load_step_weights`. Comparing
    # a fitted profile against anything else would compare it to a straw man.
    if task == "dataset_load":
        from vtscore.datasets.stages._common import load_step_weights  # noqa: PLC0415

        # `load_step_weights` reads the device from the *process*, not from an
        # argument, so this row is only comparable with the profile rows when
        # the analysis runs on the same kind of machine the sweep did. Naming
        # the resolved device in the label is what makes that checkable rather
        # than assumed.
        resolved = profile_mod.resolve_device_name()
        for case in cases:
            kw = case["kwargs"]
            shipped = load_step_weights(
                kw.get("media_type", ""),
                n=int(kw.get("n", 0)) or None,
                download_size_mb=kw.get("size_mb"),
                embedder=kw.get("embedder", ""),
            )
            total = sum(shipped) or 1.0
            cells = " | ".join(f"{w / total:.2f}" for w in shipped)
            lines.append(f"| **shipped `LOAD_COST_MODEL`** (device `{resolved}`) | {case['label']} | {cells} |")

    for label, path in sorted(profile_paths.items()):
        if path is not None and not path.is_file():
            continue
        try:
            profile_mod.reload_profile(str(path) if path else "")
            for case in cases:
                weights = profile_mod.step_weights(task, **case["kwargs"])
                cells = " | ".join(f"{w:.2f}" for w in weights) if weights else " | ".join(["—"] * spec.tracker_steps)
                name = f"`{path.stem}`"
                lines.append(f"| {name} | {case['label']} | {cells} |")
        finally:
            profile_mod.reload_profile("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True, help="experiment dir holding the profiles and rows")
    ap.add_argument("--out", default="", help="summary JSON (default <exp>/summary_timing3345.json)")
    ap.add_argument("--md", default="", help="markdown tables (default <exp>/tables.md)")
    ap.add_argument(
        "--resweep-dir",
        default="/exp/sgreenberg/resweep-3062",
        help="dir holding the #3062 resweep .filt.jsonl rows",
    )
    args = ap.parse_args()

    exp = Path(args.exp)
    fixed = exp / "fixed"
    sources = {
        "profile_generic": [exp / "rows.jsonl"],
        "profile_loadprof": [exp / "loadprof.jsonl"],
        "profile_resweep": sorted(Path(args.resweep_dir).glob("*.filt.jsonl")),
        # Leg 2: the same eight imports re-measured on the fixed tree. Named
        # apart so the before/after is visible in every table rather than
        # averaged into the totals.
        "profile_fixed": [fixed / "rows.jsonl"],
        "profile_fixed_loadprof": [fixed / "loadprof.jsonl"],
    }
    locations = {"profile_fixed": fixed, "profile_fixed_loadprof": fixed}

    records: list[dict] = []
    for name, row_paths in sources.items():
        path = locations.get(name, exp) / f"{name}.json"
        if not path.is_file():
            print(f"SKIPPED {path}: not written", file=sys.stderr)
            continue
        recs = read_profile(path)
        records.extend(enrich(recs, [p for p in row_paths if p.is_file()]))

    if not records:
        print("no profiles to read", file=sys.stderr)
        return 1

    md: list[str] = []
    md.append("## Fit kinds — read this before any r²\n")
    md.extend(coverage_table(records))
    md.append("\n## Which steps are affine in *n* (exact cells, generic recorder)\n")
    md.extend(verdict_table(records, "profile_generic"))
    md.append("\n## Which steps are affine in *n* (exact cells, #3062 resweep)\n")
    md.extend(verdict_table(records, "profile_resweep"))
    for profile in sorted({r["profile"] for r in records}):
        md.append(f"\n## Per-step detail — `{profile}`\n")
        md.extend(step_table(records, profile))
    md.append("\n## The price of the rollup cells\n")
    md.extend(specificity_table(records))

    md.append("\n## What each profile does to the bar it was fitted to pace\n")
    md.append(
        "Normalized `step_weights` for `dataset_load`. Step 2 is the model load; step 3 the embed; step 4 finalize.\n"
    )
    md.extend(
        pacing_table(
            {
                "profile_generic": exp / "profile_generic.json",
                "profile_fixed": fixed / "profile_fixed.json",
                "profile_resweep": exp / "profile_resweep.json",
            },
            "dataset_load",
            [
                {
                    "label": "image/siglip, n=412, 131 MB",
                    "kwargs": {
                        "device": "cuda",
                        "media_type": "image",
                        "embedder": "siglip",
                        "n": 412,
                        "size_mb": 131.0,
                    },
                },
                {
                    "label": "audio/clap_general, n=245, 600 MB",
                    "kwargs": {
                        "device": "cuda",
                        "media_type": "audio",
                        "embedder": "clap_general",
                        "n": 245,
                        "size_mb": 600.0,
                    },
                },
            ],
        )
    )

    md_path = Path(args.md or exp / "tables.md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    out_path = Path(args.out or exp / "summary_timing3345.json")
    out_path.write_text(json.dumps({"records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {md_path}\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
