#!/usr/bin/env python
"""Figures for #3345 — what the timing profile's r² does and does not say.

Four, each carrying one of the run's claims:

1. **Fit kinds per step** — how many cells got an affine fit, a median
   fallback, or a per-MB rate. This has to come first: an absent r² is not a bad
   fit, and a bar chart of r² alone would silently drop every step that was
   never fitted as a line.
2. **r² against prediction error** — the #3329 question applied to its own
   action item. If the two agreed, the top-left quadrant would be empty.
3. **The price of the rollups** — error by cell specificity. The fitter's
   docstring justifies the rollups on the grounds that the least-specific cell
   always matches; this is what that guarantee costs.
4. **Observed against predicted seconds** — the fits themselves, per step, so a
   reader can see which points a coefficient is actually made of.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DPI = 130
KIND_COLOURS = {
    "affine": "#2b6cb0",
    "median fallback": "#d68910",
    "byte rate": "#7f8c8d",
}
SPEC_COLOURS = {
    "exact": "#2b6cb0",
    "media rollup": "#d68910",
    "device rollup": "#c0392b",
}
SPEC_ORDER = ("exact", "media rollup", "device rollup")
GOOD_R2 = 0.90
GOOD_PRED_ERROR = 0.20


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load(summary: Path) -> list[dict]:
    return json.loads(summary.read_text(encoding="utf-8"))["records"]


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def fit_kinds(records: list[dict], outdir: Path) -> list[str]:
    """Stacked composition of fit kinds per (task, step), one panel per profile."""
    plt = _plt()
    profiles = sorted({r["profile"] for r in records})
    fig, axes = plt.subplots(1, len(profiles), figsize=(4.6 * len(profiles), 4.6), layout="constrained", squeeze=False)
    for ax, profile in zip(axes[0], profiles):
        recs = [r for r in records if r["profile"] == profile]
        labels = sorted({f"{r['task']}\n{r['step']}" for r in recs})
        bottoms = [0.0] * len(labels)
        for kind, colour in KIND_COLOURS.items():
            counts = [
                sum(1 for r in recs if f"{r['task']}\n{r['step']}" == label and r["kind"] == kind) for label in labels
            ]
            ax.barh(labels, counts, left=bottoms, color=colour, label=kind, height=0.7)
            bottoms = [b + c for b, c in zip(bottoms, counts)]
        ax.set_title(f"`{profile}`", fontsize=10)
        ax.set_xlabel("profile cells")
        ax.tick_params(labelsize=8)
        ax.invert_yaxis()
    axes[0][0].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("How each step was fitted — an absent r² is a branch, not a bad fit", fontsize=11)
    out = outdir / "fit_kinds_by_step.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out.name]


def r2_vs_error(records: list[dict], outdir: Path) -> list[str]:
    """r² against in-sample median APE. The disagreement is the finding."""
    plt = _plt()
    pts = [r for r in records if r["kind"] == "affine" and _finite(r.get("r2")) and _finite(r.get("pred_error"))]
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 5.0), layout="constrained")
    # Rescoring a clamped fit can land r2 well below zero; clip the view so the
    # populated 0-1 band stays readable, and say how many fell off.
    floor = -1.0
    clipped = sum(1 for r in pts if r["r2"] < floor)
    for spec in SPEC_ORDER:
        sel = [r for r in pts if r["specificity"] == spec]
        if not sel:
            continue
        ax.scatter(
            [max(r["r2"], floor) for r in sel],
            [max(r["pred_error"], 1e-4) for r in sel],
            s=[18 + 40 * min(1.0, math.log10(max(r.get("median_seconds", 1.0), 1.0)) / 2.5) for r in sel],
            c=SPEC_COLOURS[spec],
            alpha=0.72,
            edgecolors="none",
            label=spec,
        )
    ax.axvline(GOOD_R2, color="#555", ls="--", lw=0.9)
    ax.axhline(GOOD_PRED_ERROR, color="#555", ls="--", lw=0.9)
    ax.set_xlim(floor - 0.08, 1.06)
    ax.set_yscale("log")
    if clipped:
        ax.annotate(
            f"{clipped} at or below r² {floor:.0f} (clipped)",
            xy=(0.03, 0.62),
            xycoords="axes fraction",
            fontsize=7.5,
            color="#c0392b",
        )
    ax.set_xlabel("r² kept by `StepCoeffs` (goodness of the line)")
    ax.set_ylabel("median |predicted − observed| / observed  (what the bar feels)")
    ax.set_title("A good fit and a well-paced step are not the same claim", fontsize=11)
    # The two quadrants that carry the claim, labelled where they actually sit.
    ax.annotate(
        "fits well,\nstill mis-paced",
        xy=(0.97, 0.88),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.5,
        color="#c0392b",
    )
    ax.annotate(
        "fits badly,\npaces fine",
        xy=(0.03, 0.30),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7.5,
        color="#2b6cb0",
    )
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    ax.grid(alpha=0.25, lw=0.5)
    out = outdir / "r2_vs_prediction_error.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out.name]


def rollup_cost(records: list[dict], outdir: Path) -> list[str]:
    """Error by cell specificity — the price of the cell that always matches."""
    plt = _plt()
    profiles = sorted({r["profile"] for r in records})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")

    ax = axes[0]
    width = 0.8 / max(len(profiles), 1)
    for i, profile in enumerate(profiles):
        vals = []
        for spec in SPEC_ORDER:
            sel = sorted(
                r["pred_error"]
                for r in records
                if r["profile"] == profile and r["specificity"] == spec and _finite(r.get("pred_error"))
            )
            vals.append(sel[len(sel) // 2] if sel else float("nan"))
        ax.bar(
            [x + i * width - 0.4 + width / 2 for x in range(len(SPEC_ORDER))],
            vals,
            width=width,
            label=f"`{profile}`",
        )
    ax.axhline(GOOD_PRED_ERROR, color="#555", ls="--", lw=0.9)
    ax.set_xticks(range(len(SPEC_ORDER)))
    ax.set_xticklabels(SPEC_ORDER)
    ax.set_ylabel("median prediction error")
    ax.set_title("Cost of falling through to a broader cell", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    ax = axes[1]
    # r2 can be negative once a clamped fit is rescored against the coefficients
    # actually stored ("worse than predicting the mean"), and a single -7 would
    # otherwise flatten every other point onto one line. Clip the view and say
    # how many fell off, rather than dropping them silently.
    floor = -1.0
    clipped = 0
    for i, spec in enumerate(SPEC_ORDER):
        sel = [r["r2"] for r in records if r["specificity"] == spec and r["kind"] == "affine" and _finite(r.get("r2"))]
        if not sel:
            continue
        clipped += sum(1 for v in sel if v < floor)
        ax.scatter(
            [i + 0.16 * ((j % 7) - 3) / 3 for j in range(len(sel))],
            [max(v, floor) for v in sel],
            s=16,
            c=SPEC_COLOURS[spec],
            alpha=0.7,
            edgecolors="none",
            marker="o",
        )
    ax.set_ylim(floor - 0.08, 1.06)
    ax.axhline(0.0, color="#999", lw=0.8)
    if clipped:
        ax.annotate(
            f"{clipped} below {floor:.0f} (clipped)",
            xy=(0.02, 0.03),
            xycoords="axes fraction",
            fontsize=7.5,
            color="#c0392b",
        )
    ax.axhline(GOOD_R2, color="#555", ls="--", lw=0.9)
    ax.set_xticks(range(len(SPEC_ORDER)))
    ax.set_xticklabels(SPEC_ORDER)
    ax.set_ylabel("r²")
    ax.set_title("…and what it does to the fit statistic", fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    out = outdir / "rollup_cost.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out.name]


def observed_vs_predicted(
    records: list[dict], rows_path: Path, outdir: Path, profile: str = "profile_generic"
) -> list[str]:
    """The fits themselves: one panel per step of the generic recorder's run."""
    import sys

    repo = Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from vtscore.timing.fit import load_rows, normalize_row
    from vtscore.timing.profile import StepCoeffs

    if not rows_path.is_file():
        return []
    plt = _plt()
    samples = [n for n in (normalize_row(r) for r in load_rows([str(rows_path)])) if n and not n["slot"]]
    # The most specific cells this profile actually has. Leg 1 has none that are
    # exact — that is the defect it measured — so falling back to the media
    # rollup is what keeps the figure from being empty for the very run whose
    # point is that the exact cells are missing.
    pool = [r for r in records if r["profile"] == profile]
    for level in ("exact", "media rollup", "device rollup"):
        exact = [r for r in pool if r["specificity"] == level]
        if exact:
            break
    keys = sorted({(r["task"], r["step"]) for r in exact})
    keys = [k for k in keys if any(s["task"] == k[0] and s["step"] == k[1] for s in samples)]
    if not keys:
        return []
    cols = 4
    rows_n = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 2.9 * rows_n), layout="constrained", squeeze=False)
    # One colour per cell. A panel must never draw one cell's line through
    # another cell's points: an image import at 0.014 s/item and an audio one at
    # 0.102 s/item share a `(task, step)` and nothing else, and pooling them is
    # the very confusion the rollup section is about.
    cell_colours = ("#2b6cb0", "#c0392b", "#38a169", "#8e44ad", "#d68910")
    for ax, (task, step) in zip([a for row in axes for a in row], keys):
        cells = [r for r in exact if r["task"] == task and r["step"] == step]
        labels: list[str] = []
        for idx, rec in enumerate(cells):
            device, media, embedder = (rec["cell"].split("|") + ["", ""])[:3]
            pts = [
                s
                for s in samples
                if s["task"] == task and s["step"] == step and s["media_type"] == media and s["embedder"] == embedder
            ]
            if not pts:
                continue
            colour = cell_colours[idx % len(cell_colours)]
            ax.scatter([p["n"] for p in pts], [p["seconds"] for p in pts], s=18, c=colour, alpha=0.8, zorder=3)
            coeffs = StepCoeffs(a=rec["a"], b=rec["b"], per_mb=rec["per_mb"])
            xs = sorted({p["n"] for p in pts})
            if len(xs) >= 2:
                mb = pts[0]["size_mb"]
                ax.plot(xs, [coeffs.seconds(n=x, size_mb=mb) for x in xs], lw=1.2, color=colour, zorder=2)
            r2 = float(rec.get("r2", float("nan")))
            shown = embedder or media or device
            labels.append(f"{shown} r² —" if math.isnan(r2) else f"{shown} r² {r2:.2f}")
        ax.set_title(f"{task}·{step}\n" + "  ·  ".join(labels), fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.22, lw=0.5)
    for ax in [a for row in axes for a in row][len(keys) :]:
        ax.axis("off")
    fig.supxlabel("n (items)", fontsize=9)
    fig.supylabel("seconds", fontsize=9)
    fig.suptitle(f"Every fitted step of `{profile}`, with its line", fontsize=11)
    out = outdir / f"observed_vs_predicted_{profile.replace('profile_', '')}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out.name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True, help="summary_timing3345.json from the analyzer")
    ap.add_argument("--rows", default="", help="the generic recorder's rows.jsonl")
    ap.add_argument("--fixed-rows", default="", help="the post-fix leg's rows.jsonl")
    ap.add_argument("--outdir", required=True, help="directory to write the PNGs into")
    args = ap.parse_args()

    records = _load(Path(args.summary))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    written += fit_kinds(records, outdir)
    written += r2_vs_error(records, outdir)
    written += rollup_cost(records, outdir)
    if args.rows:
        written += observed_vs_predicted(records, Path(args.rows), outdir, profile="profile_generic")
    if args.fixed_rows:
        written += observed_vs_predicted(records, Path(args.fixed_rows), outdir, profile="profile_fixed")
    for name in written:
        print(f"wrote {outdir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
