"""Do the RANSAC fit statistics tell a true match from a false one? (#3349)

#3349 left the structural fit's statistics unvalidated because every corpus on
the cluster was category-labelled, and on such a corpus a "positive pair" is two
different objects that share no structure. FullMarks is instance-level (a class
is one physical stamp or logo), so here a positive pair is real.

Reads #4162's template matrices: every template of a class (the query crop and
each positive page's box) verified against every pool page, with the 9
match statistics of each fit (``match_stats_to_features``). For pairs (template,
page), labelled by whether the page carries the class, it reports:

* how often RANSAC returns a sane model (``model_ok``), by label;
* each statistic's distribution for true and false fits (``model_ok`` fits only,
  where the geometry is defined);
* each statistic's AUC at separating them, over all fits and **among the fits
  that clear the shipped 8-inlier gate**. That second case is the hard-negative
  set the gate lets through on documents (#4367). A statistic that separates
  there could sharpen the accept decision;
* the AUC per class, so a pooled number is not mistaken for a uniform one.

A template's own page is excluded (a self-match). Writes ``summary.md``,
``auc.csv``, ``per_class_auc.csv`` and the figures beside them.

    python fit_quality.py --matrix <votes-4162>/matrix-s --out <dir>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

FEATURES = (
    "inlier_count",
    "inlier_ratio",
    "tentative_count",
    "mean_reproj_error",
    "median_reproj_error",
    "scale",
    "reflection",
    "inlier_spread",
    "model_ok",
)
#: Statistics whose *lower* values mean a better fit (AUC is reported as "higher = more positive").
LOWER_IS_BETTER = {"mean_reproj_error", "median_reproj_error"}
GATE = 8
#: Negatives sampled per class and template set, to keep the pooled arrays small.
MAX_NEG = 200_000


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive outscores a random negative (ties count half)."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort").astype(np.float64) + 1
    # average ranks over ties
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    _, first, counts = np.unique(sorted_v, return_index=True, return_counts=True)
    avg = np.empty_like(ranks)
    for f, c in zip(first, counts):
        avg[order[f : f + c]] = (f + 1 + f + c) / 2.0
    r_pos = avg[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def class_pairs(npz: Path, which: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``(stats [P, 9], label [P])`` for one class's pairs; *which* is ``crop`` or ``all`` templates."""
    z = np.load(npz)
    stats = z["stats"].astype(np.float32)  # [T, N, 9]
    pos = z["positives"].astype(bool)
    pool = [str(p) for p in z["pool_ids"]]
    col = {p: i for i, p in enumerate(pool)}
    templates = [0] if which == "crop" else list(range(stats.shape[0]))
    rows, labels = [], []
    for t in templates:
        tid = str(z["template_ids"][t])
        keep = np.ones(len(pool), dtype=bool)
        if tid in col:
            keep[col[tid]] = False  # a template against its own page is a self-match
        rows.append(stats[t][keep])
        labels.append(pos[keep])
    s, y = np.concatenate(rows), np.concatenate(labels)
    neg = np.flatnonzero(~y)
    if neg.size > MAX_NEG:
        drop = rng.choice(neg, size=neg.size - MAX_NEG, replace=False)
        mask = np.ones(len(y), dtype=bool)
        mask[drop] = False
        s, y = s[mask], y[mask]
    return s, y


def decoded(s: np.ndarray) -> dict[str, np.ndarray]:
    """The statistics in their natural units (counts were stored ``log1p``)."""
    out = {f: s[:, i].astype(np.float64) for i, f in enumerate(FEATURES)}
    out["inlier_count"] = np.expm1(out["inlier_count"])
    out["tentative_count"] = np.expm1(out["tentative_count"])
    return out


def feature_auc(d: dict[str, np.ndarray], y: np.ndarray, mask: np.ndarray, f: str) -> float:
    v = d[f][mask]
    if f in LOWER_IS_BETTER:
        v = -v
    return auc(v[y[mask]], v[~y[mask]])


def main(argv: Optional[Sequence[str]] = None) -> int:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    files = [f for f in sorted(args.matrix.glob("*.npz")) if not f.name.startswith("vectors-")]

    lines = [f"{len(files)} classes, matrix `{args.matrix}`.", ""]
    auc_rows: list[dict[str, object]] = []
    per_class: list[dict[str, object]] = []
    pooled: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    for which in ("crop", "all"):
        ss, ys = [], []
        for f in files:
            s, y = class_pairs(f, which, rng)
            if not y.any():
                continue
            d = decoded(s)
            ok = d["model_ok"] > 0.5
            gate = ok & (d["inlier_count"] >= GATE)
            row: dict[str, object] = {"class_id": f.stem.replace("__", "/", 1), "templates": which}
            for feat in ("mean_reproj_error", "inlier_ratio", "inlier_spread", "scale"):
                row[f"{feat}_auc_gate"] = feature_auc(d, y, gate, feat)
            row["gate_pos"] = int((gate & y).sum())
            row["gate_neg"] = int((gate & ~y).sum())
            per_class.append(row)
            ss.append(s)
            ys.append(y)
        s, y = np.concatenate(ss), np.concatenate(ys)
        d = decoded(s)
        pooled[which] = (d, y)
        ok = d["model_ok"] > 0.5
        gate = ok & (d["inlier_count"] >= GATE)
        lines += [
            f"## Templates: {'the query crop only' if which == 'crop' else 'every template (crop + Good boxes)'}",
            "",
            f"{int(y.sum()):,} true pairs, {int((~y).sum()):,} false (negatives capped at {MAX_NEG:,} a class).",
            f"A sane model (`model_ok`): **{ok[y].mean():.2f}** of true pairs, **{ok[~y].mean():.2f}** of false.",
            f"Past the 8-inlier gate: {int((gate & y).sum()):,} true, {int((gate & ~y).sum()):,} false.",
            "",
            "| statistic | median, true (model_ok) | median, false (model_ok) | AUC, all fits | AUC, model_ok | **AUC, past the gate** |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        everything = np.ones(len(y), dtype=bool)
        for feat in FEATURES:
            if feat == "model_ok":
                continue
            a_all = feature_auc(d, y, everything, feat)
            a_ok = feature_auc(d, y, ok, feat)
            a_gate = feature_auc(d, y, gate, feat)
            mt = float(np.median(d[feat][ok & y])) if (ok & y).any() else float("nan")
            mf = float(np.median(d[feat][ok & ~y])) if (ok & ~y).any() else float("nan")
            lines.append(f"| `{feat}` | {mt:.4g} | {mf:.4g} | {a_all:.2f} | {a_ok:.2f} | **{a_gate:.2f}** |")
            auc_rows.append(
                {"templates": which, "statistic": feat, "auc_all": a_all, "auc_model_ok": a_ok, "auc_gate": a_gate}
            )
        lines.append("")

    # per-class spread of the past-the-gate AUC, every-template case
    pc = [r for r in per_class if r["templates"] == "all" and int(r["gate_neg"]) >= 20 and int(r["gate_pos"]) >= 5]
    lines += [
        "## Past the gate, per class (every template; classes with >= 20 false and >= 5 true gate-passing fits)",
        "",
        "| statistic | classes | median AUC | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for feat in ("mean_reproj_error", "inlier_ratio", "inlier_spread", "scale"):
        xs = np.array([float(r[f"{feat}_auc_gate"]) for r in pc])
        xs = xs[~np.isnan(xs)]
        if xs.size:
            lines.append(f"| `{feat}` | {xs.size} | {np.median(xs):.2f} | {xs.min():.2f} | {xs.max():.2f} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, rows in (("auc.csv", auc_rows), ("per_class_auc.csv", per_class)):
        with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # figures: reprojection error and inlier ratio, true vs false, all model_ok fits and past the gate
    d, y = pooled["all"]
    ok = d["model_ok"] > 0.5
    gate = ok & (d["inlier_count"] >= GATE)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7), dpi=130)
    for row, (mask, label) in enumerate(((ok, "all sane fits"), (gate, "fits past the 8-inlier gate"))):
        for col, (feat, bins) in enumerate(
            (("mean_reproj_error", np.linspace(0, 0.012, 61)), ("inlier_ratio", np.linspace(0, 1, 51)))
        ):
            ax = axes[row][col]
            for sel, color, lab in ((y, "#2a78d6", "true pairs"), (~y, "#eb6834", "false pairs")):
                v = d[feat][mask & sel]
                if v.size:
                    ax.hist(v, bins=bins, density=True, histtype="step", linewidth=2, color=color, label=lab)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{feat}: {label}", loc="left", fontsize=10)
            ax.set_xlabel(feat, fontsize=9)
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "distributions.png")
    plt.close(fig)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
