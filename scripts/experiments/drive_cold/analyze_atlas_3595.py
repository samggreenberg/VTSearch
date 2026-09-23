#!/usr/bin/env python3
"""How the coverage-atlas rebuild scales past n = 2954 (#3595).

Reads the ``rows.jsonl`` that ``run_atlas_3595.sbatch`` records (every
``dataset_open`` run carries a ``coverage`` row tagged ``restored`` or
``rebuilt``, and an ``items`` row) and writes:

* ``tables.md``: per (media, n), the rebuild seconds for each rep, its rate in
  s/item, the restore seconds, and the share of an open's bar that the coverage
  step takes on each branch; plus a per-media fit;
* ``figures/rebuild_scaling.png``: rebuild seconds against n, on linear axes
  (a log axis would hide the curvature this study exists to look for), with
  #3521's four points and the old extrapolation overlaid;
* ``figures/coverage_share.png``: the coverage step's share of the bar, per
  branch, against n, with the shipped 0.85 drawn in.

Both are rebuilt from the committed ``measurements/rows.jsonl`` alone:

    python scripts/experiments/drive_cold/analyze_atlas_3595.py \\
        --rows docs/experiments/2026-09-22-atlas-rebuild-3595/measurements/rows.jsonl \\
        --out docs/experiments/2026-09-22-atlas-rebuild-3595
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: #3521's rebuilds, same node class (V100, cuML), caltech101 tiers.
PRIOR_3521 = {412: 0.98, 838: 2.0, 1704: 4.0, 2954: 7.7}
PRIOR_SLOPE = 0.0026
SHIPPED_WEIGHT = 0.85
THRESHOLD = 50_000  # COVERAGE_ATLAS_AUTO_THRESHOLD


def load(path: Path) -> dict:
    """``{(media, n): {"rebuilt": [...], "restored": [...], "items": [...]}}``."""
    cells: dict = defaultdict(lambda: defaultdict(list))
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task") != "dataset_open":
            continue
        key = (r["media_type"], int(r["n"]))
        if r["step"] == "coverage":
            cells[key][r.get("branch") or "unmarked"].append(float(r["seconds"]))
        elif r["step"] == "items":
            cells[key]["items"].append(float(r["seconds"]))
    return cells


def fit(ns: list[float], ys: list[float]) -> dict:
    """Least squares ``y = a + b n`` and through-origin ``y = b n``, with r^2."""
    x, y = np.asarray(ns, float), np.asarray(ys, float)
    b, a = np.polyfit(x, y, 1)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - (a + b * x)) ** 2).sum()) / ss_tot
    b0 = float((x * y).sum() / (x * x).sum())
    return {"a": float(a), "b": float(b), "r2": r2, "b0": b0}


def _share(cov: float, items: float) -> float:
    return cov / (cov + items)


def tables(cells: dict) -> tuple[str, dict]:
    out = ["# #3595 — coverage-atlas rebuild past n = 2954 (generated)\n"]
    out.append(
        "| media | n | rebuild, rep 1 / rep 2 | s/item | restore | items step | "
        "coverage share, rebuilt | coverage share, restored |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    fits: dict = {}
    by_media: dict = defaultdict(lambda: ([], []))
    for (media, n), c in sorted(cells.items()):
        reb, res, items = c.get("rebuilt", []), c.get("restored", []), c.get("items", [])
        if not reb:
            continue
        rb, rs, it = statistics.mean(reb), statistics.mean(res), statistics.median(items)
        for v in reb:
            by_media[media][0].append(n)
            by_media[media][1].append(v)
        out.append(
            f"| {media} | {n} | {' / '.join(f'{v:.1f}' for v in reb)} s | {rb / n:.4f} | "
            f"{rs * 1000:.0f} ms | {it:.1f} s | {_share(rb, it):.2f} | {_share(rs, it):.3f} |"
        )
    out.append("\n## Fits, rebuild seconds against n (every rep a point)\n")
    out.append("| media | points | n range | a + b·n | r² | through origin | predicted at 50 000 |")
    out.append("|---|---:|---|---|---:|---:|---:|")
    for media, (ns, ys) in sorted(by_media.items()):
        f = fit(ns, ys)
        fits[media] = f
        out.append(
            f"| {media} | {len(ns)} | {min(ns)}–{max(ns)} | {f['a']:+.1f} + {f['b']:.4f}·n | {f['r2']:.3f} | "
            f"{f['b0']:.4f} s/item | {f['a'] + f['b'] * THRESHOLD:.0f} s |"
        )
    return "\n".join(out) + "\n", fits


def figures(cells: dict, fits: dict, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    colors = {"image": "#1f6f78", "audio": "#b0402e"}

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for media in ("image", "audio"):
        pts = [(n, v) for (m, n), c in cells.items() if m == media for v in c.get("rebuilt", [])]
        if not pts:
            continue
        ns, vs = zip(*pts)
        ax.scatter(ns, vs, s=28, color=colors[media], label=f"{media}, this run (2 reps)", zorder=3)
        f = fits[media]
        xs = np.linspace(0, THRESHOLD, 50)
        ax.plot(xs, f["a"] + f["b"] * xs, color=colors[media], lw=1, alpha=0.7)
    ax.scatter(
        list(PRIOR_3521), list(PRIOR_3521.values()), marker="x", color="k", s=36, label="#3521 (caltech101)", zorder=4
    )
    xs = np.linspace(0, THRESHOLD, 50)
    ax.plot(xs, PRIOR_SLOPE * xs, ls="--", color="grey", lw=1, label="#3521 fit, extrapolated (0.0026 s/item)")
    ax.axvline(THRESHOLD, color="grey", lw=0.8, ls=":")
    ax.text(THRESHOLD, 5, " auto-build\n threshold", fontsize=7, color="grey", ha="right")
    ax.set_xlabel("items in the dataset")
    ax.set_ylabel("atlas rebuild (s)")
    ax.set_title("Coverage-atlas rebuild, V100 + cuML: linear to 36 500 items", fontsize=10)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(figdir / "rebuild_scaling.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for media in ("image", "audio"):
        keys = sorted(k for k in cells if k[0] == media and cells[k].get("rebuilt"))
        if not keys:
            continue
        ns = [n for _, n in keys]
        it = [statistics.median(cells[k]["items"]) for k in keys]
        reb = [_share(statistics.mean(cells[k]["rebuilt"]), i) for k, i in zip(keys, it, strict=True)]
        res = [_share(statistics.mean(cells[k]["restored"]), i) for k, i in zip(keys, it, strict=True)]
        ax.plot(ns, reb, "o-", color=colors[media], label=f"{media}, atlas rebuilt")
        ax.plot(ns, res, "s--", color=colors[media], alpha=0.6, label=f"{media}, atlas restored")
    ax.axhline(SHIPPED_WEIGHT, color="k", lw=1, ls=":", label="shipped default (0.85)")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("items in the dataset (log)")
    ax.set_ylabel("coverage step's share of the open")
    ax.set_title("What the 0.85 weight should be, per branch", fontsize=10)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(figdir / "coverage_share.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cells = load(args.rows)
    text, fits = tables(cells)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables.md").write_text(text)
    figures(cells, fits, args.out / "figures")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
