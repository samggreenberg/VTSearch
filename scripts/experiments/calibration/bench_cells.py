"""Shared, pure-pandas reading and pairing of overview-benchmark cells.

Deliberately importless of `vtscore` (and of `common.setup_env`): the analysis
and figure scripts only need the CSVs a run left behind, so they can be run on a
laptop against a copied results tree rather than only on the cluster.

Two things live here because the analyzer and the figure script must agree on
them or a table and a figure can disagree about the same claim:

* **what counts as a loaded cell** — a header-only CSV is a *starved* cell (the
  run never found a positive, so no row was ever emitted), not a corrupt one,
  and it must be counted rather than silently skipped.  That reading is
  `_cells_io.load_cells`'; this module only adds the bench study's own
  `embedder_suffix` tagging on top of it.
* **how an arm-vs-arm difference is measured** — paired on (category, seed),
  with a standard error, so the report can say which differences the sample
  actually resolves.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from _cells_io import describe_load
from _cells_io import load_cells as _load_cells

#: Identifies one run of the loop: a (dataset, embedder, category, seed) cell.
CELL_KEY = ["dataset", "embedder", "category", "seed"]


def load_cells(results: Path, embedder_suffix: str = "", quiet: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load a run's per-step rows plus a provenance dict of what was dropped.

    *embedder_suffix* tags an arm that differs from its twin by something the
    row itself does not record - the binary-voting run is `dinov3_patch` in
    every column, and only the run it came from says no box was drawn.
    """
    results = Path(results)
    out, prov = _load_cells(results / "cells", where=f"bench_cells.load_cells({results})")
    if embedder_suffix and not out.empty:
        out["embedder"] = out["embedder"] + embedder_suffix
    prov["results"] = str(results)
    if not quiet:
        print(f"  {results}: {describe_load(prov)}")
    return out, prov


def per_cell_means(deep: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Collapse each cell's deep-regime steps to one row, so pairs are cells."""
    return deep.groupby(CELL_KEY)[metrics].mean().reset_index()


def paired_contrasts(
    deep: pd.DataFrame,
    contrasts: list[tuple[str, str]] | None = None,
    metrics: tuple[str, ...] = ("cost", "average_precision", "auroc"),
) -> pd.DataFrame:
    """Mean difference +- standard error for arm pairs, paired on (category, seed).

    Pairing matters more than the sample size here: with 3 seeds an unpaired
    comparison of two arms is dominated by which categories each drew, and that
    variance is exactly what a paired difference cancels. `contrasts` defaults
    to every embedder pair present in each dataset.
    """
    per = per_cell_means(deep, list(metrics))
    rows = []
    for ds, g in per.groupby("dataset"):
        embs = sorted(g["embedder"].unique())
        pairs = contrasts or [(a, b) for i, a in enumerate(embs) for b in embs[i + 1 :]]
        for a, b in pairs:
            left = g[g["embedder"] == a].set_index(["category", "seed"])
            right = g[g["embedder"] == b].set_index(["category", "seed"])
            common = left.index.intersection(right.index)
            if len(common) < 3:
                continue
            row = {"dataset": ds, "contrast": f"{a} - {b}", "n_cells": len(common)}
            for metric in metrics:
                d = left.loc[common, metric] - right.loc[common, metric]
                row[metric] = f"{d.mean():+.3f} +-{d.sem():.3f}"
            rows.append(row)
    return pd.DataFrame(rows)
