# Tuning the map — per-embedder UMAP defaults for VTSBrowse

**2026-07-22.** The full write-up is the self-contained page beside this file:
[`report.html`](report.html) (open it in a browser — GitHub renders raw HTML as
source). This study predates the CSV-and-figures convention, so the page *is*
the record; there are no cell CSVs to re-analyse. The analysis code is committed
under `scripts/experiments/umap_params/`.

**Question.** How should UMAP's dials be set for the Browse projection?
~5,000 scored fits over 23 embed sets.

**Verdict.** `n_neighbors` tracks the embedder (10 image / 15 audio), `min_dist`
barely matters, and compaction consistently costs layout quality — so it ships
off.

Cited from [`docs/plans/vtsbrowse-empirical-tuning.md`](../../plans/vtsbrowse-empirical-tuning.md)
and [`docs/plans/vtsbrowse.md`](../../plans/vtsbrowse.md).
