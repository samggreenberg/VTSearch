# Structural search on screenshots & scans

**2026-07-13.** The full write-up is the self-contained page beside this file:
[`report.html`](report.html) (open it in a browser — GitHub renders raw HTML as
source). This study predates the CSV-and-figures convention, so the page *is*
the record; there are no cell CSVs to re-analyse.

**Question.** The [OpenLogo](../2026-07-11-structural-search-openlogo/REPORT.md)
follow-up on flat rasters.

**Verdict.** SIFT is the bottleneck on line art (5.1% vs SuperPoint+LightGlue's
41% true-pair verify); SP+LG as ranker beats SigLIP on both document corpora —
the first time structural search wins on a real corpus. Also: 3-DoF is a free
precision win, LightGlue needs its own inlier floor (~24), and 224 px tiling
rescues small targets.

Cited from [`docs/plans/structural-embedder.md`](../../plans/structural-embedder.md).
