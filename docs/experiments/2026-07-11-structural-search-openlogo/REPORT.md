# Structural search in the wild — VTSearch × OpenLogo

**2026-07-11.** The full write-up is the self-contained page beside this file:
[`report.html`](report.html) (open it in a browser — GitHub renders raw HTML as
source). This study predates the CSV-and-figures convention, so the page *is*
the record; there are no cell CSVs to re-analyse.

**Question.** Seeded with one real-world logo crop over 27k unstaged photos: how
far does SIFT+VLAD get, does 6-DoF beat 4-DoF, and what do labels buy?

**Verdict.** Stage-1 VLAD recall is the ceiling — 2.5% of true instances reach
the top-50 — and SigLIP cosine is ~10× stronger. 4-DoF wins. On the structural
path labels are calibration, not learning. The run also caught the deterministic
**30th-vote transient** in the shared MLP trainer.

Cited from [`docs/plans/structural-embedder.md`](../../plans/structural-embedder.md).
