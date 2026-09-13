# `gmm_init/` — the #3585 gate on the unanchored mixture fit

The question: `fit_score_gmm` fitted two Gaussians over one dimension with
sklearn's `GaussianMixture`, and that call is the largest single term in a
cosine/text sort and the larger half of a calibration fold. Replacing it is
small. Establishing that the replacement is safe is not, because **it moves the
fit** — a different init lands EM elsewhere inside its tolerance, and that
reaches the green/red line through the fold quantile, `np.quantile` and
`snap_cut_to_sample`.

The report is
[`docs/experiments/2026-09-13-gmm-init-3585/REPORT.md`](../../../docs/experiments/2026-09-13-gmm-init-3585/REPORT.md).

## The shape of it

**Capture once, gate many times.** The corpus is arrays, not verdicts, so a
candidate that did not exist when the grid ran can still be gated against the
same real inputs — which is also why the pre-#3585 fit stays in the tree as
`fit_score_gmm_sklearn` rather than only in git history.

| Stage | Script | What it produces |
|---|---|---|
| capture (fold path) | `capture_folds_3585.py` | one `.npz` per cell: every input the shipped fold-anchored fit saw, on a stride |
| capture (sort path) | `capture_sorts_3585.py` | one `.npz` per prepare grid: each query's whole cosine-sort score array, plus the sort's own cost and both fits timed in the same process |
| gate | `gate_3585.py` | `gate_cuts.csv` (threshold + admitted set per arm) and `gate_fits.csv` (fitted parameters + log-likelihood per arm) |
| cost | `bench_3585.py` | min-of-k per-call timings at the sample's own size and at resampled sizes |
| tables | `analyze_3585.py` | the gate, environment, estimator, latency and cost tables |
| figures | `figures_3585.py` | the trade-off scatter, the change ECDF, cost against *n* |
| trajectory A/B | `run_cells_arm_3585.py` | an ordinary cell with one named fit installed for the whole run, for `analyze_ab.py` |
| the arms | `arms_3585.py` | every candidate, and `swap_fit`, which installs one everywhere |
| self-test | `selftest_analyze_3585.py` | planted answers for the analyzer's joins and counts |

`launch_gmm_3585.sh` drives all of it on SLURM (`list`, `size`, `capture`,
`sorts`, `gate`, `bench`, `ab`, `abanalyze`, `status`). Set `DEP=afterany:<id>`
to chain a stage behind a running one GRID-side.

## Two things to know before reusing this

**`swap_fit` patches every binding, not the definition.** `fit_score_gmm` is
imported by name in several modules and fetched off the package by the ones that
import it inside a function, so replacing it in `thresholds.gmm` alone leaves
half the call sites on the old fit — silently, and in the direction that makes
an arm look like the baseline. `run_cells_arm_3585.py` asserts the swap took
before spending an hour measuring the wrong arm.

**An arm must reach the implementation directly.** An arm that looked
`fit_score_gmm` up at call time called *itself* once installed, and the first
real corpus turned that into a `RecursionError` rather than a wrong number —
which was luck. `_NATIVE_FIT` is bound at import for that reason.
