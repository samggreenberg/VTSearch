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

## `arms_3825.py` and friends — the #3825 gate on the **anchored** refit

Same corpus, a different thing swapped. #3585 replaced the *initialiser*;
measuring the loop next to it showed that `_anchored_em` had become ~90% of a
calibration fold's fit, running 97-200 iterations of a parameter-delta
criterion at 1e-8 — and on a large minority of real folds exiting on `max_iter`
rather than converging at all.

**Nothing needed re-running to ask this.** The 84 cells of
`capture_folds_3585.py` output are every array the shipped fold path saw, and
the anchored refit is exactly what consumes them. That is what capturing
*arrays* instead of *verdicts* buys: a candidate that did not exist when the
grid ran, gated against inputs nobody re-ran.

| Stage | Script | What it produces |
|---|---|---|
| gate | `gate_3825.py` | `gate3825_cuts.csv` (the shipped chain's threshold + admitted set per arm), `gate3825_fits.csv` (per-fold refit: iterations, convergence, both objectives, init/refit cost split), `gate3825_trace.csv` (the objective, iteration by iteration) |
| cost | `bench_3825.py` | min-of-k per-call cost, init and refit timed separately in one process |
| tables | `analyze_3825.py` | the gate, incumbent, estimator, monotonicity and cost tables |
| figures | `figures_3825.py` | the trade-off scatter, the iteration boxplot, the move ECDF, the flat ridge |
| trajectory A/B | `run_cells_arm_3825.py` | an ordinary cell with one stopping rule installed for the whole run |
| the arms | `arms_3825.py` | every rule, and `swap_anchored_stop`, which installs one |
| self-test | `selftest_analyze_3825.py` | planted answers for the analyzer's joins, rates and counts |

`launch_3825.sh` drives it (`gate`, `bench`, `ab`, `abanalyze`, `analyse`,
`status`).

### Three things to know before reusing this

**The arm intercepts the loop, not the fit.** `swap_anchored_stop` wraps
`_anchored_em` and substitutes the stopping rule **only when the call has
anchors**. `_plain_em` — the unanchored init every anchored fit runs first — is
the same loop with both anchor arrays empty and must keep the rule *it* ships
with, or an arm measures two changes and attributes both to the refit.

**Which likelihood is a second axis, not a detail.** An anchored EM ascends a
*weighted semi-supervised* objective; the free sample's likelihood is a
different quantity with no theorem making it monotone under that M-step.
Borrowing the sort path's rule would have watched the second one. Both are arms,
`monotonicity.csv` is the check, and the two coincide exactly when there are no
anchors — which is why `_plain_em` is unaffected either way, bit for bit.

**There is no cushion here.** #3585's own gate found the fold path forgiving,
but that was with the anchored refit *unchanged*, re-converging from whatever
init it was handed. This refit **is** the shipped threshold: between it and the
green/red line there is a quantile and a snap, and nothing else. Two arms
(`param1e-6`, `iter400`) exist purely so the candidates' numbers can be read
against what perturbing the incumbent does.
