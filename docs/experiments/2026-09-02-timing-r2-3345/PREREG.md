# Pre-registration: read the r² the timing fitter keeps (#3345)

**Written 2026-09-02, before the run** (job 609428 was queued at 10:43 EDT and
this was written while it queued). One thing was already known and is declared
here rather than presented later as a prediction: **the #3062 resweep rows had
already been refit locally through `vtscore.timing.fit` before this was
written**, so the `dataset_load` half of H1 and all of H4 were *seen* first.
They are recorded below as observations, not predictions. H2, H3 and H5 concern
the generic recorder, which had never been run, and are genuine predictions.

## What the issue asked for, and why it cannot be done as written

[#3345](https://github.com/samggreenberg/VTSearch/issues/3345) says: run **one
dataset load** with the recorder set, then read the per-step r². Two corrections
had to be made before anything could be launched.

1. **The env var does not exist.** The issue and the [#3329
   report](../2026-08-30-fit-quality-3329/REPORT.md) both name
   `VTSEARCH_RECORD_TIMING`. The recorder's variable is `VTSEARCH_TIMING_RECORD`
   (`vtscore/timing/recorder.py:53`, and `docs/DEPLOYMENT.md:127`). The
   transposition appears nowhere in the code, so an admin following the report
   would have armed nothing and recorded nothing — silently, since an unarmed
   recorder is the normal state.

2. **One dataset load produces no r² at all.** r² is only carried when
   `fit_step` takes the OLS branch, and that branch needs *spread in `n`*:
   `affine_fit` returns `(mean, 0, 0)` the moment `sxx <= 1e-9`, `fit_step` then
   sees `slope <= 0` and returns the median fallback, which deliberately drops
   the r². A single load gives one `n` per cell, so **every step in the profile
   would come back a median with no r² to read**, and `fit_profile`'s
   `min_samples=2` would drop most cells before that. The experiment as
   specified returns an empty answer that looks like a measurement.

So the design here is: **four size tiers per media type**, which is the smallest
grid that can answer the question the issue was actually asking.

The report's third claim — "there is no recorded timing profile on this cluster
to read an r² off" — is also only half true. There is no *profile*, but there
are **4 771 rows** of recorded load timings from the #3062 resweep sitting on
`/exp/sgreenberg/resweep-3062/`, and `normalize_row` exists precisely to fold
that older row shape in. Refitting them costs nothing and covers five media
types, which no fresh run of this size can.

## The three profiles

| profile | source | families | what it isolates |
|---|---|---|---|
| `profile_generic` | this run, `VTSEARCH_TIMING_RECORD` | `dataset_load`, `dataset_stage`, `dataset_open`, `text_sort`, `dataset_promote` | the recorder that has never been run on real data |
| `profile_loadprof` | this run, `VTSEARCH_PROFILE_LOAD` | `dataset_load` | **the same loads**, so recorder-vs-recorder is not a machine comparison |
| `profile_resweep` | #3062 resweep rows, refit | `dataset_load` | the widest `n` spread on disk: 17 datasets, 5 media types |

`profile_loadprof` is the arm that makes this a study rather than a reading.
Both recorders watch the same eight imports on the same node in the same
process, so any difference in fit quality between them is a property of **what
each recorder records**, not of the hardware or the workload.

## The grid

| axis | values |
|---|---|
| media × embedder | `image`/`siglip` (default), `audio`/`clap_general` (default) |
| size tiers | `caltech101_{s,m,l,a}` = 412 / 838 / 1704 / 2954 items |
| | `esc50_{s,m,l,a}` = 245 / 588 / 1127 / 1960 items |
| node | one V100 (`gpu:v100:1`), 8 CPUs, cuML present → device key `cuda+cuml` |

Four tiers, one rep. The spread that makes an r² possible comes from the
**sizes**, not from repetition — two runs at the same `n` add no x-variance and
would still collapse the fit to a median. The read-only families get `--reps 2`
because they are cheap and their per-run noise is what their intercepts are
made of.

`find`, `detector_load` and `train_and_score` are **expected to be skipped**: a
scratch data dir holds no detectors. That is a coverage gap in the result, and
it is declared here so it is not read afterwards as a failure.

## BLUF — the prediction

**The steps that dominate a progress bar are the steps the fitter cannot fit,
and the cell that always matches is the cell that predicts worst.** The r²
that #3334 started keeping will be excellent exactly where nobody needed
reassurance and absent exactly where the bar's error comes from.

| # | claim | bar | status |
|---|---|---|---|
| H1 | `embed` and `finalize` are genuinely affine in `n` | median r² ≥ 0.90 across exact cells, both steps | **seen** for `dataset_load` in the resweep refit; predicted for the generic recorder |
| H2 | the **model-load** step is never fitted as a line | `load` (and `text_sort`'s `load_model`) is a median fallback in ≥ 80 % of exact cells | prediction |
| H3 | the generic recorder fits `load` **worse** than the legacy profiler does, on the same loads | `profile_loadprof` yields more affine `load` cells, or a lower median APE on it, than `profile_generic` | prediction |
| H4 | the rollup cells cost real accuracy | median APE rises monotonically exact → media rollup → device rollup, and the device rollup exceeds 0.20 | **seen** in the resweep refit |
| H5 | a high r² does not imply a well-paced step | at least one step has median r² ≥ 0.90 **and** median APE > 0.20, or the converse | prediction |

**H2 and H3 are pre-registered as a pair**, for the same reason #3329's B4/B5
were. "The model load is unfittable" is a claim about the *world* (a cold weight
load costs what it costs regardless of `n`); "this recorder cannot fit it" is a
claim about the *instrument*. Only both readings together say which one is being
observed, and the legacy profiler — which stamps `cold_model` on each row and so
can hold the cold and warm populations apart — is the control that separates
them.

**H5 is the #3329 lesson applied to its own action item.** That report's most
durable finding was that "is this a good fit?" and "is this fit doing its job?"
are close to opposite questions. r² is the first. The thing a progress bar needs
is the second, so every fitted step is also scored on the median absolute
percentage error of `StepCoeffs.seconds()` against the samples it was fit from.
If r² and APE disagree anywhere, **the r² #3334 persisted is not the statistic
this profile should be read for**, and saying so is the most useful outcome
available here.

## What is measured

Per `(task, cell, step)` in each profile:

- **`kind`** — `affine` (an r² is present), `median fallback` (no slope was
  credible), or `byte rate` (`download`/`extract`, fit as seconds per MB and so
  never a line against `n`). *The counts of these are reported before any r²*,
  because a missing r² is not a bad fit and the two readings are constantly
  confused.
- **`r2`** — as persisted by `StepCoeffs.to_json`.
- **`pred_error`** — median `|predicted − observed| / observed` over the step's own
  samples, using the shipped `StepCoeffs.seconds()`.
- **`n_distinct`, `n_min`, `n_max`, `median_seconds`** — the coverage behind
  the number. A cell whose `n` never moved cannot have been fit as a line
  whatever it reports, and a step that costs 0.2 s does not deserve a verdict.

## Limits, declared now

- **One node, one afternoon.** rack7n06 (V100-SXM2, E5-2698 v4, 8 CPUs). The
  resweep arm's GPU rows came from the same node family, which is what makes
  the two comparable at all; nothing here transfers to the L40S or H200 nodes.
- **Two media types in the fresh run.** Image and audio only — the text and
  video sources are not in the shared demo cache. The resweep arm carries all
  five, which is the division of labour between the two arms.
- **Default embedders only.** `drive_dataset_load` passes `embedder: ""`, so
  each media type is measured at its default. `b_fin` tracks embedding
  dimension (#3062), so a one-embedder-per-media grid says nothing about how
  finalize's slope moves across encoders.
- **APE is in-sample.** Every residual is measured against the samples the
  coefficient was fit from, so it is a floor on the error a real load would
  see, never an estimate of it. It is the right statistic for "does this fit
  describe its own data", and the wrong one for "how far off will the bar be".
