# Safe-threshold GMM study — measuring the blend under region voting (issue #2799)

**Status: harness ready, awaiting a Grid run.** All code for this study is on
`dev`: the eval harness emits the variant rows, the runner has a launch
wrapper, and the analyzer computes every pre-registered deliverable below. The
only open work is running it and writing the verdict.

## Question

The safe-threshold GMM blend (`calculate_safe_threshold`,
`vtscore/training/thresholds.py`) has **full** authority over the decision
threshold below 6 votes and majority authority up to ~13, yet the #2781
calibration study (`docs/experiments/calibration/REPORT.md`) ran every arm with
`safe_thresholds False` — the blend has never been measured under region
voting. Two fixes shipped ahead of measurement, on geometry/theory alone:

- **#2797** (PR #2800): production now fits the GMM on the region **max-pooled**
  per-media scores (the distribution the threshold actually cuts) instead of the
  image-level scores, whose distribution sits systematically lower.
- **#2798** (PR #2801): `calculate_gmm_threshold` now cuts at the two fitted
  components' **equal-density crossing** instead of the midpoint between their
  means (midpoint over-includes when the Bad mode is a wide, heavy
  extreme-value lump — the standing shape under max-pooling).

This study measures what those fixes bought (and whether they cost anything),
plus the one idea #2798 deliberately left unshipped: fitting the GMM in **logit
space** to undo sigmoid saturation before the components are estimated.

## Design

One factorial sweep, computed *within* each simulated voting step so every
variant is paired on the identical model, votes, and held-out test scores. At
each trainable step the harness fits a GMM per (fit-geometry, fit-space) pair,
derives both cut rules from each fit, blends each cut with the step's conformal
threshold on the production label ramp, and scores every resulting threshold on
the same max-pooled test distribution. Axes:

- **Fit geometry** — `pooled` (sim-set scores through the style's inference
  max-pool; production post-#2797) vs `image` (whole-image vector scores; the
  historical geometry).
- **Cut rule** — `cross` (equal-density crossing; production post-#2798) vs
  `mid` (midpoint-of-means; historical).
- **Fit space** — `sig` (sigmoid scores, production) vs `logit` (scores
  logit-transformed before the fit, cut mapped back through the sigmoid).

Emitted variants (`gmm_variant` column; `vtscore/eval/voting_iterations.py::_SAFE_GMM_VARIANTS`):

| variant | fit | cut | space | reading |
|---|---|---|---|---|
| `xcal_only` | — | — | — | no-blend control: the raw conformal cut at the same step |
| `image_mid` | image | mid | sig | **historical production** (pre-#2797/#2798) |
| `image_cross` | image | cross | sig | cut-rule fix alone (counterfactual: #2798 without #2797) |
| `pooled_mid` | pooled | mid | sig | geometry fix alone (#2797 without #2798) |
| `pooled_cross` | pooled | cross | sig | **current production** — must equal the base row's blend |
| `pooled_mid_logit` | pooled | mid | logit | logit-space fit, historical cut |
| `pooled_cross_logit` | pooled | cross | logit | logit-space fit, production cut — the open #2798 idea |

Each variant row records `threshold` (blended), `gmm_cut` (pre-blend),
`xcal_threshold`, `blend_weight`, and the full FPR/FNR/cost + oracle/regret
metric set against the shared test scores. The `pooled_cross` row doubles as a
harness sanity check: the analyzer asserts it reproduces the production blend
(base row) bit-for-bit.

### Arms

Visual Genome region voting only (the setup where the geometry/shape concerns
live; Caltech binary voting is deliberately out of scope — single-vector GMM
behaviour is partially covered by the control arm):

| Dataset | Embedder | Style | Role |
|---|---|---|---|
| `visual_genome_m` | `dinov3_patch` | `max_patch` | the production region-vote strategy — the arm the decisions read |
| `visual_genome_m` | `siglip` | `whole_image` | single-vector control: pooled ≡ image fit, isolates the pure cut-rule effect (#2798 changed *all* GMM callers, including the text/cosine sort) |

### Fixed config

`safe_thresholds=True` (the object of study), inclusion 0, `sim_fraction` 0.5,
`calibrate_count` 2, `calibration_fraction` 0.5, MLP trainer,
`autopilot_fidelity=True` (the runner default — vote counts must mean
app-visible votes, since the ramp is keyed on them), **30 voting steps**
(above ~20 votes the blend is pure cross-cal and #2781 already covers it),
**8 seeds** (cells are ~5x cheaper than #2781's 150-step cells and the
sub-20-vote regime is the noisy one), same scale-band VG category selection as
the Max-Patch/#2781 studies (shared pickles).

### Windows

Vote count `n = n_good + n_bad` (votes, not flooded rows — the ramp's own
unit): **2–5** (pure-GMM window, blend weight 0) and **6–20** (ramp window).
The 6–20 cells are the primary ones (issue #2799); above 20 nothing the GMM
does matters.

## Pre-registered deliverables

`analyze_safe.py` computes all of these; the report draft lands in
`results/REPORT.md`:

- **FPR / FNR / cost vs vote count per variant, per arm** (`variant_vs_votes.csv`,
  `safe_{cost,fpr,fnr}_vs_votes.png`), plus per-window means
  (`window_by_variant.csv`).
- **Paired contrasts** (`contrasts.csv`), each as mean per-cell delta +
  Wilcoxon over (category, seed) cells (t-axis collapsed first so cells are the
  independent units):
  - `pooled_mid − image_mid` — the size of the bias #2797 removed;
  - `pooled_cross − pooled_mid` — what the #2798 cut rule buys under region
    voting;
  - `image_cross − image_mid` — same cut-rule contrast in single-vector
    geometry (reads on the text/cosine-sort callers);
  - `pooled_cross_logit − pooled_cross` — the logit-space question;
  - `pooled_cross − xcal_only` — does the blend beat the raw conformal cut at
    low votes at all?
- **Threshold diagnostics**: mean `gmm_cut` / `threshold` vs `oracle_threshold`
  per variant, degenerate rate per variant.
- **Sanity**: max |`pooled_cross` − base threshold| (must be 0 to the CSV's
  6-dp grain).

## Pre-registered decision rules

Read on the **`max_patch` arm, ramp window (6–20)** unless stated:

- **Keep #2801 (equal-density crossing)** unless `pooled_cross` is
  *significantly worse* than `pooled_mid` on cost (Wilcoxon p < 0.05 with a
  positive mean Δcost). The revert is a one-line swap:
  `_weighted_gaussian_crossing` returning `None` already falls back to the
  midpoint, so reverting = cutting at `fit.midpoint()` unconditionally.
- **Adopt the logit-space fit** only if `pooled_cross_logit` beats
  `pooled_cross` by mean Δcost ≤ −0.02 at p < 0.05 (both windows agreeing in
  sign). Otherwise record the number and close the idea.
- **#2797 sizing** is reporting, not a decision (the fix is already justified
  on geometry): expect `image_mid` to carry higher FPR than `pooled_mid`
  (image-level fit → cut biased low → over-inclusion). If the measured Δ is
  ~0, say so in the report — it bounds how much the geometry fix mattered in
  practice.
- **Cold-start plan note**: if `pooled_cross` is *worse* than `xcal_only` on
  ramp-window cost, the "Cold-start calibration" item in
  [`inclusion-calibration-bias.md`](inclusion-calibration-bias.md) absorbs a
  GMM-specific note (the blend as shipped is then part of the cold-start
  problem, not its mitigation); if better, that item's "hard GMM blend"
  framing should cite this study as evidence the blend earns its keep.

## Grid runbook

```bash
# worktree with this branch/dev at /exp/$USER/projects/vts-calib
cd /exp/$USER/projects/vts-calib/scripts/experiments/calibration
bash launch_safe.sh
```

`launch_safe.sh` chains the standard pipeline (`prepare` → cells array →
`analyze_safe.py`) with `CALIB_SAFE_THRESHOLDS=1`, VG-only arms
(`siglip,dinov3_patch` × `whole_image`/`max_patch`), 30 steps, 8 seeds, and
results under `/exp/$USER/calibration-safe` (the #2781 outputs stay untouched;
Max-Patch pickles/crops are reused in place). Expected size: 2 embedders × ~23
scale-band categories × 8 seeds ≈ 368 cells, each ≪ the #2781 cells (30 steps,
no 150-step tail, no re-pool arms). Knobs: `CALIB_N_SEEDS`, `CALIB_MAX_STEPS`,
`CALIB_VG_EMBEDDERS`, `CALIB_PATCH_STYLES` — all overridable in the usual
`CALIB_*` way.

## After the run

- Write `docs/experiments/safe-thresholds/REPORT.md` from the analyzer's draft
  (numbers + verdicts on the decision rules above), mirroring the #2781 report's
  shape.
- Act on the decision rules (revert/keep #2801; file or close the logit idea;
  add or don't add the cold-start note to `inclusion-calibration-bias.md`).
- Close out issue #2799 per the repo's issue workflow, and prune this plan file
  down to whatever follow-ups the data creates (delete it outright if none).
