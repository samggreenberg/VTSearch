# Price a sign-dependent tilt: the `hinge` cut rule (#3557)

**Status: pre-registered 2026-09-22, before the first cell ran. Run and reported: see [REPORT.md](REPORT.md) - nothing ships.** The decision
rule below is fixed now; the report records the measurement against it.

## The question

[#2865](../2026-08-21-inclusion-cut-rule/REPORT.md) swept five fold-anchored cut
rules across the Inclusion knob and kept the shipped `mid_tilt`. One loser had a
signed asymmetry: `cross_tilt` - the rule that keeps the fitted mixture weights
as class priors - beat `mid_tilt` **below** inclusion 0 in three of four
environments (to −0.034 ± 0.005 at k=−1 on `coco_val × siglip`) and lost above it
(to +0.073 ± 0.012). The obvious rule that follows is a hinge:

> `cross_tilt` for k < 0, `mid_tilt` for k ≥ 0.

Two things stand between that and a ship, and this study answers both:

1. **The nesting contract.** `threshold_at` is monotone by construction, so the
   admitted sets are nested (everything included at k stays included at k+1).
   A hinge at k=0 is exactly the shape that can break that at the seam.
2. **Today's stack, the whole knob, and the trajectory.** #2865 ran on the
   2026-08-21 stack (acquisition offset −1, before #3308, #3319, #3825, #3585),
   and priced `cross_tilt` as a *re-cut* of the incumbent's trajectory. A
   shipped hinge also moves the **acquisition** cut, which re-cuts the same
   estimator at k + `ACQUISITION_INCLUSION_OFFSET` (−4 today, so below the seam
   at every reporting k < 4). A re-cut cannot see that.

## The rule, and the contract

Write `M(k)` for `mid_tilt`'s combined fold quantile and `C(k)` for
`cross_tilt`'s. Both are non-increasing in k (the chain in
`FoldAnchoredCut.threshold_at`; `C` is `_rate_cut` at `lam = 2^k`, monotone in
`lam` by the same sup argument as `rate`). The literal hinge is nested **iff**
`C(0−) ≥ M(0) = q_mid`, and nothing orders those two: `C(0)` is the Bayes
misclassification-count boundary, `q_mid` the midpoint of the component means.
With a rare, wide Good component `C(0)` falls well *below* the midpoint - the
unit test `test_hinge_cut_rule.py` carries such a fit (5% prevalence, Good 25×
wider) on which the literal hinge cuts at 0.27 at k=−1 and 0.40 at k=0: asking
for fewer false alarms admits *more*.

So three hinge arms, all identical to `mid_tilt` at every k ≥ 0:

| arm | below k=0 | nested? |
|---|---|---|
| `hinge` (**the candidate**) | `max(C(k), M(k))` | **yes, by construction**: a max of non-increasing functions is non-increasing, and for k < 0 ≤ k′, `H(k) ≥ M(k) ≥ M(0) ≥ M(k′) = H(k′)` |
| `hinge_raw` (the issue's literal rule) | `C(k)` | only where `C(0−) ≥ q_mid`; here to **count** violations on real fits |
| `hinge_cont` (decomposition) | `q_mid + C(k) − C(0)` | yes: `C(k) ≥ C(0)`, so it stays ≥ `q_mid` |

The guarded `hinge` follows `cross_tilt` wherever `cross_tilt` is the *stricter*
rule and `mid_tilt` elsewhere, so below zero it can only ever admit **less** than
the incumbent - the direction the knob asks for there. Where #2865's win came
from `cross_tilt` being stricter, the guard costs nothing; where it came from
`cross_tilt` being *more* inclusive, the guard gives it up, and the
`hinge`-vs-`hinge_raw` contrast measures how much that is. `hinge_cont` splits
the win into location (`C(0) − q_mid`) and slope.

The proof is in `FoldAnchoredCut._quantile_at`'s docstring and is checked by a
property test over 40 random two-fold estimators on a 0.05-step grid that
approaches the seam from both sides (`test_hinge_cut_rule.py`).

## Design

**Two run-level arms**, one `CALIB_EXP` each, paired on cell identity
(dataset × embedder × category × seed):

| arm | live cut rule | what it prices |
|---|---|---|
| **A `incumbent`** | unset → `FOLD_ANCHOR_CUT_RULE` (`mid_tilt`) | the shipped trajectory; its `__cutincl` re-cuts price every rule **reporting-only**, paired within the step |
| **B `hinge`** | `hinge` (declared `--diverges live_cut_rule`) | what a user of the shipped hinge gets: reporting *and* acquisition |

Re-cut rules on both arms: `mid_tilt, mid, rate, cross_tilt, hinge, hinge_raw,
hinge_cont`, at every integer k in [−10, 10] plus −0.5 and −0.25 (the seam).
`mid` is the instrument check: it must come back inert.

**Environments** - no `vg_scale`, FullMarks or `coco_better` (other sessions are
rebuilding those today). #2865's four exactly, plus one it never ran:

| environment | voting |
|---|---|
| `visual_genome_m × siglip+dinov3_patch × max_patch` | region |
| `coco_val × siglip+dinov3_patch × max_patch` | region |
| `visual_genome_m × siglip × whole_image` | binary |
| `coco_val × siglip × whole_image` | binary |
| `caltech101_m × siglip × whole_image` | binary (boxless) |

Shipped defaults everywhere else (head, κ=0.3, `qmean`, 2 folds, acquisition
offset −4, per-mode blend, exclusion floor), checked by preflight 12. 300 steps,
4 seeds, seed-major. Launcher: `scripts/experiments/calibration/launch_hinge_3557.sh`.

**Statistics.** The unit is the **cell**: per cell, average over the deep band
(n_votes > 100, #2865's band) and then difference; 95% bootstrap CI over cells
(2000 resamples). Every cost and regret is on the **rate scale** (divided by
`2^|k|`, #2865's units fix). Every table is **per environment**; nothing is
pooled across environments.

## Decision rule (pre-registered)

`hinge` ships only if **all** hold:

1. **Contract.** Zero measured violations of nesting for `hinge` - no cell-step
   at which its threshold rises between two adjacent stops of the 23-stop grid,
   seam included - on both arms. (Expected by construction; a single violation
   means the proof or the code is wrong, and nothing ships.)
2. **No material harm, anywhere.** For the **full ship** (arm B's `hinge` rows
   against arm A's `mid_tilt` rows, paired on cell), at every one of the 21
   integer stops in every environment, the **upper** 95% bound of Δcost is below
   **+0.010** - the tolerance #2891/#3319 decided the acquisition offset on. The
   same bar on the reporting-only re-cut (arm A, `hinge` vs `mid_tilt`, Δregret).
3. **A real gain.** Δcost (full ship) is resolvably negative (CI excludes 0) at
   one or more k < 0 in at least **three of the five** environments - the
   "three of four" that motivated the issue, held to the same share.
4. **The trajectory does not pay for it.** At the reporting inclusion users sit
   at (k=0), the full ship's end-of-session cost, cost-AUC over clicks and
   median clicks-to-target (#3319's speed lens) are each within +0.010 (cost) /
   not resolvably worse (clicks), per environment.
5. **The knob survives.** Distinct admitted sets across the 21 integer stops is
   not lower than the incumbent's by more than 0.5 of a stop in any environment.

Reported whether or not it ships: the crossover count (stops significantly
better vs significantly worse, #3196's caution about pooled means); how often
`hinge_raw` would have broken nesting and by how much; the location/slope
split (`hinge_cont`); and the mechanism read from the seam columns - prior odds
(`fit_log2_prior_odds`) and variance asymmetry (`fit_log2_var_ratio`) - against
where the hinge wins. That last item is also #3557's "cheap half": #2865's run
directory no longer exists, so its `__cutdiag` frame cannot be read, and this
run records the fold-anchored fits' own quantities instead of the final model's
unanchored ones.
