# Decoupling the acquisition threshold from the reporting threshold — buy back the positives the fused cut costs

**Status:** Designed, not run. Prepared 2026-08-07 off the #2847 result
([`docs/experiments/spike-check-2847/REPORT.md`](../experiments/spike-check-2847/REPORT.md),
PR #2873). Blocks nothing; needs no new data.

## Background — the finding that motivates it

The #2847 study measured today's stack against the configuration that issue
reported on. Production is **7.7× cleaner** on the cost-spike axis it was built
to fix, but it carries a cost nobody chose:

> **Production finds half as many positives.** Median 9 → 4 per 100 votes
> (Wilcoxon on 147 `(category, seed)` cells, p=1e-20), while the final cost is
> statistically unchanged (0.133 → 0.137, p=0.09).

Positives are the scarce resource in this whole line of work. #2790 traced the
deep threshold spikes to positive starvation; #2825 traced the *sustained*
wrong-way runs to a vote mix of 9 bad : 1 good, with 34% of bad segments
receiving **zero** new positives. Both recommended positive-seeking acquisition.
It was never built. The #2847 run now shows production moving in the wrong
direction on exactly that axis, so the item has a measured price tag.

## Mechanism — why one number is doing two jobs

The threshold is used for two unrelated purposes, and they want opposite things.

**Job 1 — reporting.** The decision line the user sees, and what
`cost = FPR + FNR` is scored at. Correct answer: the inclusion-0 cut.

**Job 2 — acquisition.** Autopilot's `hard` pick calls
`_hard_pick_by_index(ctx, ctx.scores, ctx.threshold)`
(`vtscore/eval/al_strategies.py:192`), which ranks every item **descending**,
finds the first rank position whose score is at or below the threshold, and
takes the unlabelled item closest to that position *in rank space*. The atlas's
`New` pick takes the threshold too (`atlas.next_sample(ctx.scores, ctx.threshold)`).
So the threshold does not act as a decision boundary here at all — **it selects
a sampling position in the ranking.**

That makes the direction the opposite of the intuition from the cost weights:

| | threshold | cut's rank position | Hard samples | positives found |
|---|---|---|---|---|
| inclusion **> 0** | *lower* (a miss is priced higher) | further **down** the ranking | deeper, lower-scored | **fewer** |
| inclusion **< 0** | *higher* (a false alarm is priced higher) | further **up** the ranking | nearer the top | **more** |

`FoldAnchoredCut.threshold_at` states this directly: *"raising inclusion lowers
the threshold and admits more."* So the arm that buys positives is a **negative**
inclusion for acquisition — inclusion −2, as proposed.

### The mechanism is already visible in the #2847 data

No new run is needed to confirm the lever exists or its sign. From the 56,880
base rows of that study:

| | `threshold_percentile` (median) | share of pool above the cut | positives found (median) |
|---|---:|---:|---:|
| `A_mlp_xcal` (conformal) | 0.959 | 4.1% | 9 |
| `D_lin_fused` (production) | 0.885 | 11.5% | 4 |

Production's fused cut sits **lower in the ranking** than the conformal cut it
replaced (per-step `threshold − xcal_threshold` median **−0.154**; higher in only
20% of steps), so Hard samples roughly 11.5% deep instead of 4.1% deep, and
brings back fewer positives. And within a category, the trajectories that ran a
*higher* threshold found *more* positives: median within-category Spearman
**+0.80** (arm A) and **+0.29** (arm D) across 19 categories.

That is the whole hypothesis in one line: **production did not decide to explore
more; it drifted deeper into the ranking as a side effect of a threshold change
made for a different reason.**

## Hypotheses

- **H1 (lever).** Cutting acquisition at inclusion `k < 0` while reporting at 0
  increases positives found by t=100, monotonically in `−k`.
- **H2 (free lunch, or not).** There is a `k` at which positives rise **without**
  a regression in final cost or in deep-spike incidence. *This is the one that
  matters, and it is genuinely uncertain — see the tension below.*
- **H3 (spikes).** Because positive starvation is the #2790/#2825 spike
  mechanism, `k < 0` reduces deep-spike incidence further rather than trading
  against it.
- **H4 (schedule beats a constant).** A `k` that starts negative while positives
  are scarce and relaxes to 0 as they accumulate beats any constant `k` — the
  same shape #2841 found for the blend schedule.

### The tension H2 exists to resolve

**More positives is not obviously better, and the experiment is worth running
precisely because the argument cuts both ways.** The `hard` pick samples near
the boundary *because boundary items are the most informative* — that is the
active-learning rationale. Biasing the sampling position up the ranking buys
positives by sampling where the model is already confident, which is
label-inefficient in the classic exploration/exploitation sense. So there are
two credible outcomes:

- **The optimum is interior and negative**: starvation is currently the binding
  constraint, so the informativeness lost is worth less than the positives
  gained, and cost at t=100 improves.
- **The optimum is 0 and the shipped cut is already right**: positives rise but
  the labels are redundant, cost is flat or worse, and #2847's "half as many
  positives" is a description of the method working rather than a defect.

A result of "no interior optimum" is a real answer, not a failed run, and would
close the positive-seeking item from #2790/#2825 rather than leave it open.

## Design

### Arms

`k_acq` is the inclusion the **acquisition** threshold is cut at; reporting and
all scoring stay at inclusion 0 in every arm, so cost is comparable throughout.

| arm | `k_acq` | role |
|---|---|---|
| `prod` | 0 | **control** — today's production, both jobs on one number |
| `acq_m1` | −1 | |
| `acq_m2` | −2 | the proposed operating point |
| `acq_m3` | −3 | |
| `acq_m4` | −4 | far end; expected to be too greedy |
| `acq_p2` | +2 | **falsification arm** — the wrong direction. Must make positives *worse*; if it does not, the mechanism above is wrong and nothing else in the run is interpretable |
| `rank_pin` | — | cuts acquisition at the **rank percentile the conformal path used** (0.959) rather than at an inclusion value. Same intent, one fewer indirection; if it matches the best `k_acq` arm, prefer it as the shipped parameterisation |

`xcal` (the #2847 arm A configuration) is **not** re-run: its positives-found
figure of 9 is already measured on this exact grid and serves as the reference
for how far back the lever can get us.

The `acq_p2` arm is the load-bearing control. It costs one arm and it is the
only thing that distinguishes "the lever works" from "any perturbation of the
sampling position changes the numbers."

### Endpoints, pre-registered

**Decision endpoint (what ships or does not):** cost at t=100, paired at the
`(category, seed)` cell against `prod`.

**Mechanism endpoint (does the lever pull):** positives found by t=100 and by
t=50. This is what the experiment is *named* after, but on its own it decides
nothing — H2 is the question.

**Guardrails (a win here must not undo #2847):**
- deep-spike incidence, same rule as the #2847 study: `t ≥ 20`,
  `cost ≥ 0.25`, `cost − oracle_cost ≥ 0.20`;
- worst-step regret `max(cost − oracle_cost)`;
- `oracle_cost` itself — if the *ranking* degrades, the labels were redundant
  and that is the H2-negative outcome showing up directly.

**Ship rule, fixed in advance.** Adopt `k_acq` iff, pooled over cells:
1. positives found rises significantly (Wilcoxon, α=0.05); **and**
2. cost at t=100 does not regress (upper bound of the 95% CI on the paired
   delta below +0.01); **and**
3. deep-spike incidence does not rise (exact discordant-pair test, α=0.05).

Report the **positives-vs-cost frontier across `k_acq`**, not just the argmin.
A knob with a flat optimum is a different finding from one with a sharp one, and
#2861 showed this family of curves can be nearly flat within a decade.

### Statistics

Same discipline as #2847, for the same reason: `k_acq` steers acquisition, so
each arm is a **different trajectory** and the arms are **not step-paired**.
Pair at the `(category, seed)` cell on one summary number per trajectory. The
statistical unit is a trajectory, never a step. All 19 COCO categories, so the
per-category table can show whether the lever's value tracks prevalence — the
rare categories (`toothbrush`, `scissors`) are where starvation actually binds
and are the most likely place for an interior optimum to appear.

### Power

Taken from the #2847 run rather than assumed. On 147 cells the positives-found
contrast returned p=1e-20 and the cost contrast p=0.09 — so the mechanism
endpoint is enormously over-powered at this size and the **decision endpoint is
the binding one**. Size for cost, not for positives: 19 categories × 8 seeds =
152 cells per arm keeps the cost contrast at the same precision as #2847, where
a 0.009 median difference sat at p=0.09. Do **not** cut to 4 seeds to save time;
the whole result turns on a null that needs to be a *tight* null.

> Carry forward the lesson from #2847's near-miss: an underpowered arm that
> comes back clean is not evidence of "no regression." State the power against
> a +0.01 cost regression when reporting guardrail 2.

### Grid and cost

`coco_val` × `siglip2` × `whole_image`, 19 categories × 8 seeds × 100 labels,
identical to the #2847 grid so `prod` is directly comparable to that run's arm D
and `xcal`'s positives figure carries over.

**7 arms × 152 cells = 1,064 cells.** At the #2847 run's measured ~4.7 min/cell
single-threaded, and the 120-task `cpu_limit` QOS cap, that is **~45 minutes**,
zero GPU — prepare is reused from the #2841 mixin run exactly as #2847 did.

Add `visual_genome_m × siglip` (region voting) as a **second environment only if
the COCO result is positive**. #2861 found this family of answers does not
always transfer across voting modes, but there is no reason to pay for the
generalisation check before there is something to generalise.

## Implementation sketch

The estimator already does the work — this is plumbing, not new maths.

`FoldAnchoredCut.threshold_at(k)` exists and is **monotone by construction**, so
the arms are nested (everything Hard can reach at `k` it can still reach at
`k−1`) and `threshold_at(0)` remains exactly the shipped cut. #2865 built the
`mid_tilt` rule specifically to make the cut inclusion-aware, so re-cutting is
O(1) per step against a fit that has already been computed.

1. `vtscore/eval/voting_iterations.py` — `_safe_threshold_for_step` currently
   returns the inclusion-0 threshold. Return the `FoldAnchoredCut` alongside it
   (or a second threshold cut at `k_acq`).
2. Same file, the main loop — keep `threshold` for `_evaluate_on_test` and the
   emitted row; carry a separate `acq_threshold` into the next iteration's
   `ALContext(threshold=…)`. Two names where there is currently one.
3. Emit `acq_threshold` and `acq_threshold_percentile` as columns, so the
   analyzer can verify the sampling position actually moved rather than assume
   it. **This is the check that would have caught a sign error**, and given the
   direction is counter-intuitive, it is not optional.
4. Fall back to `acq_threshold = threshold` whenever no fold-anchored cut was
   fitted (the `gmm_blend` path, ~5.4% of steps, concentrated in the cold
   start) — the schedule blend has no inclusion-aware form.
5. `scripts/experiments/calibration/` — an arm knob (`CALIB_ACQ_INCLUSION`) and
   a launcher on the #2847 template.

### Product question this raises, to settle before shipping rather than after

Decoupling means the app samples from a position the visible decision line no
longer marks. Today a user watching Autopilot sees items arrive from around the
line they are shown; under `k_acq < 0` they would arrive from above it. That may
be entirely fine — arguably it is *better*, since the user sees more of what
they are looking for — but it is a change to what the interface implies, and it
should be a decision rather than a side effect. **Precedent worth not repeating:
that is exactly how the positives regression this plan exists to fix got in.**

## Relationship to open work

- **#2865** (the shipped `mid` cut is inclusion-blind) is the *sibling*, not the
  same item: it asks whether the reported cut tilts correctly away from
  inclusion 0. This plan holds reporting at 0 and moves only acquisition. They
  share `threshold_at`, so #2865 landing first would strengthen the tilt this
  plan rides on — worth sequencing after it if it is close.
- **#2790 / #2825** — this is the positive-seeking acquisition both recommended,
  in its cheapest possible form: no new selector, no new model, one existing
  method called at a different argument. If it fails, the more invasive
  positive-seeking selector is the next thing to try, not the thing to skip to.
- **#2847 / PR #2873** — the run that measured the regression and this plan's
  baseline. Its arm D *is* the `prod` arm here.
