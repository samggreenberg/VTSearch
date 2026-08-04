# Threshold-stability (#2790) — final report: the interactive detector should use a linear head

> Interactive version (live figures, light/dark): published Claude artifact —
> `claude.ai/code/artifact/c0146eb0-ca05-4a22-966b-a9e04476b6ef`
> (private; share from the page). This file is the durable, version-controlled record.

## BLUF

The production detector head — a small **MLP** trained on the user's good/bad votes —
makes **hidden calibration catastrophes**: a single vote can send held-out cost from
~0.01 to ~1.0 for a couple of steps, and across a run these excursions fire on ~5% of
votes.

Root cause: with only ~3–5 labeled positives the MLP is **under-determined**, so every
retrain wobbles the scores and the decision threshold lurches over the unlabeled
positives. The scarce, unrepresentative positives — not the calibration rule — are the
disease. Calibration knobs (fold count, smoothing, deferring the cut) only *blunt* the
symptom.

Replacing the MLP with a **linear (logistic) head** cuts the spike rate ~4× **and**
lowers cost and false-negative rate — a Pareto win, not a trade. It holds on **both
SigLIP embedders** and generalizes from COCO to Visual Genome. **Ship the linear head.**

## Background

VTSearch trains a detector interactively: a user votes good/bad on a handful of items,
a small model learns to rank the rest, and a split-conformal threshold (#2784) decides
the cut. Each vote retrains the model and re-scores the collection.

We measure error as **cost = FNR + FPR** on a held-out set — deliberately *not* F1,
because customers hunt *rare* things and, under heavy imbalance, F1 badly undervalues a
missed needle (a high FNR). That metric choice is what first exposed the problem: F1
looked fine while cost told a different story. The reported symptom (#2790): the trained
MLP's threshold made violent single-step jumps even as an oracle-chosen threshold barely
moved — instability in *threshold placement*, not ranking.

## The bug: hidden calibration catastrophes

The dangerous events are **deep transient excursions** mid-run (votes ~20–45): a
converged detector (cost ~0.01–0.08, pinned by a few positives) takes one boundary
false-positive "bad" vote, the cut lurches up over *all* the positives (cost → ~1.0,
FNR → 1), and snaps back within a step or two. They are "hidden" because the *labeled*
data still looks perfectly separated at that moment — the recall collapse is entirely
among the *unlabeled* positives. There is a necessary condition (a live false positive
just above the cut) but **no observable sufficient condition**: whether a vote triggers
it is an unpredictable function of how the whole label set happens to retrain.

## Mechanism: positive starvation

Boundary/uncertainty acquisition surfaces mostly negatives, so a run accrues many "bad"
votes but stays stuck at ~3–5 labeled positives — and those few are unrepresentatively
*high-scoring*. The cut ends up in the **gap** between the labeled negatives and those
high positives — exactly where the unlabeled/test positives sit, densely. With almost
nothing pinning it there, an MLP with this little data is **under-determined**: it fits
the handful of labels many ways, and each retrain re-scores everything differently. A
spike is one retrain whose operating point wobbled *up* through the hidden positives.

A direct decomposition attributes the recall collapse to **~56% model-score variance**
(the retrain) and ~44% cut movement (mostly the calibration faithfully re-cutting those
wobbling scores). Both trace to the same scarce-positive root — which is why every fix
that only touches the cut merely blunts the symptom. The spike drill-down confirms the
profile: **93%** of spikes are a **Bad vote**, **89%** a Bad vote on a `hard`-selected
boundary item, **87%** with sparse positives (median `n_good` = 3 at the spike).

## What we tried

| Lever | Touches | Result |
|---|---|---|
| `calibrate_count` 2 → 8 | the cut (more folds) | cuts spike rate + cost-sd ~15–25% — *blunts* (residual still ≈ 3× oracle floor) |
| median smoothing (`med3`) | the cut | no help (raised spike rate) |
| `defer-cut-goods` / GMM cut while sparse | the cut | crushes spikes but games the cost metric (more permissive → wrecks precision) |
| **linear / SVM head** | **the model** | **removes the spikes (~4×) and improves cost + FNR** |

The pattern is decisive: cut-only knobs blunt; reducing **model flexibility** removes.
So we put four heads on a *flexibility ladder* through the identical loop — all torch
models trained by the same balanced-BCE loop, so the linear head *is* a logistic
regression and nothing else in the pipeline changes:

| Head | Boundary | Capacity |
|---|---|---|
| MLP (baseline) | non-linear | hidden layer, grows with total votes |
| Reg-MLP | non-linear | hidden layer, capped to good-vote count |
| Linear SVM | linear | none (hinge / max-margin) |
| Linear (logistic) | linear | none (logistic loss) |

## Results — Visual Genome, 15 classes × 15 seeds, both SigLIP embedders, @ 60 votes

| Head | deep-spike | cost | FNR | FPR |
|---|---|---|---|---|
| **SigLIP 1** ||||
| MLP | 0.052 | 0.525 | 0.482 | 0.043 |
| **Linear** | 0.014 | **0.467** | **0.320** | 0.147 |
| Linear SVM | 0.013 | 0.479 | 0.347 | 0.131 |
| Reg-MLP | 0.038 | 0.524 | 0.366 | 0.159 |
| **SigLIP 2** ||||
| MLP | 0.055 | 0.497 | 0.464 | 0.033 |
| **Linear** | 0.014 | **0.451** | **0.326** | 0.125 |
| Linear SVM | 0.012 | 0.465 | 0.348 | 0.117 |
| Reg-MLP | 0.033 | 0.498 | 0.361 | 0.137 |

- **Linearity is the fix**, confirmed on all four arms and both embedders: the two
  *linear* heads collapse the spike rate to ~0.013 (vs the MLP's ~0.053, **~4×**) and win
  on cost and FNR.
- **Linear is the one to ship.** At the full 15 classes, `linear` edges `svm` on cost and
  FNR (spikes tied) — matching COCO's near-tie. *(A mid-run partial read had SVM looking
  better; that was class-selection bias — the finished classes were the common, easier
  ones. Caveat noted: peeks at these sweeps are class-biased.)*
- **Reg-MLP is dominated** — fewer spikes than the MLP but ~zero cost improvement over it.
  A constrained-but-still-non-linear model doesn't capture the win; you must go *fully*
  linear.

The FNR/FPR trade explains the cost win: linear/SVM are more permissive (higher FPR) but
miss far fewer needles (much lower FNR), landing on a lower cost = FNR+FPR contour. For
rare-event search the FNR axis is the one that matters.

## Does it generalize?

Yes, two ways. **Across embedders:** SigLIP 1 and SigLIP 2 are near-identical. **Across
datasets:** the fix was found on COCO (spike 0.055 → 0.025, ~2×) and confirmed on VG
(~4×, even larger). VG's common objects make every arm's absolute cost higher, but the
relative ordering is identical.

## Why it works

The linear head has one weight vector — a single hyperplane. It cannot reinterpret three
positives a dozen ways per retrain, so its scores are stable and the cut can't wobble up
through the hidden positives. The **Reg-MLP result is the control that proves it**:
shrinking the hidden layer cuts spikes somewhat but leaves cost unchanged — a little
non-linearity still injects enough retrain variance. And because the conformal rule is
quantile-based, it consumes the linear scores unchanged — no probability calibration
needed.

## Take-aways

- **Ship the linear (logistic) head** as the interactive detector default — best-or-tied
  on spikes, cost, and FNR, across both embedders and both datasets.
- **The disease is scarce positives, not the calibration rule.** Cut-only fixes blunt the
  symptom; reducing model flexibility removes it. You can't demand more needles from the
  user, so simplify the model.
- **Measure the metric your customer feels.** F1 hid this; cost = FNR+FPR (and watching
  FNR) surfaced it.
- **Bonus:** a linear head is cheaper to train and to score (one dot product per item),
  tightening the vote → retrain → re-rank round-trip on large collections.

## Status & follow-ups

- **Production change** (boolean/whole-image head → linear): branch
  `claude/linear-default-head`, finishing test reconciliation before its PR.
- **#2824** — extend the linear head to the region-voting path (production gap +
  experiment-harness trainer-seam refactor + {DINOv2, DINOv3} × MaxPatch validation).
- **#2825** — find & diagnose runs where held-out cost gets *sustainedly* worse with more
  labels (the persistent wrong-way curve — the opposite of the transient deep-spikes).

## Data & reproduction

Full sweep: 4 heads × 2 embedders × 15 classes × 15 seeds × 60 votes = 108,000
run-steps, plus the COCO study. Per-run, per-step schema and pandas re-analysis recipes:
[`DATA_AND_SCHEMA.md`](./DATA_AND_SCHEMA.md). Consolidated data:
`vg_bool_all.jsonl.gz` (+ durable local archive). Deep-spike = a Δcost > 0.1 excursion at
t ≥ 20 in the learned-head regime; cost = FNR + FPR on held-out positives/negatives.
The earlier calibration-knob sub-study (fold count, `med3`, `rank-transfer`; the `k8`
recommendation) is preserved in this file's git history and in `SPARSE_POSITIVE_PLAN.md`.
