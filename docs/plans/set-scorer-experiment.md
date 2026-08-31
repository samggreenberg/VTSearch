# Set-input scorer experiment — learned pooling vs linear+max for region voting

**Status:** proposed (investigation of #2890; no code yet). This designs the
experiment that answers the issue's question empirically at eval-harness tier.
Nothing here changes production, and nothing here blocks or changes the verdict
of #2886 (adopt MaxPatch, drop the HAC tree from ingest).

## Background (what the record already established)

Two studies pin down why the HAC tree lost and where the remaining headroom is:

- The max-patch study (`docs/experiments/2026-07-29-max-patch/REPORT.md`): raw-patch
  leaves fix the node-dilution problem — the raw-patch tree (MaxPatchHAC) has
  the **best ranking of any arm** (AP 0.492) — but it loses at the operating
  point because max-pooling ~392 candidates gives the max score a heavy
  false-positive tail. "The lever is the candidate set you pool over, not the
  structure of the tree."
- The calibration study (`docs/experiments/2026-07-31-calibration/REPORT.md`): the
  raw-patch tree is provably **calibration-bottlenecked** — lowest oracle cost
  of any VG arm (0.253, vs MaxPatch's trained 0.358) with significantly larger
  regret (p = 0.013). Both pre-registered fixed re-pools failed: `topk` made it
  worse; sign-corrected `pnorm` closed only ~21% of the gap.

So ~0.1 ErrorCost of measured headroom is locked behind the *pooling /
threshold*, not behind node vectorization. **Caveat:** both studies (Jul
29–31) predate the rebuilt threshold stack (fold-anchored cross-LabeledGMM
fused threshold, blend schedules, conformal inclusion, acquisition at
inclusion −3, all Aug 1–7); #2895 reruns the region-style comparison on
today's stack and re-prices this headroom. Run it first — the residual regret
it measures is this experiment's motivation. #2890 asks whether ROI Align can
convert a node (set of patches) into something scoreable, and whether the
scorer engine could accept sets directly. The investigation's conclusion: the
promising half is the **engine** — a learned set-pooling (attention-MIL)
replaces the hard max that is the measured bottleneck, and it trains in
inference geometry *by construction*, so calibration and inference cannot
disagree about how a bag collapses. ROI Align is at most a feature-extractor
arm inside that experiment: it adds the voted box's width/height (discarded by
today's nearest-patch snap) but does not touch the max-over-N tail, and it
re-creates small-object dilution on inflated union boxes.

The engine history that constrains the design: the MLP was dethroned not on
ranking but on **retrain-to-retrain threshold stability** with sparse
positives (#2790; see the `LINEAR_HEAD` comment in `vtscore/training/mlp.py`).
Any set engine adds parameters back, so it must be judged on regret and
wobble, not AP.

## Question

Can a scorer that consumes an image's **set** of patch rows — via learned soft
pooling instead of hard max over per-row linear scores — realize at the
*trained* threshold the accuracy the multi-scale candidate set already achieves
at the *oracle* threshold, without reintroducing the #2790 threshold wobble?

## Pre-registered arms

All arms run at eval-harness tier (`vtscore/eval/patch_styles.py` +
`vtscore/eval/voting_iterations.py`) on fixed candidate sets, so engine
differences are attributable to the engine:

- **Control — linear + hard max** (production geometry): the linear head
  scored per row, max-pooled per image. Run on the MaxPatch candidate set
  (CLS + raw patches).
- **Attention-MIL** (the headline): gated attention pooling (Ilse et al. 2018)
  — image score = `head(Σᵢ αᵢ·vᵢ)` with learned attention over the image's
  rows, trained bag-aware (Bad votes are bag labels, as the flooding path
  already treats them; Good region votes are instance labels). Same MaxPatch
  candidate set as the control.
- **Attention-MIL over the raw-patch tree's node set** (optional): tests
  whether learned pooling unlocks the multi-scale candidate set whose oracle
  cost is the best on record. Only worth running if the MaxPatch-set arm shows
  the engine is stable.
- **ROI-align anchors** (secondary): a small multi-scale anchor set (whole
  image + raw patches + a handful of coarser cells), each anchor featurized by
  `torchvision.ops.roi_align` over the stored `patch_grid` at small R (2–3),
  linear-or-MIL head on top. The voted box **snaps to the nearest anchor**
  (the anchor-space analog of `nearest_patch_to_box`) so the train/score
  parity invariant holds by construction; a boxless Good vote maps to the
  whole-image anchor. This arm exists to price the box's 4 DOF, not to win.

Datasets / categories / seeds / vote budget: reuse the max-patch harness setup
(`visual_genome_m` scale-band categories, paired seeds, t ≤ 150, production
cross-calibration path) so results are directly comparable to the two prior
reports.

## Pre-registered metrics and decision rule

Co-primary, paired per (category, seed), Holm-corrected Wilcoxon:

- **ErrorCost / AULC** at the trained threshold (as before);
- **regret** = trained cost − oracle cost (the calibration study's metric);
- **threshold wobble** — retrain-to-retrain dispersion of the calibrated
  threshold and of step-to-step cost (the #2790 failure signature; exact
  statistic to be fixed before any run, e.g. median absolute step-change in
  trained cost over t).

**Adopt a set engine only if** it beats the linear+max control on
ErrorCost/AULC **and** is not significantly worse on regret **or** wobble.
Ranking-only wins (AP) do not count — the tree already ranks best and it
doesn't matter at the operating point.

## Honest priors

- Attention-MIL on the MaxPatch set: modest ErrorCost gain, real regret gain
  (its calibration geometry equals its inference geometry), wobble is the risk
  that decides it.
- Attention-MIL on the tree's node set: the upside case (oracle 0.253 says the
  candidates are there) but the highest-variance arm.
- ROI-align anchors: prices the 4-DOF question; prior is neutral-to-negative
  on inflated union boxes (near-frame unions of scattered instances resample
  to R×R and dilute, the k-means-leaf failure in another guise).

## Open work

<!-- item-sep -->

- [ ] #2895 — Rerun the region-style study on today's threshold stack (Opus
  4.8; prerequisite — its residual-regret measurement is this experiment's
  motivation)

<!-- item-sep -->

- **Bag-aware trainer abstraction in the harness** — extend
  `voting_iterations` / `patch_styles` so an engine can train and score
  *bags* (an image's row set) rather than per-row-then-max only; the
  linear+max control must be re-expressible inside the same abstraction so
  the comparison is apples-to-apples. Regression-prone (the train/score
  parity and calibration-geometry invariants live here). (Opus 4.8)

<!-- item-sep -->

- **Attention-MIL engine** — gated attention pooling head + bag-aware BCE
  training, deterministic/seeded per the flaky-test rules; unit tests at
  `tests_lib/detectors/` tier mirroring the existing style parity tests.
  (Opus 4.8)

<!-- item-sep -->

- **ROI-align anchor style** — anchor set, `roi_align` featurization over
  `patch_grid` (derived on the fly; nothing persisted, per the
  no-persisted-vectors rule), snap-to-anchor vote resolution, parity tests.
  (Opus 4.8)

<!-- item-sep -->

- **Stability + regret metrics in the harness** — add oracle/regret and the
  wobble statistic to the per-step metrics records and the analyzer, fixed
  before any run. (Sonnet 5)

<!-- item-sep -->

- **Run + report** — SLURM sweep on the max-patch grid setup, REPORT.md under
  `docs/experiments/set-scorer/`, decision per the pre-registered rule.
  (Fable 5 for analysis/writing; the run itself follows
  `scripts/experiments/` conventions.)

<!-- item-sep -->
