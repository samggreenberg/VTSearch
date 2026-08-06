# Population-anchored calibration: fusing the haystack into the trained threshold

**Issues #2852 / #2853 · run 2026-08-05 · design pre-registered in
`docs/plans/population-anchored-calibration.md` · code PR #2857 · run base dev
`0a54f0d7` · 184/184 cells, 0 failures (SLURM 468311 → 468312)**

Visual report with mechanism figures:
<https://claude.ai/code/artifact/6bd39c84-5946-4dd8-ba80-334a88428920>

## BLUF

**The fold-anchored mixture ("cross-LabeledGMM") with the lightest anchor mass
(κ=1) and the rate-optimal cut is the best threshold rule this harness has
measured, and it satisfies the pre-registered adoption bar (decision rule 1)
under the measured-envelope reading of H4.** At 100–300 votes it cuts paired
regret by **−0.085** vs pure cross-calibration (winning 71–74% of ~36k paired
steps), beats the run's shipped blend everywhere they differ, and holds *both*
error rates below pure x-cal's until 200 votes. It is **2.8× steadier**
step-to-step and essentially never degenerates (99.9% fully-anchored fits).
The one caveat: in the 2–20-vote window its FNR (0.30) exceeds the 0.25
*nominal* conformal budget — but pure x-cal runs 0.45 there, so the winner is
inside the budget's *measured* envelope at every checkpoint, and from 21 votes
on it is under the nominal budget too, which no other arm achieves that early.

Hypothesis verdicts (mechanical, `summary.json`): **H1 ✓ · H2 ✓ · H3 ✓ ·
H4 ✗** (nominal reading; the ✗ is confined to the 2–20 window where the status
quo violates 1.5× harder — see "The H4 question" below).

## Take-aways

- **Fusion beats scheduling, decisively.** Every fold-anchored variant — all
  ten κ×rule combinations — beats pure x-cal in the deep regime. The ramp's
  problem is its *form* (two rivals averaged on a label-count schedule), not
  its expiry point: the blend control collapses onto x-cal past its handover
  and inherits x-cal's deficits.
- **Honest anchors matter more than anchor mass.** κ=1 — each vote counting as
  *one* haystack point among ~50k — wins, and performance degrades
  monotonically as κ grows. Label-anchored fits on the final model's own
  scores flip to *worse than x-cal* at κ≥10: heavy anchoring injects the
  votes' train-set optimism and acquisition bias. The fold repair (anchor on
  held-out scores, one scale) is worth more than any amount of anchor weight.
- **The deficit has a name — mostly sample size, partly scale.** Rank-transfer
  alone (fixing only fold→final scale transfer) recovers −0.020; anchoring the
  population fit recovers −0.073; doing both honestly recovers −0.085. The
  conformal quantile's tiny-sample noise is the dominant deficit, as the
  plan's deficit 1 predicted.
- **The winner centers the operating point; x-cal drifts to arm-dependent
  extremes.** On the production region-vote arm x-cal runs FNR-heavy at depth
  (FNR 0.27 / FPR 0.09); on the single-vector control it runs FPR-heavy
  (0.11 / 0.40). The fold-anchored cut lands near-balanced on both — that is
  what a population-scale fit buys.
- **Stability comes along free** (H3): mean |Δthreshold| per step 0.0029 vs
  x-cal's 0.0081. Pure label-anchored fits are steadier still (0.0009) but pay
  for it in accuracy at high κ.
- **Degeneracy machinery exists but almost never fires:** 5 fallbacks in
  53,710 label-anchored fits at κ=1; the fold arm ran fully anchored (2/2
  folds) on 99.91% of steps.
- **Two harness degeneracies to fix before the next run:** (a) with
  `calibrate_count=2` there are only two folds, so the qmean/qmedian
  fold-combine arms are byte-identical — the combine question is unanswered;
  (b) the run's blend control is the 6→20 ramp (production at the run base),
  not the `slow_cap50` schedule #2849 shipped days later — see caveats.

## Why this experiment exists

The threshold at the run base treated its two estimators as rivals on a
hand-tuned schedule: an unsupervised 2-component GMM on the haystack's score
distribution ramps from weight 1 at 6 votes to 0 at 20; pure cross-calibration
ships thereafter. Three prior results said the framing is wrong (owner-side
run: naive GMM still competitive at ~300 votes; the selection-bias study
cleared the labels; #2790/#2799 isolated three structural deficits of the
conformal cut, none decaying with label count):

1. **Sample size** — the conformal cut is a low quantile over *tens* of
   held-out positives; the GMM fits up to 50k scores.
2. **Scale transfer** — the x-cal cut is measured on *fold models'* score
   scales but applied to the *final model's* scores.
3. **Per-retrain variance** — fold splits redraw every vote; the x-cal cut is
   a fresh noisy estimate each step.

The reframe under test: labels and haystack hold complementary information —
labels know which side is which and which quantile matters; the haystack knows
where that lives on the final model's actual score scale. They should feed
**one estimator**, not two rivals averaged on a schedule.

## The algorithm

### Label-anchored mixture (semi-supervised EM; κ is the only knob)

`fit_anchored_score_gmm` fits the same 2-component 1-D Gaussian mixture as
production's `fit_score_gmm`, but semi-supervised: haystack scores are free;
every voted item's component membership is **clamped** to its label (Good →
high component, Bad → low). Classical anchored EM, initialized from the
unanchored seed-42 fit:

- **E-step** over free points only (log-domain responsibilities); anchors stay
  one-hot regardless of where their scores lie.
- **M-step** re-estimates means/variances/weights with each anchor counted
  **κ** times (`anchor_weight`). With n votes vs N haystack scores, the
  labels' share of the class-conditional evidence is **γ = κn / (κn + N)** —
  the labels' authority grows with data instead of a hand-tuned ramp.

Anchors *force* the component identification rather than inherit it: if labels
contradict the population modes, the fit reports a named degeneracy
(`inverted_means`, `component_collapse`, …) and falls back to the
**unanchored** fit of the same sample — never to 0.5. A variance floor
(1e-6 of total sample variance) prevents anchor-pinned collapse. The cut rule
then applies to the fitted pair: `mid` (production midpoint) or `rate` (the
#2836 rate-optimal crossing `wn·f_pos = wf·f_neg`, midpoint fallback when
rootless).

### The flaw in anchoring on the final model — and the fold repair

The label-anchored fit anchors on the **final model's** scores of the voted
items — but those items were in the final model's training set, so their
scores are optimistically separated, and votes are acquisition-biased
(Autopilot samples near the threshold). Both effects are confirmed material:
heavy final-scale anchoring (κ≥10) is *worse than not anchoring at all*.

The **fold-anchored mixture** ("cross-LabeledGMM", the #2852 comment) repairs
this with machinery cross-calibration already has. Per calibration fold k:

1. Score the haystack with fold model k (one extra scoring pass; cheap at
   linear-head scale) and fit the anchored mixture on those scores with fold
   k's **held-out** labeled scores as anchors — honest anchors, one scale.
2. Apply the cut rule to fold k's fit; read the cut's empirical quantile
   q_k in fold k's own haystack distribution.
3. Combine fold quantiles (mean/median — identical at 2 folds) and realize the
   combined quantile on the **final model's** haystack distribution
   (**rank-transfer**: two models scoring the same haystack are related by an
   approximately monotone map, and quantiles are invariant under monotone
   maps). No raw score ever crosses scales.

Degeneracy chain per fold: anchored → that fold's unanchored fit; if every
fold fails, final model's unanchored midpoint, then median — never 0.5.
`rank_transfer` also runs as its own arm (today's conformal cut carried as a
quantile onto the final distribution) — fixing *only* deficit 2, so its gain
measures how much of the problem was ever about scale.

## Experiment design

Within-step **paired variants** (the `_SAFE_GMM_VARIANTS` pattern): every arm
sees identical models, votes, and steps; contrasts are paired per
(arm, category, seed, step). Visual Genome region voting, production linear
head, safe thresholds ON, 300-vote trajectories, 4 seeds, 184 cells.

| Dimension | Values |
|---|---|
| Data arms | `dinov3_patch × max_patch` (production region-vote arm) · `siglip × whole_image` (single-vector control) |
| Estimators | `anchored_w{κ}_{rule}` · `fold_anchored_w{κ}_{rule}_{combine}` · `rank_transfer` |
| Controls | `xcal_only` (pure cross-cal) · `pooled_mid` (the run-base blend, 6→20 ramp) |
| Anchor mass κ | 1 · 3 · 10 · 30 · 100 |
| Cut rules | `mid` · `rate` |
| Fold combine | qmean · qmedian — **degenerate this run** (2 folds ⇒ identical) |
| Checkpoints | windows at 20 / 50 / 100 / 200 / 300 votes; deep regime = >100 |

## Results

### H1 — deep-regime (100–300 votes) paired Δregret vs pure x-cal

All ten fold-anchored variants win; label-anchored wins only at κ≤3;
rank-transfer wins slightly. Full ordering in `summary.json`; the shape:

| κ | fold, rate | fold, mid | label, rate | label, mid |
|---:|---:|---:|---:|---:|
| 1 | **−0.0847** | −0.0796 | −0.0725 | −0.0626 |
| 3 | −0.0699 | −0.0660 | −0.0308 | −0.0246 |
| 10 | −0.0525 | −0.0564 | +0.0259 | +0.0166 |
| 30 | −0.0435 | −0.0493 | +0.0734 | +0.0482 |
| 100 | −0.0335 | −0.0360 | +0.1054 | +0.0697 |

`rank_transfer` = −0.0196. **Attribution:** fold beats label-anchored (the
train-anchor bias is material, not theoretical); anchored ≫ rank-transfer (the
quantile's sample size, deficit 1, dominated); rank-transfer > 0 (scale
transfer, deficit 2, is real but secondary).

### The winner in numbers (`fold_anchored_w1_rate`, both arms pooled, paired)

| Window (votes) | 2–20 | 21–50 | 51–100 | 101–200 | 201–300 |
|---|---:|---:|---:|---:|---:|
| Δcost vs x-cal | −0.096 | −0.082 | −0.088 | −0.087 | −0.079 |
| ΔFNR vs x-cal | −0.112 | −0.074 | −0.035 | +0.006 | +0.031 |
| win rate (cost) | 75% | 73% | 74% | 74% | 71% |
| paired steps | 2,299 | 4,769 | 8,455 | 17,645 | 18,058 |
| FNR, production arm | 0.290 | 0.239 | 0.224 | 0.207 | 0.195 |
| … x-cal, same arm | 0.436 | 0.409 | 0.384 | 0.321 | 0.274 |

Vs the run's blend the numbers match from 21 votes on (the ramp has expired);
in 2–20 the winner still leads it by −0.036 (62% win rate). Deep ΔFNR turns
slightly positive (+0.031 at 201–300): a small recall trade for a much larger
precision gain, with absolute FNR (0.195–0.207) under the 0.25 budget
throughout that regime.

Cost by window, production arm (winner / x-cal): 0.436/0.594 · 0.377/0.515 ·
0.343/0.471 · 0.315/0.412 · 0.298/0.359. Control arm: 0.560/0.693 ·
0.448/0.540 · 0.406/0.505 · 0.396/0.503 · 0.389/0.504.

### Operating-point geometry

Deep-regime FPR/FNR: production arm — winner 0.103/0.195, x-cal 0.085/0.274
(x-cal FNR-heavy); control arm — winner 0.168/0.221, x-cal 0.397/0.106 (x-cal
FPR-heavy). The population fit centers the cut on both arms; the conformal cut
drifts to opposite extremes depending on geometry.

### H3 — stability (mean |Δthreshold| per step, past the ramp)

label-anchored κ=1 rate 0.0009 · **fold-anchored κ=1 rate 0.0029** ·
rank-transfer 0.0079 · pure x-cal / blend 0.0081.

### Estimator provenance

`fold_anchored_w1_rate`: 51,181/51,226 steps fully anchored (2/2 folds),
34 with 1/2, 11 with 0/2. `anchored_w1_rate`: 5 `inverted_means` fallbacks in
53,710 fits. Winner's degenerate-cut rate in the window tables: 0.0
everywhere.

## The H4 question, honestly

The mechanical check compares the winner's pooled FNR to the *nominal* 0.25
budget per window: 0.30 in 2–20 → fail. Framing:

- Pure x-cal — whose budget it nominally is — runs **0.45** in that window
  (0.37 at 21–50, 0.30 at 51–100). Under the plan's "measured envelope"
  wording the winner passes every checkpoint: it never gives up recall vs the
  status quo until past 200 votes, and there only +0.03 against a −0.08 cost
  gain, with absolute FNR under the nominal budget.
- Per #2799's structural finding the app's first trained-detector step is at
  7 votes — part of the failing window is user-invisible, and in the visible
  part the winner is the *closest* arm to the nominal budget.
- The nominal budget is simply not achievable at 2–20 votes by any measured
  rule — that is the sample-size deficit this line of work started from.

**Reading:** H4's intent (no recall regression) is satisfied; its letter
(under 0.25 everywhere) is not, by one window in which every arm including the
incumbent fails harder. Decision rule 1 is met under the measured-envelope
reading and blocked under the nominal one. Owner's call; the recommendation
below assumes the measured-envelope reading.

## Caveats & open threads

- **The blend control is the 6→20 ramp, not `slow_cap50`.** PR #2849 (the
  #2841 per-mode schedules; region voting now ships `slow_cap50`) merged to
  dev *after* this run's base commit. Fusion vs `slow_cap50` head-to-head is
  therefore untested; both harness arm families coexist on current dev, so the
  comparison is a cheap rerun off tip. Until then, H2 reads "fusion beats the
  ramp", not "fusion beats every schedule".
- **κ=1 is the edge of the sweep.** The winner sits at the grid boundary;
  κ<1 is unexplored and the monotone trend says it could be better still.
  Fold count 2 also degenerates the qmean/qmedian comparison — a
  κ ∈ {0.1, 0.3, 1} × folds ∈ {2, 4} follow-up closes both.
- **VG region voting only, 4 seeds, linear head, simulated voting** — same
  scope as #2799/#2836 for comparability; binary voting, other datasets, and
  the MLP head are unmeasured.
- **The fold arm costs one haystack scoring pass per fold per step** — trivial
  for the linear head; measure before enabling for heavy heads.
- **The label-anchored family is a trap at high κ.** Any adoption should ship
  the fold-anchored path (or κ≤1 label-anchored as a fallback when fold scores
  are unavailable), never heavy final-scale anchoring.
- Per-window FNR/cost aggregates weight steps equally; deep windows hold more
  steps. The paired contrasts (the decision numbers) are unaffected.

## Recommendation & next steps

1. **Adopt `fold_anchored κ=1 rate` as the production threshold path**
   (decision rule 1, measured-envelope reading), with the safe blend retained
   as fallback for datasets too small to fit the population estimator and the
   per-fold degeneracy chain as implemented. Production items per the plan:
   threshold path in `vtscore/detectors/training.py` / `thresholds.py`,
   cache-key and Stats-chart implications, tests in `tests_lib/detectors/`.
2. **First, run the cheap boundary sweep**: κ ∈ {0.1, 0.3, 1} ×
   folds ∈ {2, 4}, plus the `slow_cap50` blend as a control arm (now on dev).
   De-risks the grid-edge winner, un-degenerates the combine question, and
   closes the H2 gap left by the #2849 timing.
3. Reconcile the #2799 report's scope note (its selection-feedback attribution
   stops at 30 votes; this run shows the population term itself keeps paying
   to 300).

## Reproduction

- Worktree `/exp/sgreenberg/projects/vts-anchored-2852` @ dev `0a54f0d7`;
  launch via `launch_anchored.sh` knobs (safe+anchored ON, linear head,
  VG × {siglip, dinov3_patch}, max_patch, `CALIB_MAX_STEPS=300`, 4 seeds).
  Ops for this run: prepare reused from
  `/exp/$USER/calibration-safe-linear/results` (#2836 pattern — same arms,
  nothing to embed); cells on the CPU partition (24G / 4 cpu / %40), dodging
  the 4-GPU QOS cap. siglip cells ≈ 3 min; dinov3 max_patch cells ≈ 40 min.
- Jobs: cells array 468311 (184 cells), analyzer 468312 chained `afterany`
  GRID-side; 0 failures.
- Outputs: `/exp/sgreenberg/calibration-anchored/results/` — analyzer
  `REPORT.md` draft, `summary.json` (mechanical H1–H4), `agg/*.csv` (window /
  paired / stability / provenance), `cells/task_*.csv` (per-step rows incl.
  `threshold_provenance`).
- `selftest_analyze_anchored.py` (planted-answer analyzer self-test) passed
  before launch.
