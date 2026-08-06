# How safe should safe-thresholds be? (#2841)

Safe-thresholds is ON for everyone since #2799/#2835. That decision was binary:
blend-vs-no-blend. But the blend has a free schedule inside it — **how much GMM,
for how long** — and nobody has ever measured it. This plan is the search for the
best mix-in curve.

**Status: machinery built and verified; Phase 1 (screen) launching.**

> **The stakes shrank.** Since the population-anchored adoption, the blend is no
> longer the shipped combiner - the fold-anchored estimator is (see
> [`docs/ML.md`](../ML.md)). The schedule now decides the threshold only for
> label sets too small to form calibration folds, and there is no longer a
> `safe_thresholds` setting to turn any of it off. Re-scope this search to that
> window before spending more compute on it.

<!-- item-sep -->

## Pre-registration (written before the screen was launched)

**Disclosure:** two single-cell smoke runs were inspected first, to prove the
harness emits correct rows (`prod`'s counterfactual row reproduced the live
blended threshold and cost to 0.000e+00 on both arms). One cell is far below
any power threshold and is not evidence, but it was seen, so the ranking rules
below are fixed here rather than after the fact.

**Primary metric.** Inclusion-weighted `cost` (= `wf·FPR + wn·FNR` at
inclusion 0, i.e. FPR + FNR), averaged over **app-visible steps in the 7–20
vote window** — 7 because that is the app's first trained-detector step, 20
because that is where the production ramp ends. Secondary: FNR, FPR, regret vs
the step's own oracle cut, and the degenerate-cut rate. AP/AUROC are reported
as guardrails only: a schedule changes the *threshold*, so a ranking move means
the acquisition path changed, not the blend.

**Reported separately for region voting and binary voting**, never pooled.
The issue explicitly allows different winners, and the two arms have different
score geometry (max-pooled vs single-vector), so a pooled number would hide the
answer rather than summarise it.

**Phase 1 → Phase 2 promotion rule.** The A/B runs take the union of: the top 3
schedules by mean ramp cost on the region arm, the top 3 on the binary arm, plus
`prod` (the incumbent) and `pure_gmm` (the issue's straw man) as fixed anchors.
`pure_xcal` is *not* re-run: it is safe-thresholds OFF, which #2799 already
measured and rejected.

**Verdict rule.** A schedule replaces `prod` only if, on the **A/B** runs
(not the screen), it beats `prod` on paired-cell mean cost with p < 0.01 on its
own arm, and does not lose on the other arm at p < 0.01. Ties go to the
incumbent. If region and binary voting disagree and both effects survive, the
recommendation is a per-mode schedule, which the app can express because the
schedule is resolved per training call.

**Known limitation, stated up front.** The screen re-cuts a single trajectory —
the one `prod` produced — so it cannot see that a different schedule would have
labelled different items. #2799 showed that channel is real and can carry a gain
past the blend's own authority. The screen therefore ranks candidates; only
Phase 2 decides.

<!-- item-sep -->

## Background: what the schedule is today

`vtscore/training/thresholds.py`:

- `safe_blend_weight(n_labels)` — linear ramp on the **x-cal** weight,
  `clip((n - 6) / 14, 0, 1)`: pure GMM at ≤6 labels, pure x-cal at ≥20.
- `blend_gmm_threshold(xcal, gmm, n)` = `w·xcal + (1-w)·gmm`, plus non-finite
  guards (both non-finite → 0.5).
- `calculate_safe_threshold(xcal, all_scores, n)` = the above with
  `calculate_gmm_threshold(all_scores)` (2-component GMM, **midpoint of the
  component means** since the #2833/#2838 revert).

The 6 and 20 are unexamined constants. The linear shape is unexamined. The
choice to schedule on **total labels** (rather than positives, or a reliability
estimate) is unexamined. Those three axes are the experiment.

<!-- item-sep -->

## Structural facts that constrain the search (from #2799)

- **The app's first trained-detector step is at 7 votes.** So the entire
  "pure GMM" window (n ≤ 6) contains *zero* user-visible steps. Any candidate
  schedule is judged on 7→20+, and a schedule that only changes behaviour below
  7 is a no-op for users. Corollary: `MIN_LABELS=6` is nearly free to move
  *down*, and consequential to move *up*.
- **The blended threshold feeds acquisition.** Autopilot's Hard phase picks the
  item nearest the decision threshold (`al_strategies._hard_pick_by_index`), so
  two schedules label *different items* and trajectories diverge. Within-step
  counterfactual re-cuts (what `analyze_safe.py` does) therefore **cannot** rank
  schedules on their own — they are a screen, not a verdict. The verdict needs
  full A/B trajectory runs, paired per **(arm, category, seed)** cell.
- #2799 measured the gain **persists past 20 votes** (cost −0.0197, FNR −0.0312,
  p=0.005) where the blend has no authority at all. Only selection feedback
  carries that. So schedule changes can pay off outside their own support.
- The #2799 win was calibration + acquisition, **not ranking** (AP unchanged).

<!-- item-sep -->

## Candidate mix-in families (to be pre-registered before launch)

**A. Endpoint moves, linear shape.** `(MIN, MAX)` ∈ {(6,20) production,
(6,12) fast handoff, (6,40) slow, (2,20), (6,60), (10,30)}.

**B. Shape at fixed endpoints.** Convex `w=t²` (hold GMM longer), concave
`w=√t` (hand off early), logistic/sigmoid, hard step at the midpoint.

**C. Schedule on a different statistic.** Total labels is a proxy for
calibration reliability; the actual binding constraint on conformal calibration
is the **rarer class**. Arms: `w(n_pos)`, `w(min(n_good, n_bad))`. Motivated by
#2790 (deep spikes = positive starvation → under-determined head) and #2825
(the vote *mix* is what drives sustained wrong-way runs, not the model class).
This family is the one most likely to beat a pure re-tune of A/B.

**D. Never fully hand off.** Cap the x-cal weight at `w_max < 1` (e.g. 0.8),
leaving a permanent GMM anchor. Directly answers the issue's "HOW safe" — maybe
the answer is "always a little safe". Cheap to test as a variant of A.

**E. Bound instead of blend.** Replace the weighted average with a trust region:
clamp x-cal to within ±δ of the GMM cut (δ shrinking with labels), or
`min`/`max`/geometric-mean combiners. The GMM becomes a sanity bound rather than
a blend partner. Different failure mode: a clamp is a no-op when x-cal is
sensible and only bites when it is wild — which is the actual pathology
(#2788 cold-start "admit nothing" cuts, which #2799 showed the blend eliminates
on the whole-image arm).

The issue explicitly allows **different winners for region voting vs binary
voting**, so every arm is reported per detection style, not just pooled.

<!-- item-sep -->

## Measurement design (two-phase, mirroring #2790/#2799)

- **Phase 1 — screen (cheap).** The harness already emits `xcal_threshold`,
  `n_votes` and the sim scores the GMM was fitted on per step
  (`_blend_safe_threshold` returns them for exactly this reason). So a large
  family of curves can be re-cut on *fixed* trajectories with no new simulation.
  Screens shape/endpoints broadly; cannot see acquisition feedback.
- **Phase 2 — verdict (full A/B).** Top ~4 candidates + production baseline +
  the issue's straw man (**pure GMM forever**, `w ≡ 0`) as full trajectory runs,
  paired per (arm, category, seed) cell, reusing `analyze_ab.py` from #2799.
  Pure-x-cal (`w ≡ 1`) = safe_thresholds OFF is already measured in #2799 and
  loses; it is the low anchor and need not be re-run unless free.

Sizing reference from #2799: a dinov3 `max_patch` linear-head cell is **188 s**
on `--partition=cpu --mem=24G --cpus-per-task=4`; 552 cells/arm (24 categories ×
12 seeds × 2 arms). CPU cells dodge the 4-GPU QOS cap entirely.

<!-- item-sep -->

## App/framework fidelity — one known deviation, live already

The app **short-circuits cross-calibration below 6 votes** and never computes an
x-cal threshold there:

- `vtscore/detectors/training.py:253` (`train_and_threshold`, Find path) →
  `threshold = NO_GOOD_THRESHOLD`
- `vtscore/detectors/training.py:719` (`_train_and_score_xy`, vote/labelset
  path) → `threshold = 0.5`

The eval harness (`voting_iterations._train_and_calibrate`) has **no such
skip** — it always computes the real x-cal value. Today this is harmless
*because* `safe_blend_weight(n<6) == 0` discards the x-cal input entirely… but:

1. The two app call sites disagree with each other on the discarded sentinel
   (`NO_GOOD_THRESHOLD` vs `0.5`), and that value is **not** discarded when the
   GMM returns non-finite — `blend_gmm_threshold` falls back to the x-cal input.
   So on a degenerate GMM the Find path admits nothing and the vote path admits
   half the collection. That is a real (if rare) live divergence today.
2. **Any candidate schedule with `w > 0` below 6 labels is unimplementable in
   the app as written** (family A's `(2,20)` arm, family C's positive-count
   schedules — `n_pos` is small exactly when `n_labels` is small). The
   short-circuit must be removed, or the arm is framework-only fiction.

Per the issue's standing rule (deviations get fixed **in favour of the app**):
the app is right to want to skip 400 epochs of fold training it will throw
away, so the fix is to make the skip *derive from the schedule* —
`safe_blend_weight(n) == 0` — rather than hard-code `< 6`, and to agree on one
sentinel. Then a schedule change automatically carries the skip with it and the
harness matches by construction.

<!-- item-sep -->

## Environment / provenance

- Local worktree `/home/samiam/Code/vts-mixin-2841`, branch
  `claude/mixin-schedule-2841`, cut from fresh `dev` @ `6866b7de`
  (= PR #2840, includes #2832/#2835/#2838).
- Sibling in flight: **#2836** (GMM *cut rule* theory — where the cut goes) in
  `/home/samiam/Code/vts-gmm-cut` + GRID `/exp/sgreenberg/projects/vts-cut2836`.
  Orthogonal to this issue (cut placement vs mix-in schedule) but touches the
  same functions — coordinate before either lands.
- Cached embeddings on the GRID:
  - **VG + Caltech**: `/exp/sgreenberg/max-patch/datadir/embeddings/`
    (`visual_genome_m__{dinov3_patch,siglip,siglip_l}.pkl`, `caltech101_m__*`),
    crops at `/exp/sgreenberg/max-patch/results/crops` — this is what #2799 and
    #2781 reused, no re-embedding needed.
  - **COCO**: the #2790 sweep cache `/exp/sgreenberg/threshold-stability/cache`
    (900M, `exemplars/coco` + `regions/coco`) — reached through the `scripts/sod`
    sweep runner, **not** through `scripts/experiments/calibration`, whose
    `experiment_config.DATASETS` only knows `visual_genome_m,caltech101_m`.
    Wiring COCO into the calibration harness (or porting the schedule arms into
    the sod runner) is an open task — the issue asks for both VG and COCO.
- `/exp` now has **394G free** (was 1.4G) — the volume was expanded, so the
  #2781/#2799-era disk anxiety no longer applies.
- GRID gotcha: `export VTS_REPO=<this worktree>` in every job env, or
  `common.setup_env()` silently imports the old `/exp/$USER/projects/vts-calib`
  checkout. GRID pytest needs `-o addopts=""` (venv lacks pytest-timeout).
