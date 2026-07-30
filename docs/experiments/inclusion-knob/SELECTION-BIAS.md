# Does the conformal Inclusion budget survive VTSearch's real vote collection?

**Verdict: largely yes, under Autopilot.** The `alpha(k)` false-negative budget
is a split-conformal guarantee, so it assumes the calibration votes are
exchangeable with the inference set — and VTSearch's votes are chosen by the
most biased sampler imaginable (the detector's own sort). But measured against
the **canonical Autopilot vote order**, the realized miss rate converges to the
budget at essentially the same rate as exchangeable random voting: mean FNR
excess at Inclusion 0-3 falls 0.298 → 0.085 → 0.050 → **0.004** across 12 → 24
→ 50 → 100 votes, versus 0.208 → 0.087 → 0.027 → 0.002 for uniform sampling.
At 100 votes, Inclusion 0 delivers recall **0.930** against a promised >= 0.75.

The reason is that Autopilot's sampling biases are **self-correcting for
recall**: its Hard phase votes at the decision boundary, which pulls the
calibration positives *below* the population's (`cal_q25 - pool_q25` is
**negative**: -0.054 on AG News, -0.136 on synth:easy), placing the cut
conservatively *low*. Its New phase adds coverage-atlas diversity on top. The
knob's guarantee is therefore not in practical danger from labeling bias — the
one regime worth flagging is **very early sessions** (~12 votes), where excess
runs 0.298 and the calibration set is both tiny and policy-shaped.

> ## Retraction
>
> An earlier version of this report concluded the opposite ("the budget fails
> under real labeling ... more voting makes it worse"). That conclusion came
> from a hand-rolled `toplist` arm that was **not** VTSearch's workflow: it
> greedily voted the top of the sort for every vote after the third. Real
> Autopilot votes the top only for its first `_GOOD_TARGET` Goods, takes its
> Bads from the *bottom* of the ranking, and then spends every remaining vote
> alternating **Hard** (nearest the threshold) and **New** (atlas diversity)
> — see `vtscore/eval/al_strategies.py`, which mirrors
> `frontend/src/app/services/autopilot-state.service.ts`. Margin sampling
> biases the calibration positives in the *opposite direction* from top-greedy,
> which is why the sign of the effect flipped. That arm is retained below,
> correctly scoped as a manual-review adversarial bound.

Companion to [REPORT.md](REPORT.md) (which established the conformal rule).
Primary harness: `scripts/experiments/inclusion_knob/run_autopilot_sweep.py`
(+ `summarize_autopilot.py`), which drives the repo's own
`vtscore.eval.al_strategies.select_next` over a real `CoverageAtlas` and mirrors
`vtscore.eval.voting_iterations.simulate_voting_iterations`' loop, so the vote
order is the app's by construction rather than by imitation. Raw grid:
[`autopilot_sweep.csv`](autopilot_sweep.csv); tables:
[`autopilot_tables.md`](autopilot_tables.md). 224 cells x 3 designs x 11
inclusions = 7,392 rows; run 2026-07-30.

## Method

Arms: 4 AG News one-vs-rest categories on real E5 embeddings + 3 synthetic
separability levels; 4 seeds; checkpoints at 12/24/50/100 votes. Every item is
assigned to a **simulation half** (votable) or a held-out **test half**
(all metrics), as the eval framework does. Policies:

* **autopilot** — the canonical selector, one production model fit +
  cross-calibration *per vote* (as the app retrains per vote), with the atlas
  labelled per vote so the New phase advances. AG News seeds from a genuine E5
  `"query: ..."` embedding of text a user would type; the synthetic arms have no
  text, so the selector takes its designed random-known-good seed path. The
  Good:Bad ratio is emergent and drifts *negative* with session length
  (0.35 → 0.23 Good from 12 → 100 votes) — the opposite of the discarded
  `toplist` sim's 0.72-0.92.
* **uniform** — stratified random votes, ~1/3 positive: the exchangeable
  reference the rule assumes.

Designs per checkpoint: production `conformal`, production safe-`blend`, and an
`oracle` (the same rule fed the test half's true scores and labels) that
isolates threshold placement from model quality and from the rule's own limits.

## Findings

### 1. Autopilot tracks the budget about as well as random sampling

Violation rate at Inclusion >= 0 is **0.558 for autopilot vs 0.632 for uniform**
— autopilot is marginally *better* in aggregate. At Inclusion 0 the violation
rate falls 0.536 → 0.321 → 0.214 → 0.071 across the vote checkpoints
(uniform: 0.571 → 0.286 → 0.000 → 0.000). Threshold placement versus oracle is
statistically indistinguishable between the policies (AG News: +0.032 both) and
does **not** diverge with session length (autopilot +0.017 → +0.029 from 12 →
100 votes; uniform +0.019 → +0.046).

### 2. The sampling bias exists — and points the safe way

`cal_q25 - pool_q25` is negative under autopilot on every arm except
`synth:hard` (AG News -0.054, synth:easy -0.136, synth:medium -0.084) versus
near-zero under uniform. Margin sampling concentrates labeled positives near
the boundary, so the 25th-percentile read lands *low* and the cut is placed
conservatively. That costs precision, not recall — which is why the FN budget
survives. On the hardest arms autopilot's boundary is also simply better
learned: `synth:hard` precision **0.556 vs 0.359** for uniform at the same
inclusion.

### 3. Most residual "violation" is not selection bias at all

The oracle reference — a *perfect*, fully representative calibration set —
still violates the cap at 0.223 (k=1), 0.580 (k=3) and 0.772 (k>=7), while
sitting at just 0.049 at k=0. So beyond k=0 the cap is finer than any finite
calibration set can certify and, on overlapping tasks, finer than any threshold
can achieve. Attributing those violations to labeling bias would be a mistake;
they are the resolution/irreducibility floor already noted in REPORT.md
("~4 usable knob positions at 24 votes"). Selection bias is the *smaller*
term nearly everywhere.

### 4. The safe-blend is inert past 20 votes, and that is now fine

`calculate_safe_threshold` ramps the GMM population threshold to zero weight at
20 labels, so blend == conformal at 24+ votes. At 12 votes it halves autopilot's
excess (0.336 → 0.167). Since conformal's own excess converges to 0.007 by 100
votes, the expired ramp is no longer a gap worth closing on FN grounds — but the
12-vote regime is exactly where it still earns its keep, which argues for
keeping (not extending) the current ramp.

### 5. Cold-start is the real weak spot

At 12 votes autopilot's excess (0.298) exceeds uniform's (0.208) and Inclusion 0
recall is only 0.585. Both policies are bad here for the same finite-sample
reason, but autopilot is additionally the least exchangeable at this point in
the session (the vote set is still mostly seed/Bad-phase picks, before the New
phase has diversified anything). Any future mitigation should target the first
~20 votes specifically.

## Limitations

* The simulated user labels with ground truth and never mis-votes; a real user's
  label noise would enter the calibration quantiles directly.
* The `known_good` seed path (synthetic arms) draws its first Goods from
  ground-truth positives, standing in for the handful of examples a user
  supplies by hand. Text-seeded (AG News) and known-good (synthetic) cells reach
  k=0 recall 0.821 and 0.768 respectively, so the seed mode matters less than
  vote count.
* No region-bag (grouped) arms: the grouped path max-pools each voted image's
  regions to one calibration score and floods Bad votes with every region of the
  image, which reshapes the calibration distribution in ways this study does not
  measure. Tracked in `docs/plans/inclusion-calibration-bias.md`.
* The binary violation rate is coarse at high k (any FNR > 0.0002 counts at
  k=10); `fnr_excess` magnitudes are the load-bearing numbers.

## Superseded arm: `toplist` (manual top-of-sort review)

`run_selection_sweep.py` and [`selection_sweep.csv`](selection_sweep.csv) are
retained for the `toplist` policy: greedy top-of-sort voting in batches of 8.
That is **not** Autopilot, but it is a fair model of a user manually reviewing
and voting down a learned-sort result list (the Find-and-verify flow), and it is
a useful adversarial bound. There it behaves as badly as first reported: excess
*grows* with votes (0.254 → 0.410 from 12 → 100), Inclusion 0 recall falls to
0.474 (0.357 on AG News), and thresholds run up to +0.099 above oracle. Note the
arm also lacks a sim/test split, so its evaluation set is depleted of the high
scorers it voted on. Read it as "how bad could a purely exploitative labeling
policy get", not as a production estimate.

## Recommendations

1. **No change to the conformal rule.** The budget holds under the real vote
   order once a session has ~24+ votes; redesigning the rule to fix a bias that
   converges on its own would be unwarranted. In particular, the previously
   recommended "inject exploration votes" is **already shipped** — it is
   Autopilot's atlas-driven New phase, and it is part of why the budget holds.
2. **Scope the guarantee honestly in the docs.** `docs/ML.md` should say the
   `alpha(k)` budget is calibration-relative and needs a couple of dozen votes
   before it means much; at ~12 votes the realized miss rate can exceed the cap
   by ~0.3.
3. **Do not extend the safe-blend ramp.** Its remaining value is concentrated in
   the sub-20-vote regime it already covers (finding 4).
4. **Keep propensity-weighted quantiles on the shelf, not the roadmap.** With
   selection bias this small relative to finite-sample and irreducibility terms,
   weighted conformal would be optimizing the wrong term. Revisit only if a
   labeling flow appears that is genuinely exploitative — the `toplist` arm shows
   what that would look like.
5. **If anything is worth building, it is cold-start calibration** (finding 5),
   not bias correction.

Follow-ups are tracked in
[`docs/plans/inclusion-calibration-bias.md`](../../plans/inclusion-calibration-bias.md).
