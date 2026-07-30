# Does the conformal Inclusion budget survive biased vote collection?

**Verdict: no.** The `alpha(k)` false-negative budget the conformal quantile
rule advertises holds - and converges with more votes - when votes are
sampled uniformly, but under a simulated production labeling session (vote
on what the sort surfaces) the realized miss rate exceeds the budget in
~68-80% of cells, by ~0.33 mean FNR at the default Inclusion 0, **and more
voting makes it worse, not better.** The failure is threshold placement
caused by non-exchangeable calibration data, not model quality and not the
vote class ratio.

Companion to [REPORT.md](REPORT.md) (which established the conformal rule);
this study measures the rule's core assumption. Harness:
`scripts/experiments/inclusion_knob/run_selection_sweep.py` +
`summarize_selection.py`; raw grid in [`selection_sweep.csv`](selection_sweep.csv);
tables in [`selection_tables.md`](selection_tables.md). Run 2026-07-30,
224 cells x 3 designs x 11 inclusions = 7,392 rows.

## Question

`conformal_threshold` reads the Inclusion cut off quantiles of *held-out
calibration scores of the user's votes*. Split-conformal semantics ("miss at
most `alpha(k)` of true matches") require the calibration examples to be
exchangeable with the inference set. VTSearch's labeling loop violates that
by construction: the user votes on whatever the current sort ranks highest,
so labeled Goods are score-biased high (tail matches are never surfaced) and
labeled Bads are hard negatives. How much does that cost in realized FNR?

## Method

Same production-faithful skeleton as the original sweep (`train_model`,
`compute_fold_orderings` with `calibrate_count=2`, `calibration_fraction=0.5`,
`_auto_hidden_dim`); the **only** manipulated variable is which items get
labeled:

* **uniform** - stratified random votes, ~1/3 positive (the exchangeable
  baseline, as in `run_sweep.py`).
* **toplist** - a simulated session: a cosine query (normalized mean of 3
  random positive exemplars) seeds the first votes from the top of its
  ranking - the simulated user scrolls past surplus matches to cast the >= 2
  Bad votes training requires - then each round trains the production MLP on
  the votes so far and labels its top 8 unvoted items, until the vote budget
  (12/24/50/100) is spent. The Good:Bad ratio is emergent (it lands at
  0.72-0.92 Good).

Arms: 4 AG News one-vs-rest categories on real E5 embeddings + 3 synthetic
separability levels; 4 seeds. Two references bracket every conformal cut:

* **oracle** - the identical conformal rule fed the entire ground-truth pool
  as calibration: the threshold a perfectly representative, effectively
  infinite calibration set would produce *for the same trained model*.
* **blend** - production `calculate_safe_threshold` (GMM on the full pool
  score distribution, ramped out by 20 labels), the one production input
  immune to labeling bias.

Metrics per cell x inclusion: pool confusion numbers, the `alpha(k)` cap,
`fnr_excess = max(0, fnr - alpha)`, and composition diagnostics
(`cal_pos_q25` vs `pool_pos_q25`: where the k=0 cut is read from vs where a
representative sample would put it).

## Findings

### 1. Under exchangeable sampling the budget works - and converges

Uniform-policy mean FNR excess at Inclusion 0..3 falls from 0.188 (12 votes)
to 0.072 (24) to 0.024 (50) to **0.004 (100)**. The residual excess at low
vote counts is finite-sample slack (a 25th-percentile read off ~4 calibration
positives is coarse) plus the fold-vs-final model mismatch (the threshold is
calibrated on fold-model scores but applied to final-model scores); both
shrink as votes grow, exactly as split-conformal theory predicts. At k=0 the
uniform policy delivers mean recall 0.837 against the promised >= 0.75.

### 2. Real labeling breaks the budget, and more votes make it worse

Toplist mean FNR excess at Inclusion 0..3 *rises* with votes: 0.254 (12) →
0.357 (24) → 0.345 (50) → **0.410 (100)**. Every additional top-of-sort
round concentrates the labeled positives higher in the score distribution,
so the calibration quantiles drift further from the population's. There is
no "it washes out with more labels" - the bias compounds instead.

At the default Inclusion 0 the knob promises "keep >= 75% of true matches";
the toplist policy delivers mean recall **0.474** overall and **0.357** on
the real-geometry AG News arms (uniform: 0.854). Violation rate at k=0 is
0.679 toplist vs 0.223 uniform.

### 3. The failure is threshold placement - not model quality, not class ratio

The oracle reference uses the *same* trained model per cell and stays within
budget under both policies (mean k=0 FNR 0.172 toplist / 0.074 uniform, cap
0.25): given honest calibration data, the rule itself is sound even on a
model trained from biased votes. The gap is entirely in where the quantiles
sit: conformal thresholds run +0.066 (AG News) to +0.099 (synth:hard) above
oracle under toplist, ~2-3x the uniform gap, and the per-cell
`cal_pos_q25 - pool_pos_q25` shift averages +0.04..+0.08 under toplist
(p90 +0.26 on AG News) vs ~0 or negative under uniform.

The emergent Good-heavy vote ratio (0.72-0.92) is **not** the cause: the
rule's quantiles are class-conditional and ratio-immune. What matters is
*which* positives got labeled, not how many.

### 4. The safe-blend is not a mitigation as currently ramped

`calculate_safe_threshold` ramps the GMM population threshold out by 20
labels, so blend == conformal at 24+ votes - precisely the regime where
selection bias is worst (finding 2). At 12 votes it trims mean toplist
excess 0.247 → 0.200. The population-score information it carries is the
right kind (it cannot be biased by labeling), but the label-count ramp
removes it exactly when it is needed most.

### 5. Separately: extreme +k outruns calibration resolution under any policy

At k >= 5 the cap (`alpha <= 0.008`) is finer than what tens of calibration
positives can certify, so even the uniform policy "violates" (rate 0.77-0.88,
excess plateau ~0.116): the quantile saturates at the lowest calibration
positive. This is the known resolution limit from REPORT.md ("~4 usable knob
positions at 24 votes"), not selection bias - the toplist and uniform curves
converge there. It bounds how literally the halving-per-step semantics
should be read at high k on small vote sets.

## Limitations

* The simulated user labels with ground truth, votes in fixed batches of 8,
  and never explores off the top of the sort; real sessions are noisier and
  occasionally wander, which would *reduce* the bias. The cosine-exemplar
  query is a stand-in for a real text query.
* The evaluation pool excludes voted items, slightly depressing pool
  positive quantiles under toplist (which removes high scorers); this works
  *against* the measured effect, so the true gap is, if anything, larger.
* No region-bag (grouped) arms; the grouped calibration path max-pools per
  image but reads the same quantile rule, so the mechanism carries over.
* The binary violation rate is coarse at high k (any FNR > 0.0002 counts at
  k=10); `fnr_excess` magnitudes are the load-bearing numbers.

## Recommendations

In decreasing order of leverage:

1. **Fix the sampling, not the rule: inject exploration votes.** The uniform
   arm shows the rule needs no repair when calibration is representative -
   excess 0.004 at 100 votes. Surfacing an occasional score-stratified item
   for voting (even 1-in-5) would repair exchangeability at the source and
   also diversify training data. This is the only option that attacks the
   compounding in finding 2.
2. **Propensity-weighted conformal quantiles.** Log each vote's surfacing
   context (score/rank when shown) and take weighted quantiles
   (Tibshirani et al., weighted conformal prediction). No UX change, but
   weights estimated from tens of votes are high-variance; would need its
   own measurement pass before shipping.
3. **Rethink the safe-blend ramp.** The GMM population threshold is the one
   bias-immune signal already in production, and it is currently ramped out
   exactly where bias peaks. A permanent floor on its weight (rather than 0
   at >= 20 labels) is a cheap partial mitigation - but it dilutes rather
   than restores the budget semantics, so it complements (1)/(2) rather than
   replacing them.
4. **Until then, document the semantics honestly:** the `+k` budget is
   calibration-relative - "miss at most `alpha(k)` of matches *that resemble
   the ones you voted Good*." Under top-of-sort labeling the effective
   real-world budget at Inclusion 0 is closer to "keep ~half" than "keep
   three quarters."
