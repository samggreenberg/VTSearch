# Is 2 still the right number of cross-calibration folds?

**Issue #2897 · design pre-registered in a plan file, deleted by the PR that added this
report · harness PR #2902 · run + fixes PR #2906 · follow-ups #3115, #3116**

The decision rules below (`MARGIN`, `COST_CEILING_X`, the deep regime, H1–H4)
were fixed before the run and live as module constants in
`scripts/experiments/calibration/analyze_folds_2897.py`; the analyzer applies
them mechanically. Nothing here was chosen after seeing the numbers.

| | Screen (counterfactual) | Live A/B |
|---|---|---|
| Date | 2026-08-07 | 2026-08-07 |
| Base | `run/folds-2897` @ `7510ac91` (dev `9d004d0f`) | same |
| Cells | 208/208, 0 failures, 0 zero-byte (SLURM 480882 → 480883) | 208/208 per arm, 0 failures (481194 k=2, 481195 k=6) |
| Grid | K ∈ {1, 2, 3, 4, 6, 8, 12, 16} | `calibrate_count` ∈ {2, 6}, each carrying its own K=2 counterfactual |
| Environments | `visual_genome_m` × {`siglip`, `dinov3_patch`/`max_patch`}, `caltech101_m` × `siglip` | same |
| Sizing | 4 seeds × 150 steps, 23 VG categories (scale bands) + 6 Caltech (prevalence spread) | same |
| Path | inclusion 0, `linear` head, safe thresholds on, `prod` blend schedule, acquisition offset −3 | same |

## BLUF

**Keep `calibrate_count = 2`. The study ships nothing.**

Binary voting says so outright: no fold count beats 2 by the pre-registered
margin, and in the deep regime more folds *monotonically hurt*. Region voting
produces a mechanical recommendation of K=6, but it should not be acted on —
its cost is 2.68× production's on the interactive retrain path, its live
effect falls below the ship margin once acquisition feedback is allowed to
move, and the mechanism check that was supposed to validate it **cannot be
read**, for reasons that are a defect in the instrument rather than a result
(see *The H4 test does not work*).

Runtime does rise linearly in K exactly as predicted. The benefit half of the
prediction only holds in one of the two voting modes, and weakly.

## Verdicts

| | Binary (Caltech × siglip, VG × siglip) | Region (VG × dinov3_patch) |
|---|---|---|
| H1 — any K beats 2 by MARGIN=0.005 | **No** | Yes — K ∈ {6, 8, 12, 16} |
| H2 — affordable within 4× cost | — | K ∈ {6, 8} |
| **H3 — recommendation** | **keep 2** (`h3_kept_production`) | K=6, at 2.68× cost |
| Best K ignoring cost | K=1 | K=16 (−0.0075) |
| H4 — benefit is `rule_inefficiency` | n/a (H1 failed) | **unreadable — see below** |
| A/B `screen_agrees` | true | true (p = 0.29) |

### Binary: more folds are actively worse

Deep-window (≤200 votes) regret climbs monotonically with the fold count:

| K | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| regret | 0.00125 | 0.00200 | 0.00257 | 0.00333 | 0.00426 |

Binary voting at this depth is already at the floor — regret 0.0014 at ≤100
votes — so there is nothing for extra calibration to buy, and what remains is
cost plus the contamination discussed under #3115.

### Region: a real but small effect, at a price

Regret at ≤100 votes falls 0.1364 (K=1) → 0.1290 (K=2) → 0.1220 (K=6) →
0.1200 (K=16); paired Δ vs K=2 is −0.0067 at K=6 (p = 1.5e-4, win rate 0.63).
The exchange rate peaks at K=6 (−0.0043 regret per extra second) which is what
drives the recommendation.

Against that: **`cal_share` at K=6 is 0.76** — three quarters of every step is
calibration the user waits through after each vote — rising to 0.885 at K=16.
The pre-registered 4× cost ceiling admits this; a ceiling set with the
interactive path in mind probably would not. Region regret is ~0.12 against
binary's ~0.0014, i.e. two orders of magnitude more headroom, which is the
simplest explanation for why only this mode shows a gain.

### The A/B validates the screen, and shrinks the effect

The nested-prefix screen is structurally blind to acquisition feedback, so two
full runs lived at K=2 and K=6:

| voting | live Δregret | screen Δregret | live − screen | p (unpaired) | agrees |
|---|---|---|---|---|---|
| binary | +0.000705 | +0.000470 | +0.000234 | 1.00 | true |
| region | −0.003532 | −0.004962 | +0.001430 | 0.29 | true |

No significant disagreement, so the nested-prefix trick is sound and the cheap
screen is a valid proxy — a useful methodological result in its own right. But
the live region point estimate, **−0.0035, sits below the pre-registered
MARGIN of 0.005**. On the number that describes what a user would actually
experience, K=6 does not clear the bar either.

## The H4 test does not work

H4 asked whether K's benefit lands on `rule_inefficiency` (sampling noise in
the cut) rather than the calibration→test `shift` term, which the plan asserted
"K cannot touch at all". The run reported the benefit landing entirely on the
shift term. **That reading should not be trusted, because the test is
ill-posed.** From `vtscore/eval/voting_iterations.py:655-661`:

```python
c_thr             = oracle_cut(cal_scores, cal_labels)   # best cut ON the pooled calibration set
rule_inefficiency = cost - cal_oracle_cost
calibration_shift = cal_oracle_cost - o_cost
```

`c_thr` is estimated from the pooled calibration set, and **that set grows
linearly in K** (`n_cal_scores` 6.25 → 100 across the grid). Sweeping K moves
the reference cut that defines the split. As K rises, `c_thr` converges on the
test oracle, which shrinks `calibration_shift` and widens `rule_inefficiency`
from a single cause, in opposite directions, with the sum pinned to regret by
construction. The identity holds exactly on the shipped numbers — binary
`le_20` K=1: `−0.291185 + 0.355423 = 0.064238` against a reported regret of
`0.064237` — so the anti-correlation is algebraic, not empirical.

Two things follow. The plan's premise is false by construction:
`calibration_shift` is *defined through* an estimate taken from the K-dependent
calibration set, so of course K touches it. And `rule_inefficiency` is not a
variance — it is a signed cost gap between two cuts, and it was **negative in
every row**, rising toward zero (binary `le_20`: −0.291 at K=1 → −0.080 at
K=16). A negative value means the trained conformal cut beats the
calibration-set oracle on test, i.e. the oracle is overfitting a handful of
scores. **Nothing in this run says variance rose.** The run neither confirms
nor refutes the variance story; it failed to ask the question. #3116 carries
the fix.

## Limitations

- **The blend arm swept only half the threshold.** `_fold_count_variant_rows`
  fits the GMM cut once, outside the fold-count loop, and reuses it for every
  arm, so only the pooled cross-calibration component varied with K while the
  anchored component stayed frozen. Conclusions about what `calibrate_count`
  does to the *shipped* threshold are therefore partial. Tracked in #3116.
- **`visual_genome_m × siglip` is not a region-voting environment.** It carries
  no `patch_grid` (0 of 4193 media), so `region_voting=True` silently falls back
  to whole-image training and scoring. It is grouped with binary above. Only
  `dinov3_patch` (4193/4193) region-votes. The same trap as #2877; `dev` now
  raises a `RuntimeWarning` for it.
- **Pooling vs averaging was never tested.** The swept path pools fold scores;
  the anchored path averages per-fold quantiles, and the two carry contradictory
  claims about whether fold scores are comparable. Tracked in #3115.
- **The 4× cost ceiling was pre-registered without reference to `cal_share`.**
  At the recommended K it admits an arm that spends 76% of each step calibrating.

## Follow-ups

- **#3115** — pooled vs `qmean` vs `qmedian` fold combination. Only measurable
  at K ≥ 3, since mean and median coincide at the shipped 2.
- **#3116** — re-decompose against a fixed reference, emit `sd(threshold)` per
  K, and unfreeze the GMM cut in the fold-count arms.

Both are answered by one run and should not be run separately — that run is
[`calibration-fold-combine/`](../calibration-fold-combine/REPORT.md), which also
corrects two errata recorded above: the `voting` label is now derived per cell
from `experiment_config.region_voting_for` rather than from the dataset name (so
`visual_genome_m × siglip` groups as binary in the code and not only by hand),
and no head is pinned.
