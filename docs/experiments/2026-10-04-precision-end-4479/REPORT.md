# The document line at the precision end of the balance (#4479)

**Question.** #4458 shipped the recall end: at beta >= 2 the document line returns a superset of
beta 1's set (#4475). Beta 1/4 kept the shipped line. Round 2's rules for it gained on average
but dipped at single clicks, and its two folds disagreed. Does the mirror image work: beta 1/4
returns a **subset** of beta 1's set?

**Answer: almost.** By class-split CV the rule gains +0.084 [+0.044, +0.126] share of the best cut
over clicks 0–25, and every click from 5 to 50 is clearly positive. But click 3's lower bound is
−0.023, which misses the pre-registered per-click limit of −0.02 by 0.003. By the rule set in
advance it does not ship. **The owner overrode the bar and shipped it (2026-10-04),** on the
strength of the overall gain, the clearly positive later clicks, and the end-to-end result.

## The rule

Beta 1/4's set is beta 1's set intersected with the verified pages with at least
T = max(16, ⌈0.5 × the Goods' median leave-one-out inliers⌉, the Bad ceiling + 5) inliers.
- **Click 0** (the example sort): the line rises from 8 to 16 inliers.
- **Votes but no Bad:** H1's tight-fit gate, and inliers >= T.
- **After a Bad:** inliers >= T. The ceiling margin of +4 is the refit choice.

It is a subset of beta 1's set by construction, so the slider stays monotone. The rule was
designed after round 2 and pre-registered on #4479 before it was computed.

## Class-split CV (tier `m`, 2 replicates, every class scored by a rule chosen without it)

192 rules: t ∈ {8, …, 24}, q ∈ {off, 0.25, 0.5, 0.75}, d ∈ {0, 2, 4, 8}, geometry after a Bad on
or off. The folds chose t=20, q=0.5, d=0 and t=16, q=0.25, d=4. The refit on all classes is t=16,
q=0.5, d=4.

| clicks | shipped | held-out | difference [95%] |
|---|---:|---:|---|
| **0–25** | 0.768 | 0.851 | **+0.084 [+0.044, +0.126]** |
| 0 | 0.555 | 0.750 | +0.194 [+0.041, +0.335] |
| 1 | 0.771 | 0.811 | +0.039 [+0.006, +0.084] |
| 2 | 0.785 | 0.850 | +0.065 [+0.015, +0.123] |
| **3** | 0.798 | 0.841 | +0.043 **[−0.023, +0.111]** |
| 5 | 0.780 | 0.864 | +0.084 [+0.043, +0.126] |
| 10 | 0.762 | 0.870 | +0.108 [+0.060, +0.156] |
| 15 | 0.805 | 0.892 | +0.087 [+0.044, +0.132] |
| 25 | 0.876 | 0.926 | +0.049 [+0.013, +0.093] |
| 50 | 0.899 | 0.932 | +0.033 [+0.010, +0.061] |

The intersection with beta 1's set removed round 2's late dips: clicks 15, 25 and 50 are now
clearly positive. The scorer matched a page-by-page reference exactly on tier `s`.

## End to end (in sample)

Through the app on the proposal branch, using the refit rule (`sota_documents.py --beta 0.25`,
2 replicates):
- **The frames predicted the app** in 322 of 322 steps, and ranking AP is unchanged, since this
  line reorders nothing.
- **Against the shipped app**, F(1/4) of the returned set over clicks 0–25: **+0.094 [+0.056,
  +0.131]**.
  - Click 0: +0.217. Click 3: +0.069 [+0.005, +0.135].
  - Precision 0.82 → 0.95; pages returned, median 16 → 13.
- **This is in sample** (the rule was refit on these classes). The CV above is the
  out-of-sample estimate.

## Reproduce

```bash
cd scripts/experiments/fullmarks
python precision_end_cv.py --frames /expscratch/sgreenberg/balance-4458/m-rep1/frames \
    --frames /expscratch/sgreenberg/balance-4458/m-rep2/frames \
    --cuts /expscratch/sgreenberg/multistat-4434/cuts-s/cuts.json --out <dir>
python sota_documents.py --tier m --max-v 50 --beta 0.25 [--swap-halves] \
    --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-m \
    --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/betaq-rep1  # and rep2
```
