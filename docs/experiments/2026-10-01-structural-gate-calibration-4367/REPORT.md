# The structural returned set: a line above the Bads (#4367)

**Question.** The first State of the App: Structural Document (#4392)
compared two things on FullMarks documents, all 36 classes at ~47,000 pages:
the set the app returns, which is every verified page with ≥ 8 inliers, and the
best cut of the ranking itself. The returned set scored **F1 0.43**; the
ranking's best cut scored **0.93**. The gate erred in both directions:

- **Small classes:** it passed hard negatives. These clear 8 inliers easily on
  a page with ~6,000 keypoints.
- **Large classes:** it missed positives beyond the verified shortlist.

Which accept rule closes the gap? Rules and verdict were pre-registered on
#4367 before any run.

**Answer: return only pages that fit better than every Bad vote did.** This is
rule R1. It takes the returned set from **F1 0.43 to 0.85** at 25 clicks, and
the ranking is untouched. Measured end to end through the app:

| through the app, 36 classes, held-out half | AP | returned set: precision / recall / F1 |
|---|---:|---|
| 10 clicks, before → after | 0.90 | 0.39 / 0.92 / 0.44 → **0.55 / 0.89 / 0.55** |
| 25 clicks, before → after | 0.91 | 0.38 / 0.93 / 0.43 → **0.87 / 0.89 / 0.85** |
| 50 clicks, before → after | 0.91 | 0.37 / 0.93 / 0.42 → **0.89 / 0.87 / 0.85** |

The ranking's best cut, the oracle, is 0.92–0.93. At 10 clicks the gain is
smaller because the early votes are mostly Goods: until a Bad is cast, R1 *is*
the shipped gate.

## The rules (offline, on per-page frames at 10 / 25 / 50 clicks)

| rule | F1 @25 vs shipped (95%) | large-class (≥ 50 positives) precision @25 | verdict |
|---|---|---|---|
| R1: verified, and more inliers than any Bad | **+0.43** [+0.32, +0.53] | +0.09 | **ships (simplest that passes)** |
| R2: a threshold fit to the votes (Goods leave-one-out vs Bads) | +0.44 [+0.32, +0.55] | +0.09 | passes; not simpler |
| R3: R2 plus unverified pages by Stage-1 score | +0.26 [+0.10, +0.41] | **−0.21** | fails (large-class precision) |
| R4: oracle best cut (reference) | +0.50 | +0.26 | |

- **Why R1 works:** a Bad vote shows how well a page that is *not* the mark
  fits these templates. On documents the hard negatives fit with 35–52
  inliers, far above 8. Anything that fits no better than a Bad is no
  evidence of the mark.
- **Why R3 fails:** reaching past the verified shortlist on Stage-1 scores
  buys the large classes recall, but at a cost in precision the pre-registered
  bar rules out. The large-class misses are a Stage-1 problem (#4415), not a
  line problem.

## What changed

- **`VerificationScorer.score` is now n / (n + 8).** The order is the same, and
  so is the 0.5 point at the 8-inlier gate. The difference is that it never
  saturates. The old `min(1, n / 16)` put every fit with 16 or more inliers at
  1.0, so a line at, say, 31 inliers was not expressible.
  `threshold_for(n)` maps an inlier count onto that scale.
- **`maybe_structural_rerank(..., bad_votes=...)`**: on a tiled dataset with
  Bad votes, the returned threshold is `threshold_for(max(8, best Bad inliers
  + 1))`. Bads set the line only; they still never enter the ranking (#4169).
- **Untiled (photo) datasets keep the plain gate**, because R1 was measured on
  documents. Their scores are on the new scale, but the order and acceptance
  are unchanged.
- **Harness:**
  - `sota_documents.py --frames` saves per-page frames;
  - `gate_rules.py` scores the rules offline;
  - the review's "returned set" now reads the line the app returns.

## Reproduce

```bash
source scripts/experiments/pile/pile_env.sh
python scripts/experiments/fullmarks/sota_documents.py --tier m --max-v 50 --frames 10,25,50 \
    --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-m \
    --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/run
python scripts/experiments/fullmarks/gate_rules.py --frames <dir>/run/frames --out <dir>/rules
```

Run directory: `/expscratch/sgreenberg/gate-4367/`. `run/` holds the frames,
`rules/` the offline scores, and `validate/` the end-to-end run with R1 in the
app. `measurements/` holds the rule scores and the validation's `steps.csv`.
