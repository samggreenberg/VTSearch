# A stop-list from Bad votes in the app (#4170, #4180): not shipped

**Question.** Should each Good-box template, and the Stage-1 query built from
it, drop the descriptors a Bad vote's page also matches? The State of the App:
Structural Document (#4392) found the largest harmful click is a correct Good
whose box takes in typed text (#4170). #4162 measured a stop-list offline
(#4180). Arms and verdict were pre-registered on both issues before any run.

**Answer: no, as built.** Neither arm clears the AP bar, and both break the
per-vote budget.

| vs the shipped path, FullMarks v5.0, 36 classes, tier `m`, end to end | AP at 25 clicks (95%) | retrain median / p90 / max | clicks costing > 0.1 AP |
|---|---|---|---:|
| shipped | 0.93 | 0.2 / 3.2 / 82 s | 3 |
| S: prune against every Bad | +0.011 [−0.002, +0.030] | 3.0 / **13.2** / 44 s | 5 |
| SG: prune against Bads that clear the gate | +0.010 [+0.000, +0.027] | 2.4 / **11.2** / 54 s | 4 |

**What it does fix.**
- **#4170's class recovers.** `staver/stamp_stampds-00213_1` scores 0.71 at 50
  clicks shipped, 0.92 under S and 0.85 under SG.
- **#4180's guard works.** `tobacco800/logo_aeq93a00_1`, whose Bad shows the
  mark inside a lockup, is unchanged (0.89): keeping descriptors another Good
  also matches protects the mark.

**Why it costs so much.** A pruned template is a new template, so the
verification cache misses for it. Every Bad vote re-prunes every template it
touches (all of them under S, the ones it fits under SG), and each of those is
re-verified against the whole 2,000-page shortlist: about 1 s a template on
an L40S.

**The obvious cheaper design** (#4432). Pruning only removes descriptors, so a
pruned template's inliers on a page can only fall. It only needs re-verifying
on the pages its unpruned self passed (≥ 8 inliers), typically tens of pages,
not 2,000. Whether the +0.01 AP is real needs that cheaper arm plus more
classes or repeats; at 36 classes the gain is not resolvable.

## What changed

`STOPLIST_POLICY` (`"off"`, the default; `"all"`; `"gated"`) in
`structural_similarity.py` and `sota_documents.py --stoplist` stay, so the
arms reproduce. **The shipped path is unchanged.**

## Reproduce

```bash
source scripts/experiments/pile/pile_env.sh
python scripts/experiments/fullmarks/sota_documents.py --tier m --max-v 50 --stoplist gated \
    --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-m \
    --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/arm-gated
```

Run directory: `/expscratch/sgreenberg/stoplist-4170/`. `measurements/` holds
each arm's `steps.csv`, plus the shipped path's (#4415's arm B, the same code
with the stop-list off).
