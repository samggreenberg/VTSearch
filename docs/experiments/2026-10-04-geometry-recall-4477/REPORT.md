# The "Bad ceiling's recall cost" is H1's geometry cuts (#4477): measured, no change

**Question.** #4457 found 35–53 verified positives held back below the line in
`tobacco800/logo_ajj10e00_1`, and read them as the Bad ceiling's recall cost (#4367). Which Bad
sets that ceiling, and should one Bad be able to raise the line for the whole class?

**Answer.** There is no Bad. Every one of the class's 50 clicks is a Good, so its line all session
is H1's (#4440): at least 8 inliers **and** a tight fit (#4434's cuts). The held-back positives
are loose fits. For this class the cuts are the right trade at beta 1. Two pre-registered rules
that relax them failed class-split CV, so **the line does not change.** At beta 4, #4475 already
returns loose fits above a per-detector floor.

## The held-back positives are loose fits

At the final click (tier `m`), verified pages with at least 8 inliers:

| `ajj10e00` | positives, tight | positives, loose | negatives, tight | negatives, loose |
|---|---:|---:|---:|---:|
| replicate 1 | 153 | **35** | 9 | 155 |
| replicate 2 | 121 | **52** | 8 | 165 |

- **Accepting the loose fits here would add 35 positives and 155 negatives.** That takes the
  returned set's F1 from 0.84 to 0.68.
- **Over the 9 sessions that end without a Bad,** loose fits hold 157 positives and 356
  negatives.
- **One session goes the other way:** `ald41a00`, replicate 2, has 68 loose positives against
  30 loose negatives, and only 38 tight positives.

## Two rules that relax the cuts, both failed

Both were tested by the class-split CV of #4458 round 2: the classes split by sha256, both
replicates used, and every class scored by a rule chosen without it. The choice was made at
beta 1. The bar: at beta 1, the mean share of the best cut over clicks 0–25 above shipped, with
the lower bound > 0; at beta 1 and 1/4, which share this line, no click with a lower bound below
−0.02. Both rules change only states with votes but no Bad.

1. **Drop the cuts when most of the Goods' own fits are loose** (pre-registered before any
   result). It **never fires.** The Goods are clicked from the top of a ranking in which H1
   already demotes loose fits, so their leave-one-out fits are tight almost without exception:
   2/2, 5/5, 25/25. The one exception is `ald41a00` replicate 2 at click 50 (28/50). Held-out
   shares equal shipped at every click (`measurements/hypothesis1-cv.md`).
2. **Let a loose fit pass above a high floor**, max(t, ceil(q × the Goods' median leave-one-out
   inliers)). This rule was designed after looking at these frames, and pre-registered before it
   was computed.

   | | over clicks 0–25 [95%] | worst click |
   |---|---|---|
   | beta 1 | +0.017 [+0.003, +0.036] | click 1: −0.018 lower bound |
   | beta 1/4 | −0.001 [−0.007, +0.006] | **click 1: −0.048 lower bound** (fails) |

   The two folds chose different floors: t=64 alone, and t=24 with q=0.5
   (`measurements/hypothesis2-cv.md`).

## Reproduce

```bash
cd scripts/experiments/fullmarks
python geometry_goods_cv.py [--family loose-floor] \
    --frames /expscratch/sgreenberg/balance-4458/m-rep1/frames \
    --frames /expscratch/sgreenberg/balance-4458/m-rep2/frames \
    --cuts /expscratch/sgreenberg/multistat-4434/cuts-s/cuts.json --out <dir>
```

The frames are #4458's (tier `m`, clicks 0–50, 2 replicates).
