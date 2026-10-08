# The structural line before the first Bad vote (#4440): shipped

**Question.** #4367's Bad ceiling, the shipped line, only sharpens the
returned set once Bad votes exist. Before the first Bad it is the plain
8-inlier gate, which passes hard negatives on documents: F1 0.56 at 10 clicks.
#4434 found that fixed geometry cuts help exactly there. Can a rule use those
cuts until the Bads arrive (H_k), without losing later? The rule and verdict
were pre-registered on #4440.

**Answer: yes. H1 ships.** Until a detector's first Bad vote, a verified page
must also fit tightly: inlier ratio ≥ 0.75 and median reprojection error ≤
0.004887.

| through the app, FullMarks v5.0, 36 classes, tier `m`, held-out half | 10 clicks | 25 clicks | 50 clicks |
|---|---|---|---|
| returned-set F1, shipped → **H1** | 0.56 → **0.79**, +0.23 [+0.14, +0.32] | 0.873 → 0.880, +0.008 [−0.010, +0.034] | +0.007 [−0.000, +0.019] |
| ranking AP, shipped → H1 | 0.930 → 0.920, **−0.010** [−0.018, −0.004] | −0.001 [−0.002, +0.000] | −0.002 [−0.005, −0.000] |
| retrain p90 | | | 3.0 s |

**The pre-registered bar** was F1 at 10 clicks above the shipped line (clear of
zero), F1 at 25 clicks not lower, and the end-to-end run reproducing both. All
three hold.

**The cost the bar did not anticipate.** The app's returned set is "every page
scoring at or above the line". So to apply the cuts, a loose fit scores half
its value, which demotes it below every tight fit in the verified head (failing
fits keep their order). Some loose fits are true positives, so ranking AP at 10
clicks dips by 0.010. **The owner chose to ship** (2026-10-02): +0.23 returned-set
F1 for −0.01 AP, with no difference by 25 clicks.

## How k was chosen (tier `s`), and the out-of-sample check (tier `m`)

**Choosing k.** k (the number of Bads before handing over to the ceiling) was
chosen on **tier `s`** frames at clicks 5 / 10 / 15 / 25. The pick is the best
mean F1, ties to the smaller k:

| rule | mean F1, clicks 5–25 |
|---|---:|
| shipped (Bad ceiling) | 0.64 |
| **H1** | **0.80** |
| H2 / H3 / H5 / H8 | 0.80 / 0.80 / 0.80 / 0.80 |
| H∞ (cuts throughout) | 0.80 |

**Scoring k = 1** on **tier `m`'s** frames, which are other pages, paired over
36 classes against the shipped line:

| clicks | F1 difference |
|---|---|
| 10 | +0.225 [+0.136, +0.321] |
| 25 | +0.015 [−0.010, +0.055] |
| 50 | +0.002 [−0.008, +0.014] |

The cuts themselves (ratio 0.75, reprojection error 0.004887) come from
#4434, also fit on tier `s` only.

## What changed

- **`VerificationScorer` takes optional `ratio_min` / `reproj_max`.** A fit
  failing either scores half its value: below the line, same order.
- **`maybe_structural_rerank`, on a tiled dataset with no Bad vote,** scores
  with `GEOMETRY_RATIO_MIN` / `GEOMETRY_REPROJ_MAX`. From the first Bad on, the
  plain scorer and the Bad ceiling apply as before.
- **Unchanged:** example sort (never measured with the cuts) and untiled
  (photo) datasets.
- **`gate_rules.py`** gains the H_k rules.

## Reproduce

```bash
source scripts/experiments/pile/pile_env.sh
python scripts/experiments/fullmarks/sota_documents.py --tier s --max-v 25 --frames 5,10,15,25 \
    --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-s \
    --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/frames-s
python scripts/experiments/fullmarks/gate_rules.py --frames <dir>/frames-s/frames \
    --cuts <cuts.json from #4434> --out <dir>/rules-s
python scripts/experiments/fullmarks/sota_documents.py --tier m --max-v 50 \
    --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-m \
    --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/validate
```

Run directory: `/expscratch/sgreenberg/hybrid-4440/`. `measurements/` holds
both tiers' rule scores and the validation and shipped runs' `steps.csv`.
