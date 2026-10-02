# A multi-statistic accept rule for the structural line (#4434): not shipped

**Question.** #3349 found that among fits clearing the 8-inlier gate,
inlier ratio, reprojection error, scale and spread all help tell true pairs
from false (AUC 0.80–0.95). Does adding them to the shipped line improve the
returned set? The shipped line is #4367's Bad ceiling: a page must fit better
than any Bad vote did. Rules and verdict were pre-registered on #4434.

**Answer: not at the pre-registered point (25 clicks). But fixed geometry
cuts help a lot early, before the Bads arrive (filed as #4440).**

| returned-set F1, vs the shipped line (R1), FullMarks v5.0, 36 classes, tier `m`, held-out half | 10 clicks | **25 clicks** | 50 clicks |
|---|---|---|---|
| R1 (shipped), mean F1 | 0.56 | **0.87** | 0.87 |
| **M1:** R1 + ratio ≥ 0.75 + reprojection error ≤ 0.0049 (cuts from tier `s`) | **+0.21** [+0.11, +0.31] | −0.008 [−0.061, +0.045] | −0.005 [−0.026, +0.013] |
| **M2:** R1 + a logistic regression fit on the detector's votes | +0.001 [−0.029, +0.032] | +0.014 [−0.015, +0.044] | +0.017 [−0.012, +0.047] |
| R4: oracle best cut (reference), mean F1 | 0.94 | 0.94 | 0.94 |

- **The verdict is on 25 clicks, and neither rule clears zero there.**
  Neither ships.
- **M1 at 10 clicks.** Early on, a detector has few or no Bad votes, so the
  Bad ceiling is just the 8-inlier gate and over-accepts. M1's fixed cuts lift
  F1 from 0.56 to 0.77, and precision on large classes by +0.20.
- **M1 by 25 clicks.** The Bads are in, and the ceiling alone does as well.
- **M2** (cuts learned per detector from its own votes) never resolves.

**Out-of-sample by construction.** M1's cuts were fit on **tier `s`**: 167,545
box-template pairs past the gate, maximising pair F1 (0.80). They were then
scored on **tier `m`**'s frames, which are other pages.

## What is in the repo

- `sota_documents.py --frames` now also records each verified page's best-fit
  inlier ratio and median reprojection error, and the same for each Good
  (leave-one-out) and Bad.
- `gate_rules.py` adds `--fit-cuts`, and the M1 / M2 rules with an R1-control
  table.
- **The shipped path is unchanged.**

## Reproduce

```bash
python scripts/experiments/fullmarks/gate_rules.py --fit-cuts /expscratch/$USER/fullmarks/votes-4162/matrix-s \
    --frames /dev/null --out <dir>/cuts-s
python scripts/experiments/fullmarks/sota_documents.py --tier m --max-v 50 --frames 10,25,50 \
    --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-m \
    --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/run
python scripts/experiments/fullmarks/gate_rules.py --frames <dir>/run/frames --cuts <dir>/cuts-s/cuts.json --out <dir>/rules
```

Run directory: `/expscratch/sgreenberg/multistat-4434/`. `measurements/` holds
`rules.csv` (class × click × rule), `summary.md` and `cuts.json`.
