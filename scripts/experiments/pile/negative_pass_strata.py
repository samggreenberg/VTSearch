#!/usr/bin/env python3
"""The union rate has to come from the RANDOM stratum alone.

Half the 200 sampled images are the boundary stratum, chosen by text rank to be
suspicious. Comparing a rate over all 200 against COCO's 12.7% -- which was
computed uniformly over the whole frame -- is apples to oranges, and the report
did exactly that.

READS THE COMMITTED VERDICTS, NOT THE BANK (#4012). This used to resolve each
labelset's polarity from `polarity.json` and read the reviewer's clicks out of
`negbank/*.json`; both were deleted on 2026-09-18 (#4001) and are in no record.
`verdicts.csv` carries the resolved answer and the stratum, so the estimate is
computable from the repository alone -- and the stratum manifest this used to
open separately is no longer a second input for the same fact (that the two
agree is now a test, `test_negative_pass_verdicts.py`).

**The published number moves, and is printed rather than replaced.** #3666
reported the union as 14% +/- 7. That was over the ten passes in `negbank/`
(`Backpack` and `Umbrella` finished later and lived only in the app's detector
store, which this never read) and counted the ten COCO-seeded rows as finds.
Over all twelve passes with the seeded rows dropped it is 15%. The conclusion is
the same either way -- COCO's 12.7% sits inside the interval -- and the
historical figure is reconciled below so the difference is visible instead of
silent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import negative_pass_verdicts as npv  # noqa: E402

rows = npv.load()
man = npv.strata(rows)
found = npv.found(rows)

for st in ("random", "boundary"):
    ids = [i for i, s in man.items() if s == st]
    k = sum(1 for i in ids if i in found)
    print(
        f"{st:<10} {k:3d} / {len(ids):3d} = {100 * k / len(ids):5.1f}%"
        + ("   <- THE ESTIMATE" if st == "random" else "   (ranked, biased by design)")
    )
k = sum(1 for i in man if i in found)
print(f"{'all 200':<10} {k:3d} / {len(man):3d} = {100 * k / len(man):5.1f}%   (not an estimate of anything)")

n = sum(1 for s in man.values() if s == "random")
k = sum(1 for i, s in man.items() if s == "random" and i in found)
p_hat = k / n
se = (p_hat * (1 - p_hat) / n) ** 0.5
print(f"\nunion contamination of the shared pool: {100 * p_hat:.0f}% +/- {100 * 1.96 * se:.0f} (95% CI, n={n})")
print(
    "COCO's prediction over the same frame:   12.7%  -> "
    + ("inside the interval" if abs(p_hat - 0.127) < 1.96 * se else "OUTSIDE the interval")
)

# What #3666 published, and why this run differs from it.
hist = npv.found(rows, include_seeded=True, only=npv.NEGBANK_PASSES)
hk = sum(1 for i, s in man.items() if s == "random" and i in hist)
print(
    f"\n#3666 published {100 * hk / n:.0f}% +/- {100 * 1.96 * (hk / n * (1 - hk / n) / n) ** 0.5:.0f}: "
    f"{len(npv.NEGBANK_PASSES)} passes (the ones in `negbank/`), COCO-seeded rows counted as finds.\n"
    f"This run: all 12 passes, seeded rows dropped. Both put 12.7% inside the interval."
)
