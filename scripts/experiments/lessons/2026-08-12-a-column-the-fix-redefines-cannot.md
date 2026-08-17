# 2026-08-12 — a column the fix redefines cannot be the fix's acceptance test (#2905)

**What happened.** Verifying the #2943 fix before committing ~7 h of cluster time,
I pre-registered three pass criteria on a single re-run cell, two of which were
"`acq_pool_percentile` pinning should **fall**" and "its median should move off
the ceiling". Both failed — pinning went **0% → 60.6%**, the median **0.9905 →
1.0000** — and for a moment that read as the fix not working.

It was my criteria that were wrong. #2943 changed *what those columns measure*:
pre-fix they were computed in the pool's whole-image space (self-consistent, and
therefore healthy-looking), post-fix in the threshold's max-pooled space. They are
**not the same statistic on either side of the fix**, so no comparison across it
means anything — including a comparison designed to validate the fix.

**Cost.** Minutes, because the criteria were written down in advance and so failed
loudly instead of being quietly reinterpreted. The real risk was the opposite
outcome: had they *passed*, I would have launched on a false green.

**What actually showed the fix working** was the outcome, not the instrument — the
same cell, same trajectory, went from 6 positives / cost 0.452 / AP 0.209 to 3 /
0.625 / 0.130, i.e. the optimistic bias #2943 describes, removed.

**Still advice — validate a fix on quantities the fix does not redefine.**
Prefer end-state outcomes (positives found, cost, AP) or a within-harness contrast
(does arm A differ from arm B *on the fixed code*) over any before/after on a
column whose semantics the change touches. And write the criteria down first: the
value of a pre-registered check is not that it is right, but that when it is wrong
you find out instead of rationalising.
