# 2026-08-24 — a login-node timing nearly cut a whole arm from a study (#2883)

**Study:** #2883 transfer characterisation. **Cost:** none, but it came within
one decision of removing the study's only shippable arm.

Sizing the new estimators, profiled on the login node:

| | login node | compute node, BLAS pinned to 1 thread |
|---|---:|---:|
| `fit_score_gmm` (2000 scores) | ~0.58 s | **0.016 s** |
| the bagged-fit arm (16 refits) | 9.2 s | **0.27 s** |
| `decomposition_cuts`, whole call | 14.2 s | **0.52 s** |

**36×.** On the login-node numbers the label-free bagged-fit arm cost ~550 s per
cell against a ~100 s baseline — it would have roughly tripled a 552-cell array,
and the reasoning was already written down: the arm is exploratory, #2883 item 1
asks for the characterisation before the remedy, so drop it. On the real numbers
it costs **+19 % of a cell** and there was never a trade to make.

Two things were wrong with the login-node measurement, and only one of them is
about load:

- The login node is shared and was busy. That is the obvious half.
- sklearn's `GaussianMixture` spawns a BLAS thread pool per fit. At this size the
  pool costs more than the arithmetic, so the default threading makes each fit
  *slower*, and 40 concurrent cells each doing it would oversubscribe the node
  they land on. The run now exports `OMP_NUM_THREADS=1` and its two siblings —
  which is both a correctness-of-sizing fix and a be-a-good-citizen fix.

**The general form.** *Time a cell where the cell will run, under the environment
it will run with.* The skill already says "size it from a real cell, not a guess";
this is the sharper version — a real cell **on the wrong host** is also a guess,
and it is a more dangerous one, because it produces a number with a unit attached
that nobody thinks to doubt. The failure mode is not "the run took longer than
expected", which is visible; it is **an arm silently dropped at design time on a
bad number**, which leaves no trace in the run at all.

**Status: advice.** `preflight.sh` cannot know where a timing came from. What
would have caught it, and what this study did do in the end, is the habit the
#2865 lesson already recommends: run the cells you are sizing from through
`srun`, not through the login shell, and put the numbers in the launcher's
comments where the next reader can see what they were measured on.
