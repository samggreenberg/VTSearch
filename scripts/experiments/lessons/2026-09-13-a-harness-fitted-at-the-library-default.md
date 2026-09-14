# 2026-09-13 — a harness fitted at the library default, not the shipped one (#3825)

**Cost:** ~1 hour of re-running (a 35-min gate and a 40-min bench), and a whole
report drafted around numbers that described a configuration nobody runs.

`fit_anchored_score_gmm(arr, scores, labels)` has a default
`anchor_weight=ANCHOR_WEIGHT_DEFAULT` = **10.0**. Nothing on the shipped path
calls it that way: `fit_fold_anchored_cut` passes `FOLD_ANCHOR_WEIGHT` = **0.3**,
which the 2026-08-06 anchor-mass sweep picked and #2861 shipped.

`gate_3825.py` has two frames. Its *decision* frame calls
`fit_fold_anchored_cut` — the shipped function — and was correct throughout. Its
*estimator* frame called `fit_anchored_score_gmm` bare, so every iteration
count, convergence rate, cost split and objective comparison in the study
described the estimator at 33x the anchor mass it ships at. `bench_3825.py` had
the same call, and so did the arms module's objective scorer.

Nothing crashed, nothing was empty, and the numbers were plausible — which is
the whole problem. They were also **wrong in the flattering direction**: at
κ=10 the incumbent looks like a median of 43 iterations with 5.7% of folds
exiting on `max_iter`; at the shipped κ=0.3 it is **113** and **26.5%**. A study
about the cost of that loop had understated it by a factor of five.

**What caught it:** building the report's worked case. Reconstructing one case
by hand through fold cut → quantile → realise → snap gave 3 admitted medias
where the gate's own frame said 70. Two frames of the same run disagreeing is a
signal a summary statistic cannot produce.

**Still only advice.** The general shape is: *a harness that reaches past the
shipped entry point inherits the library default, not the shipped value.* The
shipped value is usually a named constant precisely because a study chose it, so
the tell is a call that passes neither. Not currently checkable by
`preflight.sh` — it would have to know which defaults a given path overrides.

Two things that do help, both cheap:

- **Name the configuration in the run's own output.** `gate_3825.py` now prints
  `fitting at the shipped fold anchor weight, kappa = 0.3` at startup, bound from
  `FOLD_ANCHOR_WEIGHT` rather than typed. A wrong value is then in the log of
  every run rather than only in the code.
- **Put a worked case in the report, and reconstruct it by hand.** It is the one
  check that reads the frames against each other instead of against expectation.
