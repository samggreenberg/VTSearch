# Precision floor ("MinPrecision"): replacing the Inclusion knob

**Background.** The owner's ruling on #4223 replaces the Inclusion knob's
objective. The user states the precision they will accept ("I'm willing to look
at X%-positive returns"), and the cut returns as much as it can while keeping at
least X of it right. #4224 is the umbrella. #4220 measured an estimator for
that promise
([`REPORT.md`](../experiments/2026-09-28-precision-frames-4220/REPORT.md)):
fold-rank evidence, a logistic posterior, a 10th-percentile bootstrap lower
bound, and an EM prior re-estimate, gated on 10 positives among the calibration
votes. The owner's ruling on #4267 (2026-09-29) moved the promise off that
estimator and onto a **spot check** (below). The library already holds:
- the estimator, `vtscore.training.thresholds.precision_floor_cut`, which is
  public API and stays, off the promise path;
- one re-cut seam, `vtscore.state.core.recut_detector_threshold`;
- the inverse that lets acquisition find its origin off any line,
  `FoldAnchoredCut.inclusion_for_threshold`.

What follows is what the app still owes.

## Design

- **The operating point is X; Inclusion survives only as a unit.**
  - The reporting cut is the checked candidate when a spot check has promised
    it, and the Inclusion 0 fallback otherwise (#4272).
  - Autopilot's acquisition cut stays where #3319 put it: four inclusion steps
    stricter than the line.
  - The "line's inclusion" is now *derived*: `inclusion_for_threshold(line)`,
    the strictest inclusion that reproduces the cut. So acquisition is
    `threshold_at(k(line) − 4)` (owner, 2026-09-28).
  - Inclusion's cost-weight semantics (`inclusion_cost_weights`) stay the
    internal coordinate this is measured in.
- **A spot check earns the promise** (owner, 2026-09-29, #4267, from the
  decision page built on #4257's rank frames). The promise certifies a list
  the user acts on without looking, and it costs about 5 votes.
  - **The rule** (#4257's `a:top32`). The candidate is the top 32 *unvoted*
    items of the current ranking. The user votes on m(X) of them, drawn
    uniformly at random. The whole candidate is promised iff the one-sided
    Clopper–Pearson lower bound on the hit rate, at α = 5%, is ≥ X. There is
    one round, with no redraw on the same candidate.
  - **The check grows with X** (owner, 2026-09-29). m(X) = max(5,
    ⌈ln α / ln X⌉), the fewest picks that can promise X when all are right.
    That is 5 picks up to X = 54.9%, 11 at 75% and 29 at 90%.
  - **The picks are a separate check, not the ranking** (owner, 2026-09-29).
    They come from anywhere in the top 32, so the UI must not show them as the
    top of the sort (#4273).
  - **What it buys** at 0.44% (COCO Better's default): a promise in 41%, 28%
    and 20% of sessions at X = 25%, 50% and 75%, with at most 0.28% of promises
    broken. It hands over 9.6, 6.9 and 4.0 matches per session that the user
    never saw. On recall it gets 0.38 of the oracle at X = 50%, against 0.73
    for reading the top 32. That gap is the price of the ~5-vote budget.
  - **At 0.1% a pass is almost always luck.** The check promises in 0.09% of
    sessions, and 88% of those promises break. Clopper–Pearson bounds broken
    promises per session (≤ α of all sessions), not per promise made. The
    no-cause wording below covers this case.
  - **Headless runs can't be checked**, because nobody is there to vote. AutoRun,
    CLI autodetect and the cold Find path always export the labelled fallback.
  - **Not yet decided:** what a retrain does to a promise. Check votes train
    the model and move the top 32, and repeated checks compound α. #4272 raises
    this with the owner before building it.
- **Three states, not a number.**
  - `unchecked`: no check has been run on the current candidate.
  - `promised`: a check passed; the candidate is at least X right.
  - `not_promised`: the check failed.

  Every consumer of the cut has to handle the two states with no promise on
  purpose (#4247).
- **When nothing is promised, fall back to today's cut, labelled** (owner,
  2026-09-28). The line, the matches and every match action (Export, To
  Dataset, Browse, AutoRun) keep working at the Inclusion 0 cut, with the
  control saying why no promise is made. A floor that can't be met never
  empties the results.
  - **A failed check names no cause** (owner, 2026-09-29). A check fails both
    for a sparse corpus (0.1%) and for a weak model (tv@small at 0.44%), and
    the app can't tell them apart. The wording must be true in both cases, for
    example "Too few matches near the top to promise X%."
- **The floor is per detector, seeded from the user's last value** (owner,
  2026-09-28), as Inclusion is today (#3416). A floor one detector can meet,
  another may not.
- **The control shows the floor and its state, not the estimate** (owner,
  2026-09-28). No "about 60% of these should be right": the lower bound stays
  internal.
- **Why not the estimator.** Every vote a model chose is a biased sample of
  that model's scores (#4256). The merged backend (#4245) calibrates only on
  learned-sort draws. That filter stops a long text walk from breaking promises,
  but only by keeping the gate shut. It does not make the posterior unbiased.
  With a consistent reference pool, learned-sort evidence alone still breaks 83%
  of X = 50% promises (today's g3 opening). The shipped pool looks safe (5.3%)
  only through its in-sample offset. A spot check's picks are uniform by
  construction, so its bound holds at every prevalence tested (#4257).
  - The estimator's open questions no longer bear on the promise: #4221's knobs,
    and #4261's question of whether atlas votes may calibrate. Whether either
    still earns its run is the owner's call.
- **The default floor is 50%, and every detector has one** (owner,
  2026-09-28; `null` refused since #4269, which retired Inclusion as a user
  preference).
- **The control offers four presets and no off switch** (owner, 2026-09-28,
  #4246): 25/50/75/90%, the floors #4220 priced, so every choice is a
  measured one. The API still takes any value in `[0.01, 1]`; the control
  shows a stored non-preset as it is.
  Under #4267's growing check, the presets cost 5, 5, 11 and 29 picks.
- **What waits on the GRID, and what doesn't.**
  - The spot check needs no GRID run: #4257's rank frames price it exactly.
  - #4222's opening now matters for detector quality only, since the promise no
    longer depends on the calibration gate. A moderate text walk (Good target 6)
    is the only tested change that helps the detector at both 0.44% and 0.1%.
    Its adaptive stop (`g20+dry1/8@top`) is still running.
  - The closed loop is unmeasured: check votes training the model, and
    re-checks after a retrain. Both feed #4272's lifecycle decision.

## Open work

<!-- item-sep -->

- [ ] #4245 — Backend: `min_precision` setting, endpoint, detector wiring, eval default arm (Opus 4.8)

<!-- item-sep -->

- [ ] #4247 — When no cut is promised, fall back to today's cut and label it (Opus 4.8)

<!-- item-sep -->

- [ ] #4246 — Frontend control replacing the Inclusion stepper, with its guide section and reshoots (Sonnet 5; Opus 4.8 for the write paths)

<!-- item-sep -->

- [ ] #4272 — Backend: earn the promise with a spot check of the top 32, not the estimator (Opus 4.8)

<!-- item-sep -->

- [ ] #4273 — Frontend: the spot-check step that earns the promise (Sonnet 5; Opus 4.8 for its state and write paths)

<!-- item-sep -->

- [ ] #4242 — Find Stats: estimated precision against the number returned (Sonnet 5)

<!-- item-sep -->

- [ ] #4243 — What the Smart indicator prices once the user sets no Inclusion (Opus 4.8)

<!-- item-sep -->

- [ ] #4253 — Measure four ways to price Smart's error cost under the floor (Opus 4.8)

<!-- item-sep -->

- [ ] #4248 — Precision floor under region voting (Opus 4.8)

<!-- item-sep -->

- [ ] #4244 — Slides: end *Hold the Line*'s "Preference" section on the floor (Sonnet 5)

<!-- item-sep -->

- [ ] #4261 — Should Autopilot's New-phase (atlas) votes calibrate the floor? (Sonnet 5)

<!-- item-sep -->

<!-- item-sep -->

- **Re-derive `provenance-partitioned-calibration.md` before running it.** That
  plan is motivated by the conformal miss budget. Top-of-list review votes bias
  calibration positives high, so the FNR budget over-promises. Its metrics are
  in Inclusion units (FNR excess at Inclusion 0–3). A posterior fitted on
  score-picked votes was thought unbiased under *score-only* selection, which
  is what reviewing the model's own sorted list is (#4224's feasibility note).
  #4256 measured it, and it does not hold: learned-sort draws alone, with a
  consistent reference pool, break 83% of the X = 50% promises. So the problem
  does transfer to the floor. Votes selected by a *different* ranker are worse
  (#4222: text-sort selection broke 51% of the X = 50% promises at a 20-Good
  opening). Before it runs, restate its hypotheses as violation rate at X, or
  retire it in favour of random verification (#4257). #4267 ruled for random
  verification, so retiring it is the expected outcome.

<!-- item-sep -->

- **Triage the FPR + FNR cut-rule issues against the ruling.** These tune the
  reporting line under an objective the floor replaces:
  - #4118, a hinge rule for the reporting line only;
  - #4136, the guarded text-sort line as display only;
  - #4115, where C = 0.1 ranks better but the shipped cut gives it back;
  - #4219, the production head decision, read partly through Inclusion.

  Their acquisition-side findings may survive, because acquisition still
  re-cuts `mid_tilt`. Their reporting-side ones do not. #4121 (the fused cut
  drifting into the negatives once positives are exhausted) is about the
  estimator's haystack, not the objective, and stays live.

<!-- item-sep -->
