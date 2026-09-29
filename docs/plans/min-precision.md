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
  - The reporting cut is the set the spot check ended on, or the unchecked
    starting candidate before any check (#4272).
  - Autopilot's acquisition cut stays where #3319 put it: four inclusion steps
    stricter than the line.
  - The "line's inclusion" is now *derived*: `inclusion_for_threshold(line)`,
    the strictest inclusion that reproduces the cut. So acquisition is
    `threshold_at(k(line) − 4)` (owner, 2026-09-28).
  - Inclusion's cost-weight semantics (`inclusion_cost_weights`) stay the
    internal coordinate this is measured in.
- **A spot check earns the promise** (owner, 2026-09-29, #4267, from the
  decision page built on #4257's rank frames). The promise certifies a list
  the user acts on without looking, and it costs 5 votes a round.
  - **Do your best, and say how close we got** (owner, 2026-09-29). The
    promise is not make-or-break. The line always keeps the set the check ended
    on: the confirmed set, or the top 32 after a short check. The control shows
    a **likely range** for how much of it is right. The range is a
    Clopper–Pearson interval from the labels inside the set, with each tail at
    α / R, so a check confirms X iff the range's lower end clears X.
    - At 0.44% and X = 10%, a session returns 53 items on average, 40% right,
      and meets the floor in 84% of sessions.
    - The range contains the truth in 99% of sessions. With plain 90% tails it
      sat above the truth 11% of the time after a first-round pass (4.0% at
      α / R).
    - The range is wide at 5 picks: 0.57 on average at 50%. That width is the
      honest answer to "how close?". See
      [`REPORT.md`](../experiments/2026-09-29-floor-candidate-4267/REPORT.md#do-your-best-how-close-did-we-get).
  - **The rule** (#4257's `a:top32`, extended below 50%). The candidate is
    the top K *unvoted* items of the current ranking. The user votes on m of
    them, drawn uniformly at random. The whole candidate is confirmed iff the
    one-sided Clopper–Pearson lower bound on the hit rate is ≥ X at level
    α / R, with α = 5%. A failed round halves K and draws m fresh picks, down
    to the top 32. There is no redraw on the same candidate.
  - **The check grows with X** (owner, 2026-09-29). Each round draws
    m = max(5, ⌈ln(α / R) / ln X⌉) picks, the fewest that can reach X when all
    are right: 5 up to X = 54.9%, 11 at 75% and 29 at 90%.
  - **The candidate grows as X falls** (owner, 2026-09-29), so a low floor
    returns more.
    - It starts at K = 32 · 2^max(0, ⌊log₂(0.5 / X)⌋): the top 128 at 10%, the
      top 64 at 25%, and the top 32 at 50% and above.
    - That gives R = log₂(K / 32) + 1 rounds of 5 picks each: 3 at 10%, 2 at
      25%.
    - The owner picked 5 picks a round from the grid in
      [`REPORT.md`](../experiments/2026-09-29-floor-candidate-4267/REPORT.md).
      At 10% (0.44%) it costs 12 votes and promises in 59% of sessions. A
      promise returns 68 items on average, and the check recovers 0.63 of the
      oracle's recall, against 0.43 for a fixed top 32. That beats reading the
      top 32 yourself by +0.058.
  - **The picks are a separate check, not the ranking** (owner, 2026-09-29).
    They come from anywhere in the candidate, so the UI must not show them as
    the top of the sort. A check can take up to three rounds (#4273).
  - **What it buys** at 0.44% (COCO Better's default): a promise in 59%, 43%,
    28%, 20% and 12% of sessions at X = 10%, 25%, 50%, 75% and 90%, with at
    most 0.29% of promises broken. It hands over 17, 12, 6.9, 4.0 and 0.37
    matches per session that the user never saw. On recall it gets 0.38 of the
    oracle at X = 50%, against 0.73 for reading the top 32. That gap is the
    price of the ~5-vote budget.
  - **At 0.1% a pass is almost always luck.** At X = 50% the check promises
    in 0.09% of sessions, and 88% of those promises break. Clopper–Pearson bounds broken
    promises per session (≤ α of all sessions), not per promise made. The
    no-cause wording below covers this case.
  - **Headless runs can't be checked**, because nobody is there to vote. AutoRun,
    CLI autodetect and the cold Find path export the schedule's starting
    candidate, marked unchecked and with no range (owner, 2026-09-29). At
    0.44% the top 128 is 22% right on average at 10%, and the top 32 is 53%
    right at 50%.
  - **A finished check's result is kept, and goes stale quietly** (owner,
    2026-09-29).
    - Check votes train the model, so the list at the line changes after a
      check. The line follows the new ranking at the result's count.
    - The last range stays on screen, and only its tooltip says it predates
      later votes. There is no vote count and no separate re-check button
      (#4273); the existing check affordance runs a fresh check.
    - While a check runs, its candidate's ids are fixed, so every round samples
      one list.
    - Accepted limit: a user who re-checks until one confirms keeps the lucky
      result.
- **Three states, and a range.**
  - `unchecked`: no check has run on the current candidate. The line is the
    starting candidate, with no range.
  - `confirmed`: the check's range clears X.
  - `short`: the check ended with its range below X. The line keeps the top 32
    it ended on.
- **The line never falls back** (owner, 2026-09-29). This replaces the
  2026-09-28 ruling (#4247) that put the line at the Inclusion 0 cut when nothing
  was promised; at 0.44% that cut returns about 2,300 items at 2% right. Every
  match action (Export, To Dataset, Browse, AutoRun) works on the set the line
  keeps, and a floor never empties the results.
  - **A short check names no cause** (owner, 2026-09-29). A check fails both
    for a sparse corpus (0.1%) and for a weak model (tv@small at 0.44%), and
    the app can't tell them apart. The wording must be true in both cases, for
    example "Too few matches near the top to promise X%."
- **The floor is per detector, seeded from the user's last value** (owner,
  2026-09-28), as Inclusion is today (#3416). A floor one detector can meet,
  another may not.
- **The control shows the floor, its state, and the check's likely range**
  (owner, 2026-09-29). An example: "Likely 11–73% right (checked 5)". The range
  comes only from the check's uniform picks, never from the model. This
  replaces the 2026-09-28 ruling that kept every estimate internal.
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
- **The control offers three named presets and no off switch** (owner,
  #4298 on 2026-09-29): **Lean: Complete / Centered / Correct**, for 10, 50
  and 90%, symmetric about the 50% default. The control never shows the
  number, since the check rarely delivers a floor exactly. The API still takes
  any value in `[0.01, 1]`; the control snaps a stored non-preset to the
  nearest preset.
  - Under the spot check, the presets cost at most 15 picks at 10% (3 rounds
    of 5), 5 at 50% and 29 at 90%.
  - #4220 never priced the estimator at 10%, so until #4272 replaces it, the
    10% floor runs unmeasured.
- **What waits on the GRID, and what doesn't.**
  - The spot check needs no GRID run: #4257's rank frames price it exactly.
  - #4222's opening no longer decides whether a promise can be made, since the
    promise no longer depends on the calibration gate. It still moves what the
    check returns: at 0.44% and vote 150 the shipped opening (#4282) confirms a 50%
    floor in 34% of sessions against 28%, with the top 32 58% right against 53%.
    At 0.1% it changes nothing
    ([`REPORT.md`](../experiments/2026-09-29-floor-opening-4287/REPORT.md), #4287). A moderate text walk (Good target 6)
    is the only tested change that helps the detector at both 0.44% and 0.1%.
    Its dry stop has since landed (`g3@top,g20+dry1/16@top,b4@mid`,
    [`REPORT.md`](../experiments/2026-09-29-drystop-4222/REPORT.md)). It raises
    AP at vote 150 by +0.052 at 0.44% and +0.032 at 0.1%.
  - The closed loop is unmeasured: check votes training the model, how far a
    stale range drifts from the list it now sits beside, and re-checks after a
    retrain.

## Open work

<!-- item-sep -->

- [ ] #4245 — Backend: `min_precision` setting, endpoint, detector wiring, eval default arm (Opus 4.8)

<!-- item-sep -->

- [ ] #4247 — When no cut is promised, fall back to today's cut and label it (Opus 4.8)

<!-- item-sep -->

- [ ] #4246 — Frontend control replacing the Inclusion stepper, with its guide section and reshoots (Sonnet 5; Opus 4.8 for the write paths)

<!-- item-sep -->

- [ ] #4272 — Backend: a spot check, not the estimator, decides the line and how close it got (Opus 4.8)

<!-- item-sep -->

- [ ] #4273 — Frontend: the spot-check step, and how close the line got (Sonnet 5; Opus 4.8 for its state and write paths)

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
  - #4115, where C = 0.1 ranks better but the shipped cut gives it back;
  - #4219, the production head decision, read partly through Inclusion.

  Their acquisition-side findings may survive, because acquisition still
  re-cuts `mid_tilt`. Their reporting-side ones do not. #4121 (the fused cut
  drifting into the negatives once positives are exhausted) is live only while
  an unpromised floor falls back to the Inclusion 0 cut. Close it as not
  planned when #4272 merges: the line then reads the ranking, which past
  exhaustion is unharmed.

<!-- item-sep -->
