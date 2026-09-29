# Precision floor ("MinPrecision"): replacing the Inclusion knob

**Background.** The owner's ruling on #4223 replaces the Inclusion knob's
objective. The user states the precision they will accept ("I'm willing to look
at X%-positive returns"), and the cut returns as much as it can while keeping at
least X of it right. #4224 is the umbrella. #4220 measured the estimator that
keeps that promise
([`REPORT.md`](../experiments/2026-09-28-precision-frames-4220/REPORT.md)):
fold-rank evidence, a logistic posterior, a 10th-percentile bootstrap lower
bound, and an EM prior re-estimate, gated on 10 positives among the calibration
votes. The library already holds:
- the estimator, `vtscore.training.thresholds.precision_floor_cut`;
- one re-cut seam, `vtscore.state.core.recut_detector_threshold`;
- the inverse that lets acquisition find its origin off any line,
  `FoldAnchoredCut.inclusion_for_threshold`.

What follows is what the app still owes.

## Design

- **The operating point is X; Inclusion survives only as a unit.**
  - The reporting cut is `precision_floor_cut(X, ...)`.
  - Autopilot's acquisition cut stays where #3319 put it: four inclusion steps
    stricter than the line.
  - The "line's inclusion" is now *derived*: `inclusion_for_threshold(line)`,
    the strictest inclusion that reproduces the cut. So acquisition is
    `threshold_at(k(line) − 4)` (owner, 2026-09-28).
  - Inclusion's cost-weight semantics (`inclusion_cost_weights`) stay the
    internal coordinate this is measured in.
- **Three states, not a number.**
  - `promised`: at least X right, at the lower bound.
  - `unreachable`: enough evidence, but no cut clears X. The best reachable
    bound is reported.
  - `insufficient_evidence`: fewer than 10 calibration positives. This is
    today's common case: #4220 found the gate open in 5% of COCO Better cells
    after 150 votes.

  Every consumer of the cut has to handle the two states with no promise on
  purpose (#4247).
- **When nothing is promised, fall back to today's cut, labelled** (owner,
  2026-09-28). The line, the matches and every match action (Export, To
  Dataset, Browse, AutoRun) keep working at the Inclusion 0 cut, with the
  control saying why no promise is made. A floor that can't be met never
  empties the results.
- **The floor is per detector, seeded from the user's last value** (owner,
  2026-09-28), as Inclusion is today (#3416). A floor one detector can meet,
  another may not.
- **The control shows the floor and its state, not the estimate** (owner,
  2026-09-28). No "about 60% of these should be right": the lower bound stays
  internal.
- **Only the learned sort's own draws calibrate the promise** (owner,
  2026-09-28, from #4222). Votes picked by another ranker (the text-sort
  opening, the coverage atlas's New picks, a list sorted by anything but the
  model) train the model but stay out of the evidence and the gate. Whether the
  atlas's picks are fair enough to count is #4261.
  - **What the filter buys, and what it doesn't** (#4256, report on its PR). It keeps a long
    text-walk opening from breaking promises by keeping the gate shut. With the
    opening's Good round at 20, all votes break 69% of X = 50% promises at the
    0.44% pool. Learned-sort votes break 0 of 19, but promise in only 0.71% of
    frames. The filter does **not** make the posterior unbiased: learned-sort
    draws were chosen on an earlier model's score too. With a consistent
    reference pool they still break 83% (today's g3 opening). The promise's
    safety rests on the shipped pool's in-sample offset (#4221). The
    selection-free route is random verification (#4257).
- **The default floor is 50%, and a set floor wins over Inclusion** (owner,
  2026-09-28). `null` is "no floor", which hands the line back to Inclusion.
- **The control offers four presets and no off switch** (owner, 2026-09-28,
  #4246): 25/50/75/90%, the floors #4220 priced, so every choice is a
  measured one. The API still takes any value in `[0.01, 1]` and `null`; the
  control shows a stored non-preset or `null` as it is, and never sends
  `null`.
- **What waits on the GRID, and what doesn't.**
  - #4222 answered how the opening moves the gate. A longer text walk opens it
    but breaks the promise (above). A moderate one (Good target 6) is the only
    tested change that helps the detector at both 0.44% and 0.1%. At 0.1% no
    opening reaches the gate: a pool of ~11,000 holds ~11 positives. The
    adaptive stop it recommends is still open there.
  - #4221 decides the estimator's remaining knobs: the transfer coordinate,
    pooling the folds' evidence, and the bound level. Those are already
    parameters of `precision_floor_cut`.

  The plumbing can land on #4220's defaults and take #4221's knobs as a change
  of arguments. It can't take the provenance filter that way: that changes
  which votes build the fold orderings, so it belongs in #4245 from the start.

## Open decisions (owner)

These are the questions #4224 raised that no issue below can settle alone:

- **Does the CLI get a flag for the floor?** Headless runs (AutoRun, CLI
  autodetect, the cold Find path) already cut at the user's floor.
- **Retiring Inclusion from the extension surface** (#4269). `get_inclusion` /
  `set_inclusion`, `CoreConfig.inclusion`, the `inclusion_value=` parameters on
  `train_and_score` and its siblings, and `register_setting_persister("inclusion")`
  are public `vtscore` API. Per CLAUDE.md they are deprecated with an
  `[Unreleased]` note, not deleted, and the break is raised before it is made.
- **Verify the floor by audit?** #4257 priced random verification
  ([`REPORT.md`](../experiments/2026-09-28-random-verification/REPORT.md)). The
  user audits a uniform sample of the returned set. That keeps the promise at
  every prevalence tested, with no calibration-positive gate. But the sets a
  floor can honestly return are small, so at 0.44% and 0.1% most audited
  promises end with the user having read the whole set. Two questions follow for
  #4246 and #4247. Does the control ever ask for audit votes? And below what
  promisable size does it just show the top k and let the user's votes be the
  verification?

## Open work

<!-- item-sep -->

- [ ] #4245 — Backend: `min_precision` setting, endpoint, detector wiring, eval default arm (Opus 4.8)

<!-- item-sep -->

- [ ] #4247 — When no cut is promised, fall back to today's cut and label it (Opus 4.8)

<!-- item-sep -->

- [ ] #4246 — Frontend control replacing the Inclusion stepper, with its guide section and reshoots (Sonnet 5; Opus 4.8 for the write paths)

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

- [ ] #4269 — Retire Inclusion as a user preference (Sonnet 5; Opus 4.8 for the `vtscore` deprecation)

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
  retire it in favour of random verification (#4257).

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

## Where Inclusion is documented today

This is a reference for the issues above. Each PR prunes what it replaces, and
this list goes when the last one lands. It excludes the user guide and
screenshots, which #4246 and #4242 own.

- **App docs:**
  - [`docs/ML.md`](../ML.md) § Threshold Calibration: how Inclusion reaches the
    cut, and the conformal rule's budget semantics;
  - [`docs/EVAL.md`](../EVAL.md): `acq_inclusion_offset`, the `@k` startup
    rounds, and "inclusion-weighted" cost;
  - [`docs/api/labeling.md`](../api/labeling.md) § Inclusion & Thresholds;
  - [`docs/api/settings.md`](../api/settings.md), the `inclusion` rows;
  - [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md), the settings example;
  - [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md), per-detector inclusion.
- **Library docs:**
  - [`vtscore/docs/faq.md`](../../vtscore/docs/faq.md);
  - [`vtscore/docs/concepts.md`](../../vtscore/docs/concepts.md);
  - `vtscore/docs/packages/` (`config`, `state`, `training`, `detectors`,
    `eval`);
  - the executed samples in `quickstart.md`, `tutorials/train-and-score.md` and
    `integration.md`, which pass `inclusion`.
