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
  model) train the model but stay out of the evidence and the gate, because the
  posterior is unbiased only under score-only selection. Whether the atlas's
  picks are fair enough to count is #4261.
- **The default floor is 50%, and a set floor wins over Inclusion** (owner,
  2026-09-28). `null` is "no floor", which hands the line back to Inclusion.
- **What waits on the GRID, and what doesn't.**
  - #4222 decides how often the gate opens.
  - #4221 decides the estimator's remaining knobs: the transfer coordinate,
    pooling the folds' evidence, and the bound level. Those are already
    parameters of `precision_floor_cut`.

  Neither blocks the plumbing, which can land on #4220's defaults and take
  #4221's answer as a change of arguments.

## Open decisions (owner)

These are the questions #4224 raised that no issue below can settle alone:

- **Whether X is free or a few presets** in the control (#4246). #4220 priced
  25/50/75/90%. The backend takes any value in `[0.01, 1]`.
- **Does the CLI get a flag for the floor?** Headless runs (AutoRun, CLI
  autodetect, the cold Find path) already cut at the user's floor.
- **Retiring Inclusion from the extension surface.** `get_inclusion` /
  `set_inclusion`, `CoreConfig.inclusion`, the `inclusion_value=` parameters on
  `train_and_score` and its siblings, and `register_setting_persister("inclusion")`
  are public `vtscore` API. Per CLAUDE.md they are deprecated with an
  `[Unreleased]` note, not deleted, and the break is raised before it is made.

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

- **Re-derive `provenance-partitioned-calibration.md` before running it.** That
  plan is motivated by the conformal miss budget. Top-of-list review votes bias
  calibration positives high, so the FNR budget over-promises. Its metrics are
  in Inclusion units (FNR excess at Inclusion 0–3). A posterior fitted on
  score-picked votes is unbiased under *score-only* selection, which is what
  reviewing the model's own sorted list is (#4224's feasibility note), so that
  half of the problem does not transfer to the floor. The other half does:
  votes selected by a *different* ranker break the promise (#4222: text-sort
  selection broke 51% of the X = 50% promises at a 20-Good opening), which is
  why the floor already calibrates only on the learned sort's draws. Before
  it runs, restate its hypotheses as violation rate at X by provenance, or
  retire it.

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
  - [`docs/api/find.md`](../api/find.md), the Find Stats `sweep`;
  - [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md), the settings example;
  - [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md), per-detector inclusion.
- **Library docs:**
  - [`vtscore/docs/faq.md`](../../vtscore/docs/faq.md);
  - [`vtscore/docs/concepts.md`](../../vtscore/docs/concepts.md);
  - `vtscore/docs/packages/` (`config`, `state`, `training`, `detectors`,
    `eval`);
  - the executed samples in `quickstart.md`, `tutorials/train-and-score.md` and
    `integration.md`, which pass `inclusion`.
