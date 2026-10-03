# The balance: an F-beta preference in place of the precision floor (#4413)

**Owner, 2026-10-01 03:42**, after #4411's pricing: "WE invented this 'Min
Precision Thresh' notion. It's just one way to record the users' preference.
(Obviously everyone prefers a perfect precision AND a perfect recall if such a
thing is possible.) Instead of asking 'What precision should we aim for in your
results?', I'm totally fine asking 'What balance of precision and recall
should maximize?'. We could abandon this 'Precision Floor' question and just
have users self-categorize by F-beta. We'd find the threshold based on F-beta
and progress the SVM based on that F-beta threshold and everything." Ruled:
**switch to F-beta.**

This supersedes the precision floor's design (#4224, #4267, #4383, #4389;
the floor and its plan were removed in #4421, and git history keeps both).
What carries over from the floor is the machinery: the band walk with uniform audits, the
vote-anchored mixture as the no-vote fallback, the memoised `LineRanking`,
the per-preference review. What changes is the stop rule, the no-vote rule,
what the state reports, and the words.

## What was priced (#4411, `docs/experiments/2026-10-01-fbeta-line-4411/REPORT.md`)

- The F-beta optimum is `argmax_k (1 + b^2) tp(k) / (b^2 n_pos + k)`: unlike
  the floor's cut it needs **n_pos**, the corpus's positives, which the mixture
  over-counts 10-28x at 0.1% (#3827). So the mixture alone over-returns 2-4x
  on large sparse corpora (0.22-0.48 of the best F1 at 113k items), while the
  **audited walk** (tp from the picks, n_pos from the mixture) reaches
  0.66-0.92 of the best F-beta from 3,200 items up and never collapses.
- The audited walk costs about a third more votes than the floor's (28 vs 20
  on the 11.6k bench, 52 vs 32 at 100k), and its own F-beta estimate is off by
  0.15-0.21 (the floor walk's precision estimate: 0.10-0.14).
- No single beta lands on a precision target: an F1 cut sits at 91% precision
  on large objects and 22% on small. That is why the preference is a balance,
  not a floor, and why the floor's presets never were an F-beta.

## Design

- **The preference is a balance, `beta`.** Three presets: **precision-leaning
  0.5**, **balanced 1** (the default), **recall-leaning 2**; any beta in
  `[0.25, 4]` is accepted. It is per detector, seeded from the user setting,
  as the floor was.
- **The line keeps a set**, as it has since #4272: the top *count* unvoted
  items of the ranking the last retrain scored.
  - **Checked:** the band edge where the walk's F-beta estimate peaked.
  - **Unchecked** (AutoRun, the CLI, a cold Find, every session before its
    first check): the mixture's F-beta argmax over the unvoted ranking,
    **capped** by the same schedule count the floor used for its no-vote
    line (32 at beta <= 1, 128 at beta 2; #4389's cap, which is what stops
    the mixture's over-return on large sparse corpora).
- **The walk** (`SpotCheck`, bands 8, 8, 16, 32, ...; 5 uniform picks a band,
  a census where a band is smaller):
  - It estimates, for the top *b* bands, `tp = sum over bands of (share right
    among the band's picks) x (band size)` and `F-beta = (1 + b^2) tp /
    (b^2 n_pos + edges[b])`, with `n_pos` the mixture's positives over the
    unvoted ranking, fixed at the start.
  - It starts at the band holding the unchecked count; it goes **deeper while
    the estimate does not fall** (auditing the new band), and when it first
    falls it ends at the peak; from a start whose first deeper step falls, it
    goes **shallower while the estimate rises** (no new picks: shallower sets
    are subsets), and ends at the peak. The kept set is the peak's edge.
  - It reports the kept set's **estimated precision and recall, each with a
    likely range** (precision: the bands' Clopper-Pearson intervals weighted
    by size, as today; recall: tp's range over n_pos), and the estimated
    F-beta. There is no `short`: a balance has nothing to fall short of. The
    states are `unchecked` and `checked`.
  - A result belongs to its beta, as a floor result belonged to its floor; it
    goes `stale` when the ranking under it moves.
- **Acquisition** stays at line − 4 (half the argmax's depth shipped on
  2026-10-01 and was reverted on 2026-10-02: on the objective it was worse at
  every preset, `../experiments/2026-10-01-acquisition-fbeta-4409/REPORT.md`
  section 4); the rank cut remains the harness arm of #4409
  (`acq_p_crossing`) with the target depth swapped from the P crossing to the
  F-beta argmax, priced against today's line - 4 at each preset before it
  ships.
- **The metric** (the review and every A/B): the returned set's F-beta over
  the best F-beta of any cut of the same ranking, per beta over clicks, with
  its precision and recall beside it. Sessions run once per beta (#4408's
  per-floor machinery, re-keyed). AP stays the ranking's measure.
- **The words.** The Threshold control's three radios are the presets, with
  the spectrum's ends as they are (False Positives ... False Negatives). The
  note under it: *Top 32 kept, unchecked* / *Checked: top 48 kept, about 70%
  right, about half of them found (15 picks)*. The check button stays
  **Check 5 picks**.

## What is given up, on purpose

- The checkable promise "at least P of what you see is right". Under a
  balance the check says what it estimated and how sure it is; nothing is met
  or missed.
- About a third more audit votes a check.
- A self-estimate that leans on the mixture's count of the positives, so the
  recall half of "how close we got" is a wider range than the precision half.

## Steps (each a PR that keeps the suite green; the floor path stays until the last)

Status (2026-10-01): steps 1-4 landed with PR #4414 and PR #4416; step 6
(the UI and the docs, with the switch of `line_preference` to the balance)
is the PR this revision belongs to; step 5's acquisition cut shipped on
2026-10-01 at half the F-beta argmax's depth (#4409, the PR after #4420).

1. **Library core** (`vtscore/training/thresholds/spot_check.py`, tests in
   `tests_lib/sorting/`): `mixture_positives` and `fbeta_count` beside
   `mixture_count`; `SpotCheck.start(..., beta=, n_pos=)` with the F-beta
   stop; `balance_count` / `balance_line` / `balance_state` beside the floor's
   three, sharing `LineRanking`, the bands and the ranges. `BalanceState`
   carries beta, status, count, the estimated precision and recall ranges,
   the F-beta estimate, `stale`, and the schedule.
2. **The setting and the detector** (`vtsearch/settings_models.py`,
   `vtscore/config/core_config.py`, `vtscore/state/*`, the retrain, the
   re-cut, the cold Find path, AutoRun, the CLI): `beta` per detector, the
   line drawn by `balance_line`, the state by `balance_state`, `min_precision`
   the deprecated alias. The API's `floor` object becomes `balance`.
3. **The eval** (`vtscore/eval/voting_iterations.py`, `run_cells.py`,
   `experiment_config.py`): `beta` as the arm knob (`CALIB_BETA`), the
   default arm the app's default balance, the end-of-run check the F-beta
   walk; the sync gate re-pinned.
4. **The review and the metric** (`scripts/experiments/state_of_app/`,
   `_rank_metrics.py`): F-beta over the best cut per beta over clicks; one set
   of sessions per beta; the skill's standing decisions.
5. **Acquisition** (#4409 re-scoped): price the F-beta-cut arm at each preset
   against line - 4; ship if it raises F-beta without losing Goods.
6. **The UI and the docs** (`precision-floor` -> `balance` component, the
   check modal's copy, Find Stats, `docs/ML.md`, `docs/api/labeling.md`,
   `docs/user/USER_GUIDE.md`, this plan's predecessor marked superseded).

## Decisions (2026-10-01, as shipped)

- **The objective** (owner, 2026-10-01, while #4427 was being measured): the
  F-beta of the images the line keeps, scored on the withheld test half.
  Gathering more positives counts only through that; the share of the best
  cut and the user's own unvoted corpus (the review's Diagnostic section) are
  diagnostics.

- **The presets are 0.5 / 1 / 2, default 1.** Any beta in `[0.25, 4]` is
  accepted by the API and the setting; a value outside is clamped.
- **The unchecked cap is the floor's schedule count:** 32 at beta 1 and 0.5,
  128 at beta 2 (`balance_schedule`). It is also where a walk starts.
- **The note says "about half of them found", not "recall".** The phrase is
  read off the recall range's midpoint: *few of them found* / *about a
  quarter of them found* / *about half of them found* / *about three
  quarters of them found* / *nearly all of them found*. The checked note
  reads `Checked · likely 55–80% right, about half of them found (checked
  15) · 48 kept`; the unchecked one `Top 32 kept, unchecked`.
- **The control keeps its "Threshold" heading and its spectrum** (False
  Positives ... False Negatives) with three unlabelled radios; the component
  is `vt-balance`. The check button stays **Check 5 picks**.
- **The check modal is `vt-spot-check-modal`.** Its walk notes are `Better
  so far: checking the next 32.` and `Past the peak: checking a shorter
  list.`; its result reads `Checked: likely 55–80% right, about half of them
  found (checked 15).` and `The line keeps these 48: the set where the
  check's balance peaked. The ranges are what the picks say about them.`
- **The Find Stats chart draws no floor line.** Its legend reads `Line:
  checked (48 kept)` or `Line: the top 32, unchecked`.
- **The states are `unchecked` and `checked`.** There is no `short`.
- **Headless runs export the preference's unchecked set and say so:** the
  `detector_unchecked` event reads `Detector 'det' exports its top 32
  unchecked (at F1); nobody is here to check it.`, and carries `beta`.
- **The eval's default arm is the balance at beta 1**
  (`simulate_voting_iterations(beta=None)`; `CALIB_BETA` unset);
  `CALIB_BETA=0.5|1|2` pins a balance, and `CALIB_BETA=off` is the Inclusion
  arm. The State-of-the-App review runs one set of sessions per beta
  (`SOTA_BETA`).
