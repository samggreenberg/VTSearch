# Test mode: Find becomes a decision, not a browse (#4520)

**Status:** design, decided. The test sample (#4527) and the Test autopilot
inside the Find view (#4524) have shipped; the rest has not. #4520 asks three
questions: how a "Test" would work, how the open Find interface becomes a
constrained Test interface, and which metric and statistics it needs. This
file is the answer; the owner settled its open decisions on 2026-10-05, and
the work is filed as the issues at the end.

**Background, what already exists.** Three pieces, which this design assembles
rather than replaces:

- **Find** (`frontend/src/app/components/find-view/`,
  `vtsearch/routes/detectors/scoring.py`) scores a dataset, draws the balance's
  line on that corpus, flood-fills every item with the detector's call, and
  opens a three-pane view whose work queue walks the line alternately above and
  below it (the **Review** tab since #4524, beside the Test autopilot's). Votes
  there are session-only (`find_mode` keeps them out of the labelset) and reach
  the detector only through **Add Corrections**. They used to feed a Stats
  modal whose *Kept rate* and *Checked by you* curve counted only what the walk
  happened to serve; the walk is biased toward the line by design, so neither
  number estimated anything about the set as a whole. #4524 retired the modal
  into the Test autopilot's result pane.
- **The spot check** (`vtscore/training/thresholds/spot_check.py`,
  `vtsearch/routes/precision_check.py`,
  `frontend/src/app/components/modals/spot-check-modal/`) is the only measured
  precision in the app: uniform picks from doubling rank bands (8, 8, 16, 32,
  ...), a band-stratified estimate, a Clopper-Pearson likely range per band, and
  a walk that stops where the balance's F-beta peaks. It runs in Train only
  (#4317), its picks become training votes, and its *found* phrase divides by
  the mixture's count of positives, which no pick ever measures.
- **The active auditor** in [`coverage-atlas.md`](coverage-atlas.md) §6.3
  already specifies the estimation statistics this design needs: strata with
  Beta posteriors, Neyman or Thompson allocation, score-proportional sampling
  of the predicted negatives with Horvitz-Thompson reweighting for recall, a
  stop at a target interval width. That section is the statistical spec; this
  file is the product that calls it.

Train's Autopilot is the model for the shape: a phase panel, one item at a
time, a light per phase, and a **Done!** that says what it found. The guided
flow is called **Autopilot** in Test as in Train (the owner's choice): one
word for "the guided flow" wherever it appears, with the phases differing by
view, so the Autopilot / Review tab pair in Test mirrors Autopilot / Manual in
Train.

## 1. The question Test answers, and the metric

Train asks *is this detector good enough yet?* Find asks nothing: it hands
over a ranked list and a pair of piles. Test asks one question with a yes or
no answer:

> If I move this detector to AutoRun, the matches it ships from a future
> dataset like this one will be what share right, and what share of the real
> matches?

Both halves are needed, because AutoRun ships the set above the line unchecked
and nobody downstream can tell a wrong match from a missed one. The **metric
is the line's precision and recall on this corpus, at the balance the user
picked**, each as a likely range, and **F-beta at that balance** as the single
headline, from the same draws. The verdict is a reading of the ranges, not a
threshold the app enforces: *do your best, and say how close we got* (#4267)
carries over unchanged. Test does not promise; it measures, and tells the user
what it measured so they can decide.

The denominator the ranges are over is the corpus in front of the user,
treated as a sample of the future datasets it stands in for. The two trust
checks the Stats modal already carries, **Training-domain overlap** and
**Evidence coverage**, are exactly the caveat on that extrapolation and belong
on the verdict page, not behind a button.

Two constraints that the metric imposes, and that Find does not meet today:

- **Test votes must not train the detector.** The point of a test set is that
  the detector never saw it. Find already keeps session votes out of the
  labelset; Test keeps that and adds a provenance flow of its own (`test` in
  `vtscore/datasets/vote_provenance.py`), so a later merge can tell a test
  vote from a check vote. **Add Corrections** stays as the *failed-the-test*
  exit, and the moment it is used the test result is stale: the detector has
  now seen the test set. `find_eval_stale` already models this.
- **The picks must be a sample, not the top of the ranking.** Every number
  Test reports is a function of uniform picks within rank bands, as the spot
  check's are (#4257: model-chosen votes broke 83% of the old estimator's
  promises). The boundary walk, the piles and the ranked list are therefore
  not where the metric comes from, and the interface says so by not showing
  them while the test runs.

## 2. The Test autopilot

A phase machine in the shape of Train's
(`frontend/src/app/services/autopilot-state.service.ts`), with the phase
derived from state on every poll rather than accumulated, and ported to the
harness like `vtscore/eval/autopilot_flow.py` so the eval's default arm runs
the same flow.

1. **Score.** No votes. Find's pass as it is today: train or load the head,
   score the corpus, draw the balance's line on it. The light is the pass's
   progress bar.
2. **Check the matches.** Precision. Uniform picks from the bands *above* the
   line, 5 a round (`CHECK_MIN_PICKS`), starting with the band holding the
   line and working up, so the first rounds land where the detector is least
   sure and the set is largest. Later rounds go to whichever band's posterior
   would shrink the F-beta range most (the greedy face of Neyman allocation;
   `coverage-atlas.md` §6.3). The light tracks the precision range's width
   against its target: red while wider than twice it, yellow within twice,
   green at or under.
3. **Check the misses.** Recall. Picks from the bands *below* the line, the
   first band under it first, then deeper while a band is still turning up
   matches or its posterior mass is not yet negligible against the matches
   found above the line. Estimation is model-assisted (the item posteriors
   from the labels line as the auxiliary, corrected band by band by the picks;
   Horvitz-Thompson under the band design), because uniform sampling of a
   10,000-item tail at 0.4% prevalence can bound nothing. The deep tail the
   walk never reaches is taken from the model and said to be, as the *found*
   phrase is today. This phase has a dry-run stop like the document stop of
   #4488: a band with no match in its picks and little posterior mass ends
   the walk. The light is the recall range's width.
4. **Done!** The verdict, with three exits, mirroring the **Detector
   Trained** dialog:
   - **Move to AutoRun**, the reason the test exists. Available whatever the
     ranges say; the verdict sits beside the button.
   - **Lean the Threshold**, when precision and recall trade the wrong way.
     Because the picks are uniform within bands and the bands are nested, the
     same draws re-estimate the line at every band edge: the verdict page
     shows the precision and recall ranges each preset would ship, from the
     picks already taken, with no extra votes. This replaces the Stats modal's
     *Checked by you* curve with an unbiased one at band resolution.
   - **Add Corrections and retrain**, the failed-the-test exit. It marks the
     result stale, as today.

   And one more state, **Nothing to test**: the line keeps nothing on this
   corpus, or fewer items than one round. The verdict then says so and offers
   the Threshold and the Train view, as Autopilot's *exhausted* does.

**The stop rule.** Each phase ends when its range is narrower than a target
set behind the scenes, or at a per-phase pick budget, or when its bands are
exhausted. Proposed values to price, not to ship untested: 95% ranges, a
precision width of 0.20 (the spot check's 5-pick band ranges are about 0.57
wide, and three to four rounds on one band bring it near 0.20), a recall
reported in the five *found* words rather than a width (the honest resolution
of a model-assisted tail), budgets of 40 picks above the line and 40 below.
Intervals are Beta posteriors per band drawn jointly by Monte Carlo (Jeffreys
prior), so precision, recall and F-beta come from one set of draws and
optional stopping does not change what the posterior means; frequentist
coverage of the stop is priced in the eval, as every check so far was
(#4267, #4383). The user clicks until Done, as the issue asks, and Done is
where the ranges stop moving usefully, not where the corpus runs out.

**The verdict persists** (the owner's choice). A finished test is kept on
the detector, one entry per tested dataset: the picks' ids, labels and bands,
the balance, the date, and the ranges and F-beta at Done; ids, labels and
numbers only, never a vector. A retrain marks every entry stale, as the spot
check's range goes stale, and the entry stays flagged. The detector's Stats
and the Dashboard's AutoRun tab read it (*tested on drawings-new: likely
70–85% right, about half found*), which is the reason to move a detector to
AutoRun. A later Test on the same dataset resumes from the kept picks when
the ranking is unchanged.

**What a vote does.** It is recorded against the pick's band, refreshes the
ranges on the right, and advances to the next pick. There is no retrain, no
re-sort and no line move between votes: the ranking is frozen for the whole
test, which is what makes the band design valid. The ↓ key goes back one pick
in the round, as in the spot check.

## 3. The interface: from three open panes to one constrained one

The Test view keeps the Train view's chassis (left panel, centre viewer, right
panel) and the pair route (`/test/:datasetId/:detectorId`, today's `/find`),
and swaps what each pane holds while the test runs:

- **Left.** The phase panel (Score, Check the matches, Check the misses, Done),
  with the balance control above it. The balance is live between phases and
  frozen within one: changing it mid-phase would move the line and the bands
  under the picks. The ranked media list is **not shown** during the test. A
  pick is drawn from a band, and showing the list would show its rank, which
  the spot check already forbids for the same reason (#4267, "the picks are a
  separate check, not the ranking").
- **Centre.** The current pick with Good and Bad, the round's dots, and the
  phase's one-line prompt, as the spot-check modal shows today, but as the
  view rather than a modal over a list.
- **Right.** The result as it forms: picks so far by band, the precision and
  recall ranges with their lights, the F-beta headline, and the two trust
  chips. This is the Stats modal's content, live, in place of the two piles.

After Done, a **Review** tab, where today's Find lives in full: the ranked
list with the line, the boundary walk, Verified Good and Verified Bad, To
Dataset, Export and Browse. The test's picks are already in the piles. This is
Train's Autopilot / Manual split applied to Test: the guided flow is the
default, the open one is a tab away, and nothing a user can do today is lost.
A user who wants the old Find without a test clicks Review first.

The Stats modal goes: its counts and the 2×2 table move to the right pane,
its *Checked by you* curve is replaced by the band-resolution curve of §2's
verdict, and its trust chips move to the verdict. The balance control's state
line does not show Train's check range here (the owner's choice): that range
measured the training corpus, and showing it in Test invites reading it as
this corpus's. In Test the line reads this corpus's result, or *untested*
before Done.

## 4. What Find becomes

The Dashboard's big button reads **Test**. The third big button, labelled
**AutoRun** while today's Find is still on the Dashboard (it runs the
background AutoRun task on every ticked dataset with every ticked detector,
and is a button, not a view), is relabelled **Find**. The row menu's **Run
AutoRun** stays.

The *Scoring a dataset* and *Find* sections of the user guide become a
*Testing a detector* section; the step-by-step's Step 4 becomes the test. The
screenshots move with it (queued under `docs/reshoot-queue/`, not reshot).

## 5. What has to be written

Library tier, `vtscore`, Flask-clean: shipped as
`vtscore/training/thresholds/line_test.py` (#4527) - the test sample, its
estimators, the allocation rule, the phase machine and stop rule, and the
`test` vote flow. Its targets and budgets (`LineBudgets`) are the §2
proposals until #4523 prices them; its pin in `scripts/check-eval-app-sync.py`
is that slice's.

App tier:

- Routes under `vtsearch/routes/` for start, round votes, cancel and state,
  on the pattern of `vtsearch/routes/precision_check.py`, minus the line move
  and the training write.
- The Test view and its phase panel, the right-pane result, the Review tab,
  the route rename, the Dashboard buttons, the hint text.
- The persisted verdict on the detector, and its two readers.
- Docs: the user guide and how-tos named in §4, and the eval doc's
  description of the new mirror.

## 6. Price it before shipping it

The stop rule and the recall estimator are claims about coverage and cost
that only a simulated user can check, as #4267 priced the floor check and
#4383 the band walk. The harness already measures *train on X, Find on Y*: the
withheld half is the test corpus, and ground truth is in hand. A Test arm in
`vtscore/eval/voting_iterations.py` runs the Test autopilot on the withheld
half after the training run's last click and records, per session: picks to
Done per phase, the ranges at Done, whether they held the truth, the F-beta
the verdict would read against the oracle's, and how often the verdict's
reading (ship, lean, retrain) matches what the oracle would say. Run on COCO
Better at 0.1%, 0.44% and 5% and on a document class, at the three presets.
That run picks the targets and budgets §2 proposes, and it is an
`experiment`-labelled issue, so it is laptop work.

## Slices

Each is independently shippable. The test sample and the eval arm carry no visible change, and the AutoRun button needs nothing else here, so it can ship first.

<!-- item-sep -->

- [x] #4527 — Test mode: the test sample and its estimators in vtscore (Opus)

<!-- item-sep -->

- [ ] #4523 — Test mode: eval arm and the pricing study for the stop rule (Opus; `experiment`)

<!-- item-sep -->

- [x] #4524 — Test mode: the Test autopilot inside the Find view, with Find kept as a Review tab (Opus)

<!-- item-sep -->

- [ ] #4529 — Dashboard: a big button that runs AutoRun on the selected datasets and detectors (Sonnet; depends on nothing here)

<!-- item-sep -->

- [ ] #4525 — Test mode: rename Find to Test, and relabel the AutoRun button Find (Sonnet)

<!-- item-sep -->

- [ ] #4526 — Test mode: persist a finished test's verdict on the detector (Sonnet)
