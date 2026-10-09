# What did each step of the F-beta era buy, in the score the user sets? (issue #4668)

**Two steps carry the F-beta era, and three more each add a little.** On COCO Better with the user's pool at 1%,
720 paired runs per rung, each preset on its own sessions and scored at its own beta:

- **The typed query's line at the preset (#4603)** is the biggest step: +0.26 / +0.19 / +0.16 over votes 1–50 at beta
  1/4 / 1 / 4. Before a detector shows, the user's set scores 0.48 / 0.37 / 0.47 instead of 0.01 / 0.02 / 0.16.
- **The labels line at the preset (#4452)** carries the precision-minded user: +0.13 at vote 150 at beta 1/4. At
  beta 1 it is +0.008, and at beta 4 0.000.
- **The weak check (#4496) and even-odds asking (#3546)** each add +0.011 to +0.017 at vote 150 at every preset.
- **The relative floor (#4492)** adds +0.009 at vote 150 at beta 1/4, nothing at beta 1, and −0.005 at beta 4. Its
  priced gain at votes 5–10 was on detectors the app does not show (#4605).

So at beta 1/4, at vote 150, each shipped step in order raises the score: 0.46 → 0.59 → 0.60 → 0.61 → 0.63. The deck's
slide ("Up and Up") draws the two big steps at beta 1/4, and its notes name the rest.

## The question

The owner (2026-10-08) asked for the slide that shows how the work improved the app, redrawn in F-beta curves on COCO
Better, with the takeaway "the score they care about improved over and over". Cherry-picking a beta or a prevalence was
allowed. "If this slide isn't easy to draw, I want to know why. (It might mean that steps which we thought improved
the app actually didn't?)" And: "If we have steps that we thought were better than the state before them, I want you
to tell me now."

## Why the deck's own ladder cannot be drawn

The deck's Calibration section is a sequence of line rules, each presented as a repair of the one before. Measured in
F-beta, each step against the one before it, at vote 150 (#4582's re-score of #4184's cells at 0.44%, 720 paired runs;
#4519's 1% run has the same shape):

| step (slide) | Δ F-beta 1/4 / 1 / 4 | verdict |
|---|---|---|
| mixture midpoint (Oops! All Haystack, Great Expectations), against cross-calibration | **−0.38 / −0.40 / −0.33** | bad |
| blend (Cross Examination) | +0.016 / +0.021 / +0.044 | better than the midpoint, still ~0.36 under cross-calibration |
| fused cut (Above Average, Vote of Confidence) | +0.031 / +0.042 / +0.076 | same |
| rank transfer (No Mean Feat) | **−0.052 / −0.072 / −0.14** | bad: the raw-mean strawman beats it |
| 70/30 split (Train More, Check Less) | +0.001 / +0.002 / +0.004; ≤ 0.006 under the labels line (#4583) | null |
| Second Cut (line − 4) under the labels line, against asking at the line (#3546) | objective 0.525 vs 0.528 at beta 1 | null |

Every one of those rules estimated the line that minimises FPR + FNR, and at low prevalence that line returns 25 times
too much. None of them is on today's path. The labels line (#4452) reads the folds' held-out scores and its own fits of
the corpus. So the deck's ladder falls off a cliff at its first step and recovers only at the labels line. #4670 moves
those slides past the end of the deck.

Other ideas that measured bad or null in F-beta: the Smart light as a stop signal (#4359, #3560), the Binary opening
hand-off (#4604, after #4603), half-argmax acquisition (#4409, reverted), and band picks mixed into Autopilot (#4482).
Two have never been priced in F-beta: the Coverage Atlas walk and the dry-stop opening (#4671).

## What ran: the F-beta era as a cumulative build-up

The owner picked a build-up of what the app does today, at all three presets. Each rung adds one shipped change to
the rung before it, so two adjacent rungs differ in exactly that change.

| rung | adds | knobs, against today's app |
|---|---|---|
| b1 | cross-calibration on today's folds, asking at its own line, no check | `CALIB_BETA=off CALIB_LIVE_THRESHOLD=xcal_mincost CALIB_ACQ_INCLUSION_OFFSET=0 CALIB_SPOT_CHECK=off`; one set of sessions, scored at each beta |
| b2 | the labels line at the preset (#4452), with its absolute spread floor; asking at line − 4, as the app did 10-02 to 10-07; no check | `CALIB_BETA=<b> CALIB_SIGMA_FLOOR=absolute CALIB_ACQ_TARGET_P=off CALIB_SPOT_CHECK=off` |
| b3 | + the relative spread floor (#4492) | drop `SIGMA_FLOOR` |
| b4 | + the weak-separation check (#4496) | drop `SPOT_CHECK` |
| b5 | + even-odds asking (#3546, #4632): today's sessions | drop `ACQ_TARGET_P` |
| b6 | + the typed query's line at the preset (#4136, #4603): **today's app** | b5's sessions, scored before the hand-over at today's line |
| b7 | + the More walk on the detector's top (#4637), **priced, not shipped** | `CALIB_MORE_WALK=detector`, scored with the typed query on screen until Hard |

- **Bench:** `coco_better`, binary SigLIP, 144 class@band cells × 5 seeds, 150 votes. The user's pool is thinned to 1%
  positive (`CALIB_HAYSTACK_PREVALENCE=0.01`); the withheld half Find searches stays at 0.44%.
- **Code:** one frozen commit for every rung (`vts-buildup-4668-run` @ 2624dbad1), launched by
  `scripts/experiments/calibration/launch_buildup_4668.sh`. The new knob `CALIB_SIGMA_FLOOR=absolute`
  (`live_threshold_rules.sigma_floor`) restores the floor before #4492.
- **Reproduction check:** b1 at the old 50/50 split reproduces #4519's `r1_xcal` cell vote for vote (all 147 steps).
  `CALIB_BETA=off` is therefore the path `CALIB_MIN_PRECISION=off` took before #4421 removed it.
- **Score:** the objective, F-beta at the run's own preset of the withheld half above the line the app shows (#4427).
  Until a run shows a detector it scores the typed query's set: at the mixture midpoint for b1–b5 (the display line
  until #4590) and at today's per-preset line for b6–b7 (#4603). Both come from `text_baseline.py` rebuilt at 1%, which
  also settles #4625. Steps are paired by (category, seed).
- **Analysis:** `scripts/experiments/calibration/analyze_buildup_4668.py`.

## Result

The objective at each preset, as the user experiences it (filled from the typed query's set until a detector shows),
for every rung. Each rung adds one change to the rung above it.

| rung | beta 1/4: mean 1–150 | at 150 | beta 1: mean 1–150 | at 150 | beta 4: mean 1–150 | at 150 |
|---|---:|---:|---:|---:|---:|---:|
| b1 cross-calibration | 0.38 | 0.46 | 0.38 | 0.49 | 0.46 | 0.60 |
| b2 + the labels line at the preset | 0.44 | 0.59 | 0.37 | 0.50 | 0.48 | 0.60 |
| b3 + the relative floor | 0.45 | 0.60 | 0.38 | 0.50 | 0.47 | 0.59 |
| b4 + the weak check | 0.45 | 0.61 | 0.38 | 0.51 | 0.48 | 0.61 |
| b5 + even-odds asking | 0.45 | 0.63 | 0.38 | 0.53 | 0.48 | 0.62 |
| **b6 + the typed query's line: today's app** | **0.54** | **0.63** | **0.45** | **0.53** | **0.54** | **0.61** |
| b7 + the detector walk (not shipped) | 0.55 | 0.64 | 0.45 | 0.53 | 0.54 | 0.62 |

![Every rung's objective over votes, one panel per preset](figures/buildup_curves.png)

*Each line is the mean of 720 runs at that preset's own beta. Read a step as the gap between a line and the one
before it. b2–b5 lie within 0.04 of each other everywhere (widest at vote 150), so they overlap. b6 and b7 start at
today's typed-query line, and the rest start at the old midpoint.*

**Each step against the one before**, paired by (category, seed). Bold means more than 2 SE from zero:

| step | beta 1/4: mean 1–150 | at 150 | beta 1: mean 1–150 | at 150 | beta 4: mean 1–150 | at 150 |
|---|---|---|---|---|---|---|
| b1 → b2 the labels line | **+0.064** ± 0.003 | **+0.131** ± 0.005 | **−0.003** ± 0.001 | **+0.008** ± 0.002 | **+0.014** ± 0.002 | −0.000 ± 0.003 |
| b2 → b3 the relative floor | **+0.005** ± 0.001 | **+0.009** ± 0.003 | +0.001 ± 0.001 | +0.000 ± 0.002 | **−0.005** ± 0.001 | **−0.005** ± 0.002 |
| b3 → b4 the weak check | **+0.005** ± 0.001 | **+0.011** ± 0.004 | **+0.005** ± 0.001 | **+0.013** ± 0.002 | **+0.005** ± 0.001 | **+0.017** ± 0.002 |
| b4 → b5 even-odds asking | −0.000 ± 0.001 | **+0.017** ± 0.004 | **+0.003** ± 0.001 | **+0.013** ± 0.002 | **+0.009** ± 0.001 | **+0.011** ± 0.002 |
| b5 → b6 the typed query's line | **+0.092** ± 0.003 | −0.000 | **+0.065** ± 0.002 | −0.000 | **+0.054** ± 0.002 | **−0.003** ± 0.001 |
| b6 → b7 the detector walk | **+0.008** ± 0.002 | +0.005 ± 0.003 | **+0.004** ± 0.001 | **+0.006** ± 0.002 | **+0.003** ± 0.001 | +0.004 ± 0.002 |

Over votes 1–50 the typed query's line is +0.258 ± 0.007 / +0.186 ± 0.005 / +0.164 ± 0.006, and nothing else
reaches 0.02 there. The full table, with the windows 1–25, 1–50 and 51–150 and the share of runs better or worse by
more than 0.05, is `paired.csv`.

![The mean over votes 1–150, rung by rung](figures/buildup_steps.png)

**What the user gets at vote 150** (runs showing a detector; `returned.csv`):

| | beta 1/4: precision / recall / returned, median / over 200 | beta 1 | beta 4 |
|---|---|---|---|
| b1 cross-calibration | 0.48 / 0.69 / 69 / 25% | 0.48 / 0.69 / 69 / 25% | 0.48 / 0.69 / 69 / 25% |
| b2 the labels line | 0.67 / 0.47 / 36 / 5% | 0.53 / 0.63 / 54 / 20% | 0.37 / 0.80 / 125 / 42% |
| b6 today's app | 0.76 / 0.40 / 24 / 0% | 0.59 / 0.58 / 48 / 7% | 0.37 / 0.75 / 126 / 37% |

Cross-calibration returns the same set whatever the user wants. Today's app returns a third as many at beta 1/4, at
precision 0.76, and twice as many at beta 4, at recall 0.75. The weak check is what shrank the beta 1 tail (over 200:
20% → 11%).

## Readings

- **The labels line is a beta 1/4 win.** Cross-calibration's line is long and recall-heavy. At beta 4 that already
  suits the user, so the labels line ties it at vote 150. It is +0.014 over the session, spread across it and gone by
  the end. At beta 1 it
  is −0.003 over the session and +0.008 at the end. This is Follow Suit's finding (#4548), now on today's harness.
- **The relative floor is not the win it was priced as.** #4492 measured +0.25 to +0.39 at votes 5–10, on detectors
  scored from the first vote. The app shows the typed query's set until Hard (#4605), so a user never sees those
  detectors. In what the user sees, the floor is +0.005 / +0.001 / −0.005 over the session. The beta 4 loss is small
  but resolvable. It is not owed a revert on this evidence, but it belongs on the list of ideas whose priced gain did
  not reach the user.
- **The weak check and even-odds asking are late, small and consistent:** each adds +0.011 to +0.017 at vote 150 at
  every preset, and less than 0.01 over the session.
- **The typed query's line is the largest single step of the F-beta era.** It also makes the hand-over visible: where
  today's app first shows a detector (from vote 23), that detector's set is briefly worse than the typed query's. The
  dip is 0.48 → 0.45 at beta 1/4, deepest at vote 26 (37% of runs have handed over by vote 25), and gone by vote 31.
  At beta 1 and 4 it is 0.01 deep and gone by vote 28. #4508 and #4604 own the hand-over.
- **The detector walk (#4637, not shipped)** adds +0.003 to +0.008 over the session, in line with #4637's own
  +0.011 / +0.008 / +0.004.

## The slide

`slides/fragments/up-and-up.md` ("Up and Up", Section VI, before Photo Finish). The figure is
`slides/figs/src/make-up-and-up-fig.py`, from this study's `buildup_curve.csv`. It shows one panel at beta 1/4, built
in three pages: cross-calibration (b1), then the labels line (b2), then today's app (b6). The owner picked it over a
staircase of every rung and over three rows of every rung: the floor, the check and even-odds asking move the curve
by less than 0.02, which no line on the slide can show.

## Ops

- **The cluster's MaxJobCount (10,000) counts every array task.** Sixteen whole 720-task arrays would have needed
  11,520 job records. The first twelve took 85% of the cluster's records, the last four were refused, and other
  users' arrays would have been too. The run was finished by a driver (`/expscratch/sgreenberg/buildup-4668/drive.sh`)
  that submits 48-task chunks per rung, at most 6 running per rung, and stays under ~800 records. Its log, `drive.log`,
  shows nothing retried and nothing given up.
- **Cells:** ~4.5–5 min and 1.5 GB each; 11,520 runs between 14:55 on 10-08 and 01:36 on 10-09.

## Files

- `buildup_curve.csv`: per preset, rung and vote, the objective's filled mean and SE, the share of runs showing a
  detector, and AP.
- `paired.csv`: each rung against the one before, and against b1, per preset, at fixed votes and over windows.
- `returned.csv`: precision, recall and the returned set's size at vote 150.
- `REPORT_buildup.md`: the analyzer's machine summary, including what each arm ran with, read back off its rows and
  its run logs.
- `provenance.json`: the files read per arm.
- `figures/`: the two figures above, and the mandatory AP-over-clicks figure (`average_precision_vs_clicks.png`).
- The interactive viewer (14 MB) and the per-run AP figure (4 MB) are too large to commit. They stay at
  `/expscratch/sgreenberg/buildup-4668/analysis/`, as do the cells.

Follow-ups: #4670 (the deck's claims, done in #4679), #4671 (the atlas walk and the dry-stop opening, never priced in
F-beta). Refs #4519, #4548, #4582, #4625.
