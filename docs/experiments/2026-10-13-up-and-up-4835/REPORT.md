# Up and Up, re-measured: today's app on the build-up's sessions (issue #4835)

**The slide's "today's app" line was the app of 2026-10-08. Re-run on the same 720 sessions a preset with the app of
2026-10-13, it holds the text query's set on screen to vote 30 instead of handing over near vote 23, never dips below
it, and gains +0.021 / +0.013 / +0.010 over the session at beta 1/4 / 1 / 4** (paired, ±0.003 / 0.002 / 0.002 SE),
most of it in votes 51–150. After the spot check it ends at 0.64 / 0.53 / 0.63.

## The question

The owner, reviewing the deck (#4835, slide 43): "Is this graph current? It looks like we stick with text until low
20s, but [Lay of the Land] says we stick with text until 30?" It was not current. The slide drew #4668's build-up,
measured 2026-10-08, whose "today's app" handed over at about vote 23 and dipped below the text query for a few votes.
The text query's hold (#4736, merged 2026-10-13) keeps the query's set on screen until 30 labels on the Binary path,
and the Spot Check before Done (#4836, merged 2026-10-13) changed the end of the session. The owner also asked for the
chart at beta 1 and beta 4, and an x axis of text query, 25, 50, 75, 100, 125, 150, ∞.

## What ran

#4668's rung `b5_app_<preset>`, which sets nothing but the preset's beta and so is the app as it ships, re-run with
`launch_buildup_4668.sh` from a frozen worktree at dev `d046982b2` (the merge of #4836), into its own base,
`/expscratch/sgreenberg/upandup-4835`. Everything else is #4668's: COCO Better, every class at every size (144 cells),
5 seeds, the user's pool at 1%, SigLIP binary votes, 150 votes and the end-of-run spot check. 2,160 runs, 0 lost,
2026-10-13 16:19–17:03. The text query's baseline at today's per-preset line came out byte-identical to #4668's.

`scripts/experiments/calibration/analyze_upandup_4835.py` reads the runs the way `analyze_buildup_4668.py` reads its
b6 rung: the objective (F-beta, at the preset's beta, of the withheld half above the line the app shows) at every
vote, every run in every average, filled with the text query's own set until the app shows a detector. Its curve
matches `analyze_buildup_4668.py`'s own b6 read of the same runs exactly (largest difference 0.0). It adds the
session's end after the check, the State of the App's rule (`state_of_app/analyze.py`): a run scores its last check
row's withheld set if the app shows that detector, and keeps its last vote otherwise. 717 of 720 runs ran a check at
every preset. `today_curve.csv` is the output; `REPORT_machine.md` the analyzer's own summary.

## Results

The objective, mean over the 720 runs. "10-08" is #4668's b6 (today's app as of 2026-10-08); cross-calibration and
the labels line are #4668's b1 and b2, which did not run a check.

| beta | line | mean, votes 1–150 | mean, votes 1–50 | vote 50 | vote 150 | after the check |
|---|---|---|---|---|---|---|
| 1/4 | cross-calibration | 0.38 | 0.21 | 0.43 | 0.46 | |
| 1/4 | + the labels line | 0.44 | 0.22 | 0.46 | 0.59 | |
| 1/4 | today's app, 10-08 | 0.54 | 0.48 | 0.51 | 0.63 | |
| 1/4 | **today's app, 10-13** | **0.57** | **0.50** | **0.54** | **0.64** | **0.64** |
| 1 | cross-calibration | 0.38 | 0.20 | 0.40 | 0.49 | |
| 1 | + the labels line | 0.37 | 0.19 | 0.40 | 0.50 | |
| 1 | today's app, 10-08 | 0.45 | 0.38 | 0.42 | 0.53 | |
| 1 | **today's app, 10-13** | **0.46** | **0.39** | **0.44** | **0.53** | **0.53** |
| 4 | cross-calibration | 0.46 | 0.29 | 0.46 | 0.60 | |
| 4 | + the labels line | 0.48 | 0.31 | 0.49 | 0.60 | |
| 4 | today's app, 10-08 | 0.54 | 0.48 | 0.51 | 0.61 | |
| 4 | **today's app, 10-13** | **0.55** | **0.48** | **0.52** | **0.62** | **0.63** |

Today's app against 10-08's, paired run for run (± SE, n = 720):

| beta | votes 1–150 | votes 1–50 | votes 51–150 | votes 26–35 |
|---|---|---|---|---|
| 1/4 | +0.021 ± 0.003 | +0.017 ± 0.002 | +0.023 ± 0.003 | +0.024 ± 0.004 |
| 1 | +0.013 ± 0.002 | +0.010 ± 0.002 | +0.014 ± 0.002 | +0.010 ± 0.004 |
| 4 | +0.010 ± 0.002 | +0.004 ± 0.002 | +0.013 ± 0.002 | +0.002 ± 0.004 |

- **The hand-over moved from vote 23 to vote 30 at every preset, and the dip went with it.** On 10-08 the first
  detector shown scored below the text query for a few votes (0.45 against 0.48 at beta 1/4); on 10-13 no vote scores
  below the text query's 0.48 / 0.37 / 0.47.
- **Most of the gain is late**, votes 51–150. This run cannot say which of the changes since 10-08 bought it: the
  hold, the head's count line while the class model is thin (#4786, #4798), the head started from the Goods' average
  (#4785), the centroid count (#4759, #4779, #4783) and the check before Done (#4836) all landed in between.
- **At beta 1 the labels line ties cross-calibration** (0.37 against 0.38 over the session), as #4668 found; the slide
  says so rather than claim a step there.
- **The check barely moves the end**: +0.001 / −0.003 / +0.008 against vote 150.

The slides: `slides/figs/up-and-up.png` (beta 1/4, a three-page build) and `slides/figs/up-and-up-b1.png`,
`slides/figs/up-and-up-b4.png`, drawn by `slides/figs/src/make-up-and-up-fig.py` from this CSV and #4668's.
