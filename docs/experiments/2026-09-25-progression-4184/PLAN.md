# The calibration ladder on COCO Better: one cost curve per rung (#4184)

**Status: planned 2026-09-25. The harness, launcher, analyzer and slide figure
script are in the tree; no cell has run yet, so there is no verdict.** The
cells need the GRID (the COCO Better pile lives there). Everything below is
fixed before the first cell, including what the figure does if a rung does
not come out lower than the one before it.

## What this is for

The *Hold The Line* deck (`slides/decks/hold-the-line.deck`) walks the
threshold through a ladder of ideas. Each one repairs what the last starved on.
This study draws that ladder as a single figure: one mean-cost-over-votes curve
per rung, all on COCO Better (`coco_better` in code, renamed in #4183), and all
leaving from the same click-0 notch, the typed query's own ranking.

It is an **illustration, not an A/B**. No knob ships or retires on it, and the
decision rules below are about what the slide may say, not what the app does.

## The rungs

Each rung is the app **as the deck draws it** at that point, not as it
historically shipped. The owner accepted that in #4184. The real order ran
cross-calibration and the GMM in one February commit, left the blend off by
default for five months, shipped the acquisition offset before the 70/30 split,
and swapped the head twice in between. Two adjacent rungs differ in exactly the
thing the slide between them changes. Everything no slide discusses is held at
today's production.

| rung | slide | live threshold | calibration split | acquisition |
|---|---|---|---|---|
| `r1_xcal` | *Grading Your Own Homework* | each fold's cost-minimising cut, averaged (`xcal_mincost`) | 50/50 | at the reporting cut |
| `r2_gmm` | *Oops! All Haystack* | GMM over the haystack, midpoint of the means (`gmm_mid`) | 50/50 | at the reporting cut |
| `r3_blend` | *Cross Examination* | `r1`'s cut and `r2`'s under `corridor20` (`blend`) | 50/50 | at the reporting cut |
| `r4_rawmean` | *The Rank & File*, build b | anchored fold mixtures, midpoints averaged as raw scores (`anchored_rawmean`) | 50/50 | at the reporting cut |
| `r5_anchored` | *The Rank & File*, build d | the shipped fused cut: anchored folds, quantile transfer | 50/50 | at the reporting cut |
| `r6_split70` | *Train More, Check Less* | the shipped fused cut | 70/30 (production) | at the reporting cut |
| `r7_acq4` | *Second Cut* | the shipped fused cut | 70/30 (production) | 4 inclusion steps below (production), i.e. **today's app** |

The live-threshold names are `CALIB_LIVE_THRESHOLD` values, implemented in
`vtscore/eval/live_threshold_rules.py`. The table is `launch_progression_4184.sh
plan`'s output, and the analyzer reads each rung's premise back off its own rows.

**Why every rung is closed-loop.** The line decides twice: what comes back, and
what autopilot asks next. A rung replayed as a re-cut of another rung's
trajectory would have chosen none of its own questions, so each rung is its own
`CALIB_EXP` and its own trajectory. The rungs pair on (category, seed): same
split, same test set, same opening. They diverge from the first fitted step.

**Two sections of slides are deliberately not rungs.**

- *Preference* (walk, tilt): at inclusion 0 `mid_tilt` **is** `mid`, bit for
  bit, so these slides move nothing the curve can show. The tilt is what makes
  `r7` possible at all, because the offset is inert under a rule that is
  constant in k.
- *Exploration* (the Coverage Atlas): the owner chose to keep today's
  atlas-driven autopilot in every rung. Its cost effect has never been
  measured. Pricing it would take an atlas-off switch in the ported autopilot,
  which this study does not add.

**Known departures from history**, each deliberate:

- `r1`'s folds are the app's stratified re-draws, not two complementary halves.
  The *Grading Your Own Homework* slide itself calls redrawn splits "polish, not
  the idea". The cut rule is February's (`b5033a152`): `>=` at an observed
  score, ties toward the higher cut. It is reused as `oracle_cut`.
- The head is today's linear SVM throughout. The February app ran a 64-unit MLP.
- The opening is today's text sort throughout. The harness seeded from a boxed
  crop until #3269.
- `r2`'s cut never shipped on its own for a detector. It was only ever the text
  sort's cut and the blend's cold-start end. It is the deck's rung, not a
  historic one.

## The grid

- **Data:** `coco_better`, every designated cell (`CALIB_CATEGORY_MODE=all`):
  49 classes in 144 class@band cells, 100 positives against a shared 9,900
  negatives (~1%).
- **Representation:** `siglip` only, whole-image, **binary voting**. The figure
  sits before the deck's *Regions* section.
- **Opening:** the typed query (`CALIB_REQUIRE_OPENING=text`).
- **Seeds:** 5, as #4184 proposed. **Horizon:** 150 votes.
- **Size:** 144 × 5 = 720 cells per rung, 5,040 in all.

**Resolution.** At the GRID skill's σ ≈ 0.04 per paired cell, an adjacent-rung
difference over 720 pairs carries SE ≈ 0.0015. That resolves a 0.004 step at
2 SE. Five seeds is enough, and each curve point is a mean over 720 runs, so the
drawn curves will be smooth without smoothing.

**Cost: not measured on this grid.** The nearest measurement is #3551's
`caltech101_m × siglip` binary cell at 150 votes: 46 s and 1.4 GB. A COCO Better
cell holds ~11,000 images to caltech's ~8,700, and the retired rules add no
training. Run `size` on `r1` and `r7`, and read `sacct`, before quoting a wall
clock or raising `CALIB_CONC` above the launcher's 15 per rung.

## Running it

From a worktree at `/expscratch/$USER/worktrees/vts-4184` (see the
`grid-experiments` skill for creating it):

```bash
H=scripts/experiments/calibration
bash $H/launch_progression_4184.sh plan        # read the rung table; submits nothing
bash $H/launch_progression_4184.sh prepare     # once
bash $H/launch_progression_4184.sh size 0 r1_xcal
bash $H/launch_progression_4184.sh size 0 r7_acq4
bash $H/launch_progression_4184.sh baseline    # the click-0 notch, shared by every rung
bash $H/launch_progression_4184.sh rungs       # all seven arrays; preflight per rung
bash $H/launch_progression_4184.sh status
bash $H/launch_progression_4184.sh analyze     # after every rung drains
```

## What the analysis reports

`analyze_progression_4184.py` (planted-answer test:
`selftest_analyze_progression_4184.py`):

- **The curve the slide draws is the *filled* mean over every cell.** A cell
  with no app-visible detector yet is still showing the typed query's ranking,
  so its cost at that click is its text-sort cost. That also makes the click-0
  notch the natural left end of every curve. Averaging only the cells that
  happen to have a detector would put an easier subset of the grid on the left.
  A cell that never found a positive in one rung is filled at every click.
  A lost cell file is dropped from its rung and counted.
- **Paired steps.** Each rung is compared with the one before it and with `r1`,
  at 10, 20, 30, 50, 100 and 150 votes and over the whole trajectory. Each
  comparison reports mean Δcost with its SE, at two significant digits.
- **Premises**, read back off each rung's rows: the live threshold, the
  calibration split, one head. The last check is that acquisition leaves the
  reporting cut on `r7` and never on any other rung. A failed premise makes the
  analyzer exit non-zero.
- The mandatory quality-over-clicks pair (`curves.py`) and `viewer.html`.

## What the slide may say (fixed now)

- **The figure shows what the run measures, in the deck's order.** No rung is
  reordered, dropped or smoothed to make the ladder descend.
- **`r4` is expected to sit above `r3`.** *The Rank & File* introduces it as a
  strawman: three models, three scales. If it comes out above, the slide says
  that is the point. If it comes out below, the slide's strawman claim is
  measured wrong, and that goes to the owner before the slide is written.
- **A step smaller than 2 SE is drawn but not narrated as an improvement.** The
  presenter notes say "not resolvable here" for it.
- **Adjacent rungs may cross.** The deck's own argument predicts it for
  `r1`/`r2`: cross-calibration "reads nothing but labels, and so starves early",
  while the mixture reads no labels and cannot starve. If curves cross, the
  notes name the click where they do, rather than quoting one number for the
  pair.

## After the run

1. Commit `REPORT.md` (question as H1, verdict in the first paragraph),
   `progression_curve.csv`, `paired.csv`, `figures/` and `viewer.html` to this
   directory.
2. `python slides/figs/src/make-progression-fig.py` reads the committed CSV and
   writes `figs/progression.png` plus `progression.build1…6.png`, one rung per
   page. Preview the layout at any time with `--demo DIR`, which draws
   synthetic curves and refuses to write into `figs/`.
3. Add a `progression` fragment, a seven-page build (a–g, one rung per page),
   at the end of section 5 in `hold-the-line.deck`, after `data-set-coco-better`.
   That is the only place in the deck that comes after COCO Better is
   introduced and before *Regions*. Its notes name each rung's slide, its paired
   step, and the click-0 notch.
4. Render `hold-the-line` and attach the PDF, per `CLAUDE.md`'s slide rule.
