# Fitting the 3-part fit's Bad component on the corpus recovers about half of the chances' shortfall at beta 1/4 and 1 (issue #5040)

**Status (2026-10-17): priced offline on #4937's snapshots (Binary Photo, 702 sessions at each of the three presets,
nine clicks from vote 10 to 150; Region Photo seed 0 at beta 1), every candidate the issue names and their
combinations, on the withheld half and on three corpora cut from it. The best two are in a closed-loop A/B at all
three presets: CLOSED-LOOP PENDING.**

**The labels line's chances (#4937) can be moved most of the way the truth moves them on the negatives' side, and
almost none of the way on the Good side.**

1. **The Bad component, fitted on the corpus.** Today the 3-part fit's Bad component is a normal fixed at the
   held-out Bads' mean with the Goods' spread. Fitting its mean and spread on the corpus being decided, the held-out
   Bads counted as labelled points of it, gains +0.056, +0.048 and +0.017 F-beta at beta 1/4 (votes 40, 100 and 150)
   and +0.025, +0.022 and +0.007 at beta 1, on the same heads and on the class model's cut. That is about 45% and
   30% of the true chances' ceiling, and 55% to 60% of what the negatives' own (true) shape gains. At beta 4 it loses
   0.009 to 0.011. On a corpus with no positives it keeps fewer images than today at every preset.
2. **With the counted bound loosened from 2x to 4x** the same fit gains +0.061, +0.056 and +0.020 at beta 1/4,
   +0.033, +0.033 and +0.012 at beta 1, and +0.004, +0.003 and -0.001 at beta 4 (+0.017 at vote 50). It still keeps
   fewer images than today on a corpus with no positives (25 against 32 at beta 4, vote 40). Removing the bound gains
   a little more on a full corpus, but at beta 4 an empty one then keeps 84 and 148 wrong images at votes 40 and 100
   (54 and 87 with the fitted Bad component) where today keeps 32 and 47, so it fails #4452.
3. **No Good-side candidate pays.** The Goods' own spread gains at beta 1/4 and 1 (+0.03 and +0.02 at vote 40) and
   loses 0.02 at beta 4. Fitting the Good component on the corpus runs away: the prevalence estimate reads 13 to 14
   times the truth and an empty corpus keeps up to 1,600 images. The spot check's Goods lose 0.03 at beta 4. The
   held-out Goods are selected, and nothing the app holds tells the corpus's hard positives from its negatives near
   the line.
4. **Region overshoots.** On Region Photo the fitted Bad component puts a chance of 0.87 at the best set's edge,
   where the truth is 0.37 (today 0.23), and the set runs 1.3 times the best early. F-beta is flat at votes 25 to 50
   (-0.006, -0.007) and +0.031 at vote 100. Region's line was not short to begin with (#4937).

| class model's cut, F-beta gain over today (vote 40 / 100 / 150) | beta 1/4 | beta 1 | beta 4 |
|---|---|---|---|
| the true chances (ceiling, #4937) | +0.127 / +0.124 / +0.084 | +0.086 / +0.073 / +0.045 | +0.070 / +0.052 / +0.037 |
| the negatives' own shape (truth, #4937) | +0.098 / +0.087 / +0.048 | +0.041 / +0.041 / +0.017 | -0.006 / -0.002 / -0.005 |
| **Bad component fitted on the corpus** | **+0.056 / +0.048 / +0.017** | **+0.025 / +0.022 / +0.007** | **-0.010 / -0.009 / -0.011** |
| **… and the bound at 4x** | **+0.061 / +0.056 / +0.020** | **+0.033 / +0.033 / +0.012** | **+0.004 / +0.003 / -0.001** |
| **its mean alone fitted, the bound at 4x** | **+0.054 / +0.052 / +0.024** | **+0.036 / +0.031 / +0.012** | **+0.008 / +0.005 / -0.000** |
| the bound at 4x alone | +0.017 / +0.033 / +0.012 | +0.017 / +0.019 / +0.005 | +0.014 / +0.013 / +0.005 |
| no counted bound | +0.029 / +0.051 / +0.018 | +0.031 / +0.026 / +0.005 | +0.015 / +0.013 / +0.006 |
| the Goods' own spread | +0.031 / +0.044 / +0.007 | +0.019 / +0.025 / +0.007 | -0.021 / -0.001 / -0.002 |
| images kept on the corpus with no positives (vote 40): today / fitted / fitted, 4x / mean, 4x / no bound | 3.5 / 1.8 / 2.1 / 2.5 / 6.3 | 8.1 / 4.3 / 5.5 / 6.1 / 25 | 32 / 19 / 25 / 29 / 84 |

## The question

#4937 found that on the class model's cut the labels line's chances underrate the positives at the line: the true
chances would gain +0.13 (beta 1/4), +0.07 to +0.10 (beta 1) and +0.05 to +0.08 (beta 4) through vote 100. It split
the error into a Good side (the held-out Goods are the session's easy positives, so the Good component sits 0.3 logit
too high and too narrow) and a negatives' side (the 3-part fit's Bad component and bulk put too much negative weight
at the line), in proportions that depend on the preset. The issue asks for deployable fixes on each side, and for the
counted bound, priced in F-beta at each preset, on the same heads and then in closed loop, with Find on a corpus with
no positives still keeping close to nothing (#4452).

## What was run

| | |
|---|---|
| snapshots | #4937's: today's app (dev 73c887c27), Binary Photo, `coco_better`, 144 class@band cells x 5 seeds, at beta 1/4, 1 and 4; snapshots at votes 10 to 150 with every fold model's scores. Region Photo: #4936's seed-0 run at beta 1 (139 sessions with a head) |
| pricer | `scripts/experiments/state_of_app/chances_5040.py`: Find's line on the withheld half (`fit_labels_line(...).on_corpus(scores)`) redrawn with one candidate swapped in. Today's replay matches the library's threshold and prevalence to the last bit on all 14,253 Binary and 926 Region snapshots |
| corpora | the withheld half (~10,900 images); its negatives alone (`empty`, ~10,850, no positive); 200 of its negatives (`empty200`, #4452's case); 1,000 of its images at their own prevalence (`small1k`) |
| library arm | `corpus_posteriors(bad="corpus")`, set for a run by `labels_chances()` (`CALIB_LINE_BAD`, `CALIB_LINE_BOUND_MARGIN`). Drawn through the library on every snapshot it keeps the pricer's set exactly (`chances_5040.py verify`) |
| closed loop | `chances_5040.sh`: arms `ctl` (the app), `bssl` (the Bad component fitted on the corpus) and `bssl4` (the same, the bound at 4x), Binary Photo, beta 1/4, 1 and 4, 5 seeds x 144 cells, 200 clicks and the closing check, paired by cell and seed, from a frozen worktree at 4783a51eb |

Every gain is on the same heads, at the session's own preset; every ± is one SE over classes (a class's bands and
seeds averaged first). "The class model's cut" is the snapshots where the class model draws the line (60% at vote
35, 82% at vote 100, 86% to 88% at vote 150); on the rest the head's count line serves (#4786, #5039), which no
candidate here changes. "What Find returns" counts both.

## The candidates

The issue's candidates, as each was built. Every one reads only what the app holds: the labels' held-out fold scores,
their surfacing provenance, and the scores of the corpus being decided.

| side | candidate | what changes |
|---|---|---|
| Good | the Goods' own spread | the Good component's spread is the held-out Goods' own, not pooled with the Bads (floored on the corpus as today) |
| Good | fitted on the corpus | its mean and spread fitted by EM on the corpus, the held-out Goods counted as labelled points of it, in both the 2-part total and the 3-part chances (semi-supervised) |
| Good | the spot check's Goods | its mean and spread from the held-out Goods a spot check picked (uniform within rank bands; their votes carry `flow: check`), where a session holds 3 or more |
| Bad | its own spread | the Bad component's spread is the held-out Bads' own |
| Bad | the Bads' density | the Bad component is a kernel density of the held-out Bads (#4490's shape), its weight fitted by the 3-part EM, which takes up their selection |
| Bad | fitted on the corpus | its mean and spread fitted on the corpus, the held-out Bads counted as labelled points of it; split into its mean only and its spread only |
| total | the bound at 4x / no bound / the chances' sum | the 2-part estimate may exceed the counted bound by 4x rather than 2x / is uncapped / is replaced by the chances' own sum |
| both | combinations | both classes' own spreads; both fitted; each fitted Bad with the bound at 4x or none; with the Goods' own spread |

The truth swaps from #4937 ride along on the same rows: the true chances (isotonic on the truth: the ceiling), the
negatives' own shape and the Good component's true mean and spread.

## The Bad component fitted on the corpus

![Each candidate's gain over today's line, its set size and what it keeps with no positives](figures/binary/candidates.png)

*Per preset over votes, on the class model's cut: each candidate's F-beta gain over today's line on the same heads
(top), its median set size against the best set's (middle), and the mean number of images it keeps on the withheld
half's negatives alone (bottom, log scale). Dotted: the true chances.*

**Why it works.** Today's Bad component is pinned where the held-out Bads sit, and the Bads are the ones Autopilot
picked near the line and from the top. The corpus's own negatives near the line sit lower and spread differently, so
the 3-part EM can only shrink the Bad component's weight or let it eat the positives' lower tail. Fitted on the
corpus, the component moves to where those negatives are, its weight roughly doubles (0.022 to 0.096 of the corpus
at vote 50, beta 1/4, in a 25-cell sample), and the chances' sum moves to the true count of positives (32 to 45 at
vote 50 and 42 to 51 at vote 100, against about 51). At the best set's edge the chance goes from 0.69 to 0.76
(vote 40) and from 0.31 to 0.71 (vote 100) at beta 1/4, where the truth reads 0.79 and 0.75; at beta 1 from 0.15 to
0.30 (vote 50), truth 0.38.

| beta 1/4, class model's cut | vote 40 | vote 50 | vote 100 | vote 150 |
|---|---|---|---|---|
| set ÷ best (median): today / fitted / fitted, 4x | 0.94 / 0.98 / 1.00 | 0.79 / 0.98 / 1.00 | 0.67 / 0.97 / 1.00 | 0.90 / 1.02 / 1.05 |
| chance at the best set's edge: today / fitted / truth | 0.69 / 0.76 / 0.79 | 0.52 / 0.81 / 0.79 | 0.31 / 0.71 / 0.75 | 0.46 / 0.76 / 0.74 |
| expected F-beta minus F-beta: today / fitted / fitted, 4x | -0.06 / +0.03 / +0.00 | -0.11 / +0.02 / -0.00 | -0.21 / +0.00 / -0.04 | -0.17 / -0.04 / -0.06 |

- **Both moves matter, the mean more.** Fitting only the mean gains +0.046 / +0.038 at beta 1/4 (votes 40 / 100);
  only the spread +0.036 / +0.012; both +0.056 / +0.048.
- **At beta 4 it keeps fewer.** The recall cut sits deep in the positives' lower tail, where the fitted component now
  also claims some of the positives; the median set falls from 0.72 to 0.65 of the best at vote 40 and the F-beta
  falls 0.010. The Good side is the shortfall at beta 4 (#4937), and nothing here reaches it.
- **The line's own expected F-beta gets honest.** Today it underrates its set by 0.06 to 0.21 through vote 100; with
  the fitted component the error is within ±0.04 at beta 1/4 and +0.08 to -0.01 at beta 1. The stop (#4878) reads it.
- **What Find returns** (every snapshot, the head's count line where it serves): +0.034 to +0.038 at beta 1/4 (votes
  35 to 100), +0.012 to +0.026 at beta 1, -0.009 to +0.007 at beta 4.

**Its mean alone.** Fitting only the Bad component's mean, its spread kept at the class model's, moves the chances
less: at the best set's edge 0.80, 0.69 and 0.57 at votes 40, 50 and 100 (beta 1/4; the full fit 0.76, 0.81, 0.71,
the truth 0.79, 0.79, 0.75). With the bound at 4x it prices with the full fit on Binary: +0.054 / +0.052 / +0.024 at
beta 1/4 (votes 40 / 100 / 150), +0.036 / +0.031 / +0.012 at beta 1 and +0.008 / +0.005 / -0.000 at beta 4; on what
Find returns +0.031 to +0.041, +0.021 to +0.028 and +0.004 to +0.012 (votes 35 to 100). It leaves more of the
expected F-beta's error (-0.05 and -0.11 at votes 50 and 100, beta 1/4), so the stop moves less. It is the arm that
holds on Region (below), and it was priced, and added to the closed loop, because of that.

## The counted bound

The bound caps the 2-part estimate of the positives at twice the fewest any cut could hold (#4452), and #4937 found
it binds in three quarters of early snapshots and sets a prevalence of half the truth. It matters to the cut wherever
the total is larger than the chances' own sum.

| (vote 40 / 100) | beta 1/4 | beta 1 | beta 4 |
|---|---|---|---|
| prevalence ÷ truth: today / bound 4x / no bound | 0.52 / 0.88 / 1.27; 0.70 / 1.09 / 1.52 | 0.50 / 0.87 / 1.26; 0.67 / 1.07 / 1.53 | 0.51 / 0.88 / 1.26; 0.63 / 1.05 / 1.52 |
| empty corpus, images kept (mean; p90): today | 3.5; 7 / 1.7; 3 | 8.1; 19 / 8.4; 19 | 32; 74 / 47; 112 |
| … the bound at 4x | 3.7; 8 / 2.3; 5 | 10; 24 / 13; 33 | 43; 100 / 69; 168 |
| … no bound | 6.3; 17 / 6.0; 17 | 25; 57 / 38; 118 | 84; 230 / 148; 412 |
| … the Bad component fitted, the bound at 4x | 2.1; 4 / 1.9; 4 | 5.5; 11 / 6.7; 15 | 25; 47 / 33; 76 |
| … its mean alone fitted, the bound at 4x | 2.5; 5 / 1.8; 3 | 6.1; 11 / 7.8; 18 | 29; 56 / 43; 110 |

- **At 4x the estimate reads 0.87 to 0.93 of the truth early** and 1.05 to 1.09 at vote 100; uncapped it reads 1.2 to
  1.5. The bound at 4x alone gains +0.013 to +0.033 at every preset.
- **On a corpus with no positives the bound is what keeps the line small**, and that is the price of loosening it:
  removing it multiplies what an empty corpus keeps by three to four at beta 1 and 4. The fitted Bad component pays
  for the 4x bound: together they keep fewer than today at every preset, at both clicks; with its mean alone fitted,
  about as many as today (fewer at beta 1 and 4, 2.5 against 3.5 and 1.8 against 1.7 at beta 1/4).
- **#4452's own case, 200 images with no positives**, is unchanged by every Bad-side and bound candidate: 1.0 to 2.8
  images on average, the same as today.

## The Good side

| (vote 40 / 100), class model's cut | beta 1/4 | beta 1 | beta 4 | prevalence ÷ truth (vote 40) | empty corpus, beta 4 (vote 40) |
|---|---|---|---|---|---|
| the Goods' own spread | +0.031 / +0.044 | +0.019 / +0.025 | -0.021 / -0.001 | 0.78 to 0.80 | 25 |
| fitted on the corpus | -0.120 / -0.006 | -0.110 / -0.013 | -0.104 / -0.015 | 13 to 14 | 1,330 |
| the spot check's Goods | +0.033 / +0.026 | +0.015 / +0.000 | -0.036 / -0.030 | 0.84 to 0.85 | 25 |
| the Good component true (truth, #4937) | -0.027 / +0.010 | -0.008 / +0.004 | +0.028 / +0.012 | 0.92 to 0.95 | - |

- **The Goods' own spread is narrower than the pooled one** (pooling with the Bads widens it, #4937), so it moves the
  chances the wrong way for a recall cut and loses at beta 4. Its gains at beta 1/4 and 1 come through the counted
  bound: a narrower Good component reads a looser bound (0.52 to 0.80 of the truth). The bound at 4x gives the same
  without the loss at beta 4.
- **Fitted on the corpus, the Good component is not identifiable.** With its mean free the EM slides it into the
  negatives' upper tail; the held-out Goods (a median of 10 at votes 35 to 50, against about 50 positives in the
  corpus) are too few to hold it.
- **The spot check's Goods are less selected than the rest, but not by enough**: they come from the top bands of the
  ranking (the top 32 at beta <= 1, 128 above), the next-easiest positives. 86% to 87% of class-model snapshots at
  votes 35 to 50 hold 3 or more (a median of 7). They cannot be read on Region (the grouped folds carry no vote ids).

## Also tried

Three further families were prototyped before the full pricing, on the first 20 (semi-parametric) or 30 (logistic)
sessions of each preset, every class-model snapshot (scripts in the work dir, not in the repo). Each lost at beta 1 or
4 and kept hundreds of wrong images on an empty corpus at beta 4, so none went further.

| prototype (F-beta gain over today, class model's cut, all votes) | beta 1/4 | beta 1 | beta 4 | empty corpus, beta 4 (mean kept; today 30) |
|---|---|---|---|---|
| chances from a logistic fit of the held-out labels on their scores (Firth), no prevalence shift | +0.005 | -0.051 | -0.097 | 1,024 |
| … its slope, its level set by the 2-part estimate (uncapped) | +0.017 | -0.009 | -0.023 | 271 |
| a logistic fit on the spot check's held-out picks only, its level capped the same way | +0.013 | -0.002 | -0.017 | 377 |
| negatives as a leave-one-out kernel density of the corpus weighted by each image's chance of a negative (semi-parametric EM) | -0.074 | -0.069 | -0.055 | 740 |

The logistic fits read the labels' share of Goods at a score as the corpus's, and a session's labels hold far more
Goods at a given score than the corpus does. The semi-parametric negatives are not identifiable: any share of the top
of the corpus can be called negative.

## Literal examples

Sessions at vote 50 picked by rule among those the fitted Bad component with the bound at 4x moves (best set of 10
or more; `chances_5040.py examples`): the one it hurts most, the median, the one it helps most.

![Beta 1/4, vote 50](figures/binary/examples_bad_ssl_bound_x4_b025_vote50.png)

*Left: the withheld half's top on a log count scale, positives green, negatives grey; today's cut (black), the arm's
(orange) and the best cut (purple dashes). Right: each image's chance of a positive under today's 3-part fit, the
arm's, and the truth's (isotonic).*

| preset | picked as | session | seed | vote | kept today / arm / best | F today / arm / best | chances' sum today / arm / positives |
|---|---|---|---|---|---|---|---|
| beta 1/4 | hurt most | keyboard@medium | 3 | 50 | 7 / 53 / 18 | 0.74 / 0.55 / 0.81 | 0 / 96 / 48 |
| beta 1/4 | median | snowboard@medium | 0 | 51 | 10 / 11 / 13 | 0.83 / 0.85 / 0.88 | 27 / 31 / 44 |
| beta 1/4 | helped most | mouse@medium | 3 | 50 | 1 / 21 / 23 | 0.00 / 0.74 / 0.77 | 2 / 45 / 53 |
| beta 1 | hurt most | snowboard@medium | 3 | 53 | 32 / 22 / 35 | 0.81 / 0.64 / 0.85 | 29 / 13 / 47 |
| beta 1 | median | frisbee@large | 0 | 50 | 45 / 49 / 53 | 0.90 / 0.92 / 0.92 | 46 / 53 / 53 |
| beta 1 | helped most | mouse@medium | 1 | 50 | 6 / 30 / 57 | 0.14 / 0.55 / 0.62 | 2 / 30 / 50 |
| beta 4 | hurt most | snowboard@small | 4 | 50 | 89 / 59 / 141 | 0.81 / 0.66 / 0.85 | 0 / 0 / 51 |
| beta 4 | median | kite@large | 3 | 51 | 55 / 50 / 48 | 0.99 / 0.99 / 0.99 | 45 / 38 / 44 |
| beta 4 | helped most | orange@medium | 2 | 51 | 19 / 52 / 84 | 0.43 / 0.71 / 0.79 | 6 / 31 / 43 |

- `mouse@medium` seed 3 (beta 1/4) is the failure #4937 named: today's chances never pass 0.35, even at the top of the
  ranking, and the line keeps one image. With the Bad component fitted on the corpus the chances follow the truth's
  curve almost exactly, and the line keeps 21 where the best set is 23.
- `keyboard@medium` seed 3 (beta 1/4) is the same collapse the other way: today's chances are 0 everywhere and it
  keeps 7 by the cut's floor; the fitted chances overshoot (a sum of 96 for 48 positives) and keep 53 where the best
  is 18.
- The same panels at beta 1 and 4: `figures/binary/examples_bad_ssl_bound_x4_b1_vote50.png`, `…_b4_vote50.png`.

## Region Photo

| beta 1, class model's cut (vote 25 / 50 / 100 / 150) | F-beta gain | set ÷ best | chance at the best set's edge (truth 0.37 to 0.39) | empty corpus kept (mean) |
|---|---|---|---|---|
| today | - | 1.00 / 0.94 / 0.84 / 0.96 | 0.23 / 0.18 / 0.08 / 0.14 | 18 / 16 / 20 / 17 |
| the Bad component fitted | -0.003 / -0.001 / +0.031 / +0.001 | 1.30 / 1.32 / 1.17 / 1.12 | 0.87 / 0.80 / 0.62 / 0.56 | 26 / 26 / 18 / 13 |
| … and the bound at 4x | -0.011 / -0.005 / +0.034 / +0.003 | 1.36 / 1.33 / 1.22 / 1.18 | 0.87 / 0.80 / 0.62 / 0.56 | 27 / 29 / 21 / 17 |
| its mean only | +0.019 / +0.018 / +0.031 / +0.002 | 1.06 / 1.05 / 0.97 / 1.04 | 0.49 / 0.42 / 0.23 / 0.33 | 20 / 19 / 15 / 13 |
| … with the bound at 4x | +0.014 / +0.017 / +0.040 / +0.007 | 1.12 / 1.09 / 1.00 / 1.07 | 0.49 / 0.42 / 0.23 / 0.33 | 22 / 23 / 20 / 18 |
| the bound at 4x alone | +0.004 / +0.014 / +0.027 / +0.010 | 1.04 / 0.98 / 0.94 / 1.03 | 0.23 / 0.18 / 0.08 / 0.14 | 21 / 22 / 27 / 26 |

Region's line was not short (#4937: 1.06 of the best set at vote 25), so chances that rise everywhere overshoot: the
fitted component reads 0.87 at the edge where the truth is 0.37. F-beta holds early only because the overshoot sits
in the set's tail. On 1,000-image Region corpora it loses 0.013 to 0.032 (vote 25 to 150). Fitting the Bad
component's mean alone overshoots far less (0.42 to 0.49 at the edge early) and gains at every click, +0.014 to
+0.040 with the bound at 4x; on 1,000-image Region corpora it loses up to 0.026 too (vote 50).

![Region Photo, beta 1, vote 25: the Bad component's mean fitted, the bound at 4x](figures/region/examples_bad_ssl_mean_bound_x4_b1_vote25.png)

*Region sessions at vote 25 picked by rule, as the Binary panels are. `toothbrush@large` keeps 98 today where the best
set is 50; the full fit keeps 261 (`figures/region/examples_bad_ssl_bound_x4_b1_vote25.png`), its mean alone 163.
`mouse@large` keeps 9 today where the best is 43; its mean alone keeps 36 at F 0.86 against the best cut's 0.91.*

## The closed loop

CLOSED-LOOP PENDING.

## What this leaves

CLOSED-LOOP PENDING.

## Provenance

| | |
|---|---|
| snapshots | #4937's runs (dev 73c887c27): `/expscratch/sgreenberg/state-of-the-app/2026-10-16-fscale4936-early4937-binary-b{025,1,4}`; Region `/expscratch/sgreenberg/state-of-the-app/2026-10-15-fscale4936-region-b1` |
| pricing | `chances_5040.py measure` passes 1 and 2 on branch `claude/chances-5040` (9331d6c42), per-snapshot tables in `/expscratch/sgreenberg/chances-5040/m{1,2}-{b025,b1,b4,region}/snapshots.csv.gz`; jobs of 32 to 96 CPUs, 10 to 25 minutes each |
| verify | `chances_5040.py verify`: the library's arm drawn on every snapshot, the same set as the pricer's on 5,044 of 5,044 snapshot rows per preset and 971 of 973 on Region (the 2 others have no line) |
| closed loop | `chances_5040.sh` from frozen worktree `vts-5040-run` at 4783a51eb; 45 cpu packs of 8 cells (AMD), 2026-10-17 21:59; dirs `/expscratch/sgreenberg/state-of-the-app/2026-10-17-chances5040-{ctl,bssl,bssl4}-b{025,1,4}` |
| analysis | `chances_5040.py report` / `examples`; `chances_5040_ab.py` (the closed loop) |

## Reproduce

```bash
export VTSEARCH_DATA_DIR=/expscratch/$USER/vts-cache/datadir OMP_NUM_THREADS=1
P=scripts/experiments/state_of_app/chances_5040.py
S=/expscratch/$USER/state-of-the-app/2026-10-16-fscale4936-early4937-binary
for b in b025 b1 b4; do
  python $P measure --results ${S}-$b/results --out <dir>/m1-$b --procs 96 --pass 1
  python $P measure --results ${S}-$b/results --out <dir>/m2-$b --procs 96 --pass 2
  python $P verify --results ${S}-$b/results --priced <dir>/m2-$b/snapshots.csv.gz --procs 64
done
D=docs/experiments/2026-10-17-chances-5040
python $P report --out <dir>/m1-b025/snapshots.csv.gz ... --out <dir>/m2-b4/snapshots.csv.gz --docs $D/tables --figs $D/figures
# closed loop, from a frozen worktree
L=scripts/experiments/state_of_app/chances_5040.sh
bash $L dirs
for arm in ctl bssl4 bssl; do for b in 0.25 1 4; do
  CALIB_PACK_JOBS=5 CALIB_PACK_PAR=8 CALIB_PACK_MEM=12G SBATCH_CONSTRAINT=amd bash $L cpack $arm $b
done; done
python scripts/experiments/state_of_app/chances_5040_ab.py --root /expscratch/$USER/state-of-the-app --seeds 5 \
  --baseline <text_baseline.csv> --out <dir>/ab --docs $D/tables
```
