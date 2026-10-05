# State of the App: Binary Photo — 2026-10-05

**Issue:** #4510. **Recipe:** `.claude/skills/state-of-the-app/SKILL.md`.
**App:** `dev` at 19ed74aaa. The line comes from the labels (#4452), and its spread floor follows the corpus
(#4492). Autopilot runs the spot check itself when the labels separate weakly (#4496, PR #4503). The presets are
beta 1/4, 1 and 4.
**Path:** SigLIP whole-image embedding, binary (Good/Bad) votes, Autopilot's shipped opening.
**Bench:** `coco_better`, all 49 classes at every size they have: 144 cells.
**Seeds:** 4. **Sessions:** one set per preset (`SOTA_BETA=0.25|1|4`), 576 runs each, 1,728 in all, plus one
full-label ceiling pass shared by the three.
**Runs:** `/expscratch/sgreenberg/state-of-the-app/2026-10-05-b025`, `-b1` and `-b4`, with the ceiling in
`-b1/ceiling`. They ran 07:53 to 10:23, at about 7 to 12 minutes a run on 1 CPU, from a frozen worktree at
19ed74aaa. Five seeds were launched in seed order; the fifth was cut so the review could finish by noon.
**Analysis:** `analyze.sh` per run on seeds 0 to 3, then `perp.py --kind balance` (`perbeta_summary.md`).

A **review**, not an experiment: the app as it ships, the way a user meets it.
- **The session.** The user types a query and sees the text sort, then votes for 150 clicks while Autopilot picks
  what to show. When the labels separate weakly, Autopilot stops to run a spot check. At the end the user runs the
  check once more.
- **Presets.** Each preset is read off its own sessions.
- **The objective** (#4427) is the F-beta, at the preset's beta, of the withheld test half above the threshold the
  app holds. The test half is the 10,900 images Find would search, and the threshold is the labels line Find
  draws there.
- **The share of the best cut** separates the line from the ranking. **AP** measures the ranking alone.

## Headline

**The objective, each preset off its own sessions** (563 trained runs each):

| preset | 25 clicks | 50 clicks | 150 clicks, unchecked | **after the check** | returned, median (unchecked / after) | returned > 200 (unchecked / after) |
|---:|---:|---:|---:|---:|---:|---:|
| 1/4 | 0.44 | 0.51 | 0.61 | **0.63** | 27 / 20 | 6% / 0% |
| 1 | 0.34 | 0.42 | 0.50 | **0.52** | 45 / 44 | 12% / 5% |
| 4 | 0.43 | 0.48 | 0.56 | **0.60** | 69 / 75 | 30% / 28% |

![The objective over clicks, per preset](figures/objective_at_own_beta.png)

1. **The presets return what they promise.** After the check:
   - the 1/4 preset returns a median of 20 images at precision 0.72 and recall 0.38;
   - the balanced preset returns 44 at 0.59 and 0.52;
   - the recall preset returns 75 at 0.43 and 0.67.
2. **The early session now returns a real set.** At clicks 5, 10 and 25 the labels line returns a median of
   15/25/31 images at 1/4, 29/40/54 at 1 and 50/76/133 at 4. Its objective there is 0.41/0.46/0.44,
   0.28/0.34/0.34 and 0.35/0.42/0.43. Under each sort's own line in the app, the detector beats the text sort's
   blind cut by click 2, 4 and 5 at the three presets; that cut returns about 4,500 images, F1 0.02.
3. **The check still pays, at the end and mid-session.** The end-of-run check adds +0.022 ± 0.004,
   +0.025 ± 0.002 and +0.033 ± 0.004, for about 21 votes (30 at beta 4). It cuts the share of runs returning more
   than 200 images from 6%, 12% and 30% to 0%, 5% and 28%. Mid-session, Autopilot ran it in 42 to 44% of sessions
   ([below](#the-spot-check)).
4. **The ceiling's own line still goes too deep (#4490).** With every label known, Find's line scores
   0.47/0.44/0.61 at the three presets, against the 150-click session's 0.60/0.49/0.55. It returns a mean of
   296 images at beta 1, at precision 0.37. Only beta 4, which rewards depth, comes out ahead.

![The returned set's share of the best cut, per preset; solid: each sort's own line, dashed: top-K on both](figures/returned_at_own_beta.png)

**The returned set, one rule per row.** "App line" is what each sort's own line returns: the text sort's blind
GMM cut, the detector's labels line, and for the full-label ceiling Find's labels line from every training label
(#4486). "Top-K" is set-constant at the old cap on every sort: 32 at beta ≤ 1 and 128 at beta 4. Compare within a
rule, never across. The values are F-beta at the preset; detector columns are the last click before the check.

| preset | rule | text sort | 25 clicks | 50 clicks | 150 clicks | full labels |
|---:|---|---:|---:|---:|---:|---:|
| 1/4 | app line | 0.01 | 0.38 | 0.46 | **0.60** | 0.47 |
| 1/4 | top 32 | 0.47 | 0.42 | 0.48 | **0.58** | 0.61 |
| 1 | app line | 0.02 | 0.30 | 0.38 | **0.49** | 0.44 |
| 1 | top 32 | 0.38 | 0.34 | 0.39 | **0.47** | 0.50 |
| 4 | app line | 0.15 | 0.39 | 0.45 | **0.55** | 0.61 |
| 4 | top 128 | 0.49 | 0.41 | 0.47 | **0.57** | 0.62 |

Under top-K, where only the rankings are compared, 150 clicks of detector beat the text sort by 0.08 to 0.11 at
every preset. Under the app's own lines the gap is far larger, because the text sort's line returns most of the
corpus.

**The ranking** barely depends on the preset (the beta-1 sessions):

| | text only | 25 clicks | 50 clicks | 150 clicks | full labels |
|---|---:|---:|---:|---:|---:|
| **AP** | 0.42 | 0.35 | 0.43 | **0.52** | 0.55 |
| **Goods found** | — | 11 | 15 | **21** | — |

![Mean AP over clicks](figures/ap_over_clicks.png)

150 clicks close about three quarters of the gap from the typed query to full labels. AP still dips after the
opening (0.42 → 0.35 at click 25, #4384) and is back at the text level by about click 50.

**Against the last review (#4474, 2026-10-04), one note.** The objective at click 25 rose from 0.42/0.29/0.35 to
0.44/0.34/0.43, the work of #4492's spread floor. After the check it is unchanged within 0.01. The final AP rose
from 0.51 to 0.52, and Goods from 20 to 21.

## The spot check

The check walks bands of the unvoted ranking, about 5 picks a band. It never moves the line; its picks are votes
the next retrain learns from.

- **At the end of the session**, every trained run checks once. Its precision range is 0.64 to 0.71 wide and held
  the truth in every run (563 of 563 per preset), as in the last review (#4483 is about the width).
- **Mid-session, Autopilot runs it when the labels separate weakly** (#4496: the labels line's d' below 1.5, from
  10 votes, again 25 votes after the last check ended). It does so only once Autopilot has left its text-sort
  opening, where no detector is trained (#4508 is about the opening).

| preset | sessions prompted | first prompt, median click (quartiles) | checks per prompted session | clicks spent checking, median | Goods among those picks |
|---:|---:|---|---|---:|---:|
| 1/4 | 44% | 66 (44, 84) | 1: 114, 2: 121, 3: 21 | 40 (23%) | 15% |
| 1 | 44% | 67 (44, 85) | 1: 118, 2: 119, 3: 18 | 40 (23%) | 15% |
| 4 | 42% | 65 (43, 81) | 1: 155, 2: 85 | 30 (27%) | 12% |

Between two thirds and four fifths of first prompts come in Autopilot's `hard` phase (67%, 76% and 82%), the rest in `new` or `done`.
Priced against the same sessions without it (#4496, round 2), this prompt adds +0.012 to +0.025 at click 150 and
nothing before about click 50.

## Where the app does well and where it does poorly

**By size band** (beta 1; the objective after the check, AP of the ranking):

| band (cells) | objective | returned, median | text AP | AP at 150 | full-label AP | Goods found |
|---|---:|---:|---:|---:|---:|---:|
| large (49) | 0.77 | 46 | 0.70 | 0.80 | 0.82 | 30 |
| medium (49) | 0.49 | 40 | 0.38 | 0.50 | 0.51 | 22 |
| small (46) | 0.28 | 37 | 0.16 | 0.25 | 0.31 | 11 |

![Every class x band](figures/per_cell.png)

**Well:** sports equipment and other things a scene is *about* (beta 1, trained runs, objective after the check):

| class | objective | text AP | AP at 150 | full-label AP |
|---|---:|---:|---:|---:|
| tennis racket | 0.97 | 0.96 | 0.99 | 0.99 |
| skateboard | 0.91 | 0.85 | 0.93 | 0.96 |
| kite | 0.90 | 0.82 | 0.95 | 0.97 |
| airplane | 0.89 | 0.83 | 0.92 | 0.91 |
| baseball bat | 0.88 | 0.91 | 0.93 | 0.94 |
| surfboard | 0.85 | 0.87 | 0.92 | 0.93 |

**Poorly:** furniture, tableware and cutlery, things that fill a scene rather than define it:

| class | objective | returned, median | text AP | AP at 150 | full-label AP | why (below) |
|---|---:|---:|---:|---:|---:|---|
| chair | 0.05 | 129 | 0.09 | 0.06 | 0.15 | clicks made it worse |
| bowl | 0.17 | 90 | 0.05 | 0.14 | 0.24 | loop |
| dining table | 0.18 | 127 | 0.11 | 0.15 | 0.19 | embedding |
| knife | 0.18 | 33 | 0.08 | 0.15 | 0.31 | loop |
| spoon | 0.20 | 27 | 0.07 | 0.19 | 0.28 | mixed |
| enclosed road vehicle | 0.24 | 79 | 0.17 | 0.24 | 0.26 | embedding |
| bench | 0.26 | 34 | 0.23 | 0.25 | 0.25 | embedding |
| bottle | 0.28 | 45 | 0.17 | 0.26 | 0.32 | mixed |
| fork | 0.29 | 46 | 0.15 | 0.25 | 0.33 | mixed |

**Why.** The ceiling separates two cases. When it is also low, the embedding can't separate the class. When it is
high but the final AP is not, the loop failed to collect the labels. The confusers are the images Autopilot most
often showed as Bad (`why.py`, `why/why.csv`), ranked by lift over the corpus. They are the same confusions as in
the last two reviews: the data has not changed, and neither has the embedding.

- **chair: the clicks make it worse (AP 0.09 → 0.06), against a ceiling of 0.15.** Bad clicks most often hold a
  **couch (19%, 10× the corpus rate) or a bench (15%, 10×)**, then a toilet (8%). Whole-image SigLIP sees "a seat"
  and cannot tell which kind. chair@small never trains a detector in all 4 of its seeds. Where it does train, the
  line returns a median of 129 images at an objective of 0.05.
  ![chair confusers](why/confusers_chair.jpg)
- **bowl: the loop.** The ceiling is 0.24 against a final 0.14. A quarter of its Bad clicks hold a **toilet (24%,
  11×)**, and broccoli (7%, 14×): round white things and food.
  ![bowl confusers](why/confusers_bowl.jpg)
- **knife: the loop.** The ceiling is 0.31 against a final 0.15. Bad clicks hold **scissors (11%, 46×)**, then
  pizza: the head learns "blade-shaped metal" before it learns "knife".
  ![knife confusers](why/confusers_knife.jpg)
- **spoon: mixed.** Its confusers are toothbrushes (32×) and knives (29×): long thin things held in a hand.
- **dining table: the embedding.** The ceiling is 0.19, close to the final 0.15. Its Bad clicks hold couches
  (13%, 7×) and chairs (11%, 9×): "a room with furniture", whichever kind.
- **enclosed road vehicle: the embedding.** 22% of its Bad clicks hold none of COCO's 80 categories. Buses and
  trains make up a quarter more, the vehicles this class's boundary excludes.
- **bench: the embedding.** 0.23 → 0.25 against a ceiling of 0.25. Its Bads are other street furniture: parking
  meters (11×) and fire hydrants (4×).
- **bottle and fork: mixed.** Bottle's confusers are toothbrushes (44×), cups and parking meters (15× each).
  Fork's are knives (29×) and broccoli (13×).

## Headroom (full labels minus 150 clicks)

The mean is 0.03 (0.55 − 0.52). It concentrates in a few classes: knife (0.15), person (0.12), keyboard (0.10),
bowl (0.10), chair (0.09), spoon (0.09) and laptop (0.08). Airplane, skis, snowboard and bicycle sit at or above
their ceiling: a head trained on 150 chosen labels can beat one trained on every label when the extra labels are
noisy.

## What the clicks bought (150 clicks minus text only)

The mean is +0.10 AP (0.42 → 0.52). Person gains the most, +0.37 (0.10 → 0.47): the typed query ranks it badly and
clicks rescue it. Skis gains +0.29 and sink +0.22. Chair loses 0.02. Bench, bag or luggage, scissors and parking
meter gain at most 0.03: for these, typing the query is as good as clicking.

![Mean Goods found over clicks](figures/goods_over_clicks.png)

Harvest is front-loaded: 11 of the 21 Goods arrive in the first 25 clicks.

## Images

1,542 images were clicked 10 or more times (beta-1 sessions). **3 are helpful and 17 harmful** past |z| > 3.
Shuffling images within each (cell, label, phase) bucket gives at most 1 helpful and 3 harmful, so the harmful
side is real (`images.md` has every one with its thumbnail). The harmful ones are Bad clicks that cost a specific
detector AP every time they are clicked:

- **84474**, seaplanes moored at a harbour dock: Bad for **boat** 9 times, −0.10 AP a click. Float planes on the
  water read as boats.
- **490264**, a white plastic utensil lifting macaroni over a sandwich: Bad for **fork** 9 times, −0.077 AP a
  click. The utensil reads as a fork. Whether COCO's label is right needs the full-size image, not the thumbnail.
- **422517**, a black-and-white photo of a man on a park bench: Bad for **parking meter** 10 times, −0.060 AP a
  click.
- **186207**, a bed with red embroidered pillows and a crest: Bad for **tennis racket** 8 times, −0.018 AP a click.

The last two were harmful in the last review too. They are what #4404 is about: early Bad votes that share a
cue with the class.

## Known regimes to flag, not to fix

- **Detectors that never train:** 13 of 576 runs per preset (2.3%), all at the small band: chair (4), bowl (2),
  knife (2), and one each of book, bottle, enclosed road vehicle, laptop and person. 150 clicks never surface a
  positive.
- **Weak sessions still over-return at beta 4:** 28% of runs return more than 200 images after the check (#4466,
  closed: no corpus-side rule was worth shipping). At beta 1/4 and 1 the check brings it to 0% and 5%.
- **No run exhausted its positives** before 150 clicks (#4121's regime did not occur).

## What to A/B next

- **#4508: the weak-separation check during Autopilot's opening.** Checking anywhere earned +0.02 to +0.05 at
  clicks 20 to 50 (#4496, round 1). The shipped check, past the opening, earns nothing there. It needs a
  background retrain during the opening.
- **#4490: with every label known, Find's labels line goes too deep.** The ceiling's line returns a mean of 296
  images at precision 0.37 at beta 1, and loses 0.13 at 1/4 and 0.05 at 1 to the 150-click session's line on a
  better ranking.
- **#4384: the ranking's early dip** (AP 0.42 → 0.35 by click 25, back by about 50). The line no longer collapses
  early; the ranking under it still does.
- **#4483: a tighter check range.** The current one is honest but 0.64 to 0.71 wide.
- **#4404: correct early Bad votes that share a cue with the class.** The same harmful images keep showing up.

## Files

- `perbeta_summary.md`: every per-preset table above, from `perp.py --kind balance`.
- `figures/`: the objective and the returned set per preset, AP and Goods over clicks, every cell.
- `images.md`, `images/`: the helpful and harmful images, with thumbnails.
- `why/`: the confuser sheets and `why.csv`.
- Runs and full analyses: `/expscratch/sgreenberg/state-of-the-app/2026-10-05-b025|b1|b4/analysis-binary/`;
  the prompt counts are `/expscratch/sgreenberg/sota-1005/prompts.py`.
