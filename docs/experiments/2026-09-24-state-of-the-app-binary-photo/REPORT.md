# State of the App: Binary Photo — 2026-09-24

**Issue:** #4159. **Recipe:** `.claude/skills/state-of-the-app/SKILL.md`.
**Path:** SigLIP whole-image embedding, binary (Good/Bad) votes.
**Bench:** `coco_quarry`, all 49 classes at every size, which is 144 cells.
**Seeds:** 7 (1008 runs). The companion report is *State of the App: Region
Photo*.

A **review**, not an experiment: the app as it ships, the way a user meets it.
The user types a query, sees the text sort, then votes for up to 150 clicks
while Autopilot picks what to show. Every curve runs from the **text-only score
at click 0**, through the clicks, to the **full-label ceiling**: the same head
trained on every label in the training half. *Cost* is the harness's weighted
FNR/FPR at the shipped threshold, where lower is better. *F1* is at the same
threshold.

## Headline

| mean over 1008 runs | text only | after 150 clicks | full labels |
|---|---:|---:|---:|
| **cost** | 0.49 | **0.33** | 0.15 |
| **F1** | 0.02 | **0.05** | — |

![Mean cost over clicks](figures/cost_over_clicks.png)

1. **Clicking roughly halves the distance to the ceiling, and no more.** Cost
   falls from 0.49 to 0.33, against a ceiling of 0.15: 150 clicks close 47% of
   the gap between typing the query and having every label.
2. **The first detector is worse than the text sort.** Mean cost jumps from 0.49
   to **0.68 at click 4**, when the first Good and Bad train a head, and does not
   beat the text sort again until **click 12**. A user who stops after a few
   clicks has a worse result than if they had not clicked at all.
3. **The clicks find few positives.** The median run finds **4** positives in
   150 clicks, of about 50 in its training half. 30 runs never find one, so no
   detector is ever trained, all of them on small-band cells: `chair@small` in
   all 7 seeds, `knife@small` in 6, `bench@small`, `laptop@small` and
   `spoon@small` in 3, and five more cells once or twice. The easy classes find
   far more: `tennis racket` 31, `baseball bat` 29, `skis` 26.
4. **The shipped threshold buys recall with precision.** Median recall at 150
   clicks is **0.96**; median precision is **0.02**. At about 1% prevalence, a
   user sees roughly 50 false positives for every true one. Cost rewards that
   trade and F1 does not, which is why cost looks moderate while F1 is 0.05.

![Mean F1 over clicks](figures/f1_over_clicks.png)

## Where it does well and where it does poorly

| band (mean) | text only | after 150 clicks | full labels | F1 | positives found |
|---|---:|---:|---:|---:|---:|
| large | 0.41 | **0.19** | 0.057 | 0.062 | 8.0 |
| medium | 0.47 | **0.31** | 0.17 | 0.046 | 6.8 |
| small | 0.58 | **0.50** | 0.24 | 0.033 | 5.5 |

**Well:** sports equipment and other objects with a distinctive shape.

| class | text only | after 150 | full labels | positives found |
|---|---:|---:|---:|---:|
| baseball bat | 0.46 | **0.082** | 0.018 | 29 |
| skis | 0.43 | **0.089** | 0.015 | 26 |
| snowboard | 0.34 | **0.10** | 0.014 | 14 |
| surfboard | 0.39 | **0.11** | 0.016 | 7.8 |
| tennis racket | 0.43 | **0.10** | 0.013 | 31 |

**Poorly:** furniture, tableware and people, all of them things that fill
scenes rather than stand out in them.

| class | text only | after 150 | full labels | positives found |
|---|---:|---:|---:|---:|
| chair | 0.74 | **0.78** | 0.36 | 2.8 |
| bench | 0.75 | **0.66** | 0.31 | 3.6 |
| vase or potted plant | 0.54 | **0.64** | 0.31 | 3.2 |
| knife | 0.69 | **0.53** | 0.14 | 2.8 |
| bottle | 0.70 | **0.55** | 0.29 | 3.4 |
| person | 0.74 | **0.53** | 0.17 | 3.2 |

- **Clicking made things worse than typing** in 122 of 1008 runs (12%). Averaged
  over seeds, four classes end above their text score: `vase or potted plant`
  (+0.09), `parking meter` (+0.05), `stop sign` (+0.04) and `chair` (+0.04).
- **Headroom** (final − ceiling) is largest where positives are hardest to find:
  `chair` 0.42, `knife` 0.40, `person` 0.35, `bench` 0.35. The head could learn
  these classes with the labels; the loop never collects them. That makes the
  loop, not the class, the limit.

Every cell, text only → final → full labels:

![Per class and size](figures/per_cell.png)

## Images: what a click does, and when

Each scored step's change in the held-out test cost is split equally among the
clicks since the previous scored step. The first step is measured from the
text-only score and split among the opening clicks. The credits sum exactly to
each run's net change.

**When a positive is clicked matters more than which one it is.**

| mean cost removed per click | clicks 1–30 | clicks 31–90 | clicks 91–150 |
|---|---:|---:|---:|
| a **positive** (Good) | **−0.027** (hurts) | **+0.015** (helps) | +0.001 |
| a negative (Bad) | +0.005 | +0.001 | +0.001 |

An early positive hurts on average because it trains the first detector, and
that detector is worse than the text sort (finding 2). The same positive helps
once there are enough labels to fit a reasonable head.

**Some images hurt in their own right; few help in their own right.** Early
positives hurt and some classes are harder than others, so an image's raw mean
credit mostly records *when* and *where* it was clicked. The test below nets
both out: each click's credit minus the mean of every click with the same cell,
label and phase. Of 21,880 clicked images, 4,742 were clicked 10 or more times:

| images past \|z\| > 3 | observed | shuffled images (5 draws) |
|---|---:|---:|
| **harmful** | **39** | 9–12 |
| helpful | 11 | 3–8 |

About 28 harmful images are beyond chance; the helpful excess is too small to
name individual images. The harmful ones share a pattern: each hurts **one
class, as a negative, in every size band** of that class. Most are correct
negatives from an unrelated scene, such as a bear for `kite`, cows for
`airplane`, an elephant for `stop sign`, a horse for `tie` and zebras for
`tennis racket`. One is not: **490264**, a negative for `fork`, shows a fork,
which makes it a candidate label error in the bench.

See [images.md](images.md) for the 12 strongest in each direction, with
thumbnails.

## Flagged regimes

- **Never found a positive:** the 30 runs above, all small-band cells.
- **Positive exhaustion (#4121)** did not arise. Runs find too few positives to
  run out.

## What to A/B next (proposed)

1. **Don't let the first detector replace a better text sort.** Blend the text
   score with the early detector until the detector beats it (#3944). This
   targets finding 2 and the "early positives hurt" row.
2. **Find positives faster:** the acquisition offset and the text-seed dwell
   (#3261, #2910). Starvation is the first-order problem on small-band cells.
3. **The operating point:** a cut that trades some recall for precision, judged
   on F1 as well as cost (#4118).

## Provenance

- **Run directory:** `/expscratch/sgreenberg/state-of-the-app/2026-09-23/`.
  Analysis is in `analysis-binary/`, including `viewer.html`, the interactive
  curves for every cell.
- **Shipped defaults** throughout: text opening, fused threshold, linear SVM
  head, 150 clicks.
- **CPU count:** seeds 0–2 and part of 3–6 used 2 CPUs; the rest used 1. Thread count changes low-order
  floating-point digits, and so can change individual picks, but a check on one
  cell gave identical costs at clicks 25, 50 and 150.

```bash
cd scripts/experiments/state_of_app
SOTA_PATH=binary srun -p cpu --mem=48G -c 4 bash analyze.sh <run dir>
```
