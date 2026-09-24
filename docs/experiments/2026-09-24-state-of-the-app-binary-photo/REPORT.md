# State of the App: Binary Photo — 2026-09-24

**Issue:** #4159. **Recipe:** `.claude/skills/state-of-the-app/SKILL.md`.
**Path:** SigLIP whole-image embedding, binary (Good/Bad) votes.
**Bench:** `coco_quarry`, all 49 classes at every size, which is 144 cells.
**Seeds:** 3 (432 runs). Four more seeds are running, and the numbers here will
be refreshed at **7 seeds**. The companion report is *State of the App: Region
Photo*, which is still running.

A **review**, not an experiment: the app as it ships, the way a user meets it.
The user types a query, sees the text sort, then votes for up to 150 clicks
while Autopilot picks what to show. Every curve runs from the **text-only score
at click 0**, through the clicks, to the **full-label ceiling**: the same head
trained on every label in the training half. *Cost* is the harness's weighted
FNR/FPR at the shipped threshold, where lower is better. *F1* is at the same
threshold.

## Headline

| mean over 432 runs | text only | after 150 clicks | full labels |
|---|---:|---:|---:|
| **cost** | 0.49 | **0.33** | 0.15 |
| **F1** | 0.02 | **0.05** | — |

![Mean cost over clicks](figures/cost_over_clicks.png)

1. **Clicking roughly halves the distance to the ceiling, and no more.** Cost
   falls from 0.49 to 0.33, against a ceiling of 0.15: 150 clicks close 48% of
   the gap between typing the query and having every label.
2. **The first detector is worse than the text sort.** Mean cost jumps from 0.49
   to **0.68 at click 4**, when the first Good and Bad train a head, and does not
   beat the text sort again until **click 12**. A user who stops after a few
   clicks has a worse result than if they had not clicked at all.
3. **The clicks find few positives.** The median run finds **4** positives in
   150 clicks, of about 50 in its training half. 12 runs never find one, so no
   detector is ever trained: `chair@small` and `knife@small` in all 3 seeds,
   `bowl@small` and `person@small` in 2, and `laptop@small` and `spoon@small` in
   1. The easy classes find far more: `tennis racket` 30, `baseball bat` 27,
   `skis` 24.
4. **The shipped threshold buys recall with precision.** Median recall at 150
   clicks is **0.96**; median precision is **0.02**. At about 1% prevalence, a
   user sees roughly 50 false positives for every true one. Cost rewards that
   trade and F1 does not, which is why cost looks moderate while F1 is 0.05.

![Mean F1 over clicks](figures/f1_over_clicks.png)

## Where it does well and where it does poorly

| band (mean) | text only | after 150 clicks | full labels | F1 | positives found |
|---|---:|---:|---:|---:|---:|
| large | 0.41 | **0.19** | 0.06 | 0.06 | 8.3 |
| medium | 0.47 | **0.31** | 0.17 | 0.05 | 6.4 |
| small | 0.59 | **0.49** | 0.24 | 0.03 | 5.5 |

**Well:** sports equipment and other objects with a distinctive shape.

| class | text only | after 150 | full labels | positives found |
|---|---:|---:|---:|---:|
| baseball bat | 0.46 | **0.08** | 0.02 | 27 |
| skis | 0.43 | **0.09** | 0.01 | 24 |
| snowboard | 0.34 | **0.10** | 0.01 | 14 |
| surfboard | 0.39 | **0.11** | 0.02 | 8.6 |
| tennis racket | 0.43 | **0.11** | 0.01 | 30 |

**Poorly:** furniture, tableware and people, all of them things that fill
scenes rather than stand out in them.

| class | text only | after 150 | full labels | positives found |
|---|---:|---:|---:|---:|
| chair | 0.74 | **0.78** | 0.36 | 2.9 |
| bench | 0.74 | **0.68** | 0.29 | 2.8 |
| vase or potted plant | 0.55 | **0.64** | 0.33 | 3.6 |
| knife | 0.69 | **0.55** | 0.15 | 2.4 |
| bottle | 0.69 | **0.54** | 0.30 | 4.2 |
| person | 0.74 | **0.51** | 0.18 | 2.9 |

- **Clicking made things worse than typing** in 46 of 432 runs. Averaged over
  seeds, three classes end above their text score: `vase or potted plant`
  (+0.09), `stop sign` (+0.05) and `chair` (+0.04).
- **Headroom** (final − ceiling) is largest where positives are hardest to find:
  `chair` 0.42, `knife` 0.40, `bench` 0.39, `person` 0.33. The head could learn
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
| a **positive** (Good) | **−0.027** (hurts) | **+0.017** (helps) | +0.002 |
| a negative (Bad) | +0.005 | +0.001 | +0.001 |

An early positive hurts on average because it trains the first detector, and
that detector is worse than the text sort (finding 2). The same positive helps
once there are enough labels to fit a reasonable head.

**Individual images are not yet distinguishable from noise.** Of 19,361 clicked
images, 383 were clicked 10 or more times, and only **4** of those show an
effect beyond |z| > 3, close to the number chance alone would give. The
thumbnails below are therefore **examples of the format, not findings**. At 7
seeds more images reach the repeat counts where a per-image claim can hold.

See [images.md](images.md) for the top and bottom 12 among images clicked at
least 5 times.

## Flagged regimes

- **Never found a positive:** the 12 runs above, all small-band cells.
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
- **CPU count:** these runs used 2 CPUs. Thread count changes low-order
  floating-point digits, and so can change individual picks, but a check on one
  cell gave identical costs at clicks 25, 50 and 150.

```bash
cd scripts/experiments/state_of_app
SOTA_PATH=binary srun -p cpu --mem=48G -c 4 bash analyze.sh <run dir>
```
