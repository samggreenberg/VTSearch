# SAM3's zero-shot text prompt on Region Photo: it out-ranks SigLIP's typed sort and 150 Region votes (issue #5213)

**Status (2026-10-25): zero-shot only. SAM3 (`facebook/sam3`) is given each class's text and nothing else, and
scores every image of #4711's grid (COCO Better as #5191 rebuilt it, 24,507 images, 144 class x band cells). Each
cell is scored on its own pool and withheld half, exactly as SigLIP's typed sort is in `text_baseline.csv`, and set
against #4711's Region arms. Nothing ships from this PR. The follow-ups at the end are filed as issues.**

**SAM3 with no votes ranks better than Region voting does after 150 votes.**

1. **Against SigLIP's typed sort, the opening every cell starts from:** AP **0.72 vs 0.42** (+0.30 ± 0.03, ahead
   in 47 of 49 classes). The best cut of the ranking (oracle F1) is 0.73 vs 0.45.
2. **Against Region after 150 votes (seed 0, paired by cell):** SAM3's noun prompt is AP **+0.051 ± 0.017** over
   `max_patch` (0.72 vs 0.67) and +0.044 over the HAC tree, ahead in 73% of classes. Region never catches it on
   small or medium objects. On large objects it does: the tree passes SAM3 at vote 40, `max_patch` at vote 109.
3. **What the user would see:** SAM3's own operating point (keep an image if a detection scores over 0.5) gives
   **F1 0.57** (precision 0.47, recall 0.86), against **0.37** for SigLIP's typed set and **0.23** for Region's
   line at vote 150. At 0.5, SAM3 sits 0.15 below its own oracle; Region's line sits 0.44 below its oracle.
4. **The gain is biggest where SigLIP is weakest.** Small objects: AP 0.17 -> 0.52 (Region at 150: 0.39). Clutter
   classes SigLIP cannot rank at all (chair 0.06, bowl 0.05, knife 0.07, person 0.10) go to 0.42-0.67.
5. **SAM3 wants short noun phrases.** The typed queries were written for SigLIP. "a person wearing a necktie"
   scores AP 0.48 and "necktie" 0.91. "a car or truck" scores 0.54, and the better of "car" and "truck" 0.71. "a
   television screen" scores 0.45 and "television" 0.68. Over all classes the two prompt sets tie (0.72), because
   one noun loses: "skis" 0.50 vs "a pair of skis" 0.72.
6. **It is expensive, and the app's text line does not fit it.** Scoring one query over the pool costs a full
   backbone pass per image unless ~65 GB of features are cached (see Cost). The app's text-sort line, drawn on
   SAM3's scores, cuts at ~0.002 and keeps nearly everything (F1 0.04). SAM3 needs its own line.

**Training overlap: checked (#5240), and it does not explain the lead.** The pool is COCO train2017. SAM3's training
sets (paper Table 26) name no COCO, but SA-Co/EXT holds Visual Genome, whose images are COCO images: 41.5% of this
pool (10,161 of 24,507). Graded apart, SAM3 does no better on the images it trained on: AP 0.72 on them vs 0.73 on
the rest (noun prompts), and its lead over SigLIP is +0.30 on the images it never had labels for. See
[Training overlap](#training-overlap-5240).

**SAM3 as the opening: measured (#5241), as an opt-in path beside SigLIP's (SigLIP stays the default: it is fast).**
A session that opens on SAM3's set at 0.5 and hands Test to the detector at the 20th Good shows F1 +0.12 (± 0.015)
over votes 1-150 on Binary and +0.07 (± 0.016) on Region, against today. Handing over at the label quota, as SigLIP's
opening does, keeps only +0.02-0.03. See [SAM3 as the opening](#sam3-as-the-opening-5241). #5311 confirmed it at
beta 1. #5565 makes it right at every preset: SAM3's cut moves with the preset (0.85 / 0.75 / 0.4 at beta 1/4 / 1 /
4), with a hand-over at the 20th Good, and no hand-over at 4. On a fresh seed that is +0.11 / +0.13 / +0.23 on
Binary and +0.06 / +0.09 / +0.14 on Region. See [A SAM3 cut per preset](#a-sam3-cut-per-preset-5565).

| seed 0, mean over cells (± SE over classes) | all (144) | small (46) | medium (49) | large (49) |
|---|---|---|---|---|
| **AP**, SigLIP typed sort (vote 0) | 0.42 (± 0.04) | 0.17 | 0.38 | 0.70 |
| **AP**, SAM3 typed prompt (vote 0) | **0.72** (± 0.03) | **0.52** | 0.78 | 0.84 |
| **AP**, SAM3 noun prompt (vote 0) | **0.72** (± 0.03) | 0.48 | **0.79** | 0.88 |
| AP, Region `max_patch` at vote 150 | 0.67 (± 0.03) | 0.39 | 0.72 | 0.90 |
| AP, Region `max_patch_hac` at vote 150 | 0.68 (± 0.03) | 0.38 | 0.72 | **0.91** |
| Oracle F1, SigLIP / SAM3 typed / SAM3 noun | 0.45 / 0.72 / 0.73 | 0.21 / 0.58 / 0.56 | 0.43 / 0.76 / 0.78 | 0.70 / 0.81 / 0.85 |
| Oracle F1, Region `max_patch` / `hac` at vote 150 | 0.67 / 0.68 | 0.43 / 0.43 | 0.71 / 0.72 | 0.87 / 0.89 |
| **F1 shown**, SigLIP typed set (app's line) | 0.37 | 0.16 | 0.35 | 0.58 |
| **F1 shown**, SAM3 typed / noun, score > 0.5 | **0.57** / 0.52 | 0.51 / 0.47 | 0.59 / 0.54 | 0.60 / 0.55 |
| F1 shown, Region `max_patch` / `hac` at vote 150 | 0.23 / 0.25 | | | |
| Vote at which Region's AP reaches SAM3 noun's (`mp` / `hac`) | never / never | never | never | 109 / 40 |

*Zero-shot rows are seed 0, to pair with #4711's Region arms (seed 0 only). Over seeds 0-2 the zero-shot means move
by under 0.01: the scores do not depend on the seed, only which images land in the withheld half does. Paired gains
are in `tables/sam3_5213_paired.csv`, every cell in `tables/sam3_5213_per_cell.csv`, per class in
`tables/sam3_5213_per_class_AP.csv`. Region's F1 by band is in #4711's report.*

![AP and oracle F1 by object size: zero-shot sorts at vote 0, Region at vote 150, ±2 SE](figures/sam3_5213_bands.webp)

![Region's AP, oracle F1 and shown F1 over the votes, against each zero-shot sort as a flat line](figures/sam3_5213_over_votes.webp)

## The question

#5213 asked for #4711's Region-voting comparison with SAM3 in it, zero-shot text queries first, plus a cost
estimate and a look at how SAM3 could enter the vote loop. The owner scoped the first pass on 2026-10-25: zero-shot
SAM3 on all 144 cells, measured cost, and a memo of follow-ups filed as issues.

## Setup

- **Model:** `facebook/sam3` through transformers 5.19 (`Sam3Model`), image input 1008 x 1008. An image's score
  for a prompt is its best detection: the max over SAM3's 200 object queries of sigmoid(query logit) x
  sigmoid(presence logit). This is the product SAM3's own `post_process_object_detection` thresholds.
- **Prompts:**
  - `typed` is the text every cell opened on (`seed_query_text`, the same string SigLIP embedded).
  - `noun` is a short SAM3-style phrase ("necktie", "television", "computer mouse"). A merged class gets one phrase
    per member, and the image takes the max: "car"/"truck", "cup"/"glass", "bag"/"suitcase", "vase"/"potted plant".
  - `both` takes the max over the two sets. It matches `noun` to within 0.002 everywhere, so it is not discussed.
- **Pool and scoring:** `text_baseline.py --sam3` swaps SigLIP's similarities for SAM3's scores and changes
  nothing else: same `evaluable_pool` (a wrong-band image is excluded, so a class-level score is the whole task),
  same seeded split, same columns. The SAM3 rows' test sets and positive counts were checked identical to
  SigLIP's, cell by cell (432 of 432).
- **Region:** #4711's committed tables (`max_patch` and `max_patch_hac`, seed 0, 150 votes on today's stack).
- **Runs:** 12 shards on V100 at fp16 (`state_of_app/sam3_5213.sh score 12`), ~63 min each, no failures.
  fp16 vs fp32 on 200 images x 102 prompts: Spearman >= 0.999 for every prompt, and the 0.5 cut agrees on 100% of
  image-prompt pairs. The mask head is skipped (`--masks` turns it back on); with it on, the scores are
  bit-identical.

## Reading the result

**Ranking.** SAM3 is a detector trained for "find every instance of this noun phrase". Its score answers "is there
one in this image", which is the Region question. SigLIP's typed sort compares a caption-like text with the whole
image, and is weakest exactly where the object is a small part of a busy scene. The per-class table shows it.
SigLIP's worst classes are tableware, furniture and people in clutter (bowl 0.05, chair 0.06, knife 0.07, spoon
0.08, person 0.10). Those are SAM3's largest gains, +0.36 to +0.54. Where SigLIP is already good (tennis racket
0.96, surfboard 0.88), SAM3 ties.

**Against voting.** Region's 150 votes buy AP 0.42 -> 0.67. SAM3 starts at 0.72. The votes' advantage is a
user-specific concept and the band. SAM3 knows neither, and on large objects the votes still win, from vote 40
(the tree) or 109 (`max_patch`). On small objects the vote loop never reaches SAM3: at vote 150 Region is at 0.39
and SAM3 at 0.48-0.52.

**The line.** SAM3's scores are near-calibrated probabilities. At 0.5 it keeps 146 images per withheld half
(typed), for ~50 positives. That is precision 0.47 and recall 0.86, F1 0.57: 0.15 below the best cut of the same
ranking. Today's lines are fitted to cosine similarities and to the class model, and do not transfer. The text
route's line on SAM3's scores cuts at a median of 0.002 and returns nearly the whole pool (F1 0.04). Using SAM3 in
the app means using its own cut, or fitting a line to its scores.

**Prompt wording.** SAM3 reads a prompt as one noun phrase. A relative clause ("a person wearing a necktie") or a
disjunction ("a car or truck") reads as a different and rarer concept. Splitting a disjunction into one prompt per
member and taking the max recovers it: car/truck +0.16, vase/potted plant +0.24. One noun went the other way:
"skis" (0.50) loses to "a pair of skis" (0.72). "A pair of" is a count phrase SAM3 evidently knows for skis.

## Cost

Measured on 200 images x 102 prompts (`state_of_app/sam3_5213.sh smoke`):

| | V100 fp16 | L40S bf16 | L40S fp32 |
|---|---|---|---|
| Backbone, per image (once, shared by every prompt) | 126 ms | 77 ms | 240 ms |
| Per prompt per image, mask head skipped | 16.7 ms | 4.7 ms | 22.6 ms |
| Images/s at 102 prompts | 0.55 | 1.77 | 0.39 |
| The whole 24.5k pool at 102 prompts | ~12 GPU-h | ~3.8 GPU-h | |

SigLIP's typed sort, for comparison, is one dot product per image against a vector embedded at ingest. For the app
the question is the cost of a new query:

- **Without a cache,** every query pays the backbone: 24.5k x (77 + 4.7) ms = **~33 min on one L40S**.
- **With the backbone's detector-level features cached,** a query pays only the prompt step: **~2 min on one
  L40S** (~7 min on V100). The detector reads one 256 x 72 x 72 level, 2.65 MB an image at fp16: **~65 GB** for
  this pool, against ~0.3 MB a `max_patch` image today (#4711's bench).
- **Ingest** adds 77 ms an image on L40S (126 ms V100) for the backbone, on a GPU. The model is ~850M parameters.
- Lighter variants exist but were not measured: EfficientSAM3 (~90M, replaces both encoders), SAM3-LiteText
  (smaller text encoder only, so the same image cost). SAM 3.1 changes video only.

## How SAM3 could enter the vote loop: what is known

From a literature pass (sources in `research_notes.md` beside this report):

- **Exemplar prompts, positive and negative,** are native to SAM3: a box with a label, pooled from the image and
  concatenated with the text tokens. **But an exemplar must be in the image being searched.** The paper's exemplar
  results take the box from the target image itself. Cross-image exemplars are an open request
  (facebookresearch/sam3 issue 426). The workaround pastes the reference and the target on one canvas, about 2x
  the backbone per pool image per exemplar set. One paper (arXiv 2604.05433) reports that negative exemplars often
  hurt in few-shot use. So a Good vote's box cannot simply be handed to SAM3 for the rest of the pool.
- **Fine-tuning** pays in the paper's few-shot detection: ODinW13 box AP 61.0 zero-shot -> 66.1 / 69.1 / 71.8 at 1 /
  3 / 10 shots (full fine-tune, 40 epochs). That costs a training run per detector, far heavier than today's SVM.
  LoRA variants exist (text encoder kept frozen). **No light head on frozen SAM3 features was found**; that would be
  new.
- **SAM masks + DINO features:** "No Time to Train" (arXiv 2507.02798) pools DINOv2 inside SAM2 masks and matches to
  few-shot prototypes. It gets COCO 10-shot nAP 36.6 without training, ahead of fine-tuned few-shot detectors. In
  that paper **DINOv3 did worse than DINOv2** (32.8-33.8 vs 35.7). No head-to-head of mask pooling against a
  max-over-patches ranking was found.

## Training overlap (#5240)

**Question.** Is SAM3's lead memorisation of COCO? The paper's training-set table (Table 26) names no COCO and calls
COCO and LVIS zero-shot. But SA-Co/EXT holds Visual Genome, and VG's 108,077 images include 51,208 COCO 2014 images
(VG's `coco_id`). SAM3 saw those with VG's dense object labels, mapped to its ontology.

**Design.** That splits this pool, inside the same cells, into **seen** (in VG: 10,161 images, 41.5%) and **unseen**
(14,346). `text_baseline.py --keep-ids` grades each test half's seen and unseen images apart; the pool, split and
cuts stay the whole cell's. So the two slices partition #5213's rows item for item. Every arm (SigLIP typed, SAM3
typed, SAM3 noun) is graded on the same items. All 144 cells x 3 seeds are paired (none dropped). Prevalence is
0.43% on the seen slice and 0.46% on the unseen slice (8,680 and 13,096 test positives). SigLIP is the control: it
had no VG labels, so any difference between the slices themselves shows up in it too. The "all" run reproduces
#5213's CSVs (SAM3 exactly; SigLIP to 6e-5, text-tower float noise). Means here are over 3 seeds, not seed 0 alone.

| AP, mean over cells (± SE over classes) | seen (in VG) | unseen | seen - unseen |
|---|---|---|---|
| SigLIP typed sort | 0.42 | 0.42 | -0.001 |
| SAM3 typed | 0.71 | 0.72 | -0.009 (± 0.009) |
| SAM3 noun | 0.72 | 0.73 | -0.013 (± 0.009) |
| **SAM3 noun lead over SigLIP** | +0.30 | **+0.31** (± 0.02) | **-0.012 (± 0.010)** |

- **No lift on seen images.** If SAM3 remembered COCO it would rank seen images better. It ranks them slightly worse
  (difference-in-difference -0.012 ± 0.010 noun, -0.007 ± 0.010 typed). This holds in every band: small -0.015 ±
  0.019, medium -0.010 ± 0.015, large -0.011 ± 0.013 (noun). AUROC: -0.004 ± 0.003. Best-cut F1: +0.001 ± 0.009.
- **No class hides an effect.** Per class the difference-in-difference is spread both ways: 22 of 49 classes above 0
  (noun). The largest are knife +0.12 and bird +0.11; the smallest are bicycle -0.17 and traffic light -0.15. That is
  the spread of noise at a class's positive count, not a tail.
- **The unseen slice is the clean number for #5213's headline:** AP 0.73 (noun) vs SigLIP 0.42, lead +0.31.

**What this does not rule out.** Unseen means "never given labels for". SAM3's image encoder was pretrained on web
images (MetaCLIP is CommonCrawl), and COCO images are on Flickr, so some may have been crawled without labels. SigLIP's
WebLI has the same exposure, so the comparison stays even. Labelled exposure is the strong form, and it bought
nothing here. A corpus collected after SAM3 was trained would close the gap fully; none with COCO-style labels is on
the GRID.

Tables: `tables/sam3_5240_summary.csv` (every metric x band x arm x slice, with leads and differences),
`tables/sam3_5240_per_class.csv`, `tables/sam3_5240_counts.json`.
Reproduce: `sam3_5213.sh overlap DIR` (one CPU job, ~15 min).

## SAM3 as the opening (#5241)

**Status (2026-10-26): measured on seed 0, beta 1, both paths. SAM3 is an opt-in alternative to SigLIP's typed
sort, not a replacement: SigLIP stays the default because it is fast (see [Cost](#cost)). Nothing ships from this
PR. The run adds the knobs that open a session on SAM3, and prices takeover rules on the sessions it ran.**

**Question.** SAM3's set at its own 0.5 cut shows F1 0.57, against 0.37 for SigLIP's typed set. If a session opens on
SAM3, what does the user see over votes 1-150? When does the detector beat SAM3's set? And which rule should decide
when the detector takes over Test, now that the bar is 0.57 and not 0.37?

**Design.** Three arms on #4711's grid (COCO Better, 144 cells a path, seed 0, beta 1, 150 votes, trajectory pass):

- **`sig`:** today's app. SigLIP's typed sort opens; its set is shown until the label quota (Binary: the query hold
  and the hand-over), and then the head.
- **`sam3`:** SAM3's typed-prompt scores order the opening's picks (`CALIB_SEED_SAM3`). Its set at 0.5 is what Test
  shows at every vote on both paths (`CALIB_QUERY_LINE=fixed:0.5`, `CALIB_QUERY_HOLD` past any session,
  `CALIB_QUERY_HOLD_PATCH=1`, no query miss). The head still trains on every vote and is scored on the same withheld
  half, in rows tagged `hold:head`. Picks never depend on what Test shows, so **every takeover rule is a choice,
  vote by vote, between two rows of the same session.** Rules are priced afterwards; nothing is re-run.
- **`sam3gmm`:** SAM3's set at the GMM midpoint of its scores instead of 0.5 (one vote per Binary cell: the set
  does not change with the votes).

The query row of the `sam3` arm reproduces #5213's `text_native` row exactly (for example dining table@large: P 0.112,
R 0.889, F1 0.199). Means are over the 144 cells, ± SE over the 49 classes.

| F1 shown, mean over votes (± SE) | Binary 1-50 | Binary 1-150 | vs today, 1-150 | Region 1-50 | Region 1-150 | vs today, 1-150 |
|---|---|---|---|---|---|---|
| SigLIP opening (today) | 0.40 | 0.49 | | 0.46 | 0.55 | |
| SAM3's set at 0.5, never handed over | 0.57 | 0.57 | +0.085 (± 0.023) | 0.57 | 0.57 | +0.024 (± 0.025) |
| SAM3, head from the quota (today's hand-over) | 0.45 | 0.51 | +0.024 (± 0.005) | 0.52 | 0.58 | +0.032 (± 0.008) |
| **SAM3, head from the 20th Good** | **0.58** | **0.61** | **+0.122 (± 0.015)** | **0.59** | **0.62** | **+0.072 (± 0.016)** |
| SAM3, head once held-out Goods rank better under it (#4776 judge, log rank, sticky) | 0.58 | 0.60 | +0.114 (± 0.017) | | | |
| The better of the two at each vote (a bound, not a rule) | 0.62 | 0.64 | +0.158 (± 0.015) | 0.64 | 0.67 | +0.120 (± 0.015) |
| SAM3's set at its GMM line | 0.18 | 0.18 | -0.30 | 0.18 | 0.18 | -0.36 |

![F1 shown over the votes on each path: today, SAM3's set held, SAM3 handed over at the quota and at the 20th Good, and the per-vote bound](figures/sam3_5241_gate.webp)

**1. Opening on SAM3 pays, on both paths, if the hand-over waits.** With the head taking over at the 20th Good,
F1 shown over votes 1-150 is +0.12 on Binary and +0.07 on Region. Over votes 1-50 it is +0.18 and +0.13. Handing over
at the quota, as today, throws most of that away: about vote 7, the head is far below 0.57 (Binary +0.024, Region
+0.032).

**2. When the detector beats SAM3's set.**

- **Binary: on average, never within 150 votes.** The head's mean F1 is 0.56 at vote 150, against SAM3's 0.57. Per
  cell, the head is ahead in 25% of cells at vote 25, 36% at 50, 39% at 100 and 44% at 150.
- **Region: at vote 43 on average.** Per cell, 44% at vote 25, 48% at 50, 57% at 100 and 55% at 150.
- **It depends on object size, as #5213 found for AP.** Small objects: the head never catches SAM3 (Binary's bound
  is only +0.011 over holding SAM3). Large objects: the head wins early and by a lot (holding SAM3 costs -0.13
  Binary, -0.14 Region against today).

**3. Gating: hand over at the 20th Good.** The rule may only use the labels. Two families were priced:

- **A Goods count** (`goods>=N`, check votes included). Against holding SAM3, over votes 1-150:

  | N | 4 | 8 | 12 | 16 | **20** | 24 | 32 |
  |---|---|---|---|---|---|---|---|
  | Binary | -0.046 | -0.006 | +0.019 | +0.030 | **+0.037** (± 0.012) | +0.035 | +0.016 |
  | Region | +0.018 | +0.028 | +0.040 | +0.045 | **+0.047** (± 0.011) | +0.035 | +0.013 |

  Both paths peak at 16-24, and the top is flat. A session has 29 Goods at vote 150 (median). The count works as a
  stand-in for object size and prevalence: a class whose Goods arrive fast is one the head learns fast.
- **#4776's judge, with SAM3 in the centroid's place.** Each labelled Good is held out, the head is refitted on the
  rest, and the Good's rank in the unvoted pool under the head is set beside its rank under SAM3 (SAM3 is not fitted
  on the labels, so its rank is read straight off the pool). Binary only: the judge's rows exist on single-vector
  cells. The best variant (mean log rank, once it prefers the head it keeps it) gives +0.028 (± 0.010) against
  holding SAM3, level with the count. The plain variants give -0.002 to +0.011. **It buys nothing over the count and
  needs a refit per Good, so the count is the rule.**

By band, the 20th-Good rule against today (votes 1-150):

| | small | medium | large |
|---|---|---|---|
| Binary | +0.26 (± 0.03) | +0.11 (± 0.02) | +0.002 (± 0.010) |
| Region | +0.19 (± 0.03) | +0.05 (± 0.02) | -0.016 (± 0.027) |

The gain is all on small and medium objects. On large objects the rule ends level with today, within 1 SE on both
paths: those sessions reach 20 Goods early, so they spend most votes on the same head as today.

**4. The head itself is better after a SAM3 opening, early on.** At the same vote, the head trained after SAM3's
picks against the head after SigLIP's: Binary +0.020 (± 0.009) at vote 50, +0.012 (± 0.007) at 100, +0.003 (± 0.005)
at 150; Region +0.038 (± 0.012), +0.020 (± 0.009), +0.003 (± 0.009). SAM3's picks find more Goods early, and the
difference washes out by vote 150.

**5. SAM3 needs its own line, and 0.5 is it.** At the GMM midpoint of its scores, SAM3's set is F1 0.18: the scores
pile up near 0, the midpoint falls inside that pile, and the set keeps most of the pool.

**Caveats.**

- **The count of 20 is chosen on the data it is scored on.** The top is flat from 16 to 24 on both paths, so the
  choice is not a fine-tuned point, but it needs a fresh seed to confirm (#5311).
- **Seed 0 and beta 1 only.** The other presets move the head's line, and with it where the hand-over pays.
  #5311 checked both: see the next section.
- **Cost is #5213's.** SAM3 scores a new query in ~2 min on one L40S with cached backbone features (~65 GB for this
  pool), or ~33 min without. That is why this is an opt-in path and SigLIP stays the default.

Tables: `tables/sam3_5241_summary.csv` (every shown rule x path x band: areas, gaps to today, F1 at votes 10, 50, 100,
150), `tables/sam3_5241_gate.csv` (every takeover rule x path x band, against holding SAM3 and against today),
`tables/sam3_5241_per_click.csv`, `tables/sam3_5241_gate_per_click.csv`, `tables/sam3_5241_cells.csv` (per cell:
SAM3's F1, the head's, and the first vote the head beats SAM3), `tables/sam3_5241_info.json`.
Reproduce: `sam3_opening_5241.sh launch {sig,sam3,sam3gmm}` from a frozen worktree (288 cells a path; Binary ~10 min
a cell, Region ~1 h), then `sam3_opening_5241.py --out DIR` and `sam3_gate_5241.py {collect,price} --out DIR`.

## Out of sample and the other presets (#5311)

**Status (2026-11-05): at beta 1 the 20th-Good hand-over holds on fresh seeds: +0.12 Binary, +0.07 Region, the same
as seed 0. At the other two presets, one count of 20 is wrong. SAM3's 0.5 set is the same set at every balance, while
today's line moves with it. So at beta 4 SAM3's set should be held, and at beta 1/4 the head should take over early,
and on Region SAM3 then buys nothing. #5312 needs a count per preset, or a SAM3 line that moves with beta.**

**Design.** Same arms as #5241 (`sig`, `sam3`), on #4711's grid, 150 votes, trajectory pass, Intel kernels. Seeds 1 and
2 at beta 1 (288 cells a path), and seed 1 at beta 1/4 and 4 (144 cells a path; the owner trimmed the off-presets
to one seed). The issue asked for beta 0.5 and 2, but the app's presets are 4, 1 and 1/4 (`BALANCE_PRESETS`), so
those were run. `goods>=20` was fixed before the run. The other counts are priced beside it only to see whether the
best count moves, and the best count at a preset is chosen on the sessions it is scored on. F is F-beta at the run's
own beta, the objective the session runs on. ± is the SE over the 49 classes.

**1. Beta 1: confirmed.** F1 shown over votes 1-150, against today's SigLIP opening:

| | seed 0 (#5241, in-sample) | seed 1 | seed 2 | seeds 1-2 |
|---|---|---|---|---|
| Binary, `goods>=20` | +0.12 (± 0.015) | +0.11 (± 0.014) | +0.12 (± 0.013) | **+0.12 (± 0.013)** |
| Region, `goods>=20` | +0.07 (± 0.016) | +0.07 (± 0.012) | +0.07 (± 0.012) | **+0.07 (± 0.012)** |

The best count does not move. On Binary 20 and 24 tie (+0.115 each), 16 gives +0.11 and 32 +0.10. On Region 20 is
the top (+0.071), 16 +0.064, 24 +0.058. Holding SAM3 all session is +0.08 / +0.02, handing over at the quota +0.03 /
+0.02, and the per-vote bound +0.16 / +0.11, all as on seed 0.

**2. The count belongs to the preset.** F-beta shown over votes 1-150 against today, seed 1:

| rule | Binary 1/4 | Binary 1 | Binary 4 | Region 1/4 | Region 1 | Region 4 |
|---|---|---|---|---|---|---|
| SAM3 held, never handed over | -0.12 | +0.08 | **+0.22** | -0.17 | +0.02 | **+0.14** |
| head at the quota (about vote 7) | +0.03 | +0.03 | +0.03 | +0.01 | +0.02 | +0.05 |
| `goods>=8` | **+0.05** | +0.07 | +0.11 | +0.01 | +0.04 | +0.09 |
| `goods>=12` | **+0.05** | +0.09 | +0.15 | +0.01 | +0.06 | +0.10 |
| `goods>=20` (fixed beforehand) | +0.03 (± 0.015) | **+0.12** | +0.19 (± 0.018) | -0.01 (± 0.014) | **+0.07** | +0.13 (± 0.013) |
| `goods>=32` | -0.05 | +0.10 | +0.22 | -0.13 | +0.03 | +0.14 |
| better of the two per vote (bound) | +0.08 | +0.16 | +0.24 | +0.05 | +0.11 | +0.16 |

(The beta 1 column is seeds 1-2.)

- **Beta 4 (recall):** SAM3's set is never beaten. The head drops F4 from 0.78 to 0.72 on Binary once it takes over,
  so the best rule is to hold SAM3 (`goods>=32` or later, or no hand-over): +0.22 Binary, +0.14 Region.
- **Beta 1/4 (precision):** SAM3's set is the wrong set. Holding it costs -0.12 / -0.17. The head should take over
  by the 8th-12th Good on Binary (+0.05 ± 0.01). On Region nothing beats today by more than +0.01 (± 0.01): opening
  on SAM3 buys nothing there.

![F-beta shown over the votes at each preset and path: today, SAM3's set held, SAM3 to the head at the 20th Good, and the best count at that preset](figures/sam3_5311_by_beta.webp)

**Why: SAM3's line ignores the balance.** The 0.5 cut gives the same set at every preset. At vote 1 it has
precision 0.47 and recall 0.86. Today's typed line moves with beta: P 0.50 / R 0.25 at 1/4, 0.36 / 0.44 at 1, and
0.16 / 0.58 at 4. A recall-heavy set is just what beta 4 asks for, and too loose for beta 1/4. The head's own line
moves with beta, so the right hand-over point moves with it.

**Literal examples** (`tables/sam3_5311_cells.csv`, seed 1):

- **Beta 1/4.** SAM3's set scores F0.25 below 0.08 on dining table@small (0.05), bag or luggage@small and @medium
  (0.07) and bottle@medium (0.08). Its precision there is 0.05-0.07 at recall 0.35-1.0 (bottle@medium P 0.07, R 1.0):
  some 13-20 non-targets for each target, while beta 1/4 weights precision 16 times recall.
- **Beta 4.** It scores F4 above 0.99 on tennis racket@large, baseball bat@medium and skateboard@medium, where
  today's session reaches 0.93-0.98 over 150 votes.

**What this means for #5312.**

- One count of 20 is right only for the default balance.
- The route either needs a count per preset (about 8-12 at 1/4 on Binary, 20 at 1, hold at 4), or SAM3's line has to
  move with beta (a cut per preset), which would make one count fit all three again. #5565 tested both: see
  [A SAM3 cut per preset](#a-sam3-cut-per-preset-5565). The cut wins at 1/4 and 1, and 4 still holds.
- On Region at 1/4 the opt-in buys nothing either way.
- #5521's set-size guard is still open as well.

**Caveats.**

- **One seed at 1/4 and 4.** The per-preset best counts above are chosen on the sessions they are scored on, so they
  are a direction, not a setting. Only the beta-1 result, with N fixed beforehand, is out of sample.
- **Region at 1/4 is within noise** for every rule except the bad ones: no claim either way beyond "nothing to gain".

Tables (archived with the study): `tables/sam3_5311_gate_{b025,b1,b4}.csv` (every rule x path x band, against
holding SAM3 and against today), `tables/sam3_5311_gate_per_click_{b025,b1,b4}.csv`, `tables/sam3_5311_seeds.csv`
(each `goods>=N` against today, per seed), `tables/sam3_5311_set_pr.csv` (vote-1 precision and recall of each set),
`tables/sam3_5311_cells.csv` (per cell: SAM3's set F and today's F over 1-150). Reproduce: `S5241_DATE=2026-11-04
S5241_SEEDS="1 2" S5241_BETA=<b> sam3_opening_5241.sh launch {sig,sam3}` from a frozen worktree, then
`sam3_opening_5241.py` and `sam3_gate_5241.py {collect,price}` with `--date 2026-11-04 --beta <b>`, then
`sam3_confirm_5311.py --out DIR`.

## A SAM3 cut per preset (#5565)

**Status (2026-11-05): move SAM3's cut with the preset (0.85 at beta 1/4, 0.75 at 1, 0.4 at 4), hand over at the 20th
Good at 1/4 and 1, and hold SAM3's set at 4. On the fresh seed this beats today at every preset and on both paths,
by +0.11 / +0.13 / +0.23 on Binary and +0.06 / +0.09 / +0.14 on Region. It also beats a hand-over count per preset
at 0.5 at 1/4 (+0.075 / +0.051) and at 1 (+0.023 / +0.017), and ties it at 4. With the cut moved, the 20th Good fits
1/4 and 1 again, but at 4 holding is still better. The label-free cut (scores read as probabilities) fails, because
SAM3's scores are far from calibrated.**

**Design.** #5311's question was which of two fixes makes the opt-in route (#5312) right at every preset. Both were
fixed in the issue before any fresh cell ran:

- **Option 2, a hand-over count per preset at the 0.5 cut**, read off #5311's seed 1: `goods>=8` at 1/4, `goods>=20`
  at 1, hold at 4.
- **Option 1, a SAM3 cut per preset.** (a) is label-free: the cut that maximises F-beta's expectation when SAM3's
  scores of the withheld half are read as probabilities. (b) is one fixed cut per preset, the grid cut (0.05-0.95)
  with the best mean set F-beta on the tuning seeds (seeds 0-1 at beta 1, seed 1 at 1/4 and 4).

The fresh seed is seed 2 at every preset. New sessions, on dev plus a line knob (273907ef1), Intel kernels:

- seed 2 at 1/4 and 4, both arms and both paths;
- today's Binary arm again at every seed and preset compared, because dev shipped hold 0 (#5556) after #5311's tree.
  Region's today is #5311's run: the hold never applied on a patch space.

F is F-beta at the preset's beta over votes 1-150. ± is the SE over the 49 classes. Paired differences are per cell.

**A cut is priced offline, and that is checked.** Picks never read the set, so SAM3's set at any cut is rebuilt off
each cell's withheld half: `evaluable_pool`, then `_split_media_ids` under `RandomState(seed)`, and every Test image
scoring the cut or more (`_query_step`). Two checks:

- The rebuilt 0.5 sets match all 2,080 sessions' recorded vote-1 precision and recall, on both paths, within 5e-7
  (`tables/sam3_5565_check.csv`).
- The same 32 cells were run at `fixed:0.5` and `fixed:0.3` on this tree (24 Binary, 8 Region, seed 1). Both give
  byte-identical picks and `hold:head` rows, to each other and to #5311's sessions on the old tree. Only the set
  changes.

**1. The set: a cut per preset, the same on every seed.** SAM3's set alone (no hand-over), mean over 144 cells:

| set | beta 1/4: F, P / R | beta 1: F, P / R | beta 4: F, P / R |
|---|---|---|---|
| 0.5 (#5311's) | 0.48, 0.47 / 0.86 | 0.57, 0.47 / 0.86 | 0.78, 0.47 / 0.86 |
| (b) tuned: 0.85 / 0.75 / 0.4 | **0.68**, 0.71 / 0.58 | **0.63**, 0.63 / 0.72 | **0.79**, 0.42 / 0.89 |
| (a) label-free (median cut 0.76 / 0.32 / 0.05) | 0.62, 0.62 / 0.74 | 0.49, 0.38 / 0.91 | 0.63, 0.14 / 0.98 |

- **Seed 2 picks the same tuned cuts.** Its own best grid cut is 0.85 / 0.75 / 0.4 as well, so tuning on seed 1 costs
  it nothing. Seeds 0-2 differ by at most 0.005 in set F.
- **Even at beta 1, 0.5 is not SAM3's best cut.** 0.75 is +0.06 F1.
- **(a) keeps far too much at 1 and 4, because SAM3 is overconfident at the bottom.** Images scoring under 0.1
  average 0.004 but are Good 0.02% of the time (`tables/sam3_5565_calibration.csv`). Images scoring 0.5-0.6 are Good
  6% of the time, and those at 0.9-1.0 77%. A cell's ~11,000 withheld images scoring under 0.1 add some 47 phantom
  Goods to the expectation, so the cut sinks.

**2. The route: F-beta shown over votes 1-150 against today, seed 2 (out of sample for every choice made).**

| route (set, hand-over) | Binary 1/4 | Binary 1 | Binary 4 | Region 1/4 | Region 1 | Region 4 |
|---|---|---|---|---|---|---|
| one count: 0.5, 20th Good (#5312 as filed) | +0.026 | +0.11 | +0.19 | -0.014 | +0.068 | +0.12 |
| option 2: 0.5, 8th / 20th Good / hold | +0.034 (± 0.010) | +0.11 (± 0.014) | +0.23 (± 0.022) | +0.006 (± 0.010) | +0.068 (± 0.012) | +0.13 (± 0.017) |
| option 1a: label-free cut, 20th Good | +0.081 | +0.076 | +0.12 | +0.038 | +0.036 | +0.060 |
| option 1b: tuned cut, 20th Good | **+0.11** (± 0.016) | **+0.13** (± 0.015) | +0.20 (± 0.019) | **+0.057** (± 0.015) | **+0.085** (± 0.012) | +0.13 (± 0.014) |
| **both: tuned cut, 20th Good, hold at 4** | **+0.11** (± 0.016) | **+0.13** (± 0.015) | **+0.23** (± 0.021) | **+0.057** (± 0.015) | **+0.085** (± 0.012) | **+0.14** (± 0.015) |
| best count on seed 2, tuned cut (bound) | +0.11 (24th) | +0.14 (32nd) | +0.23 (48th) | +0.057 (20th) | +0.085 (20th) | +0.14 (hold) |
| today's F, for scale | 0.60 | 0.49 | 0.56 | 0.65 | 0.55 | 0.65 |

Seed 1 gives the same picture (within 0.01 in every cell of the table; `tables/sam3_5565_routes.csv`). Paired per
cell on seed 2 (`tables/sam3_5565_cells.csv`), "both" against option 2 is:

- +0.075 (± 0.010) and +0.051 (± 0.009) at 1/4, ahead in 83% of cells;
- +0.023 (± 0.006) and +0.017 (± 0.006) at 1, ahead in 76%;
- +0.006 (± 0.004) on both paths at 4, ahead in 47%.

Against #5312's single count at 0.5, it is +0.083 / +0.071 at 1/4, +0.023 / +0.017 at 1 and +0.040 / +0.014 at 4.

- **Does one count fit all three presets again?** At 1/4 and 1, yes: with the cut moved, the 20th Good is within
  0.01 of the best count on both paths. At 4, no. The head still pulls F4 down once it takes over (0.79 to about
  0.73 on Binary), so holding SAM3 is worth +0.035 on Binary and +0.010 on Region over the 20th Good.
- **Region at 1/4 now gains.** At 0.5 SAM3 bought nothing there (+0.006). At 0.85 the set is +0.057, and the
  route is +0.057.

![F-beta shown per vote on seed 2 at each preset and path: today, option 2 (0.5 with a count per preset), option 1b (a cut per preset, 20th Good) and both (a cut per preset, 20th Good, held at 4). At 1/4 and 1 option 1b is the same route as both, under its line](figures/sam3_5565_by_beta.webp)

**Literal examples** (seed 2, `tables/sam3_5565_sets.csv`). The tuned cut helps the set in 89% of cells at 1/4, 76%
at 1 and 47% at 4:

- **keyboard@medium (53 Goods in the withheld half).** 0.5 keeps 158 images (49 Good). 0.85 keeps 55 (44 Good):
  F0.25 +0.48. At 0.4, 190 images (51 Good).
- **dog@medium (51 Goods).** 0.5 keeps 99 (50 Good). 0.85 keeps 60 (50) and 0.75 keeps 71 (50): +0.32 at 1/4,
  +0.15 at 1.
- **skis@small (45 Goods), the worst cell.** 0.5 keeps 77 (27 Good). 0.85 keeps 3 (0 Good) and 0.75 keeps 9 (3):
  -0.36 at 1/4, -0.33 at 1. SAM3 scores small skis low, so a high cut empties the set. At 0.4 it keeps 112 (38):
  +0.20 at 4.

**What this means for #5312.** The route should take SAM3's cut from the preset: 0.85 at 1/4, 0.75 at 1 and 0.4 at
4. It hands over to the head at the 20th Good at 1/4 and 1, and does not hand over at 4 (through vote 150).

**Caveats.**

- **The tuned cuts are fitted on seed 1** (seeds 0-1 at 1), on the same 144 cells. Seed 2 is a fresh split and
  fresh sessions, but the same classes and bands. A different corpus or prompt set would need its own check.
- **The cut grid is 0.05 wide.** The set's F is flat near each optimum (within 0.01 for ±0.05 at 1 and 4), so the
  exact value matters little.
- **A high cut can empty a weak cell's set** (skis@small). That is the reverse of #5521's broad-set guard, and the
  same guard question: what to show when SAM3's set is the wrong size.
- **"Hold at 4" is through vote 150**, the eval's horizon, not the user's. The head never caught SAM3's set at 4 by
  then.

Tables (archived with the study), all in `tables/`:

- `sam3_5565_summary.csv`: the headline routes on seed 2.
- `sam3_5565_routes.csv`: every set x gate x seed x path x preset.
- `sam3_5565_cells.csv`: per cell, the headline routes.
- `sam3_5565_per_vote.csv`: per vote, the headline routes.
- `sam3_5565_sets.csv`: SAM3's set on each cell's withheld half at every cut.
- `sam3_5565_set_pr.csv`, `sam3_5565_tuned_cuts.csv`, `sam3_5565_calibration.csv`, `sam3_5565_check.csv`.

Reproduce, from a frozen worktree:

1. `S5241_DATE=2026-11-05 S5241_SEEDS=... S5241_BETA=<b> sam3_opening_5241.sh launch {sig,sam3}`, and
   `S5241_TAG=-v03 S5241_LINE=fixed:0.3` for the check.
2. `sam3_cut_5565.py sets|check|price --out DIR`.

## Follow-ups (filed)

- **SAM3 as the opening (#5241), done:** see [SAM3 as the opening](#sam3-as-the-opening-5241). As an opt-in path,
  with the head taking over at the 20th Good: +0.12 F1 shown over votes 1-150 on Binary, +0.07 on Region.
- **Confirm the 20th-Good hand-over out of sample (#5311), done:** see [Out of sample](#out-of-sample-and-the-other-presets-5311).
  It holds at beta 1. At 1/4 and 4 the count has to move, because SAM3's 0.5 line does not.
- **A SAM3 cut per preset (#5565), done:** see [A SAM3 cut per preset](#a-sam3-cut-per-preset-5565). The cuts are 0.85
  / 0.75 / 0.4, with a hand-over at the 20th Good at 1/4 and 1 and a hold at 4. That is +0.11 / +0.13 / +0.23 Binary
  and +0.06 / +0.09 / +0.14 Region on a fresh seed.
- **SAM3 as an opt-in opening in the app (#5312):** the route, the feature cache, SAM3's cut per preset and the
  hand-over (#5565). SigLIP stays the default.
- **A light head on SAM3's object queries (#5242):** each image's 200 decoder queries are region embeddings that come with
  boxes. A Good box picks its query, a Bad floods them, and the SVM trains on them: #4711's Region loop with SAM3's
  regions in place of DINOv3 patches. This is the "learnable head" question, and it would keep SAM3's prior.
- **SAM3 masks as Region geometry for DINOv3 (#5243):** mask-pooled patch features and a HAC over SAM3's objects, against
  `max_patch`, the "DINO on SAM3 masks" question.
- **Cross-image exemplars from votes (#5244):** the canvas trick with a cell's Good (and Bad) boxes, scored on the
  withheld half, against zero-shot. Includes whether negatives help or hurt.
- **Training overlap (#5240), done:** see [Training overlap](#training-overlap-5240). Labelled exposure buys SAM3 nothing,
  so the follow-ups above can be built on #5213's numbers.

## Files

- `scripts/experiments/calibration/sam3_score_5213.py`: the GPU half (prompts, scores, timing).
- `scripts/experiments/calibration/text_baseline.py`: `--sam3 DIR --sam3-prompts typed|noun|both`.
- `scripts/experiments/state_of_app/sam3_5213.sh`: `smoke GPU`, `score N`, `baseline`.
- `scripts/experiments/state_of_app/sam3_analyze_5213.py`: tables and figures here.
- `scripts/experiments/state_of_app/sam3_overlap_5240.py`: #5240's id split (`ids`) and analysis (`analyze`);
  `text_baseline.py --keep-ids`; `sam3_5213.sh overlap DIR` runs both.
- `scripts/experiments/state_of_app/sam3_opening_5241.sh` (#5241's three arms), `sam3_opening_5241.py` (F1 shown
  by rule, path and band), `sam3_gate_5241.py` (takeover rules: `collect` the judge's evidence, `price` the rules).
  Knobs: `CALIB_SEED_SAM3`, `CALIB_SEED_SAM3_PROMPTS`, `CALIB_QUERY_LINE`, `CALIB_QUERY_HOLD_PATCH`;
  `calibration/sam3_scores.py` reads the shards for both `text_baseline.py` and `run_cells.py`.
- `scripts/experiments/state_of_app/sam3_confirm_5311.py`: #5311's cross-preset figure, per-seed table, set P/R and
  per-cell examples (the launcher takes `S5241_SEEDS` / `S5241_BETA`; the two analyses take `--date` / `--beta`).
- `scripts/experiments/state_of_app/sam3_cut_5565.py`: #5565's sets at any cut off the rebuilt withheld half
  (`sets`), their check against the sessions (`check`) and every route's price (`price`). The launcher takes
  `S5241_LINE` for the sam3 arm's line.
- Raw scores (12 `.npz` shards, 24,507 x 102 x {top1, top2, n>0.5}) and the three `text_baseline` CSVs live in the
  study's results directory on the GRID, `sam3-5213/` under the owner's `/expscratch`.
