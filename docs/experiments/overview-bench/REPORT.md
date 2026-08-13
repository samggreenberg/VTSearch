# VTSearch overview benchmark: how each configuration behaves

**Run:** 2026-08-12 · branch `claude/vts-benchmark` · arrays `496044` (wave 1),
`496454` / `496673` (wave 2)
**Data:** `/expscratch/sgreenberg/bench-{overview,vgbox,vgbox2}/results`

This is a **characterization**, not a comparison. Nothing here is trying to pick
a configuration to ship. The question is what each of them *does* — what it is
good at, what it is bad at, and what regime moves it from one to the other. Where
two configurations differ, the useful output is the mechanism behind the
difference, not the sign of it.

Every *behavioural* knob is at its shipped default — head `linear`,
`safe_thresholds=False`, `calibrate_count=2`, acquisition inclusion offset `-1`,
production `max_patch` geometry. Only sizing knobs were set.

## What was exercised

| axis | levels |
|---|---|
| representation | `siglip` (shipped, whole-image, text-capable), `siglip2_l` (premium, whole-image, text-capable), `dinov3_patch` (patch geometry, region-voting where boxes exist, **no text tower**) |
| acquisition | typed query (0 clicks, GMM cut) · Autopilot clicking (150 votes) |
| haystack | `visual_genome_m` (4,193), `coco_val` (4,952), `caltech101_m` (838, boxless), `vg_box_{small,medium,large}` (12,000 each, banded on box area) |
| category | 6–10 per dataset · 3 seeds · 150 votes |

Wave 1: 189 cells / 26,538 steps. Wave 2: 99 cells / 14,042 steps. A wave-2
re-run with prevalence-spread categories (270 cells) is in flight; its numbers
slot into the same frame and are not yet included.

---

# Reference numbers

Deep regime (t ≥ 100). cost = fpr + fnr. No ordering implied.

| dataset | embedder | cost | fpr | fnr | regret | AP | AUROC | cell wall-time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `caltech101_m` | `siglip` | 0.0039 | 0.0039 | 0.0000 | 0.0039 | 1.000 | 1.000 | ~110 s |
| `caltech101_m` | `siglip2_l` | 0.0013 | 0.0013 | 0.0000 | 0.0013 | 1.000 | 1.000 | ~110 s |
| `caltech101_m` | `dinov3_patch` | 0.0047 | 0.0038 | 0.0009 | 0.0031 | 1.000 | 1.000 | ~110 s |
| `coco_val` | `siglip` | 0.2177 | 0.0711 | 0.1466 | 0.0448 | 0.695 | 0.942 | ~110 s |
| `coco_val` | `siglip2_l` | 0.2019 | 0.1154 | 0.0865 | 0.0618 | 0.711 | 0.955 | ~110 s |
| `coco_val` | `dinov3_patch` | 0.1524 | 0.0920 | 0.0604 | 0.0496 | 0.787 | 0.976 | ~19 min |
| `visual_genome_m` | `siglip` | 0.3918 | 0.3006 | 0.0912 | 0.1113 | 0.428 | 0.898 | ~110 s |
| `visual_genome_m` | `siglip2_l` | 0.3666 | 0.2438 | 0.1228 | 0.0960 | 0.457 | 0.899 | ~110 s |
| `visual_genome_m` | `dinov3_patch` | 0.3242 | 0.1819 | 0.1423 | 0.1100 | 0.525 | 0.913 | ~17 min |
| `vg_box_large` | `siglip` | 0.3677 | 0.2050 | 0.1628 | 0.0974 | 0.283 | 0.901 | ~40 s |
| `vg_box_large` | `siglip2_l` | 0.3234 | 0.1591 | 0.1643 | 0.0709 | 0.277 | 0.906 | ~40 s |
| `vg_box_large` | `dinov3_patch` | 0.3243 | 0.1554 | 0.1689 | 0.1241 | 0.300 | 0.940 | ~32 min |
| `vg_box_medium` | `siglip` | 0.7471 | 0.4062 | 0.3409 | 0.0800 | 0.097 | 0.689 | ~40 s |
| `vg_box_medium` | `siglip2_l` | 0.7365 | 0.3830 | 0.3535 | 0.0945 | 0.105 | 0.708 | ~40 s |
| `vg_box_medium` | `dinov3_patch` | 0.5902 | 0.1725 | 0.4176 | 0.0885 | 0.143 | 0.789 | ~32 min |
| `vg_box_small` | `siglip` | 0.7774 | 0.4131 | 0.3642 | 0.0934 | 0.093 | 0.668 | ~40 s |
| `vg_box_small` | `siglip2_l` | 0.7391 | 0.4363 | 0.3028 | 0.0902 | 0.086 | 0.683 | ~40 s |
| `vg_box_small` | `dinov3_patch` | 0.6463 | 0.4429 | 0.2035 | 0.1964 | 0.136 | 0.823 | ~32 min |

Zero-click typed query, same test splits (`dinov3_patch` has no text tower):

| dataset | embedder | text cost | text AP | text AUROC |
|---|---|---:|---:|---:|
| `caltech101_m` | `siglip` / `siglip2_l` | 0.1612 / 0.0444 | 1.000 / 1.000 | — |
| `coco_val` | `siglip` / `siglip2_l` | 0.3024 / 0.2768 | 0.707 / 0.743 | — |
| `visual_genome_m` | `siglip` / `siglip2_l` | 0.4894 / 0.3936 | 0.496 / 0.544 | — |

---

# The representations

## `siglip` — the shipped default

**Behaves like:** a fast, text-addressable whole-image encoder that degrades
gracefully. ~110 s per 150-vote run; a text tower, so a user can start from a
typed query at zero cost.

**Its error budget sits in false positives.** On VG its fpr (0.301) is 3.3× its
fnr (0.091) — by far the most lopsided arm in the study. It is *including* too
much, not missing things. That shape is stable across datasets: fpr ≥ fnr
everywhere except COCO.

**Where it holds up:** anything with a clean whole-image signature. On
`caltech101_m` it is at ceiling (AP 1.000). On COCO its ranking (AP 0.695) is
within 0.02 of the premium encoder.

**Where it comes apart:** as the target shrinks relative to the frame. Across the
box bands its AP is 0.283 → 0.097 → 0.093 (large → medium → small) and AUROC
0.901 → 0.689 → 0.668. At sub-patch scale it is close to uninformative, which is
the expected consequence of pooling a whole image into one vector when the
target occupies under 0.5 % of it.

## `siglip2_l` — the premium whole-image encoder

**Behaves like:** `siglip` with a uniformly better ranking and the same cost
profile. Same ~110 s. AP is higher on every non-saturated dataset (VG 0.457 vs
0.428; COCO 0.711 vs 0.695), and its *text* ranking is better too (VG text AP
0.544 vs 0.496).

**It rebalances the error budget rather than only shrinking it.** On VG it moves
fpr 0.301 → 0.244 while fnr rises 0.091 → 0.123. It is a less trigger-happy
encoder, not merely a more accurate one.

**It inherits `siglip`'s scale failure intact.** AP across box bands 0.277 →
0.105 → 0.086 — the same collapse. Capacity does not substitute for geometry:
whatever a bigger whole-image encoder buys, it is not the ability to see a
sub-patch object.

## `dinov3_patch` — patch geometry, and region voting where boxes exist

**Behaves like:** a much better ranker with a much worse threshold, at ~10–30×
the compute (17–32 min per run vs ~110 s / ~40 s).

**Its ranking is the best measured, and the margin grows as targets shrink.** AP
by box band: 0.300 → 0.143 → 0.136 against `siglip2_l`'s 0.277 → 0.105 → 0.086.
AUROC 0.940 / 0.789 / 0.823 vs 0.906 / 0.708 / 0.683. On the large band its
ranking edge is small; at medium and small it is proportionally large. The
mechanism is straightforward: box supervision only carries information the
whole-image vector lacks when the box is a small fraction of the frame. A box
covering a third of the image *is* approximately the image.

**Its threshold is its weak point, and this is the most useful thing the run
says about it.** On `vg_box_large` it is the **only arm in the entire study with
a positive `rule_inefficiency`** (+0.021, against −0.021 for `siglip2_l`), and
its regret is 0.124 vs 0.071. So on large boxes it ranks better and cuts worse,
and the two cancel to an identical cost (0.3243 vs 0.3234). Its discrimination
advantage is real and **currently unconverted** — that is a calibration
property, not a representation one.

**Where it comes apart:** cold start and cost. `too_few_default` is 7.2 % on VG
(worst of the three) and 18 % on the sub-patch band. On `vg_box_small` it is the
only arm in either wave whose **regret grows with votes** (0.129 → 0.196). And
it has **no text tower**, so there is no zero-click entry point for it at all.

**On a boxless dataset it is not a patch model.** `caltech101_m × dinov3_patch`
runs `whole_image` by construction: with no box, a Good vote has nothing to pool,
so patch rows would be negatives-only and could teach nothing but "patch-like ⇒
negative". It is DINOv3 used as a whole-image encoder, and it behaves like one
(cost 0.0047, indistinguishable from the SigLIPs at ceiling).

---

# Typing and clicking supply different things

The two acquisition modes are usually discussed as alternatives. The measurement
says they are not the same kind of thing at all.

**What a typed query supplies: discrimination, immediately, badly calibrated.**

| arm | text AP | detector AP after 150 votes |
|---|---:|---:|
| `visual_genome_m` × `siglip` | 0.496 | 0.430 |
| `visual_genome_m` × `siglip2_l` | 0.544 | 0.461 |
| `coco_val` × `siglip2_l` | 0.743 | 0.710 |
| `coco_val` × `siglip` | 0.707 | 0.700 |

On Visual Genome the *ranking* after 150 clicks is worse than the ranking you get
from typing the word. But text's operating point is poor: its GMM cut sits far
from its own oracle, so text cost (0.489) is much worse than its ranking implies.

**What the clicking loop supplies: calibration.** Its regret falls with votes and
its `rule_inefficiency` is negative on every arm but one — the shipped cut rule
already places a better threshold than the in-sample calibration optimum. The
whole of regret is `calibration_shift`, the sim→test move.

**So the two are complementary, and the composition is untested.** Text is good
at the thing clicking is bad at (getting a usable ranking from nothing) and bad
at the thing clicking is good at (placing the cut). This is sharpest for
`dinov3_patch`: it has the best ranking, the worst cold start, and no text tower
— while SigLIP text gives a usable ranking over the same medias in the pile for
free. Seeding a DINOv3 detector from a SigLIP text query is the obvious thing
this measurement points at, and nothing here tests it.

**Where each mode fails, concretely:**

| category | prevalence | text cost | detector @150 | text AP | det AP | what it shows |
|---|---:|---:|---:|---:|---:|---|
| `coco` `cat` | 0.038 | 0.017 | 0.081 | 0.991 | 0.969 | text at ceiling; clicking has nothing to add and adds threshold noise |
| `vg` `sky` | 0.188 | 0.574 | 0.682 | 0.441 | 0.376 | 19 % prevalent, and clicking still degrades a usable ranking |
| `vg` `ball` | 0.012 | 0.490 | 0.648 | 0.392 | 0.199 | rare; the clicking loop starves and the ranking collapses |
| `coco` `bear` | 0.009 | 0.215 | 0.010 | — | 0.992 | clicking's best case: rare, visually clean, threshold is everything |

Typed queries fail on **parts and mass nouns** (`nose`, `sky`, `ball`) and on
awkward label strings; they succeed on **common distinctive nouns**. The clicking
loop fails when **positives are too rare to accumulate** and succeeds when a
handful of positives is enough to fix a cut.

---

# What moves the regime

**Target scale** is the strongest axis in the study. Best-arm cost across the box
bands runs 0.323 (large) → 0.590 (medium) → 0.646 (small), and AP 0.30 → 0.14 →
0.14. Sub-patch retrieval is hard for every configuration; the patch geometry
reduces the damage but does not remove it. This is the first measurement of that
band on a real sample — the full VG vocabulary has 643 sub-patch categories
against 5 in the demo vocabulary.

**Prevalence** governs whether the clicking loop functions at all. Median
positives found in 150 votes is 4–11. One trace holds 3 positives for **120
consecutive votes**; the slowest successful cell needed **80 votes to find its
first**; seven cells never found one.

**Dataset saturation** is worth stating so it is not mistaken for a result.
`caltech101_m` is at ceiling for all four configurations *and* for text (AP
1.000). It characterizes the floor case — everything works when the task is easy
— and nothing else.

---

# Failure modes observed

| mode | rate | where |
|---|---|---|
| **Total starvation** — no positive in 150 votes, cell emits nothing | 7 / 189 (3.7 %) wave 1; 0 / 99 wave 2 | rarest categories (`ball` 51/4193, `refrigerator` 101/4952, `sports ball` 169/4952) |
| **Cold-start default threshold** (`too_few_default`) | 1–7 % wave 1; **17–20 %** on the sub-patch band | worst on `dinov3_patch` / small boxes |
| **Degenerate step** | 0.04 % wave 1; **1.96 %** wave 2 | small/medium boxes |
| **Regret rising with votes** | 1 arm | `vg_box_small × dinov3_patch` (0.129 → 0.196) |
| **Cut fallback** | **0 / 40,580** | never observed |

A starved cell is **silent**: no row is ever emitted with `n_good == 0`, so it
writes a header and exits 0. That is how these seven were nearly lost — they are
reported, not excluded, and every average above is conditioned on the run having
produced data at all.

---

# Caveats

- **Wave 2's category selection was collapsed by my error.** The scale-band
  selector was left on for datasets already banded by box size, so it re-banded
  within each set: 5 / 4 / 2 categories out of 40 available, with `vg_box_medium`
  resting on two. The scale trend is large enough to survive it, but the medium
  row is not a point estimate and no between-band difference under ~0.05 should
  be read as real. The re-run (270 cells, 10 categories per set) is in flight.
- Text queries are **raw category names** (`car_side`, `sports ball`) and
  `embed_text_enriched` was not used, so text numbers are a lower bound.
- 3 seeds. Differences under ~0.02 in cost are not resolvable here.
- `caltech101_m × dinov3_patch` is a pairing not present in `dev`.
- COCO's `sub_patch` band had 1 candidate against a target of 2, and that
  category (`sports ball`) is one of the two that starved.

# What this points at next

1. **Compose typing and clicking** rather than choosing: seed a detector from a
   text ranking, especially for `dinov3_patch`, which cannot be typed at.
2. **`dinov3_patch`'s unconverted ranking advantage** — a positive
   `rule_inefficiency` on large boxes is a specific, addressable calibration
   defect, not a property of the representation.
3. **Acquisition is the scarce resource, not the cut rule** — `rule_inefficiency`
   is already negative nearly everywhere.
4. **Make a starved run say so** — a `starved` column and a warning; this is the
   shape that hid #2877.

---

# The constraint scenarios

Nobody gets to pick freely. Some environments cannot run the premium encoder;
some users will not draw boxes; real text queries are not the toy ones. So the
comparison that matters is *within* a constraint, not across.

Region voting needs **both** a patch embedder and a user willing to draw. That
makes the interaction axis measurable independently of the encoder axis, which
every earlier table in this report confounded:

| | binary only (Good/Bad on whole images) | draws boxes |
|---|---|---|
| `siglip` | measured | impossible — no patch grid |
| `siglip2_l` | measured | impossible |
| `dinov3_patch` | **measured (array 496762)** | measured |

`dinov3_patch` binary was run on the *same* datasets, categories, seeds and
splits as its box-drawing counterpart — cell-for-cell paired, differing only in
whether a box is dragged.

## Under a binary interaction, DINOv3 is the worst of the three

Deep regime (t ≥ 100), all binary except the last row:

**`visual_genome_m`**

| configuration | cost | fpr | fnr | AP | AUROC |
|---|---:|---:|---:|---:|---:|
| `siglip2_l` binary | **0.3666** | 0.2438 | 0.1228 | 0.457 | 0.899 |
| `siglip` binary (shipped default) | 0.3918 | 0.3006 | 0.0912 | 0.428 | 0.898 |
| `dinov3_patch` binary | **0.4838** | 0.3069 | 0.1769 | **0.402** | 0.856 |
| *`dinov3_patch` with boxes* | *0.3242* | *0.1819* | *0.1423* | *0.525* | *0.913* |

**`coco_val`**

| configuration | cost | fpr | fnr | AP | AUROC |
|---|---:|---:|---:|---:|---:|
| `siglip2_l` binary | **0.2019** | 0.1154 | 0.0865 | 0.711 | 0.955 |
| `siglip` binary (shipped default) | 0.2177 | 0.0711 | 0.1466 | 0.695 | 0.942 |
| `dinov3_patch` binary | **0.2930** | 0.2067 | 0.0863 | **0.642** | 0.937 |
| *`dinov3_patch` with boxes* | *0.1524* | *0.0920* | *0.0604* | *0.787* | *0.976* |

**Strip the boxes and the expensive encoder finishes last** — on cost *and* on
ranking, on both datasets, against the cheap shipped default:

| | VG | COCO |
|---|---:|---:|
| `dinov3` binary vs `dinov3` boxes | **+0.160 cost, −0.123 AP** | **+0.141 cost, −0.145 AP** |
| `dinov3` binary vs `siglip` binary | **+0.092 cost, −0.026 AP** | **+0.075 cost, −0.053 AP** |

So essentially the whole DINOv3 advantage reported earlier in this document is
**box supervision, not encoder quality.** Read as a constraint: *if your users
will not draw boxes, DINOv3 is not a weaker version of the win — it is worse
than the default you already ship.*

The mechanism is consistent with what the models are. DINOv3 is self-supervised
and vision-only: its strength is spatial correspondence between patches, and its
whole-image vector was never trained to separate semantic categories the way
SigLIP's language-contrastive embedding was. Give it a region to point at and
that spatial strength is usable; take the region away and you are using the part
of it that is weakest — which is also why it has no text tower to fall back on.

## What this says per constraint

**Compute-limited (stuck on `siglip`).** You lose ~0.026 cost against
`siglip2_l` on VG and ~0.016 on COCO — small. Your characteristic failure is
**over-inclusion**: fpr is 3.3× fnr on VG (0.301 vs 0.091), the most lopsided
error budget in the study. The lever is the operating point, not the encoder.
And `siglip` has a text tower, so a typed query gives you a usable ranking for
free — see the text section.

**Box-averse (users answer Good/Bad only).** Your ceiling is `siglip2_l`, not
DINOv3, and the gap to the shipped default is small (0.037 on VG, 0.016 on
COCO). Spending on a patch encoder you cannot point at makes things **worse**.
This is the clearest actionable finding in the report.

**Can draw boxes and can afford DINOv3.** The advantage is real (−0.042 cost vs
`siglip2_l` on VG, −0.050 on COCO) and grows as targets shrink (see the box-band
section), but it costs ~10× wall-clock, has the worst cold start
(`too_few_default` 7.2 % on VG, 18 % on the sub-patch band), and cannot be
seeded from text.

## Starvation is a property of the data, not the interaction

The binary arm starved on exactly the same two cells as the box-drawing arm —
`coco_val`/`refrigerator`/seed 0 and `coco_val`/`sports ball`/seed 1 — 2 of 45.
Identical categories and seeds. Whether the user draws boxes has no bearing on
whether Autopilot ever surfaces a first positive; that is set by prevalence and
the split.

---

# The datasets, their classes, and what the model actually gets wrong

## What the `vg_box_*` sets are

Not size tiers. `visual_genome_m`'s `_m` is a *dataset size* tier and says
nothing about boxes; these three are a **box-area axis**, built specifically so
the scale question could be asked.

Bands are fractions of image area, anchored to the patch embedder's geometry
rather than chosen round numbers — one DINOv3 patch is 1/196 of the image and
the smallest HAC leaf is 1/12:

| set | box area | meaning |
|---|---|---|
| `vg_box_small` | 0 → **1/196** (0.51 %) | below what the patch grid can resolve at all |
| `vg_box_medium` | 1/196 → **1/12** (8.3 %) | resolvable by patches, smaller than one HAC leaf |
| `vg_box_large` | 1/12 → **0.80** | above 80 % a box is not a region, it is the image (mirrors `MAX_VOTED_AREA`) |

Construction (`scripts/experiments/pile/scan_vg_boxes.py`, PR #3123):

- Scans the **whole VG source** — all ~108k images across `VG_100K` and
  `VG_100K_2`, with the full free-text object vocabulary from `objects.json`.
  Not the demo pipeline, which uses 100 curated categories over a 4 % slice.
- `objects.json` stores boxes in **pixels** and carries no image dimensions, so
  areas are normalised against dimensions read from each JPEG header.
- **40 categories and 12,000 images per band**, categories stratified *within*
  the band so a band is not silently all one size.
- Categories whose union box is >1.5× a single instance are dropped — scattered
  instances are not a region a user would drag.
- Minimum 50 images per category.

The motivating number: the demo vocabulary puts **5** categories in the
sub-patch band; the full source has **643**. The "starved sub-patch band" was a
vocabulary artefact.

## Classes used, by dataset

**Wave 1** — `visual_genome_m` (4,193 images, scale-banded): `ball` (51),
`bed` (100), `bus` (67), `cat` (30), `laptop` (60), `nose` (146), `sink` (60),
`sky` (793).

`coco_val` (4,952, scale-banded): `bear` (49), `bed` (149), `cat` (184),
`clock` (204), `microwave` (54), `refrigerator` (101), `sports ball` (169).

`caltech101_m` (838, prevalence-spread): `airplanes` (228), `car_side` (35),
`grand_piano` (28), `starfish` (24), `ibis` (23), `cougar_face` (20).

**Wave 2, first run** (scale bands — the collapsed selection):
`vg_box_small`: `hands` (799), `lips` (155), `mask` (154), `mustache` (115).
`vg_box_medium`: `chest` (122), `collar` (369) — **only two**.
`vg_box_large`: `barn` (57), `court` (429), `dresser` (94), `sheet` (174),
`station` (157).

**Wave 2, re-run** (prevalence spread, 10 per set):
`vg_box_small`: `nose` (2741), `glasses` (1259), `watch` (581), `camera` (461),
`tip` (333), `outlet` (264), `drain` (178), `mask` (154), `mustache` (115),
`tusks` (52).
`vg_box_medium`: `hair` (2628), `shorts` (987), `clock` (662), `lamp` (590),
`truck` (507), `backpack` (346), `basket` (306), `frisbee` (231), `holder` (160),
`chairs` (116).
`vg_box_large`: `fence` (2621), `hill` (807), `lady` (591), `couch` (483),
`court` (429), `walkway` (255), `runway` (202), `station` (157),
`intersection` (95), `barn` (57).

**Binary-voting arm**: same categories as wave 1's VG and COCO.

## Are the errors the model's, or the labels'?

Aggregate fpr cannot tell those apart, so the final step of five representative
cells was dumped per-media: score, label, threshold, source image id, and every
category the dataset annotates on that image.

**The test.** Pick categories that *cannot* occur without the target — you
cannot have clouds without sky. If the images the model flags are enriched for
those relative to the images it correctly rejects, the "false" positives are
largely un-annotated instances. Enrichment is measured against the model's own
**true negatives**, so it is not confounded by the model merely preferring
outdoor scenes: both groups are dataset-negative, and the only difference is
what the model said.

### `visual_genome_m` / `sky` — the labels are wrong

| | `cloud`/`clouds` on FPs | on true negatives | enrichment |
|---|---:|---:|---:|
| `siglip` (682 FPs / 1021 TNs) | 6.6 % | 0.4 % | **16.8×** |
| `dinov3_patch` (504 FPs / 1199 TNs) | 9.5 % | 0.1 % | **114×** |

Outdoor context (`tree`, `building`, `grass`, `mountain`, `roof`, `water`,
`field`, `road`) appears on **84 %** of false positives against 36–43 % of true
negatives (~2×).

Literal examples — VG annotates the clouds and omits the sky:

```
score   image        annotated categories
0.7075  498364.jpg   cloud, grass, pole, road, sign, truck
0.6568    4056.jpg   building, clouds, grass, pole, tree, water
0.6346    4877.jpg   cloud, tree, trunk
0.6291    4211.jpg   boat, building, bush, cloud, field, leaf, shadow, tree
0.5781    2628.jpg   bird, clouds, leaf, mountain, shadow, tree, trunk
```

**These are missing labels, not model errors.** `sky` is 18.8 % prevalent as
annotated; the true rate is plainly higher. Every `sky` number in this report —
fpr 0.30, cost 0.39 — is therefore a *lower bound on the model* and an upper
bound on its apparent error.

The false negatives look genuine, and differ in kind: images that *do* carry a
`sky` annotation but where sky is a thin strip behind a person or a building —
`713003.jpg` (building, hat, line, man, people, shirt, sky, wall),
`712994.jpg` (ear, face, hair, hand, horse, man, nose, people, shadow, shirt,
sky). Small-region failures, which is consistent with the box-band result.

### `coco_val` / `clock` — the model is wrong

The same test is **not applicable**: COCO's 80-class vocabulary contains no term
that entails `clock`, so there is nothing to be enriched for. But the evidence
points the other way anyway. COCO's annotation is exhaustive over its classes,
the false positives are indoor scenes with no plausible hidden clock
(`chair`; `couch, tv`; `chair, couch, bed, remote, tv`), and only 46 of 2,364
negatives are flagged at all — 1.9 %, versus VG `sky`'s 40 %.

The false negatives are the informative half: images that *do* contain a clock,
scored ~0.003–0.012, and every one is a cluttered scene where the clock is one
small object among many —

```
0.0028  000000350148.jpg  person, carrot, chair, car, dining table, cake, clock, cup, fork, knife...
0.0058  000000084674.jpg  person, couch, book, clock, donut, tv
0.0099  000000441247.jpg  chair, vase, couch, dining table, orange, oven, person, backpack, banana...
```

That is a genuine **scale** failure by a whole-image encoder, and it is the same
mechanism the `vg_box` bands measure directly.

### `visual_genome_m` / `bus` — a threshold collapse

The starkest single failure in the study: **1,210 of 2,030 negatives flagged
(59.6 %) and 0 of 67 positives missed.** The threshold has fallen so far that
almost everything passes. The ranking is not necessarily broken; the cut is.
This is the over-inclusion signature of `siglip` (fpr ≫ fnr) in its extreme
form, on a rare category (67 positives, 3.2 %).

*A caution about one heuristic*: the error report flags false positives carrying
a category name that contains the target, and for `bus` this matched 80 images
annotated **`bush`**. That is a substring coincidence, not evidence — `bush`
does not entail `bus`. It is reported here as a known false lead rather than
quietly dropped, because the same heuristic is genuinely useful for annotation
*granularity* cases and will keep firing.

## What this means for the rest of the report

- **VG-derived numbers are pessimistic by an unknown amount**, worst for common
  scene categories (`sky`, and plausibly `tree`, `building`, `grass`). COCO
  numbers do not have this problem.
- Cross-dataset comparisons of *absolute* cost between VG and COCO are therefore
  not safe. Within-dataset comparisons — which is what every configuration
  contrast in this report is — remain valid, since all configurations see the
  same labels.
- **A label-noise audit belongs upstream of the next VG study**, not inside it.
  The entailment test is cheap, mechanical, and now scripted
  (`scripts/experiments/calibration/label_noise.py`).
