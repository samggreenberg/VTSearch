# VTSearch overview benchmark: how each configuration behaves

**Run:** 2026-08-12 · branch `claude/vts-benchmark` · arrays `496044` (wave 1),
`496454` (wave 2), `496673` (wave 2 re-run, drained 2026-08-13 00:00)
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

Wave 1: 189 cells / 26,538 steps. Wave 2: 99 cells / 14,042 steps. **Wave 2
re-run: 270 cells / 37,844 steps** — 10 prevalence-spread categories per box
band, replacing the collapsed 5 / 4 / 2 selection wave 2 ran on.

**Every `vg_box_*` number in this report is from the re-run**; wave 2's are
superseded and kept in `ANALYSIS_TABLES_vgbox.txt` for comparison. The two are
not interchangeable — the re-run's categories are more prevalent (0.047–0.054
against 0.015–0.027), so costs are not comparable *across* the waves even though
the arm ordering *within* each is. Where the two disagree, the re-run is the
larger and better-constructed sample, and three of the report's original box-band
claims did not survive it; each is called out where it appears.

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
| `vg_box_large` | `siglip` | 0.4574 | 0.2719 | 0.1856 | 0.0898 | 0.354 | 0.851 | ~2 min |
| `vg_box_large` | `siglip2_l` | 0.4376 | 0.3342 | 0.1034 | 0.0932 | 0.359 | 0.868 | ~2 min |
| `vg_box_large` | `dinov3_patch` | 0.3669 | 0.2062 | 0.1607 | 0.0928 | 0.405 | 0.906 | ~23 min |
| `vg_box_medium` | `siglip` | 0.6297 | 0.3966 | 0.2331 | 0.1033 | 0.297 | 0.776 | ~2 min |
| `vg_box_medium` | `siglip2_l` | 0.6036 | 0.3675 | 0.2362 | 0.0988 | 0.340 | 0.785 | ~2 min |
| `vg_box_medium` | `dinov3_patch` | 0.4594 | 0.3426 | 0.1169 | 0.1085 | 0.391 | 0.881 | ~15 min |
| `vg_box_small` | `siglip` | 0.6339 | 0.3654 | 0.2685 | 0.0943 | 0.221 | 0.755 | ~2 min |
| `vg_box_small` | `siglip2_l` | 0.6479 | 0.4147 | 0.2332 | 0.1043 | 0.232 | 0.751 | ~2 min |
| `vg_box_small` | `dinov3_patch` | 0.4910 | 0.3395 | 0.1514 | 0.0971 | 0.294 | 0.850 | ~15 min |

The three `vg_box_*` sets are the **re-run** (`496673`); the rest is wave 1. The
box-band rows sit at prevalence 0.047–0.054, the wave-1 rows at 0.026–0.071, and
the superseded wave-2 box rows at 0.015–0.027 — so read down a column within a
band rather than across the table, and never against wave 2's numbers.
Wall-time is the median cell, and a step is what actually differs: **6.4–8.5 s
for `dinov3_patch` against ~0.6 s whole-image**, an 11–13× per-step ratio that
the per-cell figure understates.

**Every wall-time in this report is cache-warm and excludes the encoder.** The
harness sources `pile_env.sh` and reads pre-embedded cells, so no forward pass
runs during a cell — the per-step cost is the 150-vote loop over cached vectors,
and the only encoder-dependent term in it is vector dimension. Read these
numbers as *study* cost, never as the cost of deploying an encoder; the
`dinov3_patch` step is 11–13× not because its backbone is large (it is a ViT-B,
*smaller* than `siglip2_l`) but because each item carries a patch grid. See
[Cost is two different things](#cost-is-two-different-things).

Zero-click typed query, same test splits (`dinov3_patch` has no text tower):

| dataset | embedder | text cost | text AP | text AUROC |
|---|---|---:|---:|---:|
| `caltech101_m` | `siglip` / `siglip2_l` | 0.1612 / 0.0444 | 1.000 / 1.000 | — |
| `coco_val` | `siglip` / `siglip2_l` | 0.3024 / 0.2768 | 0.707 / 0.743 | — |
| `visual_genome_m` | `siglip` / `siglip2_l` | 0.4894 / 0.3936 | 0.496 / 0.544 | — |

---

# Cost is two different things

This report uses "cost" in two senses and they point in opposite directions.
Everywhere except this section, **`cost` = fpr + fnr** — a labelling-error rate,
with no compute in it. The wall-times are *study* cost on cached vectors. Neither
is the cost of running an encoder, and a reader picking a configuration to deploy
needs that third number, which the benchmark cannot see.

Measured separately (2026-08-17, jobs `507149`/`507150`, 384 VG images through
the real `vtscore` decode + processor + forward path):

| stage, V100 · batch 32 · fp32 | `siglip` | `siglip2_l` |
|---|---:|---:|
| decode | 3.9 s (54 %) | 3.4 s (12 %) |
| processor | 2.1 s (28 %) | 1.9 s (7 %) |
| GPU forward | 1.3 s (18 %) | 22.1 s (81 %) |
| throughput | 52.5 img/s | 14.0 img/s |

**Forward-only, `siglip2_l` costs 17× `siglip`** (22.1 s vs 1.3 s), consistent
with ~5× the parameters over ~3.7× the tokens. The pile build logs show only
6.1× end-to-end (830 s vs 135 s for 12,000 images) because a fixed
decode-and-preprocess tax — 82 % of the small model's run — dilutes it. So the
indexing multiplier is a property of the *pipeline*, not the encoder: **~17× of
GPU compute, 2–6× of wall-clock** depending on how much of the pipeline is
overhead. The same multiplier applies per typed query at serve time, where it is
user-visible latency rather than amortised indexing.

The pile was built un-tuned — V100, fp32, batch 32, serial decode — and that is
where the tax comes from, not from anything intrinsic. On an L40S with fp16,
batch 64 and threaded decode the same 12,000 `siglip2_l` images take 112 s
instead of 855 s (7.6×). Four fixes are filed: #3144 (the pile builds on the
cluster's slowest GPU) and #3145 (decode is serial, so the GPU idles 82 % of a
`siglip` run) are numerically safe; #3143 (fp32, no autocast) and #3146 (slow
image processor) perturb the vectors and are gated on a drift experiment. Because
the pile is a shared cache, **it must not be partially rebuilt** — a pile with
some cells fp16 and some fp32 is a confound that would surface months later as an
unexplained arm difference.

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
box bands its AP is 0.354 → 0.297 → 0.221 (large → medium → small) and AUROC
0.851 → 0.776 → 0.755, against `dinov3_patch`'s 0.906 → 0.881 → 0.850. The
degradation is the expected consequence of pooling a whole image into one vector
when the target occupies under 0.5 % of it — and it is *steeper* than the patch
arm's, which is the comparison that matters.

## `siglip2_l` — the premium whole-image encoder

**Behaves like:** `siglip` with a uniformly better ranking and the same *error*
profile — at **~17× the encode compute**. AP is higher on every non-saturated
dataset (VG 0.457 vs 0.428; COCO 0.711 vs 0.695), and its *text* ranking is
better too (VG text AP 0.544 vs 0.496).

Its `cost` (= fpr + fnr) is within 0.03 of `siglip`'s on every dataset, which is
what "same cost profile" means here and **all** it means. It is not a statement
about compute: the ~110 s per run is identical to `siglip`'s only because the
harness reads pre-embedded cells (see [Cost is two different
things](#cost-is-two-different-things)). `siglip2_l` is
`siglip2-so400m-patch14-384` against `siglip`'s `siglip-base-patch16-224` — ~5×
the vision-tower parameters over ~3.7× the tokens.

**It rebalances the error budget rather than only shrinking it.** On VG it moves
fpr 0.301 → 0.244 while fnr rises 0.091 → 0.123. It is a less trigger-happy
encoder, not merely a more accurate one.

**It inherits `siglip`'s scale failure intact.** AP across box bands 0.359 →
0.340 → 0.232, tracking `siglip`'s 0.354 → 0.297 → 0.221 far more closely than
either tracks the patch arm. Capacity does not substitute for geometry: whatever
a bigger whole-image encoder buys, it is not the ability to see a sub-patch
object.

## `dinov3_patch` — patch geometry, and region voting where boxes exist

**Behaves like:** the best ranker in the study, and — once measured on a proper
category sample — the best *cost* on every boxed set too, at 11–13× the compute
per step (15–23 min per box-band run against ~2 min; 17–19 min on wave 1's sets).

Note which compute that is. Its per-step cost is 11–13× because every item
carries a patch grid (a 3.65 GB cell against `siglip2_l`'s 59 MB); *encoding* it
is cheap — a ViT-B at 297 s per 12,000 images, **2.8× faster than `siglip2_l`'s
830 s** on the same job. DINOv3 is the expensive arm to search and the middling
arm to index; `siglip2_l` is the reverse. The two rankings are inverted, so
neither the wall-time column nor the per-step figure predicts what a deployment
pays.

**Its ranking is the best measured, and the margin grows as targets shrink.** AP
by box band: 0.405 → 0.391 → 0.294 against `siglip2_l`'s 0.359 → 0.340 → 0.232,
a margin of +0.046 → +0.051 → +0.062. AUROC 0.906 / 0.881 / 0.850 vs 0.868 /
0.785 / 0.751. The mechanism is straightforward: box supervision only carries
information the whole-image vector lacks when the box is a small fraction of the
frame. A box covering a third of the image *is* approximately the image.

**It also wins on cost in every band**, by 0.071 (large), 0.144 (medium) and
0.143 (small) against the best whole-image arm in that band. Its error budget is
the reason: `fnr` is roughly half the whole-image arms' on medium and small
(0.117 / 0.151
against 0.233–0.269), so the patch geometry is buying recall on exactly the
targets a whole-image vector dilutes.

> **Wave 2 said otherwise on the large band, and the re-run overturns it.** On
> the 5-category wave-2 sample, `dinov3_patch` on `vg_box_large` was the *only
> arm in the study with a positive `rule_inefficiency`* (+0.021), its regret was
> 0.124 against `siglip2_l`'s 0.071, and the ranking and threshold effects
> cancelled to an identical cost (0.3243 vs 0.3234). At 10 categories none of
> that reproduces: `rule_inefficiency` is **−0.011**, regret is 0.093 against
> 0.093 — level — and the cost gap opens to 0.367 vs 0.438 in DINOv3's favour.
> The "unconverted ranking advantage" was a property of five large-box
> categories, not of the arm. What survives is the *ranking* claim, which the
> larger sample strengthens.

**Where a positive `rule_inefficiency` does show up** is the **medium** band, on
all three encoders (+0.017 `dinov3_patch`, +0.010 `siglip`, +0.004 `siglip2_l`)
— the only band where all three are positive, i.e. where the shipped cut rule is
beaten by its own in-sample optimum regardless of representation. (`vg_box_small
× siglip` is the one other positive, at +0.005; every remaining arm is negative.)
That is a narrower and better-supported version of the same finding: the defect
tracks the *band*, not the encoder.

**Where it comes apart:** cold start and cost. `too_few_default` is 7.2 % on VG
(worst of the three) and 5.8–16.1 % across the box bands — though on the
sub-patch band the worst cold start is **`siglip2_l`'s** (16.1 %), not
DINOv3's (9.9 %), another wave-2 claim the re-run reverses. Regret rising with
votes does not reproduce either: on `vg_box_small × dinov3_patch` regret now
*falls* 0.117 → 0.097 over the ramp. And it has **no text tower**, so there is
no zero-click entry point for it at all.

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
bands runs 0.367 (large) → 0.459 (medium) → 0.491 (small), and AP 0.41 → 0.39 →
0.29. Sub-patch retrieval is hard for every configuration; the patch geometry
reduces the damage but does not remove it. This is the first measurement of that
band on a real sample — the full VG vocabulary has 643 sub-patch categories
against 5 in the demo vocabulary.

The monotone trend held across both samples, but its *shape* changed: on wave 2's
five-per-band selection the large→medium step was a cliff (0.323 → 0.590), and on
ten prevalence-spread categories it is a slope (0.367 → 0.459 → 0.491). Scale is
still the strongest axis; it is not the step function the first sample drew.

**Prevalence** governs whether the clicking loop functions at all. Median
positives found in 150 votes is 4–11. One trace holds 3 positives for **120
consecutive votes**; the slowest successful cell needed **80 votes to find its
first**; twelve cells across the three waves never found one.

**Dataset saturation** is worth stating so it is not mistaken for a result.
`caltech101_m` is at ceiling for all four configurations *and* for text (AP
1.000). It characterizes the floor case — everything works when the task is easy
— and nothing else.

---

# Failure modes observed

| mode | rate | where |
|---|---|---|
| **Total starvation** — no positive in 150 votes, cell emits nothing | 7 / 189 (3.7 %) wave 1; 0 / 99 wave 2; **5 / 270 (1.9 %)** re-run | rarest categories (`ball` 51/4193, `refrigerator` 101/4952, `sports ball` 169/4952, `intersection` 95/12000) |
| **Cold-start default threshold** (`too_few_default`) | 1–7 % wave 1; **5.8–16.1 %** across the box bands | worst on small boxes; worst *arm* is `siglip2_l`, not `dinov3_patch` |
| **Degenerate step** | 0.04 % wave 1; 1.96 % wave 2; **2.01 %** re-run | small/medium boxes |
| **Regret rising with votes** | **0 arms** (was 1 in wave 2, did not reproduce) | — |
| **Cut fallback** | **0 / 78,424** | never observed |

A starved cell is **silent**: no row is ever emitted with `n_good == 0`, so it
writes a header and exits 0. That is how the first seven were nearly lost — they
are reported, not excluded, and every average above is conditioned on the run
having produced data at all. The re-run's five were caught the same way, by the
analyzer counting loaded cells separately from files found (265 vs 270).

---

# Caveats

- **Wave 2's category selection was collapsed by my error, and the re-run
  replaces it.** The scale-band selector was left on for datasets already banded
  by box size, so it re-banded within each set: 5 / 4 / 2 categories out of 40
  available, with `vg_box_medium` resting on two. The re-run (270 cells, 10
  categories per set) is what every `vg_box_*` number here now comes from. The
  scale trend survived; three narrower claims built on the medium and large rows
  did not, which is the caveat working as intended rather than a surprise.
- **The two box-band samples differ in prevalence**, 0.047–0.054 against
  0.015–0.027, because the re-run spread categories by prevalence rather than
  re-banding by size. Absolute costs therefore moved for reasons unrelated to
  scale, and only within-sample arm ordering carries across.
- Text queries are **raw category names** (`car_side`, `sports ball`) and
  `embed_text_enriched` was not used, so text numbers are a lower bound.
- 3 seeds. Differences under ~0.02 in cost are not resolvable here.
- `caltech101_m × dinov3_patch` is a pairing not present in `dev`.
- COCO's `sub_patch` band had 1 candidate against a target of 2, and that
  category (`sports ball`) is one of the two that starved.

# What this points at next

1. **Compose typing and clicking** rather than choosing: seed a detector from a
   text ranking, especially for `dinov3_patch`, which cannot be typed at.
2. **The medium band's positive `rule_inefficiency`** — on `vg_box_medium` the
   shipped cut rule is beaten by its own in-sample optimum on all three encoders
   (+0.017 / +0.010 / +0.004), the only band where that happens. This replaces
   the wave-2 version of this item, which pinned the defect on `dinov3_patch` and
   the large band and did not reproduce.
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
`siglip2_l` on VG and ~0.016 on COCO — small, and what you keep is real:
`siglip2_l` costs **~17× the GPU compute per image** to index a corpus and the
same multiplier per typed query at serve time. Nothing in the benchmark's
wall-times shows that, because the encoder ran once into the pile long before any
cell did. Your characteristic failure is **over-inclusion**: fpr is 3.3× fnr on
VG (0.301 vs 0.091), the most lopsided error budget in the study. The lever is
the operating point, not the encoder. And `siglip` has a text tower, so a typed
query gives you a usable ranking for free — see the text section.

**Box-averse (users answer Good/Bad only).** Your ceiling is `siglip2_l`, not
DINOv3, and the gap to the shipped default is small (0.037 on VG, 0.016 on
COCO) — and costs ~17× the encode compute to close. Spending on a patch encoder
you cannot point at makes things **worse**. This is the clearest actionable
finding in the report.

**Can draw boxes and can afford DINOv3.** The advantage is real (−0.042 cost vs
`siglip2_l` on VG, −0.050 on COCO) and grows as targets shrink (see the box-band
section, where the re-run puts it ahead on cost in all three bands by 0.071 /
0.144 / 0.157), but it costs 11–13× per step, has the worst cold start on VG
(`too_few_default` 7.2 %, worst of the three) and 9.9 % on the sub-patch band,
and cannot be
seeded from text.

## Starvation is a property of the data, not the interaction

The binary arm starved on exactly the same two cells as the box-drawing arm —
`coco_val`/`refrigerator`/seed 0 and `coco_val`/`sports ball`/seed 1 — 2 of 45.
Identical categories and seeds. Whether the user draws boxes has no bearing on
whether Autopilot ever surfaces a first positive; that is set by prevalence and
the split.

**The re-run extends this from the interaction to the encoder.** Its five starved
cells are two (category, seed) pairs, and they starve *across embedders*:
`vg_box_small`/`tip`/seed 2 starved on all three (`siglip`, `siglip2_l`,
`dinov3_patch`), and `vg_box_large`/`intersection`/seed 0 on two of three. The
third — `intersection` on `siglip` — is the exception that shows the mechanism
rather than breaking it: it survived, but emitted only 63 of 150 steps, i.e. it
found its first positive around vote 87. Starvation is set by the draw, not by
what is doing the ranking. `intersection` is the rarest category in the large
band (95 of 12,000, prevalence 0.008); `tip` is a free-text label whose positives
are genuinely ambiguous.

This is worth stating as a design consequence: **a starvation fix has to act on
acquisition, because changing the encoder demonstrably does not move it.**

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

**Wave 2, re-run** (prevalence spread, 10 per set — the source of every
`vg_box_*` number above; `tip` and `intersection` are the two that starved):
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
