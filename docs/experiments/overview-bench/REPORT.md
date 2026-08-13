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
