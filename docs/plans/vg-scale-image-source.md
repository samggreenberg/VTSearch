# What `vg_scale` should draw its images from

**The question this plan holds open:** Visual Genome was chosen as `vg_scale`'s
image pool, and the construction has since migrated almost entirely off VG's
annotation without anyone re-asking whether VG is still the right *source*. This
records what VG demonstrably contributes today, what the cleanup costs, and what
the alternatives are — so the source decision is made on measurement rather than
on inertia.

It is a source question only. The band construction lives in
[`vg-scale-bands-and-corrections.md`](vg-scale-bands-and-corrections.md); the
plan to answer the off-COCO half by hand lives in
[`vg-scale-exhaustive-annotation.md`](vg-scale-exhaustive-annotation.md).

## Background: how little of VG is left in the construction

Each of these arrived as a separate, well-argued issue. Together they amount to
a migration nobody planned:

| Layer | Source today | Where |
|---|---|---|
| Class list | 100% COCO — all 25 `SCALE_CLASSES` are COCO-2017 classes | `pile_config.SCALE_CLASSES` |
| Labels on the overlap | COCO's, overwriting VG's | `anchor_to_coco`, `vg_scale.py` |
| Negative pool | 100% COCO-scored since #3670; VG-silence contamination 0 by construction | `pile_config.SCALE_NEG_COMPOSITION` |
| Positives | ~57% COCO-anchored, ~43% off-COCO | `pile_config` §`SCALE_NEG_COMPOSITION` |

So VG's **free-text vocabulary buys zero classes**, while being the direct cause
of `SCALE_VG_NAMES`, `SCALE_VG_AMBIGUOUS`, `SCALE_VG_NAMES_AUDITED`,
`name_evidence.py`, `coco_folds.py`, `vg_name_families.py`,
`scan_name_overlap.py`, `name_misspellings.py`, `pool_contamination.py` and
`withheld_difficulty.py`. There is no cheaper read available: every one of VG's
2,516,939 objects carries a `names` list of length **one**, so there is no
synonym field to consult instead (#3618).

What that has cost, as of 2026-09-17: **4,709 correction rows** against VG's own
annotation, **5,904 human judgements** in a single four-day review, and a
double-digit issue count. On top of the structural error rates the corrections
log records — **8.3%** of stored-box positives box a smaller instance than the
frame's main one and **1.2%** box the wrong object (#3924, #3925) — against VG's
measured recall over *C* of **0.61**.

VG still carries real weight in exactly one place: the **~43% of positives that
are off-COCO**. That is the whole of the live argument for keeping it, and it is
a *supply* argument, not a diversity or vocabulary one.

## The open questions

<!-- item-sep -->

- [ ] #3983 — the "COCO half does not reach" measurement was taken over a pool VG defines (Opus 4.8)

<!-- item-sep -->

- **Price VG's non-COCO half as diversity, not as supply.** `anchor_to_coco`'s
  docstring justifies keeping the whole of VG on the grounds that restricting to
  the COCO half "would make this a COCO subset with extra steps, losing VG's
  non-COCO diversity for nothing". That claim has never been measured, and it is
  load-bearing: it is the standing answer to every proposal to narrow the pool,
  and #3983 will not settle it (that one asks about counts). Diversity here has
  a testable meaning — whether off-COCO positives are *harder*, or differently
  distributed in scene composition, than COCO-anchored ones at matched band and
  class. `provenance_probe.py` and `provenance_shortcut.py` already read
  provenance off the vectors (AUC 0.53–0.56) and are the natural instrument. If
  the answer is "no measurable difference", the docstring should say so and stop
  being cited; if yes, it becomes the real argument for VG and the vocabulary
  cost is worth paying knowingly. (Sonnet 5)

<!-- item-sep -->

- **Decide whether the class axis wants a source COCO cannot give.** Every
  question above is about doing the *current* 25 classes better. A different
  question is whether the benchmark wants classes COCO does not have at all —
  #3603 closed with "any design that needs a wider easy end needs a different
  image source", and that has not been taken up. The candidates are ranked in
  the next section. This is a scope decision for the owner before it is a
  measurement; it is recorded here so that the next person to feel the pull
  toward VG's free-text vocabulary finds the comparison already made. (human)

<!-- item-sep -->

## If the class list is to grow beyond COCO's 80

Ranked for *this* construction, whose requirements are unusual: per-class
evidence of **absence**, box-size bands with real small-band supply, and a
vocabulary whose boundaries are written down rather than inferred.

**LVIS — the first choice, and by some distance.** It is built on the *same*
COCO images, so `COCO_ROOT`, `coco_anchor.py` and the box coordinate space all
carry over unchanged; it also retires a whole bug class, since VG ships
downscaled 500px copies of COCO's 640px originals and every box therefore needs
`box_dims` renormalisation (the origin of #3281). Its 1,203 categories are
WordNet synsets **carrying written definitions**, which is precisely the artifact
`SCALE_CLASS_RULES` exists to hand-write and precisely what `book`/magazines and
`cell phone`/landlines split on. Most importantly it is **federated**: every
category ships an explicit *negative image set* — images a human verified do not
contain it. That is `labels_exhaustive` as a delivered artifact rather than
something manufactured by a 3,391-image pass. Its long tail also skews
small-object, which is the band that binds everywhere else.

Two real caveats. Federated is **not** per-image exhaustive, so #3667's
cross-class negative rule has to be restated per category rather than inherited.
And tail categories are thin by design — at 100 positives per band the usable
vocabulary is the head few hundred, not all 1,203.

**Objects365 v2** — 365 densely-boxed classes over ~2M images. Best raw supply
and no vocabulary cleanup at all, but no explicit negatives and no COCO overlap,
so it is a new pool to embed and a domain shift to argue about.

**Open Images V7** — 600 boxable classes over ~1.9M images, and the one source
with **human-verified negative image-level labels** at scale, which is the single
artifact VG most lacks. Costs: box exhaustiveness is per-verified-class rather
than per-image, the `group-of` flag complicates a box→size band rule, and Flickr
framing differs from COCO's.

**ADE20K** — the only genuinely exhaustive option (every pixel, ~3.6k classes)
but ~27k images, far too thin to build cells from. Its real use here is as a
**gold audit set** for silence and box-quality rates, which is what #3696's
exhaustive human pass was hand-rolling against no reference.

## What makes this hard to act on

Recorded so that a future reader does not mistake inaction for oversight:

- Five-plus studies (#3115, #3196, #3287, #3290, #3318, #3319) are conditioned on
  the shipped cells. A source change makes those numbers unreproducible, in the
  way `pile_config` already warns a rebuild does.
- The corrections and `human_record` verdicts are the pile's **one
  non-rebuildable artifact** (#3729) and are keyed to VG image ids. Any pool
  change has to carry them across without invalidating one.
- The off-COCO 43% is live supply. Dropping it takes the thinnest cell from
  #3818's 1.4x over-subscription to roughly 0.8x — a fail. Nothing narrows the
  pool until #3983 says something else can fill it.

## Related

- [`vg-scale-exhaustive-annotation.md`](vg-scale-exhaustive-annotation.md) — the
  plan to answer the off-COCO half by hand. #3983 questions its founding
  measurement directly; read the two together.
- [`vg-scale-bands-and-corrections.md`](vg-scale-bands-and-corrections.md) — the
  band construction and correction loop any source change must preserve.
- [`docs/experiments/2026-09-17-vg-scale-corrections/REPORT.md`](../experiments/2026-09-17-vg-scale-corrections/REPORT.md)
  — the running log of what VG gets wrong, and the source of the rates quoted above.
- [`docs/experiments/2026-09-03-vg-scale-classes/REPORT.md`](../experiments/2026-09-03-vg-scale-classes/REPORT.md)
  — why VG's easy end cannot be widened, and the definition-risk instrument.
