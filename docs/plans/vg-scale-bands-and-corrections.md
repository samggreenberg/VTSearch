# Same-class scale bands for VG, and the label corrections they need

Refs [#3156](https://github.com/samggreenberg/VTSearch/issues/3156).

## Why the current `vg_box_*` sets can't answer the scale question

`scan_vg_boxes.py` measures each VG category's **median** voted-box area, and
`build_pile.py::_band_categories` assigns the category to whichever band that
median lands in. A category therefore lives in exactly one band, and the three
sets carry disjoint vocabularies — the overview benchmark ran `nose`, `glasses`,
`watch` against `fence`, `hill`, `lady`
([`docs/experiments/overview-bench/REPORT.md`](../experiments/overview-bench/REPORT.md)).

So the published small-vs-large gap confounds **box size** with **class
identity**: it cannot distinguish "small regions are hard" from "noses are
harder than fences". The question worth asking is *how well can we find buses in
the middleground*, which needs one class list held fixed while only size varies.

## The construction that does answer it

One image pool *I* and one class list *C*. For each class `c` and band `B`,
every image in *I* is one of three things:

| | condition | eval role |
|---|---|---|
| **positive** | the voted box for `c` falls in band `B` | positive |
| **negative** | no instance of `c` at any size | negative |
| **excluded** | `c` present, but at some *other* size | dropped from the cell |

The excluded state is the part that does not exist today.
`vtscore.eval.labels.media_is_positive` is closed-world two-valued, so a
wrong-band image would silently become a negative and the detector would be
penalised for finding a real bus — the exact failure #3156 is about. The cheap
implementation is to filter the media pool **once per cell** rather than thread
a third value through every scorer: `calibration/prepare_data.py`,
`voting_iterations.py`, `text_baseline.py`.

**Size means the union box.** `region_box_for_category` already returns the
union over a category's instances, because that is what one Good vote drags.
An image holding one foreground bus and three background buses is therefore a
foreground-bus image — which is the honest reading of "find buses in the
middleground".

**One pickle, not three.** Since *C* and the negative pool are shared, the bands
differ only in which images are positive vs excluded. Build a single pickle and
carry the band on the category name (`bus@small` / `bus@medium` / `bus@large`):
a harness cell is already `(dataset, category)`, so this needs no harness
change, cuts embedding ~3× (the `dinov3_patch` cells run ~1400 s each), and
makes small-vs-large paired on *identical* negatives. Sample a fixed
`n_pos`/`n_neg` per cell so prevalence cannot drift between bands — unequal
prevalence is what already made wave 1 and wave 2 non-comparable.

## Which classes are fit to be in *C*

Supply is necessary and nowhere near sufficient. A scale study asks two things
of a class that mere objecthood does not, and `pile_config.is_object_category`
— which defines the published `vg_box_*` sets — tests neither:

- **Its size must be its own.** A part's box is set by its host, so a "small
  nose" is just a distant face. Banding it measures the host's distance and the
  arm quietly becomes a different experiment.
- **Its absence must be checkable.** Negatives are ~95% of *I* and rest on "no
  instance here". For a part that is unverifiable at any scale — every image
  with a person has a nose whether or not VG annotated one — so the negatives
  are poisoned by construction and no amount of review repairs them. That is
  the worst case for the correction pass, not a candidate for it.

`pile_config.scale_study_exclusion` layers the stricter policy on top, keeping
`is_object_category` intact so the published sets stay reproducible. It rejects
**parts** (`nose`, `tip`, `hair`, `collar`, `roof`, `tree trunk`), **places**
(`court`, `station`, `intersection` — a location has no principled box extent),
**polysemous** bare names (`trunk`, `bat` — one string, several objects, so the
class cannot be scored as one; matched whole-name, since a modifier is what
resolves the ambiguity), and **pervasive** classes, measured against
`PERVASIVE_PREVALENCE` rather than listed. `sky` needs no rule: it is already a
mass noun and never entered `vg_box_*` in the first place.

The shortlist **reports** these with reasons instead of dropping them silently.
The list is curated, so a wrong exclusion shrinks the study and a wrong
inclusion changes what it measures — both need a human to look.

## Near-synonyms: measure the vocabulary, don't trust it

`glasses` / `sunglasses` / `reading glasses` would be a genuinely interesting
fine-grained target, but only if the labels can be trusted, and free text gives
no guarantee that they can: the names might be nested, disjoint, or overlapping
per annotator, and those want three different experiments.

`scan_name_overlap.py` decides it from geometry rather than from strings. On
images where both names appear, it asks how often an `a` box lands on the same
pixels as a `b` box (IoU ≥ 0.5) — same pixels under two names means one object
annotated twice:

| overlap | verdict | consequence |
|---|---|---|
| high both ways | **alias** | one label split arbitrarily; each name's negatives are poisoned by the other until merged |
| high one way only | **subtype** | a real fine-grained pair; the broad name's negatives are sound |
| near zero | **distinct** | different objects that merely co-occur |
| never co-annotated | **untestable** | genuinely unrelated and systematically split-by-annotator are indistinguishable here |

This is the principled version of the heuristic the overview benchmark tripped
over — flagging false positives whose annotations *contain* the target name,
which for `bus` matched 80 images annotated `bush`. String similarity is not
evidence about objects; box geometry is. (`--names bus,bush` refutes that lead
directly.)

## How corrections get recorded

Record them in **VG's own shape** — `(image_id, class, box)` — and merge over
`objects.json` *before* banding. A correction must be able to move an image
between bands, so an eval-time label overlay cannot do the job; a build-time
merge makes the scan, the bands, prevalence and region voting all pick it up.

Record **verdicts, not corrections**: every reviewed `(image, class)` pair gets
a row, whether or not it disagrees with VG. Corrections are then derived, and
review coverage falls out for free — without it, "no bus annotated" is
indistinguishable from "nobody looked", and every corrected metric is biased by
an unknown amount.

VTSearch supplies the loop with no new plugin code: a Good vote already carries
`region_box` through `LabeledElement` and the label export, so a `server_folder`
import → vote → `server_json_file` export round-trip emits exactly the required
record. The box fixes presence *and* band membership in one gesture.

**The negatives are the expensive half.** They are ~95% of *I* and rest on an
absence claim, which is precisely what VG cannot support (`498326.jpg` is
annotated `car, clouds` and has a bus front and centre). Review them in
descending detector score, plus a uniform random stratum so the residual noise
rate after review is bounded rather than unknown.

## What the scan measured (2026-08-17)

Run on the GRID as jobs `509905` and `509971` (CPU-only, ~70 s each) against the
full VG source; artifacts under `/expscratch/sgreenberg/vgscale-3156/`
(`vg_box_scale_bands.json`, `shortlist_*.json`, `name_overlap.json`).

**The construction is viable.** Of 4628 categories with >= 20 images, 534 hold
>= 50 images in *every* band and 310 hold >= 100 — so one class list held fixed
across three sizes is not supply-limited.

**But the per-class scatter filter cost the study its best classes, `bus`
included.** `union_inflation` is a per-*class* median, and a class that often
appears several times per image fails it however compact any single image is:
`bus` scores 1.71 against a 1.5 cap and is dropped, together with 30 of the 65
COCO classes present. That is the wrong 30 to lose — a shared class is the only
one whose VG miss rate can be measured against COCO's exhaustive annotation for
free.

Scatter is a property of an *image*, not of a class, so the scan now also counts
an image only when its union box is within `BAND_MAX_INFLATION` of that image's
own largest instance (`bands_compact`; `shortlist_scale_classes.py --compact`).
The non-compact remainder is *excluded* from the cell, exactly like a wrong-band
image — it is not turned into a negative.

| floor 50, all three bands | classes | of which COCO |
|---|---:|---:|
| per-class median inflation | 196 | 15 |
| per-image compact union | 305 | 41 |

Nothing is lost by the switch: every COCO class the per-class rule kept survives
it, and 26 more come back. It is cheap for the recovered classes — 90% of `bus`
images are compact, giving 112 / 782 / 2137 images across small / medium /
large. It is expensive only for classes that really are scattered (`person`
0.66, `car` 0.71), which is the intended effect.

**The eyewear arm does not earn a place.** `glasses` vs `sunglasses` overlap at
IoU >= 0.5 on only 0.20 of boxes each way over 280 co-annotated images —
`mixed`, not the clean `subtype` a fine-grained arm would need, and not the
alias the overview benchmark assumed either. `eyeglasses` vs `glasses` does come
back `alias` (0.32 / 0.31), so those two names must be merged before either is
used. Separately, `bus` vs `bush` is refuted outright: 87 co-annotated images,
**0.00** box overlap both ways. String similarity was never evidence.

## Open work

<!-- item-sep -->

- **Pick *C*.** ~~Run the supply scan~~ — done, see above; the ranking is in
  `shortlist_compact_floor100.json`. Choosing *C* from it, including a human
  pass over the reported exclusions, is still the gate on everything below.
  (human)

<!-- item-sep -->

- ~~**Decide whether a fine-grained pair earns a place in *C*.**~~ Measured:
  no pair in the top 30 plus the eyewear cluster came back `subtype`, so the
  fine-grained arm is dropped and `eyeglasses` is merged into `glasses`.

<!-- item-sep -->

- **Three-valued labels in the eval harness.** `media_is_evaluable(media,
  category)` beside `media_is_positive`, and per-cell pool filtering at the
  three entry points above. Needs a `vtscore/eval/labels.py` test for the
  wrong-band case, and an `MIRRORS` review — the harness's notion of ground
  truth is changing, not the app's algorithm. (Opus 4.8 — a silent
  mis-classification here invalidates every number downstream.)

<!-- item-sep -->

- **The `vg_scale` builder.** One pickle over *C*, band-suffixed category names,
  per-class exclusion sets, fixed `n_pos`/`n_neg` per cell. Replaces the three
  `vg_box_*` entries in `pile_config.DATASETS`; the published `vg_box_*` numbers
  stay valid for what they measured and are not comparable to the new sets.
  (Sonnet 5)

<!-- item-sep -->

- **Slate builder and correction ingest.** Per-class review slates from the
  score dumps (`vtscore/eval/score_dumps.py`) in three recorded strata —
  `boundary`, `extreme`, `random` — written as a folder plus manifest for
  `server_folder` import; and the reverse script turning an exported LabelSet
  JSON back into verdict rows keyed `(image_id, class)`. (Sonnet 5)

<!-- item-sep -->

- **COCO-anchored noise measurement.** For classes in both vocabularies, VG's
  miss rate can be measured against COCO val2017's exhaustive annotation with no
  human review at all — and the same comparison scores our own annotators. Worth
  running *before* the manual pass, to size it. (Sonnet 5)

<!-- item-sep -->

- **Corrected re-run and delta report.** Re-run the affected overview-bench
  cells against corrected labels and publish the before/after, so the size of
  the label-noise effect is on the record rather than assumed. (Sonnet 5)

<!-- item-sep -->
