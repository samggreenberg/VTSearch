# Same-class scale bands for VG, and the label corrections they need

## Why the current `vg_box_*` sets can't answer the scale question

`scan_vg_boxes.py` measures each VG category's **median** voted-box area, and
`build_pile.py::_band_categories` assigns the category to whichever band that
median lands in. A category therefore lives in exactly one band, and the three
sets carry disjoint vocabularies — the overview benchmark ran `nose`, `glasses`,
`watch` against `fence`, `hill`, `lady`
([`docs/experiments/2026-08-12-overview-bench/REPORT.md`](../experiments/2026-08-12-overview-bench/REPORT.md)).

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

**Size means the union box.** `region_box_for_category` returns the union over a
category's instances. That is the **harness's** choice when simulating a Good
vote from *N* ground-truth boxes, not a reproduction of the app: a real vote
carries one box a person drew (#3983 follow-up).
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

**But `region_box` is normalised and VG's shape is pixels**, so "VG's own shape"
above is a claim about the *record*, not about the numbers. A correction row
therefore declares `box_space` and the builder converts once; merging it
unconverted normalises it twice and parks the box on the frame origin, taking
its band with it (#3281). Both halves are checked at build time — see the pile
[`README.md`](../../scripts/experiments/pile/README.md).

**The negatives are the expensive half.** They are ~95% of *I* and rest on an
absence claim, which is precisely what VG cannot support (`498326.jpg` is
annotated `car, clouds` and has a bus front and centre). Review them in
descending detector score, plus a uniform random stratum so the residual noise
rate after review is bounded rather than unknown.

## Background the remaining work depends on

*C* is the twelve classes in `pile_config.SCALE_CLASSES` — every one also a
COCO-2017 class, which is what makes the correction pass affordable: COCO
val2017 is exhaustively annotated over exactly these names, so VG's miss rate
and our own annotators' accuracy can both be scored against it without extra
review. The measured supply behind the choice is
`/expscratch/sgreenberg/vgscale-3156/` (`vg_box_scale_bands.json`,
`shortlist_*.json`, `name_overlap.json`).

Two measurements constrain what comes next:

- **Scatter is filtered per image, not per class.** A class that often appears
  several times per image fails a per-class inflation median however compact any
  single image is, which cost the shortlist 30 of the 65 COCO classes (`bus`
  among them, at 1.71). The cell rule is therefore per image, and a non-compact
  image is *excluded* rather than counted as a negative.
- **No fine-grained pair survived.** No candidate pair came back `subtype`, so
  there is no fine-grained arm; `eyeglasses` is an alias of `glasses` and would
  have to be merged before either name is usable.

## Open work

**The correction loop is retired.** Every item that stood here was about bounding
the error on labels *inferred* from Visual Genome — finishing the negative
review, reporting the residual rate it bounds, the corrected re-run and delta,
and what the review guide should tell a reviewer about a better example. #3983
measured that COCO 2017 supplies every cell without VG, so the inferred half no
longer exists and none of it is owed. The construction above is unchanged and
still live; only its image source moves.

<!-- item-sep -->

- [ ] What remains is the migration itself — see [`coco-quarry.md`](coco-quarry.md)

<!-- item-sep -->
