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

## Open work

<!-- item-sep -->

- **Run the supply scan and pick *C*.** `scan_vg_boxes.py` now emits the
  per-`(class, band)` histogram and `shortlist_scale_classes.py` ranks
  categories on their binding (minimum) per-band supply, flagging COCO overlap
  and prior benchmark coverage. Both need the VG source under `DEMO_CACHE`, so
  the scan is a GRID job. Choosing *C* from its output is the gate on
  everything below. (Sonnet 5)

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
