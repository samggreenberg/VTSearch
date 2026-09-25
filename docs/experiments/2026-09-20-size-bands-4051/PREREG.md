# Pre-registration: the size-generalisation matrix (issue #4051)

**Written before the run.** Nothing below may be edited after the cells array is
submitted; the findings go in a separate `REPORT.md`. A prediction recorded
afterwards is not a prediction, and this line has a specific reason to be careful:
the quantity being added is an FNR measured on a *different population* from the
one beside it, which is exactly the shape that produces a confident wrong reading.

## BLUF — the prediction

**A detector trained at one box-size band does not transfer symmetrically to the
others, and the asymmetry runs toward larger targets.** A head trained on large
instances is expected to find small ones poorly; a head trained on small
instances is expected to lose less when tested large.

Per class, per embedder column, at the shipped operating point:

| # | claim | pre-registered bar |
|---|---|---|
| H1 | the off-diagonal is **worse than the diagonal** | mean paired `fnr_<test>` − `fnr_<own>` **> 0** at **> 2 SE**, pooled over the 25 classes, in **both** columns |
| H2 | transfer is **asymmetric** | paired (train small → test large) − (train large → test small) **< 0** at **> 2 SE** |
| H3 | the asymmetry is **not** just the thin `small` cells | H2 survives restricting to the 20 classes whose `small` band clears 400 positives |
| H4 | the shipped `@small` penalty is partly a **training-set** effect | (train medium → test small) is **better** than (train small → test small) for at least a third of classes, in both columns |

**H4 is the one that changes what anybody does.** If a `medium`-trained head
beats a `small`-trained head *on small targets*, then the `@small` cells'
difficulty is partly about what they can train on rather than about small objects
being hard, and the remedy is supply — which #4047 already made a query, not a
rebuild. If H4 fails and H1 holds, size is a genuine domain boundary and the
remedy is per-size detectors.

A null on H1 would be a finding in its own right and a large one: it would say
the representation is scale-invariant enough that the band matters only through
supply.

## What is NOT being measured, and why

**FPR per size.** It has no referent. The three bands of a class share one
negative pool by construction — the `3 *` in `pile_config.SCALE_PREVALENCE` — so
a negative holds no instance of the class and therefore has no size *for the
class*. Each arm has exactly one false-positive rate. Any table in the report
carrying three FPRs per arm is wrong by construction, not merely unsupported.

`coco_better_export.py --check-bands` asserts the shared pool for every class before
the array is submitted, and `launch_bands.sh prepare` refuses the launch if it
fails. That check is not ceremony: #3667 changed what a negative *is* and #3986
changed which pool it comes from, both after the guarantee was first written down.

**Train-side size mixes.** #4047 ships them and this run does not use them. The
matrix is the basis; any *test*-side mix is a weighted average of one of its rows,
so mixes cost nothing until a train-side question exists. Adding mix arms here
would multiply the grid to answer a question the matrix has not yet raised.

**A per-band skyline.** The supervised floor (#3322) is off. Interesting, and a
second study.

## The design

| | |
|---|---|
| dataset | `coco_better` — COCO 2017, 25 classes x 3 bands, boxes adjudicated throughout |
| categories | all 75 (`CALIB_CATEGORY_MODE=all`) |
| columns | `siglip` (whole-image) and `siglip+dinov3_patch` (region) |
| seeds | 20 |
| horizon | 150 clicks |
| everything else | shipped defaults |

**`coco_better`, not `vg_scale`.** The matrix is a claim about object size, and
VG's boxes sit on a smaller instance than the frame's main one 8.3% of the time
(#3924) — an error that moves an image's *band*, which is this study's axis. On
COCO the band is derived from exhaustive annotation.

**The encoder is a blocking factor, not a contrast.** The question is whether the
size-transfer pattern survives the representation, so the two columns are reported
separately and never pooled. "The two columns disagree" is a finding; their mean
is not a system anyone could run.

**The region arm is the pair, never bare `dinov3_patch`.** DINOv3 has no text
tower, so alone it opens on three random known-goods while the whole-image arm
opens on a text sort — a seeding difference sitting inside the voting-mode axis
(#3276, #3278).

## The one thing that could invalidate the table

**The diagonal and the off-diagonal are measured on different images, and they
have to be the *right* different images.** Each band's test cohort is the
held-out set that band's **own** arm would have used, replayed against that
band's evaluable pool at the same seed. If that replay drifted from the harness's
real split, an off-diagonal reading would be taken partly on images some arm
trained on, and the table would look entirely normal.

Three guards, in increasing order of how much they would have to fail together:

1. `scale_bands.holdout_ids` is pinned against `voting_iterations._split_media_ids`
   by running both (`test_the_own_band_cohort_is_the_harness_holdout`).
2. Every run asserts at cohort-construction time that the replayed own-band cohort
   *is* the harness's held-out positives, and raises if not.
3. In the rows themselves, `fnr_<own band>` must equal the row's `fnr`. This is
   checked in the suite and is also the first thing the report must verify on the
   real cells — it is a quantity whose value is already known, arriving beside six
   that are not.

**If guard 3 fails on the real cells, the run is void.** Not repaired in
analysis: the cohorts were built off the wrong pool and every square is suspect.

## Sizing

**Deliberately not predicted here.** `launch_bands.sh` inherits
`launch_scale.sh`'s `--mem`, `--cpus-per-task` and `CONC` so that a first `size`
run is safe rather than tuned, and they are a *starting point*, not a
measurement: different dataset, different pickle, one extra scoring pass per step.

Run `size 0,75` and read the seconds and MaxRSS off `sacct` before quoting any
wall clock. The marginal cost of `CALIB_TEST_BANDS` is expected to be small — a
cohort is positives-only, against a test set whose ~9,900 negatives dominate every
pass — but expected is not measured, and #3129 is the record of what quoting an
unmeasured ETA costs.

Grid shape at these settings: 75 categories x 2 columns x 20 seeds = **3,000
cells**. The region arm was 89% of `launch_scale.sh`'s wall clock, so the column
count is the real knob and widening it is a decision to take after `size`, not
before.

## Reading notes fixed in advance

- **Two significant digits**, and every arm-vs-arm difference quoted **paired**
  (same class, same seed, same split) with its standard error. A difference
  smaller than twice its SE is reported as "not resolvable at 20 seeds", which is
  a finding.
- **`n_test_pos_<band>` goes in every table.** A cohort is the held-out fraction
  of a band, so `bus@small` will be the thinnest square in the matrix, and a rate
  over a dozen images must be visibly that.
- **The natural band shares ride along as a covariate**, from
  `coco_better_export.py --mix-census`. A class whose small band is rare in the wild
  *and* hard in the matrix is a different finding from one that is merely rare.
- The mandatory figures: the quality-over-clicks pair via `curves.py`, the
  `viewer.html`, and the matrix itself as a heatmap **per column**.
