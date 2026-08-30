# Max-Patch experiment — MaxHAC vs MaxPatch (vs whole-image)

## Background

The study asked whether the HAC region tree earned its keep against scoring an
image by max-pooling the MLP over its **raw patch grid**. Its verdict — **ship
tree-free MaxPatch; drop the HAC tree from ingest** — shipped in #2886, and the
numbers are in
[`docs/experiments/2026-07-29-max-patch/REPORT.md`](../experiments/2026-07-29-max-patch/REPORT.md).

Two consequences shape anything that revisits this:

- **The `max_hac` arm is no longer runnable.** It delegated to the production HAC
  region tree, which #2886 deleted; the runner grid dropped it rather than
  reimplementing the code the study told us to remove. Its published numbers stand
  in the report, and `analyze.py` still labels `max_hac` rows in archived CSVs.
- **Scale must be measured on the *voted* box, never the instance box.** The eval's
  Good vote trains on the union box over all of a category's instances in the image
  (`region_box_for_category`), which is ~2.7× the median instance at the median and
  far larger for multi-instance categories. `vtscore.eval.labels.category_scale_stats`
  exposes `voted_area` for exactly this, plus `union_inflation` to flag the
  categories where the two diverge.

The styles themselves live in `vtscore/eval/patch_styles.py` (`whole_image`,
`max_patch`, `max_patch_hac`), and `style=None` resolves to the app's own geometry
rather than being a third thing — a patch dataset on the MLP trainer gets
`max_patch`, whose methods delegate to `pool_box_from_media` / `bad_negative_vecs`
/ `media_score_rows`. The runner is `scripts/experiments/max_patch/`.

## Open work

<!-- item-sep -->

- **Optional follow-up arm, only if a rerun is ambiguous** — Good vote =
  *mean of patches inside the box* instead of the single nearest patch (the
  other natural reading of "closest patch", better for multi-patch objects).
  Note it violates the train/score invariant by construction: a per-vote amalgam
  can never be a per-image scored row, so an arm measuring it would have to say
  what it scores as well as what it trains on.

<!-- item-sep -->

## Known limitations (accepted)

- The exemplar image itself stays in the dataset and may land in the held-out
  test split; the (tiny, equal-across-arms) optimism is accepted rather than
  re-plumbing the split.
- The Autopilot pool-acquisition proxy scores candidates by their whole-image
  vector under every style (matching the existing harness); only training and
  test scoring differ per style. This keeps vote-order differences attributable
  to the trained model, not to a different acquisition rule.
- `caltech101_m` is the **image-level-voting** control, not the large-target
  control — it has no boxes, so every Good vote on it is image-level regardless
  of object size. Large-target evidence comes from the top scale bands of the
  boxed datasets; do not read Caltech as covering that regime.
