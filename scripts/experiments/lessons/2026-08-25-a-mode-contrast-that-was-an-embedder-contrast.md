# 2026-08-25 — a "voting mode" headline that was equally an embedder headline (#3115)

**Study:** #3115 fold combine-rule run. **Cost:** the run's principal claim, and a
follow-up issue (#3258) filed on it before anyone caught the problem. Found by the
owner reading the result, not by any check.

The study reported its headline per **voting mode**: quantile-space combining is
worth −0.032 regret on region voting and costs +0.036 on binary. The grid held
exactly two cells:

```
vg_scale_any  whole_image  siglip        <- everything labelled "binary"
vg_scale_any  max_patch    dinov3_patch  <- everything labelled "region"
```

Every binary cell is SigLIP. Every region cell is DINOv3. **The two axes move
together, so the data cannot tell them apart.** *"Region voting likes
percentiles"* and *"DINOv3's fold models have less comparable score scales than
SigLIP's, so percentiles help"* predict the identical table, and they imply
different fixes — key the rule on the geometry, or on the encoder.

**The part that makes this worth a file: the design was sold as removing a
confound.** #2897 contrasted binary against region across *different datasets*
(Caltech vs VG). #3115 fixed that by taking both modes from one dataset, said so
in the report, and substituted the embedder without noticing. Removing a named
confound feels like rigour and reads like rigour, which is exactly what stops the
next question — *what else moves with this axis?* — from being asked.

Preflight did not help, and it is instructive that it looked like it had.
Check 6 (`--require-region-voting`) asserted that region voting genuinely
**happens** on the DINOv3 cell (`patch_grid=7749/7749`), which it does. That is a
premise check on one cell. It says nothing about whether a **contrast between**
cells is attributable to the axis it is named after. A green premise check on
each arm is not a valid contrast.

**The general form.** *Asserting that an axis is real is not asserting that a
difference is caused by it.* Before reporting "A vs B" where A and B are groups
of cells, list every column that differs between the groups. If more than one
does, the headline names a conjunction, not a factor.

**Status: prevented.** `preflight.sh` gains `--contrasts-voting-modes`: it derives
each cell's mode the way the runner does and fails when the embedder sets for the
two modes are disjoint, pointing at the cheap fix — a patch embedder can run
`whole_image` too, so adding it to `CALIB_PATCH_STYLES` gives one embedder both
modes. Opt-in, because a study that reports one cell per mode and never contrasts
across them is doing nothing wrong; the failure is claiming the contrast.
