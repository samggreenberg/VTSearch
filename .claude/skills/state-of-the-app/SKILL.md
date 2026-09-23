---
name: state-of-the-app
description: Run and write up the periodic "State of the App" review of VTSearch (#4159) - the eval harness run the way a user meets the app, on coco_quarry, every class at every size, for the two production paths (SigLIP binary, DINOv3 region), with per-image click influence. Use when asked for a State of the App / SotA / periodic review of how the app does, or to re-run last month's.
---

# State of the App

A **review**, not an experiment. It measures the app as it ships, so there is
no A and no B, nothing is tuned, and the settings are pinned so this month's
report reads against last month's. Improving the app comes after, as A/B
studies that the review points at.

## The owner's standing decisions (#4159, 2026-09-23)

Keep these unless the owner changes them, and record any change here in the
same edit.

- **Bench:** `coco_quarry`, every class at every size: all 144 cells
  (`CALIB_CATEGORY_MODE=all`). The data is the bench, not the subject. Classes
  and bands are strata to report across; a study about the data itself is out
  of scope.
- **Paths:** the two production paths, and only those:
  - **SigLIP binary:** `siglip`, `whole_image`.
  - **DINOv3 region:** `siglip+dinov3_patch`, `max_patch`, opened on SigLIP's
    text sort.
- **Shipped defaults** for everything else: the text opening (refused
  otherwise), the fused threshold, and no variant arms.
- **Scale up in one step (owner, 2026-09-23):** settle the presentation on a
  **small set of classes** at 1 seed, which is effectively an elaborate smoke
  test. Then widen the **classes and the seeds together**. The first round used
  8 classes: `airplane`, `dining table`, `book`, `bag or luggage`, `banana`,
  `traffic light`, `person` and `dog`, which is 23 cells and 46 runs.
- **Every curve runs left to right:** the **text-only score at click 0**, then
  the clicks, then the **full-label ceiling** (`skyline_train_full`) on the
  right. The region path's ceiling is supervised with each positive's
  ground-truth box (owner ruling on #3321).
- **Metrics:** F1 and cost, the standard `viewer.html`.
- **Per-image influence:** each scored step's change in the held-out test
  score is split **equally among the clicks since the previous scored step**.
  The first scored step is measured from the text-only score and split among
  the opening clicks. Then roll the credit up per image, per image × detector,
  and early (≤30 clicks) vs late (>90). One click's credit includes refit
  noise, so an image claim needs repeat observations (`n_obs`).

## How to run it

```bash
cd scripts/experiments/state_of_app
srun -p cpu --mem=8G -c 2 -t 60 bash launch.sh prepare    # its checks are too heavy for the login node
bash launch.sh subset "airplane,dining table,book"   # login node: a few classes, every band, both paths
bash launch.sh cells                                  # login node: the full array; SOTA_SEEDS=N for more seeds
bash launch.sh status
srun -p cpu --mem=24G -c 4 -t 4:00:00 bash analyze.sh     # after the array finishes
```

- **Output:** everything lands in `/expscratch/$USER/state-of-the-app/<date>/`,
  one directory per review.
- **Size (measured 2026-09-23, not the vg_scale guess):** at 1 seed there are
  288 runs. A whole-image run takes about 5 min. A **region run takes about
  1 h and peaks at 66-70 GB**, so the launcher asks for 80 GB. The memory QOS
  sets the wall clock: 144 region runs at a handful concurrent is most of a
  day. The first review lost an hour to 12 GB out-of-memory kills.
- **Submit `subset` / `redo` / `cells` from the LOGIN node** (they only call
  `sbatch`). Twice, `srun -c 1 ... launch.sh redo` submitted the array TWICE,
  so every cell would have run in duplicate and raced its twin on the same
  output files. Only the `prepare` checks need a compute node. After any
  submission, count the jobs.
- **Before a report is written,** check that `prepare_info.json` lists all 144
  cells for BOTH paths. `CALIB_REQUIRE_SEED_QUERY=1` silently drops a class
  that has no typed query.

## What the report says

The report goes in `docs/experiments/<date>-state-of-the-app/REPORT.md` and
carries these sections, in this order:

1. **Headline per path:** mean text-only cost, then cost at 25 and 50 clicks,
   then the final cost against the ceiling. Do the same for F1.
2. **Where the app does well and where it does poorly,** by class and by band.
   Name the classes, with numbers.
3. **Headroom:** final cost minus the ceiling. This is what better clicking
   could still buy; a large gap marks the loop, not the class.
4. **What the clicks bought:** text-only cost minus final cost. A class where
   clicking barely beats typing is a finding.
5. **Images:**
   - the most helpful and most harmful images, with thumbnails;
   - images that help many detectors;
   - images whose sign flips between early and late clicks.
6. **Known regimes to flag, not to fix:** runs that exhaust the positives
   before 150 clicks (#4121, where the fused threshold drifts). The
   preflight warns about this at launch.
7. **What to A/B next,** filed as issues, per the follow-ups rule.

Follow the standing report rules: two significant digits, a figure per claim,
and literal examples. Every `#N` in a message to the owner is a link.

## Changing this recipe

The owner expects the presentation to be polished over a few rounds. When the
owner states a preference, apply it and write it into **The owner's standing
decisions** above, so the next review starts from it.
