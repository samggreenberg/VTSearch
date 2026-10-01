---
name: state-of-the-app
description: Run and write up the periodic "State of the App" review of VTSearch (#4159) - the eval harness run the way a user meets the app, on coco_better, every class at every size, for the two production paths (SigLIP binary, DINOv3 region), with per-image click influence. Use when asked for a State of the App / SotA / periodic review of how the app does, or to re-run last month's.
---

# State of the App

A **review**, not an experiment. It measures the app as it ships, so there is
no A and no B, nothing is tuned, and the settings are pinned so this month's
report reads against last month's. Improving the app comes after, as A/B
studies that the review points at.

## The owner's standing decisions (#4159, 2026-09-23)

Keep these unless the owner changes them, and record any change here in the
same edit.

- **Bench:** `coco_better`, every class at every size: all 144 cells
  (`CALIB_CATEGORY_MODE=all`). The data is the bench, not the subject. Classes
  and bands are strata to report across; a study about the data itself is out
  of scope.
- **Paths:** the two production paths, and only those:
  - **SigLIP binary:** `siglip`, `whole_image`.
  - **DINOv3 region:** `siglip+dinov3_patch`, `max_patch`, opened on SigLIP's
    text sort.
- **A review runs one set of sessions per precision floor P, and scores the
  returned set at its own P (owner, 2026-10-01, #4408).** "The quality of our
  RETURNS matters more than the quality of our RANK." F1 cannot see P (the 50%
  and 90% lines keep nearly the same set and got the same F1), so the headline
  is, per P over clicks, the set the app returns **when it aims for P**: its
  **precision against P** (below P breaks the promise; far above it leaves
  recall behind) and its **recall against the oracle's recall at P** (the most
  any cut of the same ranking returns at or above P). Sessions at P = 10%, 50%
  and 90% (`SOTA_FLOOR`), each analyzed on its own, then `perp.py` reads each P
  off its own run. AP stays as the ranking's measure; F1 is secondary. The
  2026-10-01 review read 10% and 90% off P = 50% sessions; that is exact only
  while a session ignores P.
- **A review runs Binary Photo only by default (owner, 2026-09-26).** Region
  Photo is so slow (~5 h a seed) that it runs only when the owner asks for it
  explicitly: "We'll do that explicitly at some point when we need it." Never
  launch region cells as part of a routine review, a smoke run included.
- **One report per production path (owner, 2026-09-24):** "State of the App:
  Binary Photo" and "State of the App: Region Photo". A future "Document Logo"
  report follows the same shape. Each goes in its own directory,
  `docs/experiments/<date>-state-of-the-app-<path>-<modality>/`, and is written
  when ITS runs finish; one path does not wait for the other.
- **Seeds per path, sized by cost:** Binary Photo is cheap (~7 min and ~1 GB a
  run, ~10-15 min per seed for all 144 cells), so it takes **as many seeds as
  a night allows** (owner, 2026-09-24: per-image claims need repeat clicks, and
  most image-class pairs had one click at 7 seeds). Run it overnight with
  `CALIB_CPUS=1`, launched in seed order, and stop at a set time. Then analyze
  the complete seeds only (`SOTA_ANALYZE_SEEDS=N`) and rebuild
  `text_baseline.csv`, because it is per seed. SLURM caps an array at index
  10099, which is seed 34, and a user's queue at about 2000 jobs, so later seeds
  go in a second array with `CALIB_INDEX_OFFSET` (see
  `<run dir>/overnight.sh` from 2026-09-24). Region
  Photo costs ~5 h per seed under the memory QOS, so it has 3. About 60% of a
  region seed is the full-label ceiling, so extra region seeds can run the
  clicks (`SOTA_PASS=trajectory`) with a ceiling for only some seeds.
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
- **Metrics (owner, 2026-09-30, #4357): no FPR + FNR anywhere.** #4223
  retired that objective, and since #4272 the default arm's line is the
  floor's set (the top 32 unvoted at the default 50%), so a review scored in
  cost reads the change of objective as a regression. Cost is gone from the
  headline, the tables and the figures, not moved to an appendix.
  **The precision floor is written *P*, not *X*** (owner, 2026-09-30).
  The set, as `analyze.py` reports it:
  - **Ranking:** AP (threshold-free), and Goods found per checkpoint (harvest).
  - **The returned set's F1, over clicks** (owner, 2026-09-30: "Showing AP is
    nice, but it's entirely about the ranking. Using the returned-set
    threshold, show F1 over time, too."). It is the F1 of the set the line
    keeps on the test half at the default P = 50% (and at 10%), from the rank
    frames, never the harness rows' `f1`. It sits next to the best F1 any cut
    of the same ranking reaches (`f1_over_clicks.png`, `line_steps.csv`).
  - **The line on a fresh corpus** (the test half, which is what Find and a
    headless run return), at each floor the app offers (10 / 50 / 90%): the
    precision of the kept set (the top K, the floor's unchecked candidate);
    the shortfall, max(0, P − precision); the share of sessions meeting P;
    recall next to the oracle's recall at precision ≥ P (the best cut of the
    same ranking); and the kept set's F1.
  - **The Train-time spot check:** confirm rate, the range, and whether the
    range covered the truth (the share right of the check's candidate). What
    the check *should* certify is #4358; report whatever that ruling becomes.
  - **Ceiling:** the full-label model's AP, and its line, F1 and oracle recall at P.
  - **Click 0:** the text sort's AP and its top K at P.
- **The spot check is not a click.** The default arm checks the line once the
  voting steps are spent (`spot_check="end"`), so every run ends with
  `phase == "check"` rows past `max_steps`. "Final" is the last ordinary step;
  the check's rows and picks feed only the check's own columns, never a curve,
  a checkpoint or an image's credit.
- **Rank frames:** the line at every P is read off where the positives sit in
  each ranking (`task_NNNN__rankframes.csv`), which `launch.sh` records at the
  checkpoints (`CALIB_RANK_FRAME_STEPS`). For an F1 curve, record them densely:
  the 2026-09-30 review used every click to 10 and every 5 after, which is
  ~36 frames and ~10 KB per run at no measurable extra run time. A run from before #4357 has none: its
  AP, harvest and check columns are complete, but its line is known only at
  click 0.
- **Per-image influence:** each scored step's change in held-out **AP** is
  split **equally among the clicks since the previous scored step**, check
  rows and check picks excluded. The first scored step is measured from the
  text-only AP and split among the opening clicks. Then roll the credit up per
  image, per image × detector, and early (≤30 clicks) vs late (>90). One
  click's credit includes refit noise, so an image claim needs repeat
  observations (`n_obs`).
- **An image's own effect:** judge images on `resid_z`, never the raw mean. The
  raw mean mostly says *when* (early positives hurt) and *where* (hard classes)
  an image was clicked. `resid` nets out the cell, label and phase. Report the
  |z| > 3 count **next to the shuffled-image null** that `summary.md` prints.
  At 3 seeds nothing cleared it; at 7 seeds (2026-09-24) the harmful side did.

## Document Logo (the structural path): standing decisions (owner, 2026-10-01, #4392)

The document path (`sift_vlad_doc`, the tiled Stage 1 from #3928 / #4391)
differs from the photo paths, and the owner settled how its review works:

- **Harness:** `scripts/experiments/fullmarks/app_replay_tiled.py`, which calls
  the app's own `maybe_structural_rerank(_example)`. `vtscore.eval` has no
  structural path, and porting one is not a precondition for the review.
- **Bench:** FullMarks (current frozen version), its roster classes and
  #4162's `own_verified` pools. The opening (click 0) is example sort with the
  class's query crop. There is no text opening, because the embedder has no
  text side.
- **The line:**
  - report the set the app returns, which is the inlier gate (score ≥ 0.5,
    i.e. ≥ 8 inliers);
  - next to it, floor-style cuts at P = 10 / 50 / 90%;
  - the structural path has no precision-floor estimator, so a P-cut is the
    best cut of the same ranking with precision ≥ P (an oracle). Say so
    wherever it appears.
- **Repeats:** one closed-loop run per class. The path is deterministic given
  the crop, so per-image claims rest on one observation and are labelled
  single-observation.
- **No ceiling** ("full-label" notch) in structural mode for now. The Headroom
  section is omitted, and the curves end at the last click.
- **No spot check:** the structural path has none, so that section is omitted.
- Everything else (two significant digits, a figure per claim, literal
  examples, the "app as it is now" framing) is as for the photo reports.

## How to run it

```bash
cd scripts/experiments/state_of_app
srun --ntasks=1 -p cpu --mem=8G -c 2 -t 60 bash launch.sh prepare    # checks too heavy for the login node; --ntasks=1 or it submits twice
CALIB_MEM=12G bash launch.sh subset "airplane,dining table,book"   # login node: a few classes, every band; Binary only unless SOTA_PATH=region|all
bash launch.sh cells                                  # login node: the full array, BOTH paths; a Binary-only review uses redo ranges (below)
bash launch.sh status
SOTA_PATH=binary srun -p cpu --mem=48G -c 4 -t 4:00:00 bash analyze.sh   # per path -> analysis-binary/
# per floor (#4408): SOTA_FLOOR=0.1|0.5|0.9 on prepare / redo / analyze -> <date>-p10/-p50/-p90, then
python perp.py --run 0.1=<p10>/analysis-binary --run 0.5=<p50>/analysis-binary --run 0.9=<p90>/analysis-binary --out <dir>
SOTA_PATH=region srun -p cpu --mem=48G -c 4 -t 4:00:00 bash analyze.sh   # -> analysis-region/
```

- **Output:** everything lands in `/expscratch/$USER/state-of-the-app/<date>/`,
  one directory per review.
- **Size (measured 2026-09-23, not the vg_scale guess):** at 1 seed there are
  288 runs. A whole-image run takes about 5 min. A **region run takes about
  1 h and peaks at 66-70 GB**, so the launcher asks for 80 GB. The memory QOS
  sets the wall clock: 144 region runs at a handful concurrent is most of a
  day. The first review lost an hour to 12 GB out-of-memory kills.
- **Split the array by path** when widening: whole-image indices at 12 GB,
  region indices at 80 GB. `run_cells` does NOT skip finished cells, so build
  the index lists yourself, leaving out every `task_NNNN.csv` already written.
  Order is seed-major: in each seed's block of 288, 0-143 are SigLIP and
  144-287 are region. The 2026-09-23 widening (49 classes x 3 seeds) was
  409 + 409 runs: whole-image took about 30 min at 70 concurrent, and the
  region runs took ~40-50 h at 8-10 concurrent under the memory QOS.
- **Pin `SOTA_DATE` when adding to an existing review.** The run directory
  defaults to today's date, so a `redo` after midnight lands in a new, unprepared
  directory. The launcher now refuses that.
- **Submit `subset` / `redo` / `cells` from the LOGIN node** (they only call
  `sbatch`). Twice, `srun -c 1 ... launch.sh redo` submitted the array TWICE,
  so every cell would have run in duplicate and raced its twin on the same
  output files. Only the `prepare` checks need a compute node. After any
  submission, count the jobs.
- **A Binary-only review never goes through `cells`,** which queues both paths
  and sizes every task for region (80 GB). Use `SOTA_PATH`-filtered `subset`
  for a few classes, and `redo` over each seed's SigLIP block (`s*288` to
  `s*288+143`) for the full run, with `CALIB_MEM=4G`-`12G`. The 2026-09-26
  `overnight.sh` does this with a deadline. Measured 2026-09-30 on the floor-era
  default arm (rank frames plus the end-of-run check): a Binary run takes
  7.5–12 min and ~1.1 GB at 2 CPUs.
- **The line on a fresh corpus is the shipped one** (#4402): the rank frames
  record how many the unchecked line keeps on the test half
  (`test_line_k_p10/p50/p90`: the smaller of the schedule's count and the
  vote-anchored mixture's, #4389), and `_rank_metrics.frame_k` reads it. A run
  from before c5f55732c has no such column, and the analyzer then reads the
  schedule's count, which is not the app's line any more: **re-run it, don't
  re-analyze it.** Recording it is a pure read; the 2026-09-30 re-run was
  identical to the run without it, click for click.
- **Smoke the analyzer in a scratch directory.** `analyze.sh` writes
  `text_baseline.csv` into the run directory and reuses it when it has the
  current columns, and the baseline is per seed: a smoke analysis on seed 0
  there leaves a one-seed baseline the full analysis then reads. Make a
  directory whose `results/` symlinks the run's `cells/`, `prepare_info.json`
  and `crops/`, write its own `grid_shape.json` with the seeds it holds, and
  analyze that (2026-09-30: `<run dir>-smoke`).
- **Measured 2026-09-30 on the band-walk app:** a Binary run takes a median
  6 min (p90 7.3) at 1 CPU; 10 seeds (1,440 runs) took ~90 min at ~210 wide,
  and `analyze.sh` ~16 min for 10 seeds at 8 CPUs.
- **`analyze.sh` needs `results/grid_shape.json`, and only `launch.sh cells`
  writes it.** A run built from `subset` or `redo` (a smoke run, or seeds
  widened by index) has none, and `analyze.sh` then dies inside
  `experiment_config.py` on `int('')`. Write one by hand with the seeds the
  run really holds; the 2026-09-26 smoke run's is the example.
- **Before a report is written,** check that `prepare_info.json` lists all 144
  cells for BOTH paths. `CALIB_REQUIRE_SEED_QUERY=1` silently drops a class
  that has no typed query.

## What the report says

**The report is about the app as it is now (owner, 2026-09-27):** how it is
doing, where it does well or poorly, and WHY. It is not a contest with an
earlier review or a bench that no longer exists: "We're not fighting some
internal fight against the version that no longer exists." A delta against the
previous review gets one short note at most, never a section or the framing.
Spend the effort on the why: for each class that does poorly, look at its images
and say what the app gets wrong.

Each report goes in `docs/experiments/<date>-state-of-the-app-<path>-<modality>/REPORT.md`
(e.g. `2026-09-27-state-of-the-app-binary-photo`) and carries these sections, in this order:

1. **Headline: the returned set at each P, from its own sessions** (#4408):
   precision against P and recall against the oracle's recall at P, over
   clicks (`perp.py`: `returned_at_own_p.png`, `perp_summary.md`; per run,
   `returned_at_p.png`), text → 25 → 50 → final → ceiling, with the share of
   sessions meeting P. Then the ranking: mean text-only AP, AP at 25 and 50
   clicks, the final AP against the ceiling's, with Goods found at the same
   points. F1 at the line (`f1_over_clicks.png`) is secondary: it cannot see P.
2. **The spot check:** how often it confirms, its range, and how often the
   range held the truth.
3. **Where the app does well and where it does poorly,** by class and by band,
   on final AP and on the final line at 50%. Name the classes, with numbers.
4. **Headroom:** the ceiling's AP minus the final AP. This is what better
   clicking could still buy; a large gap marks the loop, not the class.
5. **What the clicks bought:** final AP minus text-only AP. A class where
   clicking barely beats typing is a finding.
6. **Images:**
   - the most helpful and most harmful images (credit on AP), with thumbnails;
   - images that help many detectors;
   - images whose sign flips between early and late clicks.
7. **Known regimes to flag, not to fix:** runs that exhaust the positives
   before 150 clicks (#4121, where the fused threshold drifts). The
   preflight warns about this at launch.
8. **What to A/B next,** filed as issues, per the follow-ups rule.

Follow the standing report rules: two significant digits, a figure per claim,
and literal examples. Every `#N` in a message to the owner is a link.

## Changing this recipe

The owner expects the presentation to be polished over a few rounds. When the
owner states a preference, apply it and write it into **The owner's standing
decisions** above, so the next review starts from it.
