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

- **The objective (owner, 2026-10-01, #4427):** the F-beta, at the session's beta, of the set the line keeps,
  scored on the WITHHELD test half. Found positives (Goods), the user's own unvoted corpus ("positives in
  hand", the `Diagnostic` section, `pools.csv` / `pool_steps.csv` / `in_hand.png`) and the share of the best
  cut explain a line's position; they are never the goal, and a ship decision cites the withheld F-beta at the
  line first.

Keep these unless the owner changes them, and record any change here in the
same edit.

- **The eval's horizon is not the user's (owner, 2026-10-04, #4482):** the
  review runs 150 clicks as a computational compromise. Users stop when they
  are bored or the app stops them, before 150 or after it. Never frame a
  finding or an A/B as "the last N clicks" or "stop at click N"; read the
  app as curves over the click a user has reached, and aim at getting the
  most users to the best detector wherever they stop. Arms that ask about
  late-session behaviour should run past 150 (to 200) rather than treat 150
  as an end.
- **Apples to apples (owner, 2026-10-04, #4474):** a comparison between the
  text sort and the detector uses ONE thresholding rule on both sides. Never
  let "text sort" mean "text sort at a fixed count" while "detector" means
  "detector at its line". The analyzer's balance tables carry a `rule`:
  `app line` (what each sort's own line in the app returns: the text sort's
  blind GMM cut, the detector's labels line, and for the full-label ceiling
  Find's labels line drawn from its full labels, #4486; the skyline's own row
  is still cut at the retired oracle on the test labels, which no Find could
  draw, so a ceiling frame from before #4486 reads blank), the primary
  reading; and `top-K` (set-constant at the old cap,
  32 at beta <= 1 and 128 above, on every sort), a secondary view shown for
  both sorts or neither. The owner is skeptical a set-constant line will ever
  ship (it does not scale with the Find corpus), so the report leads with the
  app lines. The text sort's blind cut returns most of the corpus (F1 0.02 on
  COCO Better); its top 32 happens to be a near-oracle count on this bench
  (about 50 positives a cell), which is why the count line looked good early
  and why #4452 found it fails on other corpus sizes.

- **Bench:** `coco_better`, every class at every size: all 144 cells
  (`CALIB_CATEGORY_MODE=all`). The data is the bench, not the subject. Classes
  and bands are strata to report across; a study about the data itself is out
  of scope.
- **Paths:** the two production paths, and only those:
  - **SigLIP binary:** `siglip`, `whole_image`.
  - **DINOv3 region:** `siglip+dinov3_patch`, `max_patch`, opened on SigLIP's
    text sort.
- **The preference is a balance, beta, and a review runs one set of sessions
  per beta (owner, 2026-10-01 03:42, #4413, after #4411's pricing).** The
  headline is, per beta at the app's presets (1/4 precision-leaning, 1
  balanced, 4 recall-leaning: the owner's pick of 2026-10-03 on #4448; they
  were 0.5 / 1 / 2) over clicks, the returned set's **F-beta as a share of
  the best F-beta any cut of the same ranking reaches** (`returned_at_beta.png`,
  the "returned set at each balance" table), with its precision and recall
  beside it, and the share of runs whose line keeps nothing (`empty`).
  Sessions run at `SOTA_BETA=0.25|1|4` (the eval's `CALIB_BETA` arm): this is
  the standing recipe. The analyzer's betas follow the app's presets
  (`_rank_metrics.BETAS` is the harness's `RANK_FRAME_BETAS`, which a test
  pins to `BALANCE_PRESETS`, #4471). The harness's default arm (neither `SOTA_BETA` nor
  `SOTA_FLOOR`) is the app's default, the balance at beta 1, since #4413's
  step 6 switched `line_preference` to the balance. The per-floor sessions
  below are the **floor-era control** (#4408): run them to compare against
  the floor era, not as the review.
- **The floor-era control: one set of sessions per precision floor P, the
  returned set scored at its own P (owner, 2026-10-01 02:20, #4408).** "The quality of our
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
  retired that objective, and since #4272 the default arm's line is a kept
  set (the floor's until #4413; then the balance's count line, the mixture's
  F-beta argmax capped at 32; since #4452 the balance's labels line), so a
  review scored in cost reads the change
  of objective as a regression. Cost is gone from the
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
- **Repeats:** two closed-loop runs per class (owner, 2026-10-03, #4457; was one):
  the second with `--swap-halves`, which clicks in the first run's test half and
  scores on its click half. The path is deterministic given the crop and the
  halves, so each replicate is one observation per class; per-image claims are
  labelled single-observation.
- **The balance (owner, 2026-10-03, #4457; presets 1/4, 1, 4 since #4472):**
  the review scores the returned set at beta 1/4, 1 and 4: its F-beta as a
  share of the best cut's (`returned_at_beta.png`, the "returned set at each
  balance" table), at the photo headline's presets. Since #4458 the structural
  line follows beta from beta 2 up (the recall end adds pages above a
  per-detector floor) and reorders a few loose fits before the first Bad, so
  a beta-4 session can differ from a beta-1 session. Run two session sets:
  - the default run serves beta 1/4 and 1, whose line is the same;
  - `sota_documents.py --beta 4` serves beta 4.

  Read each beta's row from its own run.
- **Hardware:** run on the app's GPU type (a V100 today), and report retrain
  time from it (`retrain.png`). One comparison (#4457, different nodes) put
  an L40S at about half the V100's p90.
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
# per beta (#4413, the standing recipe; presets #4448): SOTA_BETA=0.25|1|4 on prepare / redo / analyze -> <date>-b025/-b1/-b4
python perp.py --kind balance --run 0.25=<b025>/analysis-binary --run 1=<b1>/analysis-binary --run 4=<b4>/analysis-binary --out <dir>   # each beta off its own run: the objective first, the returned set, the early dip, the check (#4474)
# per floor (#4408, the floor-era control): SOTA_FLOOR=0.1|0.5|0.9 on prepare / redo / analyze -> <date>-p10/-p50/-p90, then
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
  record how many the unchecked line keeps on the test half.
  `test_line_k_b025/b1/b4` (#4471) is the labels line (#4452) at each
  preset, its corpus side fitted on the test half as Find fits it; at the
  sessions' own beta it is the objective's returned set exactly. It can be 0,
  which the analyzer reads as an empty set (F-beta 0), not as a missing
  count. `test_line_k_p10/p50/p90` is the floor's: the smaller of the
  schedule's count and the vote-anchored mixture's, #4389.
  `_rank_metrics.frame_k` / `frame_beta_k` read them. A run from before
  c5f55732c has no such column, and one from before #4471 recorded the
  retired count line at 0.5 / 1 / 2; the analyzer then reads the balance's
  cap, which is not the app's line any more: **re-run it, don't re-analyze
  it.** Recording it is a pure read; the 2026-09-30 re-run was
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

1. **Headline: the returned set at each balance, from its own sessions**
   (#4413): per beta at the app's presets (1/4 / 1 / 4, #4448) over clicks, the returned set's F-beta as
   a share of the best F-beta any cut of the same ranking reaches
   (`returned_at_beta.png`, the "returned set at each balance" table), with
   its precision and recall beside it, text → 25 → 50 → final → ceiling. The
   floor-era control (#4408), when it was run: precision against P and recall
   against the oracle's recall at P (`perp.py`: `returned_at_own_p.png`,
   `perp_summary.md`; per run, `returned_at_p.png`), with the share of
   sessions meeting P. Then the ranking: mean text-only AP, AP at 25 and 50
   clicks, the final AP against the ceiling's, with Goods found at the same
   points. F1 at the line (`f1_over_clicks.png`) is secondary: it cannot see
   the balance.
2. **The spot check:** where the F-beta walk ends (the kept set's F-beta
   against the best cut's), its precision and recall ranges, and how often
   the precision range held the truth; under the floor-era control, how often
   it confirms.
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
