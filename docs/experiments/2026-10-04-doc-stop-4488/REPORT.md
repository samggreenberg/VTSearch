# When should a document user stop clicking? (#4488)

**Question.** Autopilot says **Detector Trained** when Smart, Stable and Span are all green. On a
document (structural) detector none of them reads the structural detector. Smart and Stable score
the page-VLAD head, which ranks documents near chance (#3928), against the structural line, a
verification score on another scale. What should tell a document user to stop, and does it hold
when the collection's size and prevalence change?

**Answer: the dry run, 16 of the detector's best matches in a row that are not good.** The lights'
ideas, adapted to the structural detector, add nothing: the adapted Smart and Stable were always
green before the dry run. Across three sizes (5k, 50k and 200k pages) and two prevalences (all
positives, and a quarter of them), **not every cell clears the bar set in advance.** Small (5k) and
medium (50k) hold. At 200k pages 7.3% and 13.9% of positives are still unfound at the stop, against
the bar's 5%. But the stop is not what loses them:
- **It forgoes at most 0.8% in every cell,** counting the positives that clicking on to click 50
  found after the stop.
- **The rest were never found by click 50,** and they track the positives outside Stage 1's
  2,000-page shortlist, which no walk reaches (#4493).

By the rule set in advance the dry run does not ship. **The owner overrode the bar and shipped it
(2026-10-04):** within 50 clicks no stop does better, and the lights it replaces scored a model the
document ranking does not use.

## The signals, adapted (tier `m`, 36 classes × 2 replicates)

Fresh closed-loop sessions of the shipped app (`sota_documents.py --stop-log`, beta 1), clicking the
top unlabelled page each step. Per click the run logs the structural line, the clicked page's score
before the click, and the unlabelled pages above the line. The signals, as pre-registered on #4488:
- **Smart** (has clicking stopped teaching it?): each click is judged by the detector before it,
  correct if (score >= line) equals the label. Green when the last 10 clicks hold <= e errors.
- **Stable** (are unlabelled predictions still moving?): the churn of the returned set between
  clicks. Green when the last 10 clicks' maximum is <= c.
- **Span** (have the clicks walked past the positives?): the dry run, 16 clicks in a row without a
  Good, latched.

(e, c) were chosen by class-split CV (sha256 folds). Both folds chose e=0, c=0.01.

| signal | stops by 50 | stop click, median (range) | click-half positives unfound | AP at 50 − at stop |
|---|---:|---|---:|---:|
| **dry run** | 83% | **31 (17–50)** | **17 of 751 (2.3%)**; per session 5.2% | +0.018 (median 0) |
| Smart alone | 99% | 10 (10–44) | 37% | +0.030 |
| Stable alone | 96% | 24 (11–50) | 13% | +0.016 |
| all three | identical to the dry run at every (e, c) | | | |

- **Smart and Stable are always green before the dry run,** so the combined stop is the dry run.
  Smart, judged per click on top-of-ranking clicks, is uninformative: the detector calls its own
  top pages right.
- **The dry run fails in 2 sessions,** both #4170's harmful first click: a bad first Good breaks the
  ranking, 16 Bads follow, and the run fires at click 17 before the session recovers (~click 40).
  Otherwise AP lost is <= 0.02 in 58 of 60 stopping sessions.
- **The 12 sessions with no stop by 50** are the big classes (40–200 positives), where nearly every
  click is a Good. That is the right behaviour.

By the selection rule set for this table (mean unfound <= 5% and mean AP lost <= 0.02) the dry run
missed narrowly: 6.8% unfound on one fold, 0.024 AP on the other. The owner chose the dry run as the
document stop on condition that it survives changes in size and prevalence, since COCO Better showed
constant rules failing when those move.

## Does a constant 16 travel? Size and prevalence

**Knobs.**
- **Size:** FullMarks' nested tiers, s (5k pages), m (50k) and l (200k).
  - 27 of the 36 classes keep the same positives in every tier.
  - 9 recurring logos (tobacco800 and UCSF) gain positives as the tier grows: `ucsf/logo_bat_leaf`
    has 5, 200 and 213.
  - Prevalence falls at every step up either way (`tobacco800/logo_ajj10e00_1`: 1.3%, 0.81%, 0.21%).
- **Prevalence:** `sota_documents.py --thin 0.25` keeps a seeded quarter of each class's positives in
  the pool (sha256 order of page ids, at least one) and drops the rest.

A class is skipped when its click half or test half holds no positive, so the cells' session
counts differ.

**The bar (pre-registered):** every cell has positive-weighted unfound <= 5%, and at most 10% of
stopping sessions lose more than 0.05 AP.

| tier | positives kept | sessions | stop by 50 | stop click, median (range) | unfound at the stop (weighted / per session) | of which: found by 50 / never | AP lost: median, share > 0.05 | holds |
|---|---:|---:|---:|---|---|---|---|---|
| s (5k) | all | 66 | 95% | 31 (17–50) | 2.9% / 5.9% | 0.7% / 2.2% | 0.000, 3% | yes |
| s | 1/4 | 54 | 100% | 20 (17–33) | 2.8% / 3.5% | 0.8% / 2.0% | 0.000, 0% | yes |
| m (50k) | all | 72 | 83% | 31 (17–50) | 2.3% / 5.2% | 0.7% / 1.6% | 0.000, 3% | yes |
| m | 1/4 | 62 | 94% | 20 (16–45) | 3.8% / 4.9% | 0.6% / 3.2% | 0.000, 0% | yes |
| l (200k) | all | 72 | 81% | 31 (16–50) | **7.3%** / 7.1% | 0.1% / 7.2% | 0.000, 2% | **no** |
| l | 1/4 | 62 | 95% | 20 (16–50) | **13.9%** / 13.9% | 0.5% / 13.4% | 0.000, 0% | **no** |

"Of which" splits the unfound into positives that clicking on to click 50 found after the stop (what
the stop forgoes) and those never found by click 50. The split is descriptive and was added after
the runs; the bar is unchanged.

**The constant 16 is not tuned to tier m.** Other run lengths, read off the same sessions
(each clicked on to 50):

| tier | positives kept | run 8: stop / median click / forgone | run 12 | run 16 | run 24 | run 32 |
|---|---:|---|---|---|---|---|
| s | all | 97% / 23 / 0.9% | 95% / 27 / 0.6% | 95% / 31 / 0.6% | 80% / 39 / 0.5% | 67% / 45 / 0.1% |
| s | 1/4 | 100% / 12 / 1.6% | 100% / 16 / 0.8% | 100% / 20 / 0.8% | 100% / 28 / 0.0% | 96% / 36 / 0.0% |
| m | all | 85% / 23 / 0.5% | 83% / 27 / 0.3% | 83% / 31 / 0.2% | 74% / 38 / 0.2% | 62% / 45 / 0.2% |
| m | 1/4 | 97% / 12 / 0.6% | 94% / 16 / 0.4% | 94% / 20 / 0.4% | 92% / 28 / 0.0% | 84% / 36 / 0.0% |
| l | all | 85% / 23 / 0.2% | 83% / 27 / 0.1% | 81% / 31 / 0.0% | 75% / 38 / 0.0% | 62% / 45 / 0.0% |
| l | 1/4 | 95% / 12 / 0.5% | 95% / 16 / 0.4% | 95% / 20 / 0.4% | 92% / 28 / 0.2% | 85% / 36 / 0.0% |

Forgone here is weighted over all of a cell's sessions. Longer runs save at most half a point and
stop fewer sessions by click 50. Shorter runs forgo more at the small tier. At every tier and
prevalence, 16 forgoes at most 0.8%.

**What the stop cannot reach.** Test-half positives outside the 2,000-page shortlist at click 50
(verification score 0, among ~180k zero-score pages at tier l):

| tier | positives kept | beyond the shortlist | classes with any |
|---|---:|---:|---:|
| s | all | 4 of 984 (0.4%) | 3 |
| s | 1/4 | 1 of 246 (0.4%) | 1 |
| m | all | 77 of 2,092 (3.7%) | 9 |
| m | 1/4 | 19 of 526 (3.6%) | 9 |
| l | all | 215 of 2,217 (9.7%) | 15 |
| l | 1/4 | 74 of 560 (13.2%) | 13 |

The worst case is `tobacco800/logo_afm90c00-first_1_0` at tier l, full prevalence, replicate 2. It
found 14 Goods, then 36 straight non-Goods (test AP 0.27), and 46 of its 62 test positives sit
beyond the shortlist (tier m: 1 of 12). The shortlist's K was chosen at 50k pages (#4391); at 200k
it covers 1% of the pages instead of 4%. #4493 takes it up.

**Two tier-l runs hit a CUDA out-of-memory error** in the harness's sharded scorer at their 23rd
class. That scorer did not release the descriptor cache on a new matrix, as the app's path does;
the fix (dfa75e5ad) does not change scores. The harness is deterministic and saves after each
class, so the remaining 14 classes were resumed in separate runs and merged.

Tier l ran on 2 × V100 with the tile matrix (~34 GB fp16) split by rows across the cards
(`--shard-stage1`). That gives the same page scores as the app's single-GPU path (max difference
0.0 on 8,000 pages, 7 queries).

## What ships

- **Backend:** `GET /api/labeling-status` on a tiled dataset answers `stop_rule: "dry_run"`, with a
  `dry_run` readout: the standing non-Good votes cast since the last Good, green at 16 once the
  labelset holds a Good. Smart, Stable and Span are `off`. It is computed from the vote order alone,
  so it is never stale and never trains a model.
- **Autopilot on documents:** Good (3) → Bad (4) → More → Done.
  - More walks the top of the detector's own ranking, re-ranked after every vote, with no 20-Good
    target and in retrain mode too.
  - 16 misses in a row is Done and **Detector Trained**.
  - There is no Boundary or Diversity step.
- **Manual tab:** one **Dry run n/16** readout replaces the three lights.

**The app's readout matches the measured rule.** Replaying all 388 recorded sessions vote by vote
through `dry_run_status` (`dry_run_parity_documents.py`), it turns green at the measured click in
382. The other 6 found no Good in their first 16 clicks, and the readout waits for a Good by design;
Autopilot's walk starts only after 3 Goods.

**One difference from the sessions.** The harness clicks the freshly re-ranked top. The app picks
the next page from the ranking on screen when the vote lands, and re-sorts about 1.5 s later on a
V100, so a quick clicker's next pick can come from the ranking one vote earlier.

## Reproduce

```bash
cd scripts/experiments/fullmarks
F=/expscratch/sgreenberg/fullmarks
# one cell: tier, thin, replicate (--swap-halves for the second)
python sota_documents.py --tier m --max-v 50 --stop-log --thin 0.25 [--swap-halves] \
  --matrix $F/votes-4162/matrix-m --feature-cache $F/features --out <run dir>
# tier l: the pools come from class_pools_tier.py, and --shard-stage1 on 2 x V100
python stop_rules_documents.py --run <rep1> --run <rep2> --out <dir>        # the signals table
python stop_grid_documents.py --cell m,0.25=<rep1>,<rep2> ... --out <dir>     # the grid
python dry_run_parity_documents.py <run dir> ...                             # the app's readout
```

Measurements are in [`measurements/`](measurements/).
