# State of the App: Region Photo — 2026-10-06

**Issue:** #4534; seeds and follow-ups #4552. **Recipe:** `.claude/skills/state-of-the-app/SKILL.md`.
**Interactive viewer:** [`viewer.html`](viewer.html), the beta-1 run's sessions (`2026-10-06-b1`, the app's default
preset), both seeds.
**App:** `dev` at e530b4c0d (2026-10-05 evening). The app is the same one the Binary Photo review of 2026-10-05
read (#4510): the labels line (#4452), the corpus-relative spread floor (#4492), Autopilot's weak-separation
check (#4496), and presets of beta 1/4, 1 and 4.
**Path:** the region path. DINOv3 patch embeddings, with box votes trained as patches and a photo scored by
its best patch (`max_patch`). It opens on SigLIP's text sort.
**Bench:** `coco_better`, 144 cells (49 classes at every size they have).
**Seeds:** 2. **Sessions:** one set per preset (`SOTA_BETA=0.25|1|4`), 288 runs each and 864 in all, trajectory
pass only (the full-label ceiling is still running; see [Not covered](#not-covered)).
**Runs:** `/expscratch/sgreenberg/state-of-the-app/2026-10-06-b025`, `-b1` and `-b4`.
- A run takes a median of 42 minutes on one core (p90 61, the slowest 144) and peaks at 18 to 22 GB.
- So the per-user memory cap, not CPUs, sets how many run at once: at most about 43.
- **Seed 0** ran 2026-10-05 21:37 to 2026-10-06 08:04.
- **Seed 1** ran 2026-10-06 05:55 to 2026-10-07 00:32, sharing the cap with other studies.
- **Code:** every run imported e530b4c0d, except the seed-1 runs that started after 2026-10-06 07:42. Those
  imported 80bcc1b77, because that worktree was updated mid-run.
  - The only harness change between the two is #4523's Test arm, which is off unless `CALIB_LINE_TEST=1` and runs
    after the voting loop, plus output sinks.
  - So the trajectories are the same code's.
- **History:** the first version of this report (PR #4551) read seed 0's 138 cells complete at 07:31; PR #4554
  re-read it on all 144. This version adds seed 1, which moved the objective at 25 clicks by 0.01 to 0.02 and
  the objective after the check by less than 0.005.

**Analysis:** `analyze.sh` (`SOTA_PATH=region`, 2 seeds) per preset, then `perp.py --kind balance`
(`perbeta_summary.md`, `precision_recall_path.csv`).
- **The typed query (click 0) is read at the line the app draws for it today:** the guarded display line of
  #4136, about 204 images (owner, #4599). The sessions ran before #4136, but it only split the display line from
  the opening's acquisition cut, which stays the midpoint, so the clicks are unchanged.
- The analyses' own text baseline is the midpoint line the runs saw; the click-0 rows in
  `precision_recall_path.csv` and `objective_by_click.csv` are re-scored at the guarded line (#4599).
- The baseline was rebuilt for both seeds with the runs' own code; its seed-0 rows reproduce the first
  version's F-beta columns exactly.
- **Re-scored 2026-10-07 (#4605): the opening is the text sort.** The app stays on the text sort through
  Autopilot's opening and shows a detector only from the Hard phase on (`app_trained`). Earlier versions scored
  the detectors the harness trains during the opening, which no user sees. Every number before a session's
  hand-off is now the typed query's own set at the guarded line, and the analysis reads that line throughout.
  The votes and every number after the check are unchanged. The objective at 25 clicks moved from
  0.57/0.49/0.61 to 0.30/0.33/0.50.

A **review**, not an experiment: the app as it ships on the region path, the way a user meets it.
- **The session.** The user types a query and sees the text sort, then votes for 150 clicks while Autopilot picks
  what to show. When the labels separate weakly, Autopilot stops to run a spot check. At the end the user runs the
  check once more.
- **The objective** (#4427) is the F-beta, at the preset's beta, of the withheld test half above the threshold
  the app holds there.
- **AP** measures the ranking alone.

## Headline

**The objective, each preset off its own sessions** (279 trained runs each; 9 runs never trained a detector,
see [below](#where-it-does-well-and-where-it-does-poorly)):

| preset | 25 clicks | 50 clicks | 150 clicks, unchecked | **after the check** | the check's effect | returned, median (unchecked / after) | returned > 200 (unchecked / after) |
|---:|---:|---:|---:|---:|---|---:|---:|
| 1/4 | 0.30 | 0.51 | 0.69 | **0.73** | +0.048 ± 0.007 | 38 / 30 | 4% / 1% |
| 1 | 0.33 | 0.46 | 0.60 | **0.63** | +0.034 ± 0.005 | 51 / 48 | 7% / 3% |
| 4 | 0.50 | 0.57 | 0.68 | **0.72** | +0.046 ± 0.006 | 91 / 92 | 27% / 25% |

![The objective over clicks, per preset](figures/objective_at_own_beta.png)

1. **The presets return what they promise.** After the check:
   - the 1/4 preset returns a median of 30 images at precision 0.81 and recall 0.48;
   - the balanced preset returns 48 at 0.67 and 0.65;
   - the recall preset returns 92 at 0.45 and 0.82.
2. **The region path is ahead of the binary path at every preset from 50 clicks on.**
   - Binary Photo, 2026-10-05 (#4510, the same app at 10 seeds): 0.64/0.53/0.60 after the check, against
     0.73/0.63/0.72 here. At 50 clicks: 0.48/0.42/0.50, against 0.51/0.46/0.57.
   - At 25 clicks the two are within 0.03 (0.32/0.32/0.47 binary): the app is still on the typed query in about
     70% of sessions on both paths, since both open on the same text sort and hand over at the same clicks
     (31% by click 25, half by 40, #4605). Once a session hands over, its region detector is already good: at
     beta 4 the line never falls below the typed query, where the binary one does until click 38.
   - Seed 1 moved no number in this table by more than 0.02, and the objective after the check by less than
     0.005: seed 0 alone read 0.74/0.63/0.72.
   - This is a description of the two paths as they ship, not an A/B: the paths differ in embedding, votes and
     scoring.
3. **The check pays more than on the binary path:** +0.034 to +0.048, against +0.021 to +0.032. Part of it is
   the 9% of sessions that never leave the opening in 150 clicks: the check is where they first see a detector.
   It cuts the share of runs returning more than 200 images from 4% and 7% to 1% and 3% at 1/4 and 1. At 4 it barely moves
   (27% to 25%).

### The returned set through the session

Per preset, the returned set's precision against its recall from the typed query, through 25, 50, 100 and 150
clicks, to after the check. These are means over the trained runs, a session still in the opening counted at the
typed query's set (#4605); the returned size is a median.

![Precision against recall through the session, per preset](figures/precision_recall_path.png)

| preset | point | precision | recall | F-beta | returned, median | runs |
|---:|---|---:|---:|---:|---:|---:|
| 1/4 | typed query | 0.17 | 0.60 | 0.18 | 204 | 279 |
| 1/4 | 25 | 0.29 | 0.57 | 0.30 | 179 | 279 |
| 1/4 | 50 | 0.53 | 0.51 | 0.51 | 53 | 279 |
| 1/4 | 100 | 0.71 | 0.49 | 0.65 | 40 | 279 |
| 1/4 | 150 | 0.75 | 0.50 | 0.69 | 38 | 279 |
| 1/4 | after the check | 0.81 | 0.48 | 0.73 | 30 | 279 |
| 1 | typed query | 0.17 | 0.60 | 0.25 | 204 | 279 |
| 1 | 25 | 0.27 | 0.59 | 0.33 | 182 | 279 |
| 1 | 50 | 0.45 | 0.58 | 0.46 | 80 | 279 |
| 1 | 100 | 0.60 | 0.61 | 0.57 | 50 | 279 |
| 1 | 150 | 0.62 | 0.64 | 0.60 | 51 | 279 |
| 1 | after the check | 0.67 | 0.65 | 0.63 | 48 | 279 |
| 4 | typed query | 0.17 | 0.60 | 0.48 | 204 | 279 |
| 4 | 25 | 0.24 | 0.60 | 0.50 | 186 | 279 |
| 4 | 50 | 0.35 | 0.64 | 0.57 | 109 | 279 |
| 4 | 100 | 0.43 | 0.74 | 0.65 | 93 | 279 |
| 4 | 150 | 0.44 | 0.78 | 0.68 | 91 | 279 |
| 4 | after the check | 0.45 | 0.82 | 0.72 | 92 | 279 |

**As on the binary path, the clicks buy precision, not recall.**
- At 25 most sessions still show the typed query, so the 25s sit near it.
- From click 50 to 100, precision rises 0.08 to 0.18 at every preset (0.53 → 0.71, 0.45 → 0.60, 0.35 → 0.43),
  more than recall moves (-0.03 to +0.10).
- From 100 to 150 the objective still rises 0.03 to 0.04 at every preset. (Earlier versions read a flat stretch
  there: sessions joined the table only once they had a frame, and the late joiners were weak. Every session now
  counts at every click, #4605.)
- The check then adds precision at 1/4 and 1 (+0.05 and +0.04). At 4, precision and recall each move only +0.01.

### The returned set, one rule per row

The detector's own line (the labels line) against a set-constant top-K, at the preset's beta. Compare within a
rule, never across.

| preset | rule | text sort | 25 clicks | 50 clicks | 150 clicks |
|---:|---|---:|---:|---:|---:|
| 1/4 | app line | 0.17 | 0.29 | 0.50 | **0.67** |
| 1/4 | top 32 | 0.47 | 0.47 | 0.55 | **0.68** |
| 1 | app line | 0.24 | 0.32 | 0.45 | **0.58** |
| 1 | top 32 | 0.38 | 0.38 | 0.44 | **0.55** |
| 4 | app line | 0.47 | 0.49 | 0.55 | **0.66** |
| 4 | top 128 | 0.49 | 0.48 | 0.54 | **0.66** |

![The returned set's share of the best cut, per preset](figures/returned_at_own_beta.png)

**The labels line matches a fixed top-K on the same ranking.** The text sort's app line is #4136's guarded line,
the one the app draws today. Under top-K the detector beats the text sort by 50 clicks (0.55/0.44/0.54 against
0.47/0.38/0.49); at 25 most sessions still show the text sort (#4605).

### The ranking

The ranking barely depends on the preset (shown: the beta 1 sessions):

| | text only | 25 clicks | 50 clicks | 150 clicks |
|---|---:|---:|---:|---:|
| **AP** | 0.42 | 0.42 | 0.50 | **0.64** |
| **Goods found** | — | 11 | 15 | **25** |

![Mean AP over clicks](figures/ap_over_clicks.png)

**No early dip.**
- Both paths show the text sort's ranking through the opening, so AP starts at the text level (#4605).
- The binary path's AP dips a little as sessions hand over (0.42 to 0.40 at click 25, #4384) and is back above
  the text level by click 50.
- The region path's AP holds through the hand-over (0.42 at 25) and rises once sessions are on their detectors
  (0.50 at 50).
- The other presets read the same: AP 0.64 and 0.63 at 150, and Goods 25 and 23.

## The spot check

The check walks bands of the unvoted ranking, about 5 picks a band. It never moves the line; its picks are votes
the next retrain learns from.
- **At the end of the session,** every trained run checks once.
  - That is about 21 votes at 1/4 and 1, and 30 at 4.
  - The precision range is 0.63 to 0.70 wide, and it held the truth in every run (279 of 279 per preset).
- **Mid-session,** Autopilot runs the check when the labels separate weakly (#4496).
  - It does so in **23 to 24% of sessions**, against 43 to 45% on the binary path.
  - DINOv3's labels separate more often.

| preset | sessions prompted | first prompt, median click (quartiles) | checks per prompted session | clicks spent checking, median | Goods among those picks |
|---:|---:|---|---|---:|---:|
| 1/4 | 68 of 288 | 82 (58, 104) | 1: 42, 2: 22, 3: 4 | 25 | 21% |
| 1 | 69 of 288 | 82 (57, 99) | 1: 37, 2: 31, 3: 1 | 30 | 24% |
| 4 | 65 of 288 | 77 (53, 103) | 1: 45, 2: 20 | 30 | 19% |

## Where it does well and where it does poorly

**By size band** (beta 1; trained runs; the objective after the check, and the ranking's AP):

| band (runs) | objective | returned, median | text AP | AP at 150 | Goods found |
|---|---:|---:|---:|---:|---:|
| large (98) | 0.81 | 50 | 0.70 | 0.88 | 36 |
| medium (98) | 0.66 | 47 | 0.38 | 0.70 | 26 |
| small (92; 83 trained) | 0.39 | 46 | 0.18 | 0.37 | 14 |

Against the binary path (0.77/0.49/0.29 by band), the gain is in **medium and small**: +0.17 and +0.10, against
+0.04 for large.

**The binary path's hardest classes are where regions help most:**

| class | binary objective (10 seeds) | region objective (2 seeds) | region AP at 150 |
|---|---:|---:|---:|
| chair | 0.07 | 0.37 | 0.35 |
| enclosed road vehicle | 0.22 | 0.63 | 0.67 |
| book | 0.28 | 0.54 | 0.55 |
| bottle | 0.27 | 0.50 | 0.54 |
| knife | 0.20 | 0.41 | 0.41 |

**Poorly, still:** tableware and the dining table.
- Bowl 0.20 (bowl@small never trains, in either seed), spoon 0.29, dining table 0.32.
- No confuser sheets were made for the region path, so this report does not say why.

**Well:** tennis racket 0.98, airplane 0.91, baseball bat and orange 0.89, and apple, banana, kite and surfboard
0.86.

**Never trained a detector** (no Good found in 150 clicks), all at small:
- in both seeds: bowl and chair;
- in one: book, bottle, enclosed road vehicle, knife and person.

On the binary path, chair@small never trains in any of 10 seeds.

![Every class x band](figures/per_cell.png)

<a id="not-covered"></a>
## Not covered

- **The full-label ceiling.**
  - A ceiling run takes about 80 GB, against 24 GB for the clicks.
  - Seed 0's ceiling (array 891581, `-b1/ceiling`) started 2026-10-07 and had 5 of 144 cells at the time of this
    analysis, so there is still no headroom section.
  - `perbeta_summary.md`'s ceiling rows for beta 1 come from those 5 cells and are not read here.
  - #4552 tracks it.
- **Seeds 2 and up** are not run. The binary review's per-image influence lists need repeat clicks, so they are
  not reported on two seeds.

## Files

- `perbeta_summary.md`: every per-preset table above, from `perp.py --kind balance`.
- `precision_recall_path.csv`: each preset's returned set from the typed query through the clicks to after the
  check.
- `objective_by_click.csv`: each preset's objective at every click (`by_click.py`): the typed query until the
  app shows the session's detector (the end of Autopilot's opening, #4605), then its line. An earlier version
  switched at the opening's own first detector, around click 4, and drew a dip there that no user sees.
- `figures/`: the objective, the returned set and its path per preset; AP and Goods over clicks; every cell.
- `viewer.html`: the beta-1 sessions, both seeds.
- **Runs and analyses** are in `/expscratch/sgreenberg/state-of-the-app/2026-10-06-b025|b1|b4/`:
  - `analysis-region-4605/`: **this version's**, both seeds, the opening scored as the text sort (#4605), on the
    post-#4136 baseline; `perp.py` and `by_click.py` over them are in `2026-10-07-perp-4605-region/`, and
    `/expscratch/sgreenberg/notch-4599/opening/analyze_4605.sh` reruns one;
  - `analysis-region/`: both seeds, on the as-measured text baseline (before #4605);
  - `analysis-region-1seed/`: the first version's;
  - `analysis-region-2seeds-post4136baseline/`: the same sessions, with click 0 scored at #4136's line (#4599);
  - the pre-#4605 `perp.py` output is in `2026-10-07-perp-2seeds-mid/`.
- The prompt counts come from `/expscratch/sgreenberg/sota-region-4534/prompts_2seeds.py`.
