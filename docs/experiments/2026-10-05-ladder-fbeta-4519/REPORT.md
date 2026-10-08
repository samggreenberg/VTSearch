# The calibration ladder in F1, at 1% prevalence — 2026-10-06

**Issue:** #4519 (slide 32, "The Ladder"). **Owner's call (2026-10-05):** "Re-run the ladder, but do it at
1%, not 5%." **Study it redraws:** #4184 (`docs/experiments/2026-09-25-progression-4184`), which drew the
ladder in cost at 5%.

**What ran:** eight rungs, each the app as the deck draws it at that point.
- **The first seven** (#4184) run as they ran on 2026-09-25: the Inclusion arm (`CALIB_MIN_PRECISION=off`)
  and no spot check (`CALIB_SPOT_CHECK=off`). An unset preference is now the balance, whose line is the
  labels line, and that would have turned r5–r7 into today's app.
- **The eighth, `r8_labels`, is today's app:** the labels line (#4452) and the shipped spot check.

**Setup:**
- **Bench:** `coco_better`, 144 cells, 5 seeds, 150 votes: 720 runs a rung.
- **Pool:** the user's pool thinned to 1% positive (`CALIB_HAYSTACK_PREVALENCE=0.01`). The withheld half Find
  searches stays at the bench's own 0.44%.
- **Code:** launched from `scripts/experiments/calibration/launch_progression_4184.sh` on the #4519 branch
  (merged in PR #4538).
- **Runs:** `/expscratch/sgreenberg/progression-4184-h0.01/<rung>`, 2026-10-05 20:00 to 2026-10-06 01:52.
- **Analysis:** `analyze_progression_4184.py --arms ...`. Its machine summary is `REPORT_progression.md`.
  Every rung has all 720 cells. Three cells have no positive the session could find and are header-only, the
  same three in every rung.

**Score:** the returned set's F1 on the withheld half, *filled*. Until a run shows a detector, it scores the
typed query's own returned set (the text sort's blind cut), so every rung leaves the same dot.
The presets below read the typed query at the line the app draws for each preset instead (#4625); the ladder's
tables here were not re-scored, and the deck no longer draws them.

## Result

| rung | idea | F1 at 30 | 50 | 100 | 150 | step at 150 (ΔF1 ± SE, paired) |
|---|---|---:|---:|---:|---:|---|
| r1_xcal | cross-calibration | 0.33 | 0.41 | 0.47 | 0.48 | — |
| r2_gmm | mixture midpoint | 0.031 | 0.053 | 0.096 | 0.12 | −0.37 ± 0.011 |
| r3_blend | blend | 0.039 | 0.096 | 0.17 | 0.20 | +0.085 ± 0.007 |
| r4_rawmean | fused, raw average | 0.13 | 0.16 | 0.17 | 0.18 | −0.017 ± 0.005 |
| r5_anchored | fused, rank transfer | 0.027 | 0.033 | 0.043 | 0.050 | −0.13 ± 0.009 |
| r6_split70 | 70/30 split | 0.028 | 0.033 | 0.043 | 0.048 | −0.002 ± 0.002 |
| r7_acq4 | second cut | 0.043 | 0.048 | 0.054 | 0.067 | +0.018 ± 0.004 |
| r8_labels | **the labels line (today)** | 0.32 | 0.40 | 0.49 | **0.51** | **+0.45 ± 0.011** |

1. **Every curve leaves the typed query's dot at vote 23.** The typed query's own set scores F1 0.023: its
   blind cut returns thousands. No detector shows before Autopilot's opening ends.
2. **In F1, the middle of the ladder was a detour.**
   - Rungs r2 to r7 draw the line from the score mixture. With the pool at 1% positive, the mixture's cut
     returns far too much, and they end between 0.048 and 0.20.
   - The old ladder scored the same rungs in cost (FPR + FNR) at 5%, where the steps read as progress.
3. **Today's labels line restores what cross-calibration had, and a little more.** It draws the line from the
   votes, as cross-calibration did.
   - It tracks cross-calibration until vote 76.
   - It stays above it from vote 77.
   - It ends 0.03 above it, at 0.51, and +0.45 over the second cut.

The deck no longer draws this ladder. The owner asked what slide 32 should show (a cumulative build-up of how
much each addition helped), and the measurements in [The presets](#the-presets-4548) below answered it. The
slide is now `slides/fragments/follow-suit.md` (figure `slides/figs/src/make-follow-suit-fig.py`, from
`presets/progression_curve.csv`). The history chart that stood here is in git history (PR #4544).

## The presets (#4548)

**What the work since cross-calibration buys is a line that follows the user's preset.** Today's app also ran at
beta 1/4 (`r8_labels_b025`) and beta 4 (`r8_labels_b4`), 720 runs each, same pool. Each preset is set against
cross-calibration and scored at that preset's beta, paired by run (`presets/paired_at_own_beta.csv`):

| preset | Δ at vote 30 | 50 | 100 | **150** | today better / worse by > 0.05, at 150 |
|---|---|---|---|---|---|
| beta 1/4 | +0.066 ± 0.007 | +0.063 ± 0.007 | +0.124 ± 0.006 | **+0.168 ± 0.006** | 74% / 2% |
| beta 1 | 0.000 ± 0.004 | +0.001 ± 0.004 | +0.009 ± 0.003 | **+0.030 ± 0.003** | 30% / 9% |
| beta 4 | +0.008 ± 0.004 | 0.000 ± 0.004 | -0.005 ± 0.003 | **+0.001 ± 0.003** | 19% / 17% |

What the user gets at vote 150 (cross-calibration's line is the same set whatever the preference):

| | precision | recall | returned, median | runs returning > 200 |
|---|---:|---:|---:|---:|
| cross-calibration | 0.45 | 0.70 | 73 | 30% |
| today, beta 1/4 | 0.68 | 0.41 | 29 | 2.5% |
| today, beta 1 | 0.54 | 0.57 | 49 | 12% |
| today, beta 4 | 0.38 | 0.73 | 111 | 36% |

**Reading:**
- Cross-calibration's one line is long and recall-heavy. That already suits a beta 4 user, and today's app ties
  it there.
- The gain is large for the precision-minded user (+0.17, better in three sessions in four), small at the
  balance (+0.03), and zero for the recall-minded.
- Mean F1 at beta 1 hid it, because that is the preset closest to cross-calibration's one line.

**The typed query at its own line (#4625).** Before a run shows a detector, the app draws the typed query's
line per preset (#4603). Read there, the typed query scores F¼ 0.47, F1 0.37 and F4 0.47 on this pool, where
the beta-blind cut scored 0.012, 0.023 and 0.16. The first detector then ranks worse than the typed query
(#4384), so every line dips at vote 23-31 before it climbs:

| preset | typed query | today: lowest, back above at | cross-calibration: lowest, back above at | today / cross-calibration at 150 |
|---|---:|---|---|---:|
| beta 1/4 | 0.47 | 0.44, vote 34 | 0.39, never for good | 0.61 / 0.45 |
| beta 1 | 0.37 | 0.36, vote 28 | 0.35, vote 27 | 0.51 / 0.48 |
| beta 4 | 0.47 | 0.46, vote 30 | 0.43, vote 40 | 0.60 / 0.60 |

At beta 1/4, cross-calibration ends below the typed query it started from. The gaps at vote 150 do not move:
5% of runs have still not shown a detector there, and those runs carry the same notch in both lines.

**How these were made:**
- `presets/progression_curve.csv` and `presets/paired.csv` are `analyze_progression_4184.py` run on the four
  rungs together. Since #4625 the analyzer reads a baseline rebuilt on dev's `text_baseline.py`
  (`<run>/text_baseline_4625.csv`, with `CALIB_HAYSTACK_PREVALENCE=0.01`), which records each preset's line.
- The paired table at each preset's own beta was computed from the same cells, with a script kept beside the runs
  (`/expscratch/sgreenberg/slide32-4548/presets.py`).

## Files

- `progression_curve.csv`: per rung and vote, the filled means and SEs of F1, F¼, F4 and AP, with coverage.
- `paired.csv`: each rung against the one before, paired by cell and seed.
- `REPORT_progression.md`: the analyzer's machine summary, with the cost tables and the F1 tables.
- `provenance.json`: files read per rung (run paths shortened to `<run>/`).
