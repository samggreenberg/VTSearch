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

The deck's slide is `slides/fragments/progression.md` (figure `slides/figs/src/make-progression-fig.py`, from
`progression_curve.csv`).

## Files

- `progression_curve.csv`: per rung and vote, the filled means and SEs of F1, F¼, F4 and AP, with coverage.
- `paired.csv`: each rung against the one before, paired by cell and seed.
- `REPORT_progression.md`: the analyzer's machine summary, with the cost tables and the F1 tables.
- `provenance.json`: files read per rung (run paths shortened to `<run>/`).
