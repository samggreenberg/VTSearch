# #3547 — state at 2026-09-03 12:05 EDT, written before a reboot

Branch `claude/acq-deep-3547`, PR **#3598** (open); PR #3584 already MERGED to
dev (the `vg_scale_deep` pile). Worktree `/exp/sgreenberg/projects/vts-acq-3547`.

## THE GRID IS DONE — 1344/1344 cells, 0 failures

Study `/expscratch/sgreenberg/acq-3547`. Full tables in
`GENERATED_TABLES_3547.md` (this dir); regenerate with

    python scripts/experiments/calibration/frontier_3547.py \
      --base /expscratch/sgreenberg/acq-3547 --markdown <out>

**Do NOT read `analysis/REPORT_acq.md` as the verdict** — the chained analyzer's
default arm table covers only 5 of the 7 arms and silently drops `acq_m5`/
`acq_m6`. That is #3319's scope trap; `frontier_3547.py` is the study's own
analyzer and covers all seven.

### Verdicts

* **H3 — the plateau REPLICATES.** Δcost vs `prod` @100: `-1` −0.019, `-3`
  −0.038, `-4` −0.039, `-5` −0.030, `-6` −0.023. Flat across `-3`..`-5`, same
  shape as #3319. The pile added depth and nothing else, so the deep readings
  are about the horizon. **The anchor holds, so H1/H2 are readable.**
* **Falsifier BEHAVED** — `acq_p2` −11.2 positives [−11.9, −10.5], 192 pairs.
* **H1 — the optimum does NOT MOVE.** On the clean `-4` vs `-3` DiD (the only
  contrast where neither side is compressed): cost +0.0060 [−0.0016, +0.0134],
  AUC +0.0031 [−0.0013, +0.0073], clicks-to-target +2.8 [−5.9, +11.8]. All
  nulls. `-5`/`-6` lean "shallower" but are COMPRESSED and excluded as
  one-sided. **So the knob is a CONSTANT, not a schedule** — which is exactly
  the fork the issue named, answered.
* **H2 — the deep guardrail looks like EXHAUSTION, pending the control.**
  **Zero of 1344 cells acquire a first deep spike after t=100**; every first
  spike lands in t∈[21,56] (`spike_timing_3547.py`, output in
  `spike_timing_3547.txt`). `acq_m3` is 1.0% here at 19% harvest against
  #3319's 5.7% at 82%. Checked explicitly rather than trusted: an incidence
  identical at both horizons is equally consistent with a masking bug.
* **The ship is vindicated at depth**: `-4` reaches the control's final answer
  in 43.5 clicks against `prod`'s 153.5 — 3.5x, matching #3319's 3.2x, now in
  an environment where `-4`'s tail is NOT compressed (harvest 35.7%, and no
  cell anywhere above 80%).

### The one open confound — a control is RUNNING

H2 is cross-study, and TWO things differ from #3319: the pile AND 79 commits of
dev, including #3414 (which touched the very cost the inclusion knob prices).
So `/expscratch/sgreenberg/acq-3547-ctrl` re-runs the SHALLOW pile
(`vg_scale_any`, ~150 sim positives) at 400 clicks on the CURRENT commit,
2 arms x 192 cells.

    arrays 613686 (prod) + 613711 (acq_m3);  analyze 613712 (afterany)
    launch log /expscratch/sgreenberg/acq-3547-ctrl/launch.log
    driver     /exp/sgreenberg/ctrl3547.sh
    ETA        ~13:15 EDT

**Read it like this:** if late spikes REAPPEAR at ~82% harvest on this commit,
exhaustion is the cause and H2b is CONFIRMED. If they do not, the drop is dev
drift and H2 stays OPEN. Analyse with `spike_timing_3547.py` pointed at the
ctrl base (edit `BASE`), and with `frontier_3547.py --base .../acq-3547-ctrl`.

## Next

1. Read the control (~13:15) and settle H2.
2. Write `REPORT_3547.md`. Standing requirements: **2 significant digits**, and
   every report needs **figures and literal error examples**
   ([[report-precision-figures-examples]]).
3. File the follow-up the amendment names: a deep grid should size from the
   harvest of its DEEPEST arm, not its shipped one. 900 was a SUPPLY bound
   (all twelve classes) checked against a HORIZON bound (preflight 16b);
   neither is an AGGRESSION bound, and aggression is what sets harvest.
4. Issue **#3602** is already filed: `analyze_acq.py:218` computes
   `positives_100` as the trajectory's LAST row, not t=100 (while
   `positives_50` filters correctly). Invisible at a 100-click horizon, wrong
   on every deep wave — #3319's "Δ positives@100 = +90.1" is really t=400.

## Traps paid for in this session

* `git commit` FAILED rc=1 on ruff/ruff-format; run `ruff check --fix` and
  `ruff format` first and read the commit's OWN rc.
* Preflight refuses a launch with **uncommitted tracked changes** — commit
  before launching, not after.
* A literal beside a constant goes stale: the launcher's
  `--require-min-positives` now READS `CALIB_MIN_SIM_POSITIVES`.
* **Run long analyses under `sbatch` or `nohup`, not through a foreground ssh** —
  a killed watcher took an ssh with it, and a 20-minute pandas job on a LOGIN
  node is bad citizenship besides.
