# #3547 — state at 2026-09-02 16:45 EDT, written before a disconnect

Branch `claude/acq-deep-3547` @ `f2d0d687b`, **pushed**. Base dev `ba0c193cc`.
Worktree `/exp/sgreenberg/projects/vts-acq-3547`. Study `/expscratch/sgreenberg/acq-3547`.

## Verdict on the premise: AGREED, with three corrections

All three are in the commit message and in `PLAN_3547.md`. Short version:
prevalence must be held while positives grow (else k* moves 1.6 bits); the
scarcity was in the band split, not in VG (band-free designation gives 1006 on
the binding class against 414 banded); and the deeper cell needs a new name
because six studies stand on `vg_scale_any`.

## DONE and verified

* **The ceiling reproduces.** Recomputed from #3319's own cells: median harvest
  `prod` 14.7%, `-1` 24.0%, `-3` 82.0%, `-4` 85.3%; 29.2% of `-4` cells above
  90%. (`scripts/experiments/calibration/harvest_3547.py`)
* **Supply measured** over the full COCO-anchored label pass, not a pilot:
  band-free min **1006** (`stop sign`), median 1952, max 3455; thinnest single
  band 138 (`bus@small`). (`scripts/experiments/pile/measure_supply.py`)
* **Prefix-determinism CONFIRMED empirically** — 6336 cells per arm, all four
  arms identical at t=100 across #3319's two independent waves, on cost,
  n_good, thresholds and `acq_pool_percentile`. So ONE 400-step wave gives both
  horizons, paired within the cell. (`check_prefix_3547.py`)
* **`vg_scale_deep` BUILT**: `vg_scale_deep__siglip.pkl`, 75 MB, 22,363 medias,
  10,800 positives over 12 cells + 11,700 negatives + 300 spares, prevalence
  **0.071429** (identical to `vg_scale`'s, asserted at build time). 258s on a
  V100S. It is `on_request`, so it stays out of the pile's default sweep.
* Harness wired: `experiment_config.py` gives `vg_scale_deep` the same texts,
  `CALIB_VGSCALE_DEEP_EMBEDDERS` (default `siglip` alone), and the boxed flag.

## RUNNING right now (chained GRID-side, survives a disconnect)

    609831  acq3547-verify   -> 609832 acq3547-prep -> 609833 acq3547-size

`size` times ONE cell of the **deepest** arm (`k=-6`, index 0). Read it with

    sacct -j 609833 --format=JobID,JobName%18,MaxRSS,Elapsed,State
    tail /expscratch/sgreenberg/acq-3547/sizing/logs/size-609833.out

Driver: `/exp/sgreenberg/chain_3547.sh` (it carries the pinned environment).

## NOT done — the next session's first two jobs

1. **`launch_acq_3547.sh` does not exist yet.** The grid was deliberately not
   submitted. `PLAN_3547.md` pre-registers cell cost as *measured*, and the
   measurement (609833) had not landed; a 1344-cell array launched on an
   unreviewed launcher I could not watch is the wrong trade. Adapt
   `launch_acq_3319.sh`: `bin` half only, `CALIB_DATASETS=vg_scale_deep`,
   `CALIB_VGSCALE_DEEP_EMBEDDERS`, arms
   `prod,acq_m1,acq_m3,acq_m4,acq_m5,acq_m6,acq_p2`, `CALIB_MAX_STEPS=400`,
   `CALIB_MIN_SIM_POSITIVES=400`, job names `acq3547-*`, and `CALIB_TIME` set
   from 609833 rather than from #3319's 10m20s (the sim half is 3x deeper).
2. **Analyzer additions** `analyze_acq.py` needs for this plan: realised harvest
   as a first-class per-arm column, and the H1 difference-in-differences
   (`[m(deep,400) - m(shallow,400)] - [m(deep,100) - m(shallow,100)]`, paired
   per cell) rather than an argmin over a plateau.

## Traps already paid for

* **`git commit` exit code.** The first commit FAILED (rc=1) on `ruff` +
  `ruff-format`; the hook rewrote 5 files. Run `ruff check --fix` and
  `ruff format` first, and read the commit's own rc — a piped `tail` hides it.
* `scancel --name=acq3547-analyze` before reading any analysis: #3319's
  launcher-chained analyze silently clobbered a hand-run one with a
  default-scoped table that parsed cleanly and had lost half the arms.
* A pilot answers questions about the MACHINE, never about the DATA. `size` is
  for wall clock and memory only; harvest came from the full supply pass.
