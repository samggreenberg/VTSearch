# The objective's σ per cell, by preset (#4584)

The per-cell SD of the paired Δ-objective in `analyze_ab`'s decision window: scope `app_visible`, all steps (`n_votes >= 2`), each arm's own rows, one mean per run. This is what `preflight.sh --resolve-delta` sizes a balance A/B with.

| pair (same commit, coco_better, binary SigLIP, 1% pool) | beta 1/4 | beta 1 | beta 4 |
|---|---|---|---|
| #4583: split, fold count, both (685 runs each, 150 votes) | 0.054–0.059 | 0.030–0.035 | 0.038–0.044 |
| #4428: acquisition depth `x0.25` vs shipped (414 runs, 300 votes) | | 0.082 | |

The defaults set from these are 0.13 / 0.08 / 0.10. The β 1 value is the acquisition pair's, the larger of the two: arms that change what Autopilot asks part their trajectories more than arms that only move the line, here about 2.4×. β ¼ and β 4 are β 1 scaled by the ratios within #4583 (×1.6 and ×1.24).

- `sigma_ab_window.csv`: the #4583 rows, by contrast and window. The ramp window (votes 6–20) is empty because no detector shows before the opening ends, at about vote 23.
- [`sigma_objective_4584.py`](../../../scripts/experiments/calibration/sigma_objective_4584.py) reproduces both rows: `--pair $B/f05k2_b1=$B/f03k2_b1` etc. with `B=/expscratch/sgreenberg/calsplit-4583`, and `--pair $A/b1-x0.25-c300=$A/b1-xship-c300` with `A=/expscratch/sgreenberg/acq-sweep-4428`.
