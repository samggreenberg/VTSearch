36 classes, 645 frames.
- chosen on fold 1, scored on fold 0: a loose fit passes at max(64, ceil(None x the Goods' median LOO inliers))
- chosen on fold 0, scored on fold 1: a loose fit passes at max(24, ceil(0.5 x the Goods' median LOO inliers))
- refit on all classes: a loose fit passes at max(48, ceil(0.5 x the Goods' median LOO inliers))

### beta 1

| clicks | shipped | held-out | difference [95%] |
|---|---:|---:|---|
| 0–25 | 0.800 | 0.817 | +0.017 [+0.003, +0.036] |
| 0 | 0.662 | 0.662 | +0.000 [+0.000, +0.000] |
| 1 | 0.761 | 0.787 | +0.026 [-0.018, +0.082] |
| 2 | 0.798 | 0.827 | +0.028 [+0.006, +0.059] |
| 3 | 0.818 | 0.846 | +0.028 [+0.006, +0.058] |
| 5 | 0.811 | 0.834 | +0.022 [+0.007, +0.041] |
| 10 | 0.803 | 0.818 | +0.014 [+0.001, +0.030] |
| 15 | 0.843 | 0.852 | +0.009 [-0.002, +0.023] |
| 25 | 0.895 | 0.902 | +0.008 [+0.000, +0.018] |
| 50 | 0.910 | 0.918 | +0.008 [+0.000, +0.022] |

### beta 0.25

| clicks | shipped | held-out | difference [95%] |
|---|---:|---:|---|
| 0–25 | 0.768 | 0.767 | -0.001 [-0.007, +0.006] |
| 0 | 0.555 | 0.555 | +0.000 [+0.000, +0.000] |
| 1 | 0.771 | 0.764 | -0.007 [-0.048, +0.038] |
| 2 | 0.785 | 0.784 | -0.001 [-0.016, +0.011] |
| 3 | 0.798 | 0.797 | -0.001 [-0.018, +0.012] |
| 5 | 0.780 | 0.781 | +0.000 [-0.014, +0.012] |
| 10 | 0.762 | 0.760 | -0.002 [-0.016, +0.008] |
| 15 | 0.805 | 0.805 | -0.000 [-0.008, +0.007] |
| 25 | 0.876 | 0.879 | +0.002 [+0.000, +0.006] |
| 50 | 0.899 | 0.901 | +0.002 [+0.000, +0.005] |

**fails.** Frames the held-out rule changes, by class: {'spods/stamp_00129_1': 10, 'spods/stamp_00293_1': 8, 'spods/stamp_00514_1': 10, 'spods/stamp_00546_1': 4, 'spods/stamp_00577_1': 10, 'spods/stamp_00716_1': 9, 'spods/stamp_00737_1': 9, 'spods/stamp_00996_1': 13, 'staver/stamp_stampds-00230_0': 1, 'tobacco800/logo_aeq93a00_1': 6, 'tobacco800/logo_ajj10e00_1': 16, 'spods/stamp_00641_1': 1, 'tobacco800/logo_ald41a00-ernest_1': 1, 'ucsf/logo_rjr_script': 1}
