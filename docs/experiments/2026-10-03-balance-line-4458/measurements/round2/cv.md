36 classes, 645 frames, 6048 rules per beta; folds: 21 / 15 classes.

### beta 0.25: fails

- chosen on fold 1, scored on fold 0: t=20 q=0.5 d=None geo=off/on/off tail=off
- chosen on fold 0, scored on fold 1: t=16 q=0.25 d=4 geo=off/on/off tail=off
- refit on all classes: t=16 q=0.5 d=4 geo=off/on/off tail=off

| clicks | shipped | held-out chosen | difference [95%] |
|---|---:|---:|---|
| 0–25 (the bar) | 0.768 | 0.836 | +0.068 [+0.027, +0.112] |
| 0 | 0.555 | 0.750 | +0.194 [+0.041, +0.335] |
| 1 | 0.771 | 0.811 | +0.039 [+0.006, +0.084] |
| 2 | 0.785 | 0.850 | +0.065 [+0.015, +0.123] |
| 3 | 0.798 | 0.841 | +0.043 [-0.023, +0.111] |
| 5 | 0.780 | 0.864 | +0.084 [+0.043, +0.126] |
| 10 | 0.762 | 0.859 | +0.096 [+0.048, +0.144] |
| 15 | 0.805 | 0.843 | +0.038 [-0.027, +0.099] |
| 25 | 0.876 | 0.861 | -0.015 [-0.084, +0.051] |
| 50 | 0.899 | 0.878 | -0.021 [-0.086, +0.039] |

Check on the other tier (final rule, clicks [0, 1, 2, 3, 5, 10, 15, 25]): +0.124 [+0.053, +0.190]

### beta 1: fails

- chosen on fold 1, scored on fold 0: t=10 q=0.25 d=4 geo=off/on/off tail=off
- chosen on fold 0, scored on fold 1: t=16 q=0.25 d=-2 geo=off/off/off tail=off
- refit on all classes: t=12 q=0.25 d=4 geo=off/on/off tail=off

| clicks | shipped | held-out chosen | difference [95%] |
|---|---:|---:|---|
| 0–25 (the bar) | 0.800 | 0.830 | +0.029 [-0.001, +0.060] |
| 0 | 0.662 | 0.793 | +0.132 [+0.041, +0.215] |
| 1 | 0.761 | 0.744 | -0.017 [-0.067, +0.041] |
| 2 | 0.798 | 0.817 | +0.019 [-0.024, +0.067] |
| 3 | 0.818 | 0.832 | +0.014 [-0.028, +0.063] |
| 5 | 0.811 | 0.832 | +0.020 [-0.014, +0.055] |
| 10 | 0.803 | 0.827 | +0.024 [-0.016, +0.064] |
| 15 | 0.843 | 0.870 | +0.026 [-0.009, +0.062] |
| 25 | 0.895 | 0.910 | +0.015 [-0.013, +0.046] |
| 50 | 0.910 | 0.926 | +0.016 [-0.009, +0.043] |

Check on the other tier (final rule, clicks [0, 1, 2, 3, 5, 10, 15, 25]): +0.061 [+0.001, +0.115]

### beta 4: PASSES

- chosen on fold 1, scored on fold 0: t=10 q=0.25 d=None geo=off/off/off tail=off
- chosen on fold 0, scored on fold 1: t=10 q=0.25 d=None geo=off/off/off tail=off
- refit on all classes: t=10 q=0.25 d=None geo=off/off/off tail=off

| clicks | shipped | held-out chosen | difference [95%] |
|---|---:|---:|---|
| 0–25 (the bar) | 0.855 | 0.914 | +0.060 [+0.020, +0.104] |
| 0 | 0.891 | 0.909 | +0.017 [-0.011, +0.044] |
| 1 | 0.770 | 0.875 | +0.105 [+0.033, +0.191] |
| 2 | 0.824 | 0.908 | +0.084 [+0.014, +0.166] |
| 3 | 0.847 | 0.918 | +0.070 [+0.015, +0.134] |
| 5 | 0.853 | 0.920 | +0.067 [+0.018, +0.118] |
| 10 | 0.864 | 0.923 | +0.059 [+0.012, +0.110] |
| 15 | 0.880 | 0.929 | +0.049 [+0.013, +0.093] |
| 25 | 0.894 | 0.932 | +0.038 [+0.000, +0.082] |
| 50 | 0.905 | 0.954 | +0.049 [+0.005, +0.101] |

Check on the other tier (final rule, clicks [0, 1, 2, 3, 5, 10, 15, 25]): +0.073 [+0.031, +0.116]
