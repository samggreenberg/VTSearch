# #4668 build-up - machine summary

- `b1_xcal`: 717/720 cells with data, 102,725 rows, 3 starved (header-only; not data loss)
- `b2_labels_b025`: 717/720 cells with data, 102,725 rows, 3 starved (header-only; not data loss)
- `b2_labels_b1`: 717/720 cells with data, 102,725 rows, 3 starved (header-only; not data loss)
- `b2_labels_b4`: 717/720 cells with data, 102,725 rows, 3 starved (header-only; not data loss)
- `b3_floor_b025`: 717/720 cells with data, 102,725 rows, 3 starved (header-only; not data loss)
- `b3_floor_b1`: 717/720 cells with data, 102,725 rows, 3 starved (header-only; not data loss)
- `b3_floor_b4`: 717/720 cells with data, 102,725 rows, 3 starved (header-only; not data loss)
- `b4_check_b025`: 717/720 cells with data, 92,433 rows, 2,996 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `b4_check_b1`: 717/720 cells with data, 92,017 rows, 2,999 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `b4_check_b4`: 717/720 cells with data, 90,789 rows, 4,359 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `b5_app_b025`: 717/720 cells with data, 88,765 rows, 3,004 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `b5_app_b1`: 717/720 cells with data, 88,393 rows, 3,031 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `b5_app_b4`: 717/720 cells with data, 86,285 rows, 4,362 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `b7_walk_b025`: 717/720 cells with data, 88,313 rows, 2,978 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `b7_walk_b1`: 717/720 cells with data, 88,157 rows, 2,985 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `b7_walk_b4`: 717/720 cells with data, 85,637 rows, 4,361 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)

Grid: 720 cells.

| preset | rung | arm | cells | lost | coverage@25 | coverage@50 |
|---|---|---|---|---|---|---|
| b025 | b1 | b1_xcal | 720 | 0 | 0.37 | 0.63 |
| b025 | b2 | b2_labels_b025 | 720 | 0 | 0.37 | 0.63 |
| b025 | b3 | b3_floor_b025 | 720 | 0 | 0.37 | 0.63 |
| b025 | b4 | b4_check_b025 | 720 | 0 | 0.37 | 0.63 |
| b025 | b5 | b5_app_b025 | 720 | 0 | 0.37 | 0.63 |
| b025 | b6 | b5_app_b025 | 720 | 0 | 0.37 | 0.63 |
| b025 | b7 | b7_walk_b025 | 720 | 0 | 0.31 | 0.67 |
| b1 | b1 | b1_xcal | 720 | 0 | 0.37 | 0.63 |
| b1 | b2 | b2_labels_b1 | 720 | 0 | 0.37 | 0.63 |
| b1 | b3 | b3_floor_b1 | 720 | 0 | 0.37 | 0.63 |
| b1 | b4 | b4_check_b1 | 720 | 0 | 0.37 | 0.63 |
| b1 | b5 | b5_app_b1 | 720 | 0 | 0.37 | 0.63 |
| b1 | b6 | b5_app_b1 | 720 | 0 | 0.37 | 0.63 |
| b1 | b7 | b7_walk_b1 | 720 | 0 | 0.31 | 0.67 |
| b4 | b1 | b1_xcal | 720 | 0 | 0.37 | 0.63 |
| b4 | b2 | b2_labels_b4 | 720 | 0 | 0.37 | 0.63 |
| b4 | b3 | b3_floor_b4 | 720 | 0 | 0.37 | 0.63 |
| b4 | b4 | b4_check_b4 | 720 | 0 | 0.37 | 0.63 |
| b4 | b5 | b5_app_b4 | 720 | 0 | 0.37 | 0.63 |
| b4 | b6 | b5_app_b4 | 720 | 0 | 0.37 | 0.63 |
| b4 | b7 | b7_walk_b4 | 720 | 0 | 0.31 | 0.67 |

## Beta 0.25: the objective (filled), the session mean first

| rung | mean 1-150 | mean 1-50 | t=0 | t=10 | t=25 | t=50 | t=100 | t=150 |
|---|---|---|---|---|---|---|---|---|
| b1 cross-calibration | 0.38 | 0.21 | 0.012 | 0.012 | 0.25 | 0.43 | 0.47 | 0.46 |
| b2 the labels line at the preset | 0.44 | 0.22 | 0.012 | 0.012 | 0.28 | 0.46 | 0.57 | 0.59 |
| b3 + the relative spread floor | 0.45 | 0.22 | 0.012 | 0.012 | 0.28 | 0.47 | 0.58 | 0.6 |
| b4 + the weak-separation check | 0.45 | 0.22 | 0.012 | 0.012 | 0.28 | 0.47 | 0.59 | 0.61 |
| b5 + even-odds asking (today's sessions) | 0.45 | 0.22 | 0.012 | 0.012 | 0.28 | 0.46 | 0.58 | 0.63 |
| b6 + the typed query's per-preset line (today's app) | 0.54 | 0.48 | 0.48 | 0.48 | 0.45 | 0.51 | 0.58 | 0.63 |
| b7 + the detector walk (not shipped) | 0.55 | 0.48 | 0.48 | 0.48 | 0.47 | 0.51 | 0.59 | 0.64 |

Each rung against the one before (positive = the step helped; bold = beyond 2 SE):

| step | mean 1-150 | mean 1-50 | mean 51-150 | t=50 | t=150 |
|---|---|---|---|---|---|
| b1 → b2 | **+0.064** ± 0.003 | **+0.014** ± 0.001 | **+0.089** ± 0.004 | **+0.035** ± 0.005 | **+0.131** ± 0.005 |
| b2 → b3 | **+0.005** ± 0.001 | -0.000 ± 0.001 | **+0.008** ± 0.001 | +0.001 ± 0.003 | **+0.009** ± 0.003 |
| b3 → b4 | **+0.005** ± 0.001 | **-0.001** ± 0.000 | **+0.008** ± 0.002 | +0.000 ± 0.002 | **+0.011** ± 0.004 |
| b4 → b5 | -0.000 ± 0.001 | +0.000 ± 0.001 | -0.000 ± 0.002 | -0.002 ± 0.003 | **+0.017** ± 0.004 |
| b5 → b6 | **+0.092** ± 0.003 | **+0.258** ± 0.007 | **+0.009** ± 0.001 | **+0.043** ± 0.004 | -0.000 ± 0.000 |
| b6 → b7 | **+0.008** ± 0.002 | +0.001 ± 0.002 | **+0.011** ± 0.003 | -0.000 ± 0.006 | +0.005 ± 0.003 |

## Beta 1: the objective (filled), the session mean first

| rung | mean 1-150 | mean 1-50 | t=0 | t=10 | t=25 | t=50 | t=100 | t=150 |
|---|---|---|---|---|---|---|---|---|
| b1 cross-calibration | 0.38 | 0.2 | 0.023 | 0.023 | 0.24 | 0.4 | 0.48 | 0.49 |
| b2 the labels line at the preset | 0.37 | 0.19 | 0.023 | 0.023 | 0.23 | 0.4 | 0.48 | 0.5 |
| b3 + the relative spread floor | 0.38 | 0.2 | 0.023 | 0.023 | 0.25 | 0.4 | 0.48 | 0.5 |
| b4 + the weak-separation check | 0.38 | 0.19 | 0.023 | 0.023 | 0.25 | 0.4 | 0.49 | 0.51 |
| b5 + even-odds asking (today's sessions) | 0.38 | 0.19 | 0.023 | 0.023 | 0.25 | 0.4 | 0.49 | 0.53 |
| b6 + the typed query's per-preset line (today's app) | 0.45 | 0.38 | 0.37 | 0.37 | 0.36 | 0.42 | 0.49 | 0.53 |
| b7 + the detector walk (not shipped) | 0.45 | 0.38 | 0.37 | 0.37 | 0.36 | 0.42 | 0.5 | 0.53 |

Each rung against the one before (positive = the step helped; bold = beyond 2 SE):

| step | mean 1-150 | mean 1-50 | mean 51-150 | t=50 | t=150 |
|---|---|---|---|---|---|
| b1 → b2 | **-0.003** ± 0.001 | **-0.005** ± 0.001 | -0.002 ± 0.001 | **-0.009** ± 0.003 | **+0.008** ± 0.002 |
| b2 → b3 | +0.001 ± 0.001 | **+0.003** ± 0.001 | -0.000 ± 0.001 | +0.001 ± 0.002 | +0.000 ± 0.002 |
| b3 → b4 | **+0.005** ± 0.001 | **-0.001** ± 0.000 | **+0.008** ± 0.001 | -0.001 ± 0.001 | **+0.013** ± 0.002 |
| b4 → b5 | **+0.003** ± 0.001 | -0.001 ± 0.001 | **+0.005** ± 0.001 | +0.000 ± 0.002 | **+0.013** ± 0.002 |
| b5 → b6 | **+0.065** ± 0.002 | **+0.186** ± 0.005 | **+0.005** ± 0.001 | **+0.026** ± 0.003 | **-0.000** ± 0.000 |
| b6 → b7 | **+0.004** ± 0.001 | -0.001 ± 0.001 | **+0.006** ± 0.002 | -0.006 ± 0.004 | **+0.006** ± 0.002 |

## Beta 4: the objective (filled), the session mean first

| rung | mean 1-150 | mean 1-50 | t=0 | t=10 | t=25 | t=50 | t=100 | t=150 |
|---|---|---|---|---|---|---|---|---|
| b1 cross-calibration | 0.46 | 0.29 | 0.16 | 0.16 | 0.34 | 0.46 | 0.56 | 0.6 |
| b2 the labels line at the preset | 0.48 | 0.31 | 0.16 | 0.16 | 0.35 | 0.49 | 0.57 | 0.6 |
| b3 + the relative spread floor | 0.47 | 0.31 | 0.16 | 0.16 | 0.36 | 0.48 | 0.56 | 0.59 |
| b4 + the weak-separation check | 0.48 | 0.31 | 0.16 | 0.16 | 0.36 | 0.48 | 0.57 | 0.61 |
| b5 + even-odds asking (today's sessions) | 0.48 | 0.31 | 0.16 | 0.16 | 0.36 | 0.49 | 0.58 | 0.62 |
| b6 + the typed query's per-preset line (today's app) | 0.54 | 0.48 | 0.47 | 0.47 | 0.46 | 0.51 | 0.58 | 0.61 |
| b7 + the detector walk (not shipped) | 0.54 | 0.47 | 0.47 | 0.47 | 0.45 | 0.5 | 0.59 | 0.62 |

Each rung against the one before (positive = the step helped; bold = beyond 2 SE):

| step | mean 1-150 | mean 1-50 | mean 51-150 | t=50 | t=150 |
|---|---|---|---|---|---|
| b1 → b2 | **+0.014** ± 0.002 | **+0.016** ± 0.001 | **+0.013** ± 0.002 | **+0.030** ± 0.003 | -0.000 ± 0.003 |
| b2 → b3 | **-0.005** ± 0.001 | **+0.003** ± 0.001 | **-0.009** ± 0.001 | **-0.007** ± 0.002 | **-0.005** ± 0.002 |
| b3 → b4 | **+0.005** ± 0.001 | **-0.001** ± 0.000 | **+0.008** ± 0.001 | **-0.003** ± 0.001 | **+0.017** ± 0.002 |
| b4 → b5 | **+0.009** ± 0.001 | +0.001 ± 0.001 | **+0.013** ± 0.001 | **+0.009** ± 0.002 | **+0.011** ± 0.002 |
| b5 → b6 | **+0.054** ± 0.002 | **+0.164** ± 0.006 | -0.001 ± 0.001 | **+0.018** ± 0.003 | **-0.003** ± 0.001 |
| b6 → b7 | **+0.003** ± 0.001 | -0.001 ± 0.001 | **+0.006** ± 0.002 | -0.002 ± 0.003 | +0.004 ± 0.002 |

## What the user gets, over the session (every run, every vote)

Every run at every vote, filled as the curves are: until a detector shows, the user's set is the typed query's own, at the line of the rung's era (the midpoint for b1-b5, today's per-preset line for b6-b7); after, the last line shown. Each value is a mean over the votes in the window (`returned_curve.csv` has every vote). 'Over 200' and 'nothing' are shares of run-votes.

| preset | rung | votes | showing a detector | precision | recall | a run's mean returned, median | over 200 | nothing |
|---|---|---|---|---|---|---|---|---|
| b025 | b1 | 1-150 | 66% | 0.38 | 0.74 | 1104 | 47% | 0% |
| b025 | b1 | 1-50 | 28% | 0.21 | 0.83 | 2967 | 74% | 0% |
| b025 | b2 | 1-150 | 66% | 0.48 | 0.60 | 1051 | 36% | 0% |
| b025 | b2 | 1-50 | 28% | 0.25 | 0.78 | 2946 | 72% | 0% |
| b025 | b3 | 1-150 | 66% | 0.50 | 0.58 | 1048 | 37% | 0% |
| b025 | b3 | 1-50 | 28% | 0.24 | 0.78 | 2954 | 73% | 0% |
| b025 | b4 | 1-150 | 66% | 0.51 | 0.57 | 1044 | 35% | 0% |
| b025 | b4 | 1-50 | 28% | 0.24 | 0.78 | 2954 | 73% | 0% |
| b025 | b5 | 1-150 | 66% | 0.53 | 0.54 | 1042 | 35% | 0% |
| b025 | b5 | 1-50 | 28% | 0.25 | 0.76 | 2946 | 73% | 0% |
| b025 | b6 | 1-150 | 66% | 0.64 | 0.29 | 25 | 1% | 0% |
| b025 | b6 | 1-50 | 28% | 0.56 | 0.24 | 20 | 1% | 1% |
| b025 | b7 | 1-150 | 66% | 0.66 | 0.27 | 17 | 0% | 1% |
| b025 | b7 | 1-50 | 29% | 0.57 | 0.22 | 16 | 0% | 1% |
| b1 | b1 | 1-150 | 66% | 0.38 | 0.74 | 1104 | 47% | 0% |
| b1 | b1 | 1-50 | 28% | 0.21 | 0.83 | 2967 | 74% | 0% |
| b1 | b2 | 1-150 | 66% | 0.39 | 0.70 | 1089 | 43% | 0% |
| b1 | b2 | 1-50 | 28% | 0.22 | 0.82 | 2959 | 73% | 0% |
| b1 | b3 | 1-150 | 66% | 0.41 | 0.69 | 1081 | 44% | 0% |
| b1 | b3 | 1-50 | 28% | 0.22 | 0.82 | 2965 | 73% | 0% |
| b1 | b4 | 1-150 | 66% | 0.42 | 0.67 | 1080 | 40% | 0% |
| b1 | b4 | 1-50 | 28% | 0.21 | 0.82 | 2967 | 73% | 0% |
| b1 | b5 | 1-150 | 66% | 0.44 | 0.66 | 1076 | 38% | 0% |
| b1 | b5 | 1-50 | 28% | 0.22 | 0.81 | 2962 | 73% | 0% |
| b1 | b6 | 1-150 | 66% | 0.51 | 0.46 | 54 | 6% | 0% |
| b1 | b6 | 1-50 | 28% | 0.43 | 0.41 | 49 | 4% | 0% |
| b1 | b7 | 1-150 | 66% | 0.53 | 0.45 | 46 | 3% | 0% |
| b1 | b7 | 1-50 | 29% | 0.45 | 0.39 | 43 | 3% | 0% |
| b4 | b1 | 1-150 | 66% | 0.38 | 0.74 | 1104 | 47% | 0% |
| b4 | b1 | 1-50 | 28% | 0.21 | 0.83 | 2967 | 74% | 0% |
| b4 | b2 | 1-150 | 66% | 0.28 | 0.80 | 1199 | 55% | 0% |
| b4 | b2 | 1-50 | 28% | 0.17 | 0.85 | 2977 | 76% | 0% |
| b4 | b3 | 1-150 | 66% | 0.30 | 0.80 | 1212 | 55% | 0% |
| b4 | b3 | 1-50 | 28% | 0.16 | 0.86 | 2988 | 76% | 0% |
| b4 | b4 | 1-150 | 66% | 0.30 | 0.78 | 1203 | 53% | 0% |
| b4 | b4 | 1-50 | 28% | 0.16 | 0.85 | 2988 | 76% | 0% |
| b4 | b5 | 1-150 | 66% | 0.28 | 0.79 | 1190 | 53% | 0% |
| b4 | b5 | 1-50 | 28% | 0.15 | 0.86 | 2988 | 77% | 0% |
| b4 | b6 | 1-150 | 66% | 0.31 | 0.63 | 177 | 36% | 0% |
| b4 | b6 | 1-50 | 28% | 0.24 | 0.56 | 173 | 40% | 0% |
| b4 | b7 | 1-150 | 66% | 0.33 | 0.61 | 154 | 32% | 0% |
| b4 | b7 | 1-50 | 29% | 0.27 | 0.55 | 152 | 36% | 0% |
