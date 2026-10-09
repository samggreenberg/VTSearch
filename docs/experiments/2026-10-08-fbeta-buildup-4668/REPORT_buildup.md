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
| b025 | b4 | b4_check_b025 | 720 | 0 | 0.37 | 0.49 |
| b025 | b5 | b5_app_b025 | 720 | 0 | 0.37 | 0.41 |
| b025 | b6 | b5_app_b025 | 720 | 0 | 0.37 | 0.41 |
| b025 | b7 | b7_walk_b025 | 720 | 0 | 0.31 | 0.34 |
| b1 | b1 | b1_xcal | 720 | 0 | 0.37 | 0.63 |
| b1 | b2 | b2_labels_b1 | 720 | 0 | 0.37 | 0.63 |
| b1 | b3 | b3_floor_b1 | 720 | 0 | 0.37 | 0.63 |
| b1 | b4 | b4_check_b1 | 720 | 0 | 0.37 | 0.49 |
| b1 | b5 | b5_app_b1 | 720 | 0 | 0.37 | 0.41 |
| b1 | b6 | b5_app_b1 | 720 | 0 | 0.37 | 0.41 |
| b1 | b7 | b7_walk_b1 | 720 | 0 | 0.31 | 0.34 |
| b4 | b1 | b1_xcal | 720 | 0 | 0.37 | 0.63 |
| b4 | b2 | b2_labels_b4 | 720 | 0 | 0.37 | 0.63 |
| b4 | b3 | b3_floor_b4 | 720 | 0 | 0.37 | 0.63 |
| b4 | b4 | b4_check_b4 | 720 | 0 | 0.37 | 0.47 |
| b4 | b5 | b5_app_b4 | 720 | 0 | 0.37 | 0.38 |
| b4 | b6 | b5_app_b4 | 720 | 0 | 0.37 | 0.38 |
| b4 | b7 | b7_walk_b4 | 720 | 0 | 0.31 | 0.33 |

## Beta 0.25: the objective (filled) at checkpoints

| rung | t=0 | t=10 | t=25 | t=50 | t=100 | t=150 | mean 1-150 |
|---|---|---|---|---|---|---|---|
| b1 cross-calibration | 0.012 | 0.012 | 0.25 | 0.43 | 0.47 | 0.46 | 0.38 |
| b2 the labels line at the preset | 0.012 | 0.012 | 0.28 | 0.46 | 0.57 | 0.59 | 0.44 |
| b3 + the relative spread floor | 0.012 | 0.012 | 0.28 | 0.47 | 0.58 | 0.6 | 0.45 |
| b4 + the weak-separation check | 0.012 | 0.012 | 0.28 | 0.47 | 0.59 | 0.61 | 0.45 |
| b5 + even-odds asking (today's sessions) | 0.012 | 0.012 | 0.28 | 0.46 | 0.58 | 0.63 | 0.45 |
| b6 + the typed query's per-preset line (today's app) | 0.48 | 0.48 | 0.45 | 0.51 | 0.58 | 0.63 | 0.54 |
| b7 + the detector walk (not shipped) | 0.48 | 0.48 | 0.47 | 0.51 | 0.59 | 0.64 | 0.55 |

Each rung against the one before (positive = the step helped; bold = beyond 2 SE):

| step | t=50 | t=150 | mean 1-50 | mean 51-150 | mean 1-150 |
|---|---|---|---|---|---|
| b1 → b2 | **+0.035** ± 0.005 | **+0.131** ± 0.005 | **+0.014** ± 0.001 | **+0.089** ± 0.004 | **+0.064** ± 0.003 |
| b2 → b3 | +0.001 ± 0.003 | **+0.009** ± 0.003 | -0.000 ± 0.001 | **+0.008** ± 0.001 | **+0.005** ± 0.001 |
| b3 → b4 | +0.000 ± 0.002 | **+0.011** ± 0.004 | **-0.001** ± 0.000 | **+0.008** ± 0.002 | **+0.005** ± 0.001 |
| b4 → b5 | -0.002 ± 0.003 | **+0.017** ± 0.004 | +0.000 ± 0.001 | -0.000 ± 0.002 | -0.000 ± 0.001 |
| b5 → b6 | **+0.043** ± 0.004 | -0.000 ± 0.000 | **+0.258** ± 0.007 | **+0.009** ± 0.001 | **+0.092** ± 0.003 |
| b6 → b7 | -0.000 ± 0.006 | +0.005 ± 0.003 | +0.001 ± 0.002 | **+0.011** ± 0.003 | **+0.008** ± 0.002 |

## Beta 1: the objective (filled) at checkpoints

| rung | t=0 | t=10 | t=25 | t=50 | t=100 | t=150 | mean 1-150 |
|---|---|---|---|---|---|---|---|
| b1 cross-calibration | 0.023 | 0.023 | 0.24 | 0.4 | 0.48 | 0.49 | 0.38 |
| b2 the labels line at the preset | 0.023 | 0.023 | 0.23 | 0.4 | 0.48 | 0.5 | 0.37 |
| b3 + the relative spread floor | 0.023 | 0.023 | 0.25 | 0.4 | 0.48 | 0.5 | 0.38 |
| b4 + the weak-separation check | 0.023 | 0.023 | 0.25 | 0.4 | 0.49 | 0.51 | 0.38 |
| b5 + even-odds asking (today's sessions) | 0.023 | 0.023 | 0.25 | 0.4 | 0.49 | 0.53 | 0.38 |
| b6 + the typed query's per-preset line (today's app) | 0.37 | 0.37 | 0.36 | 0.42 | 0.49 | 0.53 | 0.45 |
| b7 + the detector walk (not shipped) | 0.37 | 0.37 | 0.36 | 0.42 | 0.5 | 0.53 | 0.45 |

Each rung against the one before (positive = the step helped; bold = beyond 2 SE):

| step | t=50 | t=150 | mean 1-50 | mean 51-150 | mean 1-150 |
|---|---|---|---|---|---|
| b1 → b2 | **-0.009** ± 0.003 | **+0.008** ± 0.002 | **-0.005** ± 0.001 | -0.002 ± 0.001 | **-0.003** ± 0.001 |
| b2 → b3 | +0.001 ± 0.002 | +0.000 ± 0.002 | **+0.003** ± 0.001 | -0.000 ± 0.001 | +0.001 ± 0.001 |
| b3 → b4 | -0.001 ± 0.001 | **+0.013** ± 0.002 | **-0.001** ± 0.000 | **+0.008** ± 0.001 | **+0.005** ± 0.001 |
| b4 → b5 | +0.000 ± 0.002 | **+0.013** ± 0.002 | -0.001 ± 0.001 | **+0.005** ± 0.001 | **+0.003** ± 0.001 |
| b5 → b6 | **+0.026** ± 0.003 | **-0.000** ± 0.000 | **+0.186** ± 0.005 | **+0.005** ± 0.001 | **+0.065** ± 0.002 |
| b6 → b7 | -0.006 ± 0.004 | **+0.006** ± 0.002 | -0.001 ± 0.001 | **+0.006** ± 0.002 | **+0.004** ± 0.001 |

## Beta 4: the objective (filled) at checkpoints

| rung | t=0 | t=10 | t=25 | t=50 | t=100 | t=150 | mean 1-150 |
|---|---|---|---|---|---|---|---|
| b1 cross-calibration | 0.16 | 0.16 | 0.34 | 0.46 | 0.56 | 0.6 | 0.46 |
| b2 the labels line at the preset | 0.16 | 0.16 | 0.35 | 0.49 | 0.57 | 0.6 | 0.48 |
| b3 + the relative spread floor | 0.16 | 0.16 | 0.36 | 0.48 | 0.56 | 0.59 | 0.47 |
| b4 + the weak-separation check | 0.16 | 0.16 | 0.36 | 0.48 | 0.57 | 0.61 | 0.48 |
| b5 + even-odds asking (today's sessions) | 0.16 | 0.16 | 0.36 | 0.49 | 0.58 | 0.62 | 0.48 |
| b6 + the typed query's per-preset line (today's app) | 0.47 | 0.47 | 0.46 | 0.51 | 0.58 | 0.61 | 0.54 |
| b7 + the detector walk (not shipped) | 0.47 | 0.47 | 0.45 | 0.5 | 0.59 | 0.62 | 0.54 |

Each rung against the one before (positive = the step helped; bold = beyond 2 SE):

| step | t=50 | t=150 | mean 1-50 | mean 51-150 | mean 1-150 |
|---|---|---|---|---|---|
| b1 → b2 | **+0.030** ± 0.003 | -0.000 ± 0.003 | **+0.016** ± 0.001 | **+0.013** ± 0.002 | **+0.014** ± 0.002 |
| b2 → b3 | **-0.007** ± 0.002 | **-0.005** ± 0.002 | **+0.003** ± 0.001 | **-0.009** ± 0.001 | **-0.005** ± 0.001 |
| b3 → b4 | **-0.003** ± 0.001 | **+0.017** ± 0.002 | **-0.001** ± 0.000 | **+0.008** ± 0.001 | **+0.005** ± 0.001 |
| b4 → b5 | **+0.009** ± 0.002 | **+0.011** ± 0.002 | +0.001 ± 0.001 | **+0.013** ± 0.001 | **+0.009** ± 0.001 |
| b5 → b6 | **+0.018** ± 0.003 | **-0.003** ± 0.001 | **+0.164** ± 0.006 | -0.001 ± 0.001 | **+0.054** ± 0.002 |
| b6 → b7 | -0.002 ± 0.003 | +0.004 ± 0.002 | -0.001 ± 0.001 | **+0.006** ± 0.002 | **+0.003** ± 0.001 |

## What the user gets at the horizon (runs showing a detector)

| preset | rung | precision | recall | returned, median | over 200 |
|---|---|---|---|---|---|
| b025 | b1 | 0.48 | 0.69 | 69 | 25% |
| b025 | b2 | 0.67 | 0.47 | 36 | 5% |
| b025 | b3 | 0.69 | 0.44 | 33 | 4% |
| b025 | b4 | 0.71 | 0.42 | 29 | 1% |
| b025 | b5 | 0.76 | 0.40 | 24 | 0% |
| b025 | b6 | 0.76 | 0.40 | 24 | 0% |
| b025 | b7 | 0.77 | 0.41 | 23 | 0% |
| b1 | b1 | 0.48 | 0.69 | 69 | 25% |
| b1 | b2 | 0.53 | 0.63 | 54 | 20% |
| b1 | b3 | 0.55 | 0.62 | 51 | 20% |
| b1 | b4 | 0.57 | 0.58 | 49 | 11% |
| b1 | b5 | 0.59 | 0.58 | 48 | 7% |
| b1 | b6 | 0.59 | 0.58 | 48 | 7% |
| b1 | b7 | 0.60 | 0.59 | 48 | 4% |
| b4 | b1 | 0.48 | 0.69 | 69 | 25% |
| b4 | b2 | 0.37 | 0.80 | 125 | 42% |
| b4 | b3 | 0.39 | 0.79 | 122 | 42% |
| b4 | b4 | 0.40 | 0.75 | 113 | 36% |
| b4 | b5 | 0.37 | 0.75 | 126 | 37% |
| b4 | b6 | 0.37 | 0.75 | 126 | 37% |
| b4 | b7 | 0.39 | 0.75 | 124 | 35% |
