# #4671 Autopilot pieces - machine summary

- `ctl_b025`: 717/720 cells with data, 88,765 rows, 3,004 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `open_b025`: 717/720 cells with data, 79,881 rows, 2,984 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `nowalk_b025`: 717/720 cells with data, 84,661 rows, 2,994 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `ctl_b1`: 717/720 cells with data, 88,393 rows, 3,031 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `open_b1`: 717/720 cells with data, 79,361 rows, 2,990 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `nowalk_b1`: 717/720 cells with data, 84,421 rows, 2,999 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `ctl_b4`: 717/720 cells with data, 86,285 rows, 4,362 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `open_b4`: 717/720 cells with data, 76,125 rows, 4,372 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `nowalk_b4`: 717/720 cells with data, 81,249 rows, 4,358 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)

Grid: 720 cells.

## Session means (every run, every vote)

| preset | arm | objective 1-150 | 1-50 | 51-150 | AP 1-150 | precision 1-150 | recall 1-150 | over 200 | showing a detector |
|---|---|---|---|---|---|---|---|---|---|
| b025 | ctl | 0.545 | 0.479 | 0.578 | 0.485 | 0.64 | 0.29 | 1% | 66% |
| b025 | open | 0.554 | 0.474 | 0.594 | 0.495 | 0.68 | 0.28 | 1% | 86% |
| b025 | nowalk | 0.554 | 0.481 | 0.590 | 0.487 | 0.64 | 0.31 | 1% | 66% |
| b1 | ctl | 0.449 | 0.380 | 0.483 | 0.485 | 0.51 | 0.46 | 6% | 66% |
| b1 | open | 0.455 | 0.373 | 0.496 | 0.495 | 0.56 | 0.45 | 4% | 86% |
| b1 | nowalk | 0.456 | 0.381 | 0.493 | 0.487 | 0.50 | 0.48 | 6% | 66% |
| b4 | ctl | 0.539 | 0.475 | 0.571 | 0.484 | 0.31 | 0.63 | 36% | 66% |
| b4 | open | 0.542 | 0.457 | 0.585 | 0.492 | 0.34 | 0.61 | 30% | 86% |
| b4 | nowalk | 0.542 | 0.474 | 0.575 | 0.486 | 0.30 | 0.63 | 36% | 66% |

## Each arm minus today's app (paired; bold = beyond 2 SE)

| preset | arm | metric | mean 1-150 | mean 1-50 | mean 51-150 | mean 1-25 | t=150 |
|---|---|---|---|---|---|---|---|
| b025 | open | objective | **+0.009** ± 0.003 | -0.005 ± 0.004 | **+0.016** ± 0.003 | **-0.046** ± 0.005 | +0.005 ± 0.003 |
| b025 | open | ap | **+0.009** ± 0.002 | +0.003 ± 0.003 | **+0.013** ± 0.002 | **-0.024** ± 0.003 | **+0.003** ± 0.001 |
| b025 | nowalk | objective | **+0.009** ± 0.001 | **+0.002** ± 0.001 | **+0.013** ± 0.001 | +0.000 ± 0.000 | +0.000 ± 0.003 |
| b025 | nowalk | ap | **+0.002** ± 0.000 | **-0.001** ± 0.000 | **+0.004** ± 0.000 | +0.000 ± 0.000 | **+0.003** ± 0.001 |
| b1 | open | objective | **+0.006** ± 0.002 | **-0.007** ± 0.003 | **+0.013** ± 0.002 | **-0.035** ± 0.004 | **+0.007** ± 0.002 |
| b1 | open | ap | **+0.010** ± 0.002 | +0.003 ± 0.003 | **+0.013** ± 0.002 | **-0.024** ± 0.003 | **+0.003** ± 0.001 |
| b1 | nowalk | objective | **+0.007** ± 0.001 | +0.001 ± 0.001 | **+0.009** ± 0.001 | +0.000 ± 0.000 | +0.001 ± 0.001 |
| b1 | nowalk | ap | **+0.002** ± 0.000 | **-0.001** ± 0.000 | **+0.003** ± 0.000 | +0.000 ± 0.000 | **+0.003** ± 0.001 |
| b4 | open | objective | +0.003 ± 0.002 | **-0.018** ± 0.003 | **+0.014** ± 0.002 | **-0.044** ± 0.004 | **+0.011** ± 0.002 |
| b4 | open | ap | **+0.008** ± 0.002 | +0.000 ± 0.003 | **+0.012** ± 0.002 | **-0.024** ± 0.003 | +0.002 ± 0.001 |
| b4 | nowalk | objective | **+0.003** ± 0.000 | -0.001 ± 0.000 | **+0.004** ± 0.001 | +0.000 ± 0.000 | **+0.005** ± 0.001 |
| b4 | nowalk | ap | **+0.001** ± 0.000 | **-0.001** ± 0.000 | **+0.003** ± 0.000 | +0.000 ± 0.000 | **+0.002** ± 0.001 |
