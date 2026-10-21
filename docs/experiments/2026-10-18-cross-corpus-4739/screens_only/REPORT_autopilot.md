# #4671 Autopilot pieces - machine summary

- `ctl_b025`: 720/720 cells with data, 95,540 rows, 2,960 spot-check rows set apart (not clicks)
- `nowalk_b025`: 720/720 cells with data, 96,180 rows, 2,967 spot-check rows set apart (not clicks)
- `ctl_b1`: 720/720 cells with data, 94,172 rows, 2,969 spot-check rows set apart (not clicks)
- `nowalk_b1`: 720/720 cells with data, 95,028 rows, 2,982 spot-check rows set apart (not clicks)
- `ctl_b4`: 720/720 cells with data, 88,284 rows, 4,346 spot-check rows set apart (not clicks)
- `nowalk_b4`: 720/720 cells with data, 90,772 rows, 4,354 spot-check rows set apart (not clicks)

Grid: 720 cells.

## Session means (every run, every vote)

| preset | arm | objective 1-150 | 1-50 | 51-150 | AP 1-150 | precision 1-150 | recall 1-150 | over 200 | showing a detector |
|---|---|---|---|---|---|---|---|---|---|
| b025 | ctl | 0.598 | 0.516 | 0.638 | 0.520 | 0.67 | 0.36 | 0% | 77% |
| b025 | nowalk | 0.603 | 0.524 | 0.642 | 0.524 | 0.66 | 0.38 | 0% | 76% |
| b1 | ctl | 0.487 | 0.405 | 0.529 | 0.519 | 0.52 | 0.51 | 2% | 77% |
| b1 | nowalk | 0.494 | 0.414 | 0.534 | 0.524 | 0.51 | 0.52 | 2% | 76% |
| b4 | ctl | 0.574 | 0.499 | 0.611 | 0.517 | 0.31 | 0.64 | 28% | 77% |
| b4 | nowalk | 0.580 | 0.502 | 0.620 | 0.522 | 0.32 | 0.65 | 29% | 76% |

## Each arm minus today's app (paired; bold = beyond 2 SE)

| preset | arm | metric | mean 1-150 | mean 1-50 | mean 51-150 | mean 1-25 | t=150 |
|---|---|---|---|---|---|---|---|
| b025 | nowalk | objective | **+0.005** ± 0.001 | **+0.008** ± 0.001 | **+0.004** ± 0.002 | +0.000 ± 0.000 | **-0.007** ± 0.002 |
| b025 | nowalk | ap | **+0.004** ± 0.000 | **+0.002** ± 0.000 | **+0.005** ± 0.000 | +0.000 ± 0.000 | **+0.002** ± 0.001 |
| b025 | nowalk | find_objective | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan |
| b025 | nowalk | find_ap | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan |
| b025 | nowalk | find_n_flagged | -0.066 ± 0.050 | **-0.144** ± 0.066 | -0.028 ± 0.065 | +0.000 ± 0.000 | +0.019 ± 0.117 |
| b1 | nowalk | objective | **+0.007** ± 0.001 | **+0.009** ± 0.001 | **+0.006** ± 0.001 | +0.000 ± 0.000 | **-0.004** ± 0.001 |
| b1 | nowalk | ap | **+0.005** ± 0.000 | **+0.002** ± 0.000 | **+0.006** ± 0.000 | +0.000 ± 0.000 | **+0.002** ± 0.001 |
| b1 | nowalk | find_objective | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan |
| b1 | nowalk | find_ap | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan |
| b1 | nowalk | find_n_flagged | **-0.332** ± 0.140 | -0.505 ± 0.333 | -0.245 ± 0.127 | +0.000 ± 0.000 | -0.196 ± 0.172 |
| b4 | nowalk | objective | **+0.007** ± 0.001 | **+0.003** ± 0.000 | **+0.009** ± 0.001 | +0.000 ± 0.000 | **+0.005** ± 0.001 |
| b4 | nowalk | ap | **+0.005** ± 0.000 | **+0.002** ± 0.000 | **+0.006** ± 0.001 | +0.000 ± 0.000 | **+0.002** ± 0.001 |
| b4 | nowalk | find_objective | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan |
| b4 | nowalk | find_ap | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan | +nan ± nan |
| b4 | nowalk | find_n_flagged | **-0.664** ± 0.275 | -0.558 ± 0.441 | **-0.717** ± 0.319 | +0.000 ± 0.000 | -0.665 ± 0.333 |

## Find on the second corpus (every run, every vote; #4960)

Objective NaN where the corpus holds no positive: read the returned count there.

| preset | arm | objective 1-150 | 1-50 | 51-150 | AP 1-150 | returned (median) 1-150 | returning none |
|---|---|---|---|---|---|---|---|
| b025 | ctl | nan | nan | nan | nan | 1.0 | 0% |
| b025 | nowalk | nan | nan | nan | nan | 1.0 | 0% |
| b1 | ctl | nan | nan | nan | nan | 1.3 | 0% |
| b1 | nowalk | nan | nan | nan | nan | 1.2 | 0% |
| b4 | ctl | nan | nan | nan | nan | 5.9 | 0% |
| b4 | nowalk | nan | nan | nan | nan | 5.5 | 0% |

## Domain-shift flags against the second corpus, at the snapshot clicks (#4960)

Share of runs each flag fires in; the shares behind them are means over runs. The training-domain overlap is NaN on a patch embedder, which the app refuses.

| preset | arm | click | runs | overlap fires | atypical share | evidence fires | unsupported share | low-trust share |
|---|---|---|---|---|---|---|---|---|
| b025 | ctl | 10 | 720 | 100% | 0.612 | 0% | 0.000 | 0.338 |
| b025 | ctl | 25 | 720 | 100% | 0.612 | 35% | 0.337 | 0.251 |
| b025 | ctl | 50 | 720 | 100% | 0.612 | 94% | 0.893 | 0.198 |
| b025 | ctl | 100 | 720 | 100% | 0.612 | 100% | 0.984 | 0.138 |
| b025 | ctl | 150 | 720 | 100% | 0.612 | 100% | 0.990 | 0.097 |
| b025 | nowalk | 10 | 720 | 100% | 0.612 | 0% | 0.000 | 0.338 |
| b025 | nowalk | 25 | 720 | 100% | 0.612 | 30% | 0.293 | 0.348 |
| b025 | nowalk | 50 | 720 | 100% | 0.612 | 82% | 0.775 | 0.256 |
| b025 | nowalk | 100 | 720 | 100% | 0.612 | 100% | 0.983 | 0.151 |
| b025 | nowalk | 150 | 720 | 100% | 0.612 | 100% | 0.990 | 0.104 |
| b1 | ctl | 10 | 720 | 100% | 0.612 | 0% | 0.000 | 0.338 |
| b1 | ctl | 25 | 720 | 100% | 0.612 | 36% | 0.347 | 0.239 |
| b1 | ctl | 50 | 720 | 100% | 0.612 | 99% | 0.936 | 0.180 |
| b1 | ctl | 100 | 720 | 100% | 0.612 | 100% | 0.983 | 0.133 |
| b1 | ctl | 150 | 720 | 100% | 0.612 | 100% | 0.990 | 0.096 |
| b1 | nowalk | 10 | 720 | 100% | 0.612 | 0% | 0.000 | 0.338 |
| b1 | nowalk | 25 | 720 | 100% | 0.612 | 30% | 0.293 | 0.348 |
| b1 | nowalk | 50 | 720 | 100% | 0.612 | 90% | 0.846 | 0.238 |
| b1 | nowalk | 100 | 720 | 100% | 0.612 | 100% | 0.984 | 0.148 |
| b1 | nowalk | 150 | 720 | 100% | 0.612 | 100% | 0.990 | 0.101 |
| b4 | ctl | 10 | 720 | 100% | 0.612 | 0% | 0.001 | 0.339 |
| b4 | ctl | 25 | 720 | 100% | 0.612 | 36% | 0.345 | 0.239 |
| b4 | ctl | 50 | 720 | 100% | 0.612 | 100% | 0.944 | 0.174 |
| b4 | ctl | 100 | 720 | 100% | 0.612 | 100% | 0.984 | 0.126 |
| b4 | ctl | 150 | 720 | 100% | 0.612 | 100% | 0.990 | 0.093 |
| b4 | nowalk | 10 | 720 | 100% | 0.612 | 0% | 0.001 | 0.339 |
| b4 | nowalk | 25 | 720 | 100% | 0.612 | 30% | 0.293 | 0.349 |
| b4 | nowalk | 50 | 720 | 100% | 0.612 | 100% | 0.936 | 0.206 |
| b4 | nowalk | 100 | 720 | 100% | 0.612 | 100% | 0.984 | 0.144 |
| b4 | nowalk | 150 | 720 | 100% | 0.612 | 100% | 0.990 | 0.099 |
