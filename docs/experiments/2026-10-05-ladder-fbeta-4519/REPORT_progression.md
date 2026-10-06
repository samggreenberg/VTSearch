# #4184 progression - machine summary

- `r1_xcal` (<run>/r1_xcal/results): 717/720 cells with data, 101,401 rows, 3 starved (header-only; not data loss)
- `r2_gmm` (<run>/r2_gmm/results): 717/720 cells with data, 101,401 rows, 3 starved (header-only; not data loss)
- `r3_blend` (<run>/r3_blend/results): 717/720 cells with data, 101,401 rows, 3 starved (header-only; not data loss)
- `r4_rawmean` (<run>/r4_rawmean/results): 717/720 cells with data, 101,401 rows, 3 starved (header-only; not data loss)
- `r5_anchored` (<run>/r5_anchored/results): 717/720 cells with data, 101,401 rows, 3 starved (header-only; not data loss)
- `r6_split70` (<run>/r6_split70/results): 717/720 cells with data, 101,401 rows, 3 starved (header-only; not data loss)
- `r7_acq4` (<run>/r7_acq4/results): 717/720 cells with data, 101,401 rows, 3 starved (header-only; not data loss)
- `r8_labels` (<run>/r8_labels/results): 717/720 cells with data, 90,709 rows, 2,999 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
Pool: `haystack_0.01` (prevalence arm).

Grid: 720 cells (union of every rung's cells and the baseline's).

| rung | cells | lost | of which missing | filled-only | coverage@10 | coverage@50 |
|---|---|---|---|---|---|---|
| r1_xcal | 720 | 0 | 0 | 3 | 0.00 | 0.63 |
| r2_gmm | 720 | 0 | 0 | 3 | 0.00 | 0.63 |
| r3_blend | 720 | 0 | 0 | 3 | 0.00 | 0.63 |
| r4_rawmean | 720 | 0 | 0 | 3 | 0.00 | 0.63 |
| r5_anchored | 720 | 0 | 0 | 3 | 0.00 | 0.63 |
| r6_split70 | 720 | 0 | 0 | 3 | 0.00 | 0.63 |
| r7_acq4 | 720 | 0 | 0 | 3 | 0.00 | 0.63 |
| r8_labels | 720 | 0 | 0 | 3 | 0.00 | 0.49 |

## Mean cost (filled) at checkpoints

| rung | t=0 | t=10 | t=20 | t=30 | t=50 | t=100 | t=150 |
|---|---|---|---|---|---|---|---|
| r1_xcal | 0.46 | 0.46 | 0.46 | 0.44 | 0.41 | 0.36 | 0.33 |
| r2_gmm | 0.46 | 0.46 | 0.46 | 0.43 | 0.37 | 0.31 | 0.28 |
| r3_blend | 0.46 | 0.46 | 0.46 | 0.41 | 0.35 | 0.28 | 0.25 |
| r4_rawmean | 0.46 | 0.46 | 0.46 | 0.35 | 0.31 | 0.26 | 0.23 |
| r5_anchored | 0.46 | 0.46 | 0.46 | 0.45 | 0.4 | 0.34 | 0.31 |
| r6_split70 | 0.46 | 0.46 | 0.46 | 0.45 | 0.4 | 0.35 | 0.32 |
| r7_acq4 | 0.46 | 0.46 | 0.46 | 0.41 | 0.36 | 0.3 | 0.27 |
| r8_labels | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 | 0.48 | 0.44 |

## Each rung against the one before (Δcost = to − from; negative = the step helped)

| from → to | at | n | Δ mean | SE |
|---|---|---|---|---|
| r1_xcal → r2_gmm | t=10 | 720 | +0 | 0 |
| r1_xcal → r2_gmm | t=20 | 720 | +0 | 0 |
| r1_xcal → r2_gmm | t=30 | 720 | -0.0083 | 0.0044 |
| r1_xcal → r2_gmm | t=50 | 720 | **-0.041** | 0.0049 |
| r1_xcal → r2_gmm | t=100 | 720 | **-0.046** | 0.0048 |
| r1_xcal → r2_gmm | t=150 | 720 | **-0.055** | 0.0049 |
| r1_xcal → r2_gmm | mean t=1..150 | 720 | **-0.033** | 0.003 |
| r2_gmm → r3_blend | t=10 | 720 | +0 | 0 |
| r2_gmm → r3_blend | t=20 | 720 | +0 | 0 |
| r2_gmm → r3_blend | t=30 | 720 | **-0.02** | 0.0018 |
| r2_gmm → r3_blend | t=50 | 720 | **-0.024** | 0.0019 |
| r2_gmm → r3_blend | t=100 | 720 | **-0.027** | 0.0018 |
| r2_gmm → r3_blend | t=150 | 720 | **-0.025** | 0.002 |
| r2_gmm → r3_blend | mean t=1..150 | 720 | **-0.021** | 0.0012 |
| r3_blend → r4_rawmean | t=10 | 720 | +0 | 0 |
| r3_blend → r4_rawmean | t=20 | 720 | +0 | 0 |
| r3_blend → r4_rawmean | t=30 | 720 | **-0.061** | 0.0036 |
| r3_blend → r4_rawmean | t=50 | 720 | **-0.039** | 0.0025 |
| r3_blend → r4_rawmean | t=100 | 720 | **-0.027** | 0.002 |
| r3_blend → r4_rawmean | t=150 | 720 | **-0.018** | 0.0021 |
| r3_blend → r4_rawmean | mean t=1..150 | 720 | **-0.027** | 0.0014 |
| r4_rawmean → r5_anchored | t=10 | 720 | +0 | 0 |
| r4_rawmean → r5_anchored | t=20 | 720 | +0 | 0 |
| r4_rawmean → r5_anchored | t=30 | 720 | **+0.11** | 0.0056 |
| r4_rawmean → r5_anchored | t=50 | 720 | **+0.094** | 0.0042 |
| r4_rawmean → r5_anchored | t=100 | 720 | **+0.085** | 0.0033 |
| r4_rawmean → r5_anchored | t=150 | 720 | **+0.077** | 0.0031 |
| r4_rawmean → r5_anchored | mean t=1..150 | 720 | **+0.074** | 0.0028 |
| r5_anchored → r6_split70 | t=10 | 720 | +0 | 0 |
| r5_anchored → r6_split70 | t=20 | 720 | +0 | 0 |
| r5_anchored → r6_split70 | t=30 | 720 | **-0.0079** | 0.0034 |
| r5_anchored → r6_split70 | t=50 | 720 | +6.7e-05 | 0.0023 |
| r5_anchored → r6_split70 | t=100 | 720 | **+0.0054** | 0.0018 |
| r5_anchored → r6_split70 | t=150 | 720 | **+0.0079** | 0.002 |
| r5_anchored → r6_split70 | mean t=1..150 | 720 | **+0.0028** | 0.0013 |
| r6_split70 → r7_acq4 | t=10 | 720 | +0 | 0 |
| r6_split70 → r7_acq4 | t=20 | 720 | +0 | 0 |
| r6_split70 → r7_acq4 | t=30 | 720 | **-0.032** | 0.0037 |
| r6_split70 → r7_acq4 | t=50 | 720 | **-0.044** | 0.0031 |
| r6_split70 → r7_acq4 | t=100 | 720 | **-0.043** | 0.0027 |
| r6_split70 → r7_acq4 | t=150 | 720 | **-0.051** | 0.0029 |
| r6_split70 → r7_acq4 | mean t=1..150 | 720 | **-0.035** | 0.002 |
| r7_acq4 → r8_labels | t=10 | 720 | +0 | 0 |
| r7_acq4 → r8_labels | t=20 | 720 | +0 | 0 |
| r7_acq4 → r8_labels | t=30 | 720 | **+0.042** | 0.0047 |
| r7_acq4 → r8_labels | t=50 | 720 | **+0.097** | 0.0059 |
| r7_acq4 → r8_labels | t=100 | 720 | **+0.17** | 0.0071 |
| r7_acq4 → r8_labels | t=150 | 720 | **+0.18** | 0.0069 |
| r7_acq4 → r8_labels | mean t=1..150 | 720 | **+0.12** | 0.004 |

Bold = more than 2 SE from zero.

## The returned set's F1 (filled) at checkpoints

| rung | t=0 | t=10 | t=20 | t=30 | t=50 | t=100 | t=150 |
|---|---|---|---|---|---|---|---|
| r1_xcal | 0.023 | 0.023 | 0.023 | 0.33 | 0.41 | 0.47 | 0.48 |
| r2_gmm | 0.023 | 0.023 | 0.023 | 0.031 | 0.053 | 0.096 | 0.12 |
| r3_blend | 0.023 | 0.023 | 0.023 | 0.039 | 0.096 | 0.17 | 0.2 |
| r4_rawmean | 0.023 | 0.023 | 0.023 | 0.13 | 0.16 | 0.17 | 0.18 |
| r5_anchored | 0.023 | 0.023 | 0.023 | 0.027 | 0.033 | 0.043 | 0.05 |
| r6_split70 | 0.023 | 0.023 | 0.023 | 0.028 | 0.033 | 0.043 | 0.048 |
| r7_acq4 | 0.023 | 0.023 | 0.023 | 0.043 | 0.048 | 0.054 | 0.067 |
| r8_labels | 0.023 | 0.023 | 0.023 | 0.32 | 0.4 | 0.49 | 0.51 |

## Each rung against the one before, in F1 (ΔF1 = to − from; positive = the step helped)

| from → to | at | n | Δ mean | SE |
|---|---|---|---|---|
| r1_xcal → r2_gmm | t=10 | 720 | +0 | 0 |
| r1_xcal → r2_gmm | t=20 | 720 | +0 | 0 |
| r1_xcal → r2_gmm | t=30 | 720 | **-0.3** | 0.013 |
| r1_xcal → r2_gmm | t=50 | 720 | **-0.35** | 0.013 |
| r1_xcal → r2_gmm | t=100 | 720 | **-0.38** | 0.012 |
| r1_xcal → r2_gmm | t=150 | 720 | **-0.37** | 0.011 |
| r1_xcal → r2_gmm | mean t=1..150 | 720 | **-0.3** | 0.0096 |
| r2_gmm → r3_blend | t=10 | 720 | +0 | 0 |
| r2_gmm → r3_blend | t=20 | 720 | +0 | 0 |
| r2_gmm → r3_blend | t=30 | 720 | **+0.008** | 0.0017 |
| r2_gmm → r3_blend | t=50 | 720 | **+0.043** | 0.0051 |
| r2_gmm → r3_blend | t=100 | 720 | **+0.078** | 0.0068 |
| r2_gmm → r3_blend | t=150 | 720 | **+0.085** | 0.0073 |
| r2_gmm → r3_blend | mean t=1..150 | 720 | **+0.051** | 0.004 |
| r3_blend → r4_rawmean | t=10 | 720 | +0 | 0 |
| r3_blend → r4_rawmean | t=20 | 720 | +0 | 0 |
| r3_blend → r4_rawmean | t=30 | 720 | **+0.093** | 0.007 |
| r3_blend → r4_rawmean | t=50 | 720 | **+0.065** | 0.0064 |
| r3_blend → r4_rawmean | t=100 | 720 | -0.0087 | 0.0057 |
| r3_blend → r4_rawmean | t=150 | 720 | **-0.017** | 0.0053 |
| r3_blend → r4_rawmean | mean t=1..150 | 720 | **+0.016** | 0.0034 |
| r4_rawmean → r5_anchored | t=10 | 720 | +0 | 0 |
| r4_rawmean → r5_anchored | t=20 | 720 | +0 | 0 |
| r4_rawmean → r5_anchored | t=30 | 720 | **-0.1** | 0.0073 |
| r4_rawmean → r5_anchored | t=50 | 720 | **-0.13** | 0.0078 |
| r4_rawmean → r5_anchored | t=100 | 720 | **-0.12** | 0.0078 |
| r4_rawmean → r5_anchored | t=150 | 720 | **-0.13** | 0.0086 |
| r4_rawmean → r5_anchored | mean t=1..150 | 720 | **-0.1** | 0.0062 |
| r5_anchored → r6_split70 | t=10 | 720 | +0 | 0 |
| r5_anchored → r6_split70 | t=20 | 720 | +0 | 0 |
| r5_anchored → r6_split70 | t=30 | 720 | +0.00031 | 0.0012 |
| r5_anchored → r6_split70 | t=50 | 720 | -0.00065 | 0.0014 |
| r5_anchored → r6_split70 | t=100 | 720 | +3.5e-05 | 0.0019 |
| r5_anchored → r6_split70 | t=150 | 720 | -0.0019 | 0.0019 |
| r5_anchored → r6_split70 | mean t=1..150 | 720 | -0.00068 | 0.0012 |
| r6_split70 → r7_acq4 | t=10 | 720 | +0 | 0 |
| r6_split70 → r7_acq4 | t=20 | 720 | +0 | 0 |
| r6_split70 → r7_acq4 | t=30 | 720 | **+0.015** | 0.0029 |
| r6_split70 → r7_acq4 | t=50 | 720 | **+0.016** | 0.0022 |
| r6_split70 → r7_acq4 | t=100 | 720 | **+0.011** | 0.0028 |
| r6_split70 → r7_acq4 | t=150 | 720 | **+0.018** | 0.0036 |
| r6_split70 → r7_acq4 | mean t=1..150 | 720 | **+0.012** | 0.002 |
| r7_acq4 → r8_labels | t=10 | 720 | +0 | 0 |
| r7_acq4 → r8_labels | t=20 | 720 | +0 | 0 |
| r7_acq4 → r8_labels | t=30 | 720 | **+0.28** | 0.012 |
| r7_acq4 → r8_labels | t=50 | 720 | **+0.35** | 0.013 |
| r7_acq4 → r8_labels | t=100 | 720 | **+0.43** | 0.012 |
| r7_acq4 → r8_labels | t=150 | 720 | **+0.45** | 0.011 |
| r7_acq4 → r8_labels | mean t=1..150 | 720 | **+0.33** | 0.0097 |
