# #4184 progression - machine summary

- `r1_xcal` (/expscratch/sgreenberg/progression-4184-h0.05/r1_xcal/results): 720/720 cells with data, 105,358 rows
- `r2_gmm` (/expscratch/sgreenberg/progression-4184-h0.05/r2_gmm/results): 720/720 cells with data, 105,358 rows
- `r3_blend` (/expscratch/sgreenberg/progression-4184-h0.05/r3_blend/results): 720/720 cells with data, 105,358 rows
- `r4_rawmean` (/expscratch/sgreenberg/progression-4184-h0.05/r4_rawmean/results): 720/720 cells with data, 105,358 rows
- `r5_anchored` (/expscratch/sgreenberg/progression-4184-h0.05/r5_anchored/results): 720/720 cells with data, 105,358 rows
- `r6_split70` (/expscratch/sgreenberg/progression-4184-h0.05/r6_split70/results): 720/720 cells with data, 105,358 rows
- `r7_acq4` (/expscratch/sgreenberg/progression-4184-h0.05/r7_acq4/results): 720/720 cells with data, 105,358 rows
Pool: `haystack_0.05` (prevalence arm).

Grid: 720 cells (union of every rung's cells and the baseline's).

| rung | cells | lost | of which missing | filled-only | coverage@10 | coverage@50 |
|---|---|---|---|---|---|---|
| r1_xcal | 720 | 0 | 0 | 0 | 0.86 | 0.98 |
| r2_gmm | 720 | 0 | 0 | 0 | 0.86 | 0.98 |
| r3_blend | 720 | 0 | 0 | 0 | 0.86 | 0.98 |
| r4_rawmean | 720 | 0 | 0 | 0 | 0.86 | 0.98 |
| r5_anchored | 720 | 0 | 0 | 0 | 0.86 | 0.98 |
| r6_split70 | 720 | 0 | 0 | 0 | 0.86 | 0.98 |
| r7_acq4 | 720 | 0 | 0 | 0 | 0.86 | 0.98 |

## Mean cost (filled) at checkpoints

| rung | t=0 | t=10 | t=20 | t=30 | t=50 | t=100 | t=150 |
|---|---|---|---|---|---|---|---|
| r1_xcal | 0.33 | 0.51 | 0.42 | 0.35 | 0.29 | 0.22 | 0.2 |
| r2_gmm | 0.33 | 0.35 | 0.29 | 0.27 | 0.24 | 0.21 | 0.19 |
| r3_blend | 0.33 | 0.34 | 0.28 | 0.26 | 0.23 | 0.2 | 0.18 |
| r4_rawmean | 0.33 | 0.34 | 0.28 | 0.25 | 0.22 | 0.19 | 0.18 |
| r5_anchored | 0.33 | 0.37 | 0.31 | 0.28 | 0.26 | 0.23 | 0.2 |
| r6_split70 | 0.33 | 0.36 | 0.31 | 0.28 | 0.25 | 0.22 | 0.2 |
| r7_acq4 | 0.33 | 0.36 | 0.29 | 0.26 | 0.23 | 0.19 | 0.18 |

## Each rung against the one before (Δcost = to − from; negative = the step helped)

| from → to | at | n | Δ mean | SE |
|---|---|---|---|---|
| r1_xcal → r2_gmm | t=10 | 720 | **-0.16** | 0.0077 |
| r1_xcal → r2_gmm | t=20 | 720 | **-0.13** | 0.0064 |
| r1_xcal → r2_gmm | t=30 | 720 | **-0.086** | 0.0055 |
| r1_xcal → r2_gmm | t=50 | 720 | **-0.058** | 0.0046 |
| r1_xcal → r2_gmm | t=100 | 720 | **-0.015** | 0.0032 |
| r1_xcal → r2_gmm | t=150 | 720 | -0.0046 | 0.003 |
| r1_xcal → r2_gmm | mean t=1..150 | 720 | **-0.042** | 0.0027 |
| r2_gmm → r3_blend | t=10 | 720 | **-0.013** | 0.0025 |
| r2_gmm → r3_blend | t=20 | 720 | **-0.0076** | 0.0024 |
| r2_gmm → r3_blend | t=30 | 720 | **-0.0098** | 0.0024 |
| r2_gmm → r3_blend | t=50 | 720 | **-0.0087** | 0.0025 |
| r2_gmm → r3_blend | t=100 | 720 | **-0.0099** | 0.0021 |
| r2_gmm → r3_blend | t=150 | 720 | **-0.01** | 0.0018 |
| r2_gmm → r3_blend | mean t=1..150 | 720 | **-0.0092** | 0.0014 |
| r3_blend → r4_rawmean | t=10 | 720 | **+0.009** | 0.0034 |
| r3_blend → r4_rawmean | t=20 | 720 | -0.0011 | 0.0028 |
| r3_blend → r4_rawmean | t=30 | 720 | -0.0054 | 0.0027 |
| r3_blend → r4_rawmean | t=50 | 720 | **-0.0067** | 0.0023 |
| r3_blend → r4_rawmean | t=100 | 720 | -0.0035 | 0.0022 |
| r3_blend → r4_rawmean | t=150 | 720 | -0.0034 | 0.0018 |
| r3_blend → r4_rawmean | mean t=1..150 | 720 | -0.0028 | 0.0014 |
| r4_rawmean → r5_anchored | t=10 | 720 | **+0.029** | 0.0044 |
| r4_rawmean → r5_anchored | t=20 | 720 | **+0.03** | 0.0034 |
| r4_rawmean → r5_anchored | t=30 | 720 | **+0.031** | 0.0029 |
| r4_rawmean → r5_anchored | t=50 | 720 | **+0.04** | 0.0027 |
| r4_rawmean → r5_anchored | t=100 | 720 | **+0.033** | 0.0024 |
| r4_rawmean → r5_anchored | t=150 | 720 | **+0.022** | 0.0021 |
| r4_rawmean → r5_anchored | mean t=1..150 | 720 | **+0.03** | 0.0017 |
| r5_anchored → r6_split70 | t=10 | 720 | **-0.011** | 0.0035 |
| r5_anchored → r6_split70 | t=20 | 720 | **-0.0057** | 0.0025 |
| r5_anchored → r6_split70 | t=30 | 720 | +0.00087 | 0.0023 |
| r5_anchored → r6_split70 | t=50 | 720 | **-0.0062** | 0.0025 |
| r5_anchored → r6_split70 | t=100 | 720 | **-0.0053** | 0.0022 |
| r5_anchored → r6_split70 | t=150 | 720 | -0.0017 | 0.0021 |
| r5_anchored → r6_split70 | mean t=1..150 | 720 | **-0.004** | 0.0014 |
| r6_split70 → r7_acq4 | t=10 | 720 | -0.0036 | 0.0038 |
| r6_split70 → r7_acq4 | t=20 | 720 | **-0.014** | 0.0029 |
| r6_split70 → r7_acq4 | t=30 | 720 | **-0.02** | 0.0031 |
| r6_split70 → r7_acq4 | t=50 | 720 | **-0.026** | 0.0028 |
| r6_split70 → r7_acq4 | t=100 | 720 | **-0.027** | 0.0026 |
| r6_split70 → r7_acq4 | t=150 | 720 | **-0.024** | 0.0024 |
| r6_split70 → r7_acq4 | mean t=1..150 | 720 | **-0.023** | 0.0018 |

Bold = more than 2 SE from zero.
