# #4184 progression - machine summary

- `r1_xcal` (/expscratch/sgreenberg/progression-4184/r1_xcal/results): 702/720 cells with data, 97,073 rows, 18 starved (header-only; not data loss)
- `r2_gmm` (/expscratch/sgreenberg/progression-4184/r2_gmm/results): 702/720 cells with data, 97,073 rows, 18 starved (header-only; not data loss)
- `r3_blend` (/expscratch/sgreenberg/progression-4184/r3_blend/results): 702/720 cells with data, 97,073 rows, 18 starved (header-only; not data loss)
- `r4_rawmean` (/expscratch/sgreenberg/progression-4184/r4_rawmean/results): 702/720 cells with data, 97,073 rows, 18 starved (header-only; not data loss)
- `r5_anchored` (/expscratch/sgreenberg/progression-4184/r5_anchored/results): 702/720 cells with data, 97,073 rows, 18 starved (header-only; not data loss)
- `r6_split70` (/expscratch/sgreenberg/progression-4184/r6_split70/results): 702/720 cells with data, 97,073 rows, 18 starved (header-only; not data loss)
- `r7_acq4` (/expscratch/sgreenberg/progression-4184/r7_acq4/results): 702/720 cells with data, 97,073 rows, 18 starved (header-only; not data loss)
Pool: `natural` (prevalence arm).

Grid: 720 cells (union of every rung's cells and the baseline's).

| rung | cells | lost | of which missing | filled-only | coverage@10 | coverage@50 |
|---|---|---|---|---|---|---|
| r1_xcal | 720 | 0 | 0 | 18 | 0.66 | 0.80 |
| r2_gmm | 720 | 0 | 0 | 18 | 0.66 | 0.80 |
| r3_blend | 720 | 0 | 0 | 18 | 0.66 | 0.80 |
| r4_rawmean | 720 | 0 | 0 | 18 | 0.66 | 0.80 |
| r5_anchored | 720 | 0 | 0 | 18 | 0.66 | 0.80 |
| r6_split70 | 720 | 0 | 0 | 18 | 0.66 | 0.80 |
| r7_acq4 | 720 | 0 | 0 | 18 | 0.66 | 0.80 |

## Mean cost (filled) at checkpoints

| rung | t=0 | t=10 | t=20 | t=30 | t=50 | t=100 | t=150 |
|---|---|---|---|---|---|---|---|
| r1_xcal | 0.48 | 0.56 | 0.5 | 0.45 | 0.41 | 0.37 | 0.36 |
| r2_gmm | 0.48 | 0.47 | 0.42 | 0.4 | 0.38 | 0.36 | 0.35 |
| r3_blend | 0.48 | 0.44 | 0.39 | 0.37 | 0.36 | 0.33 | 0.32 |
| r4_rawmean | 0.48 | 0.37 | 0.33 | 0.33 | 0.31 | 0.3 | 0.29 |
| r5_anchored | 0.48 | 0.49 | 0.44 | 0.42 | 0.4 | 0.38 | 0.36 |
| r6_split70 | 0.48 | 0.47 | 0.44 | 0.42 | 0.4 | 0.38 | 0.36 |
| r7_acq4 | 0.48 | 0.45 | 0.4 | 0.38 | 0.36 | 0.33 | 0.32 |

## Each rung against the one before (Δcost = to − from; negative = the step helped)

| from → to | at | n | Δ mean | SE |
|---|---|---|---|---|
| r1_xcal → r2_gmm | t=10 | 720 | **-0.097** | 0.0082 |
| r1_xcal → r2_gmm | t=20 | 720 | **-0.079** | 0.0075 |
| r1_xcal → r2_gmm | t=30 | 720 | **-0.05** | 0.0071 |
| r1_xcal → r2_gmm | t=50 | 720 | **-0.027** | 0.0068 |
| r1_xcal → r2_gmm | t=100 | 720 | -0.011 | 0.0064 |
| r1_xcal → r2_gmm | t=150 | 720 | -0.0055 | 0.006 |
| r1_xcal → r2_gmm | mean t=1..150 | 720 | **-0.025** | 0.0053 |
| r2_gmm → r3_blend | t=10 | 720 | **-0.024** | 0.0021 |
| r2_gmm → r3_blend | t=20 | 720 | **-0.024** | 0.0019 |
| r2_gmm → r3_blend | t=30 | 720 | **-0.025** | 0.0016 |
| r2_gmm → r3_blend | t=50 | 720 | **-0.026** | 0.0016 |
| r2_gmm → r3_blend | t=100 | 720 | **-0.026** | 0.0017 |
| r2_gmm → r3_blend | t=150 | 720 | **-0.029** | 0.0019 |
| r2_gmm → r3_blend | mean t=1..150 | 720 | **-0.025** | 0.0013 |
| r3_blend → r4_rawmean | t=10 | 720 | **-0.069** | 0.0043 |
| r3_blend → r4_rawmean | t=20 | 720 | **-0.06** | 0.0038 |
| r3_blend → r4_rawmean | t=30 | 720 | **-0.049** | 0.0035 |
| r3_blend → r4_rawmean | t=50 | 720 | **-0.044** | 0.0031 |
| r3_blend → r4_rawmean | t=100 | 720 | **-0.036** | 0.0029 |
| r3_blend → r4_rawmean | t=150 | 720 | **-0.03** | 0.0028 |
| r3_blend → r4_rawmean | mean t=1..150 | 720 | **-0.041** | 0.0026 |
| r4_rawmean → r5_anchored | t=10 | 720 | **+0.11** | 0.0056 |
| r4_rawmean → r5_anchored | t=20 | 720 | **+0.1** | 0.0046 |
| r4_rawmean → r5_anchored | t=30 | 720 | **+0.098** | 0.0042 |
| r4_rawmean → r5_anchored | t=50 | 720 | **+0.089** | 0.0038 |
| r4_rawmean → r5_anchored | t=100 | 720 | **+0.08** | 0.0033 |
| r4_rawmean → r5_anchored | t=150 | 720 | **+0.073** | 0.0031 |
| r4_rawmean → r5_anchored | mean t=1..150 | 720 | **+0.083** | 0.0031 |
| r5_anchored → r6_split70 | t=10 | 720 | **-0.012** | 0.0043 |
| r5_anchored → r6_split70 | t=20 | 720 | -0.00016 | 0.0026 |
| r5_anchored → r6_split70 | t=30 | 720 | -0.0019 | 0.0023 |
| r5_anchored → r6_split70 | t=50 | 720 | -0.0021 | 0.002 |
| r5_anchored → r6_split70 | t=100 | 720 | -0.0003 | 0.0019 |
| r5_anchored → r6_split70 | t=150 | 720 | -0.00082 | 0.0021 |
| r5_anchored → r6_split70 | mean t=1..150 | 720 | -0.0022 | 0.0015 |
| r6_split70 → r7_acq4 | t=10 | 720 | **-0.019** | 0.0042 |
| r6_split70 → r7_acq4 | t=20 | 720 | **-0.037** | 0.003 |
| r6_split70 → r7_acq4 | t=30 | 720 | **-0.041** | 0.003 |
| r6_split70 → r7_acq4 | t=50 | 720 | **-0.042** | 0.0029 |
| r6_split70 → r7_acq4 | t=100 | 720 | **-0.047** | 0.0029 |
| r6_split70 → r7_acq4 | t=150 | 720 | **-0.047** | 0.0029 |
| r6_split70 → r7_acq4 | mean t=1..150 | 720 | **-0.041** | 0.0023 |

Bold = more than 2 SE from zero.
