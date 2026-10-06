# #4184 progression - machine summary

- `r1_xcal` (<run>/r1_xcal/results): 717/720 cells with data, 101,401 rows, 3 starved (header-only; not data loss)
- `r8_labels` (<run>/r8_labels/results): 717/720 cells with data, 90,709 rows, 2,999 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `r8_labels_b025` (<run>/r8_labels_b025/results): 717/720 cells with data, 91,109 rows, 2,996 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
- `r8_labels_b4` (<run>/r8_labels_b4/results): 717/720 cells with data, 89,465 rows, 4,359 spot-check rows set apart (not clicks), 3 starved (header-only; not data loss)
Pool: `haystack_0.01` (prevalence arm).

Grid: 720 cells (union of every rung's cells and the baseline's).

| rung | cells | lost | of which missing | filled-only | coverage@10 | coverage@50 |
|---|---|---|---|---|---|---|
| r1_xcal | 720 | 0 | 0 | 3 | 0.00 | 0.63 |
| r8_labels | 720 | 0 | 0 | 3 | 0.00 | 0.49 |
| r8_labels_b025 | 720 | 0 | 0 | 3 | 0.00 | 0.49 |
| r8_labels_b4 | 720 | 0 | 0 | 3 | 0.00 | 0.47 |

## Mean cost (filled) at checkpoints

| rung | t=0 | t=10 | t=20 | t=30 | t=50 | t=100 | t=150 |
|---|---|---|---|---|---|---|---|
| r1_xcal | 0.46 | 0.46 | 0.46 | 0.44 | 0.41 | 0.36 | 0.33 |
| r8_labels | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 | 0.48 | 0.44 |
| r8_labels_b025 | 0.46 | 0.46 | 0.46 | 0.52 | 0.54 | 0.61 | 0.59 |
| r8_labels_b4 | 0.46 | 0.46 | 0.46 | 0.4 | 0.38 | 0.35 | 0.3 |

## Each rung against the one before (Δcost = to − from; negative = the step helped)

| from → to | at | n | Δ mean | SE |
|---|---|---|---|---|
| r1_xcal → r8_labels | t=10 | 720 | +0 | 0 |
| r1_xcal → r8_labels | t=20 | 720 | +0 | 0 |
| r1_xcal → r8_labels | t=30 | 720 | **+0.02** | 0.0044 |
| r1_xcal → r8_labels | t=50 | 720 | **+0.043** | 0.0046 |
| r1_xcal → r8_labels | t=100 | 720 | **+0.12** | 0.0058 |
| r1_xcal → r8_labels | t=150 | 720 | **+0.11** | 0.0055 |
| r1_xcal → r8_labels | mean t=1..150 | 720 | **+0.077** | 0.0027 |
| r8_labels → r8_labels_b025 | t=10 | 720 | +0 | 0 |
| r8_labels → r8_labels_b025 | t=20 | 720 | +0 | 0 |
| r8_labels → r8_labels_b025 | t=30 | 720 | **+0.068** | 0.0044 |
| r8_labels → r8_labels_b025 | t=50 | 720 | **+0.084** | 0.0046 |
| r8_labels → r8_labels_b025 | t=100 | 720 | **+0.14** | 0.0048 |
| r8_labels → r8_labels_b025 | t=150 | 720 | **+0.15** | 0.0051 |
| r8_labels → r8_labels_b025 | mean t=1..150 | 720 | **+0.1** | 0.0026 |
| r8_labels_b025 → r8_labels_b4 | t=10 | 720 | +0 | 0 |
| r8_labels_b025 → r8_labels_b4 | t=20 | 720 | +0 | 0 |
| r8_labels_b025 → r8_labels_b4 | t=30 | 720 | **-0.12** | 0.007 |
| r8_labels_b025 → r8_labels_b4 | t=50 | 720 | **-0.16** | 0.0074 |
| r8_labels_b025 → r8_labels_b4 | t=100 | 720 | **-0.27** | 0.0083 |
| r8_labels_b025 → r8_labels_b4 | t=150 | 720 | **-0.29** | 0.0078 |
| r8_labels_b025 → r8_labels_b4 | mean t=1..150 | 720 | **-0.19** | 0.0047 |

Bold = more than 2 SE from zero.

## The returned set's F1 (filled) at checkpoints

| rung | t=0 | t=10 | t=20 | t=30 | t=50 | t=100 | t=150 |
|---|---|---|---|---|---|---|---|
| r1_xcal | 0.023 | 0.023 | 0.023 | 0.33 | 0.41 | 0.47 | 0.48 |
| r8_labels | 0.023 | 0.023 | 0.023 | 0.32 | 0.4 | 0.49 | 0.51 |
| r8_labels_b025 | 0.023 | 0.023 | 0.023 | 0.28 | 0.35 | 0.42 | 0.47 |
| r8_labels_b4 | 0.023 | 0.023 | 0.023 | 0.3 | 0.36 | 0.43 | 0.44 |

## Each rung against the one before, in F1 (ΔF1 = to − from; positive = the step helped)

| from → to | at | n | Δ mean | SE |
|---|---|---|---|---|
| r1_xcal → r8_labels | t=10 | 720 | +0 | 0 |
| r1_xcal → r8_labels | t=20 | 720 | +0 | 0 |
| r1_xcal → r8_labels | t=30 | 720 | **-0.013** | 0.0034 |
| r1_xcal → r8_labels | t=50 | 720 | **-0.011** | 0.0028 |
| r1_xcal → r8_labels | t=100 | 720 | **+0.013** | 0.0029 |
| r1_xcal → r8_labels | t=150 | 720 | **+0.03** | 0.0028 |
| r1_xcal → r8_labels | mean t=1..150 | 720 | **+0.0061** | 0.0014 |
| r8_labels → r8_labels_b025 | t=10 | 720 | +0 | 0 |
| r8_labels → r8_labels_b025 | t=20 | 720 | +0 | 0 |
| r8_labels → r8_labels_b025 | t=30 | 720 | **-0.038** | 0.0037 |
| r8_labels → r8_labels_b025 | t=50 | 720 | **-0.047** | 0.0036 |
| r8_labels → r8_labels_b025 | t=100 | 720 | **-0.065** | 0.004 |
| r8_labels → r8_labels_b025 | t=150 | 720 | **-0.047** | 0.0037 |
| r8_labels → r8_labels_b025 | mean t=1..150 | 720 | **-0.047** | 0.002 |
| r8_labels_b025 → r8_labels_b4 | t=10 | 720 | +0 | 0 |
| r8_labels_b025 → r8_labels_b4 | t=20 | 720 | +0 | 0 |
| r8_labels_b025 → r8_labels_b4 | t=30 | 720 | **+0.016** | 0.0046 |
| r8_labels_b025 → r8_labels_b4 | t=50 | 720 | **+0.016** | 0.0046 |
| r8_labels_b025 → r8_labels_b4 | t=100 | 720 | +0.0059 | 0.0047 |
| r8_labels_b025 → r8_labels_b4 | t=150 | 720 | **-0.031** | 0.0044 |
| r8_labels_b025 → r8_labels_b4 | mean t=1..150 | 720 | +0.0026 | 0.0024 |
