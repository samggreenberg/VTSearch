# Size vs size (#4160): analysis summary

20 seeds, 241 cells, embedders ['siglip']; cost weights fpr x1, fnr x1.

Runs with no detector yet at click 30 (scored at the text sort): S 31%, M 2%, L 0%, SML= 0%, SMLn 1%

## Click 30, siglip

### cost

| train \ test | S | M | L | SML= | SMLn |
|---|---|---|---|---|---|
| **S** | **0.60** | 0.57 | 0.58 | 0.58 | 0.58 |
| **M** | 0.47 | **0.38** | 0.35 | 0.40 | 0.39 |
| **L** | 0.55 | 0.36 | **0.25** | 0.38 | 0.36 |
| **SML=** | 0.53 | 0.38 | 0.29 | **0.39** | 0.38 |
| **SMLn** | 0.53 | 0.37 | 0.29 | 0.39 | **0.37** |

### auroc

| train \ test | S | M | L | SML= | SMLn |
|---|---|---|---|---|---|
| **S** | **0.76** | 0.78 | 0.77 | 0.77 | 0.77 |
| **M** | 0.83 | **0.90** | 0.92 | 0.89 | 0.89 |
| **L** | 0.79 | 0.90 | **0.97** | 0.89 | 0.90 |
| **SML=** | 0.80 | 0.90 | 0.96 | **0.89** | 0.90 |
| **SMLn** | 0.80 | 0.90 | 0.96 | 0.89 | **0.90** |

### recall

| train \ test | S | M | L | SML= | SMLn |
|---|---|---|---|---|---|
| **S** | **0.71** | 0.74 | 0.72 | 0.72 | 0.73 |
| **M** | 0.80 | **0.88** | 0.92 | 0.87 | 0.88 |
| **L** | 0.68 | 0.87 | **0.97** | 0.85 | 0.86 |
| **SML=** | 0.73 | 0.88 | 0.96 | **0.86** | 0.88 |
| **SMLn** | 0.73 | 0.88 | 0.96 | 0.86 | **0.88** |

### f1

| train \ test | S | M | L | SML= | SMLn |
|---|---|---|---|---|---|
| **S** | **0.03** | 0.03 | 0.03 | 0.08 | 0.08 |
| **M** | 0.03 | **0.04** | 0.04 | 0.09 | 0.09 |
| **L** | 0.03 | 0.04 | **0.05** | 0.11 | 0.11 |
| **SML=** | 0.03 | 0.04 | 0.04 | **0.09** | 0.10 |
| **SMLn** | 0.03 | 0.04 | 0.04 | 0.10 | **0.10** |

Text sort alone (click 0): S: cost 0.55, auroc 0.81; M: cost 0.44, auroc 0.91; L: cost 0.38, auroc 0.98; SML=: cost 0.46, auroc 0.90; SMLn: cost 0.45, auroc 0.91

| metric | test | train | minus train | diff | SE | classes | a better in |
|---|---|---|---|---|---|---|---|
| cost | S | M | S | -0.124 | 0.022 | 46 | 35/46 |
| cost | S | L | S | -0.050 | 0.022 | 46 | 26/46 |
| cost | S | SML= | S | -0.071 | 0.024 | 46 | 30/46 |
| cost | S | SMLn | S | -0.073 | 0.023 | 46 | 30/46 |
| cost | M | S | M | +0.177 | 0.033 | 46 | 8/46 |
| cost | M | L | M | -0.028 | 0.011 | 49 | 29/49 |
| cost | M | SML= | M | -0.009 (not resolvable) | 0.009 | 49 | 24/49 |
| cost | M | SMLn | M | -0.013 (not resolvable) | 0.009 | 49 | 21/49 |
| cost | L | S | L | +0.326 | 0.047 | 46 | 7/46 |
| cost | L | M | L | +0.103 | 0.018 | 49 | 9/49 |
| cost | L | SML= | L | +0.041 | 0.006 | 49 | 6/49 |
| cost | L | SMLn | L | +0.044 | 0.008 | 49 | 8/49 |
| cost | SML= | S | SML= | +0.180 | 0.032 | 46 | 12/46 |
| cost | SML= | M | SML= | +0.008 (not resolvable) | 0.010 | 49 | 26/49 |
| cost | SML= | L | SML= | -0.014 | 0.005 | 49 | 31/49 |
| cost | SML= | SMLn | SML= | -0.001 (not resolvable) | 0.004 | 49 | 26/49 |
| cost | SMLn | S | SMLn | +0.194 | 0.035 | 46 | 11/46 |
| cost | SMLn | M | SMLn | +0.021 (not resolvable) | 0.012 | 49 | 26/49 |
| cost | SMLn | L | SMLn | -0.013 (not resolvable) | 0.007 | 49 | 30/49 |
| cost | SMLn | SML= | SMLn | +0.005 (not resolvable) | 0.004 | 49 | 19/49 |
| auroc | S | M | S | +0.077 | 0.014 | 46 | 35/46 |
| auroc | S | L | S | +0.031 | 0.014 | 46 | 25/46 |
| auroc | S | SML= | S | +0.043 | 0.015 | 46 | 28/46 |
| auroc | S | SMLn | S | +0.044 | 0.015 | 46 | 28/46 |
| auroc | M | S | M | -0.112 | 0.023 | 46 | 9/46 |
| auroc | M | L | M | +0.003 (not resolvable) | 0.006 | 49 | 21/49 |
| auroc | M | SML= | M | +0.003 (not resolvable) | 0.005 | 49 | 21/49 |
| auroc | M | SMLn | M | +0.004 (not resolvable) | 0.005 | 49 | 20/49 |
| auroc | L | S | L | -0.204 | 0.034 | 46 | 5/46 |
| auroc | L | M | L | -0.049 | 0.011 | 49 | 7/49 |
| auroc | L | SML= | L | -0.008 | 0.003 | 49 | 12/49 |
| auroc | L | SMLn | L | -0.013 | 0.004 | 49 | 15/49 |
| auroc | SML= | S | SML= | -0.118 | 0.022 | 46 | 10/46 |
| auroc | SML= | M | SML= | -0.004 (not resolvable) | 0.006 | 49 | 27/49 |
| auroc | SML= | L | SML= | -0.001 (not resolvable) | 0.003 | 49 | 19/49 |
| auroc | SML= | SMLn | SML= | -0.001 (not resolvable) | 0.002 | 49 | 20/49 |
| auroc | SMLn | S | SMLn | -0.125 | 0.024 | 46 | 10/46 |
| auroc | SMLn | M | SMLn | -0.010 (not resolvable) | 0.006 | 49 | 30/49 |
| auroc | SMLn | L | SMLn | -0.001 (not resolvable) | 0.004 | 49 | 19/49 |
| auroc | SMLn | SML= | SMLn | -0.001 (not resolvable) | 0.002 | 49 | 27/49 |

Runs with no detector yet at click 150 (scored at the text sort): S 8%, M 0%, L 0%, SML= 0%, SMLn 0%

## Click 150, siglip

### cost

| train \ test | S | M | L | SML= | SMLn |
|---|---|---|---|---|---|
| **S** | **0.48** | 0.45 | 0.49 | 0.47 | 0.47 |
| **M** | 0.40 | **0.30** | 0.25 | 0.31 | 0.31 |
| **L** | 0.54 | 0.31 | **0.19** | 0.34 | 0.32 |
| **SML=** | 0.46 | 0.31 | 0.23 | **0.33** | 0.31 |
| **SMLn** | 0.47 | 0.31 | 0.23 | 0.33 | **0.31** |

### auroc

| train \ test | S | M | L | SML= | SMLn |
|---|---|---|---|---|---|
| **S** | **0.82** | 0.84 | 0.81 | 0.82 | 0.82 |
| **M** | 0.86 | **0.93** | 0.96 | 0.92 | 0.92 |
| **L** | 0.79 | 0.91 | **0.98** | 0.90 | 0.91 |
| **SML=** | 0.83 | 0.92 | 0.97 | **0.91** | 0.92 |
| **SMLn** | 0.83 | 0.92 | 0.97 | 0.91 | **0.92** |

### recall

| train \ test | S | M | L | SML= | SMLn |
|---|---|---|---|---|---|
| **S** | **0.76** | 0.78 | 0.74 | 0.76 | 0.76 |
| **M** | 0.81 | **0.91** | 0.95 | 0.89 | 0.90 |
| **L** | 0.62 | 0.85 | **0.97** | 0.82 | 0.84 |
| **SML=** | 0.75 | 0.89 | 0.97 | **0.87** | 0.89 |
| **SMLn** | 0.73 | 0.89 | 0.97 | 0.87 | **0.89** |

### f1

| train \ test | S | M | L | SML= | SMLn |
|---|---|---|---|---|---|
| **S** | **0.04** | 0.04 | 0.03 | 0.10 | 0.10 |
| **M** | 0.04 | **0.05** | 0.05 | 0.12 | 0.12 |
| **L** | 0.04 | 0.06 | **0.06** | 0.14 | 0.14 |
| **SML=** | 0.04 | 0.04 | 0.05 | **0.11** | 0.12 |
| **SMLn** | 0.04 | 0.05 | 0.05 | 0.12 | **0.12** |

Text sort alone (click 0): S: cost 0.55, auroc 0.81; M: cost 0.44, auroc 0.91; L: cost 0.38, auroc 0.98; SML=: cost 0.46, auroc 0.90; SMLn: cost 0.45, auroc 0.91

| metric | test | train | minus train | diff | SE | classes | a better in |
|---|---|---|---|---|---|---|---|
| cost | S | M | S | -0.073 | 0.020 | 46 | 33/46 |
| cost | S | L | S | +0.067 | 0.019 | 46 | 12/46 |
| cost | S | SML= | S | -0.020 (not resolvable) | 0.019 | 46 | 22/46 |
| cost | S | SMLn | S | -0.010 (not resolvable) | 0.018 | 46 | 22/46 |
| cost | M | S | M | +0.147 | 0.021 | 46 | 6/46 |
| cost | M | L | M | +0.017 (not resolvable) | 0.009 | 49 | 20/49 |
| cost | M | SML= | M | +0.012 (not resolvable) | 0.007 | 49 | 18/49 |
| cost | M | SMLn | M | +0.011 (not resolvable) | 0.006 | 49 | 18/49 |
| cost | L | S | L | +0.300 | 0.038 | 46 | 3/46 |
| cost | L | M | L | +0.067 | 0.010 | 49 | 5/49 |
| cost | L | SML= | L | +0.042 | 0.005 | 49 | 6/49 |
| cost | L | SMLn | L | +0.040 | 0.005 | 49 | 4/49 |
| cost | SML= | S | SML= | +0.138 | 0.023 | 46 | 8/46 |
| cost | SML= | M | SML= | -0.012 | 0.006 | 49 | 34/49 |
| cost | SML= | L | SML= | +0.014 | 0.005 | 49 | 15/49 |
| cost | SML= | SMLn | SML= | +0.002 (not resolvable) | 0.002 | 49 | 18/49 |
| cost | SMLn | S | SMLn | +0.157 | 0.026 | 46 | 7/46 |
| cost | SMLn | M | SMLn | -0.002 (not resolvable) | 0.006 | 49 | 31/49 |
| cost | SMLn | L | SMLn | +0.010 (not resolvable) | 0.006 | 49 | 19/49 |
| cost | SMLn | SML= | SMLn | +0.005 (not resolvable) | 0.003 | 49 | 20/49 |
| auroc | S | M | S | +0.040 | 0.013 | 46 | 27/46 |
| auroc | S | L | S | -0.034 | 0.012 | 46 | 11/46 |
| auroc | S | SML= | S | +0.010 (not resolvable) | 0.012 | 46 | 17/46 |
| auroc | S | SMLn | S | +0.004 (not resolvable) | 0.012 | 46 | 19/46 |
| auroc | M | S | M | -0.086 | 0.013 | 46 | 2/46 |
| auroc | M | L | M | -0.019 | 0.005 | 49 | 11/49 |
| auroc | M | SML= | M | -0.008 (not resolvable) | 0.004 | 49 | 13/49 |
| auroc | M | SMLn | M | -0.008 | 0.003 | 49 | 11/49 |
| auroc | L | S | L | -0.170 | 0.024 | 46 | 1/46 |
| auroc | L | M | L | -0.022 | 0.006 | 49 | 5/49 |
| auroc | L | SML= | L | -0.006 | 0.001 | 49 | 6/49 |
| auroc | L | SMLn | L | -0.007 | 0.002 | 49 | 7/49 |
| auroc | SML= | S | SML= | -0.084 | 0.014 | 46 | 3/46 |
| auroc | SML= | M | SML= | +0.007 (not resolvable) | 0.004 | 49 | 37/49 |
| auroc | SML= | L | SML= | -0.015 | 0.002 | 49 | 8/49 |
| auroc | SML= | SMLn | SML= | -0.002 (not resolvable) | 0.001 | 49 | 20/49 |
| auroc | SMLn | S | SMLn | -0.094 | 0.016 | 46 | 3/46 |
| auroc | SMLn | M | SMLn | +0.003 (not resolvable) | 0.004 | 49 | 33/49 |
| auroc | SMLn | L | SMLn | -0.013 | 0.003 | 49 | 13/49 |
| auroc | SMLn | SML= | SMLn | -0.001 (not resolvable) | 0.002 | 49 | 26/49 |

