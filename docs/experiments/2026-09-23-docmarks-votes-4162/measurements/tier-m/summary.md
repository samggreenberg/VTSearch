## All classes

36 classes.

### Shared sequence: AP on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.87 | 0.85 | 0.81 (33) | 0.76 (31) | 0.67 (31) |
| a1_max | 0.87 | 0.91 | 0.90 (33) | 0.87 (31) | 0.86 (31) |
| a1p_goods_only | 0.87 | 0.91 | 0.90 (33) | 0.88 (31) | 0.86 (31) |
| a6_stoplist | nan (0) | 0.91 | 0.90 (33) | 0.90 (31) | 0.91 (31) |

### Shared sequence: P@10 on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.86 | 0.79 | 0.82 (33) | 0.81 (31) | 0.70 (31) |
| a1_max | 0.86 | 0.82 | 0.86 (33) | 0.86 (31) | 0.81 (31) |
| a1p_goods_only | 0.86 | 0.82 | 0.86 (33) | 0.86 (31) | 0.82 (31) |
| a6_stoplist | nan (0) | 0.84 | 0.88 (33) | 0.89 (31) | 0.85 (31) |

### Paired against `a1_max` (mean difference, bootstrap 95% interval over classes)

| arm | readout | v | classes | mean diff | 95% interval |
|---|---|---:|---:|---:|---|
| a0_exemplar | shared ap | 10 | 31 | -0.116 | [-0.191, -0.056] |
| a0_exemplar | shared ap | 20 | 31 | -0.184 | [-0.265, -0.108] |
| a1p_goods_only | shared ap | 10 | 31 | +0.008 | [+0.001, +0.019] |
| a1p_goods_only | shared ap | 20 | 31 | +0.007 | [+0.000, +0.018] |
| a6_stoplist | shared ap | 10 | 31 | +0.022 | [-0.013, +0.068] |
| a6_stoplist | shared ap | 20 | 31 | +0.049 | [+0.003, +0.106] |

## The 27 classes v4.3 had (the nine v5.0 added are easy for SIFT)

27 classes.

### Shared sequence: AP on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.83 | 0.81 | 0.79 | 0.72 (25) | 0.64 (25) |
| a1_max | 0.83 | 0.88 | 0.88 | 0.85 (25) | 0.84 (25) |
| a1p_goods_only | 0.83 | 0.89 | 0.88 | 0.86 (25) | 0.85 (25) |
| a6_stoplist | nan (0) | 0.89 | 0.89 | 0.88 (25) | 0.89 (25) |

### Shared sequence: P@10 on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.87 | 0.82 | 0.78 | 0.77 (25) | 0.68 (25) |
| a1_max | 0.87 | 0.86 | 0.83 | 0.83 (25) | 0.79 (25) |
| a1p_goods_only | 0.87 | 0.86 | 0.83 | 0.83 (25) | 0.80 (25) |
| a6_stoplist | nan (0) | 0.87 | 0.85 | 0.86 (25) | 0.84 (25) |

### Paired against `a1_max` (mean difference, bootstrap 95% interval over classes)

| arm | readout | v | classes | mean diff | 95% interval |
|---|---|---:|---:|---:|---|
| a0_exemplar | shared ap | 10 | 25 | -0.133 | [-0.222, -0.058] |
| a0_exemplar | shared ap | 20 | 25 | -0.199 | [-0.298, -0.111] |
| a1p_goods_only | shared ap | 10 | 25 | +0.010 | [+0.001, +0.024] |
| a1p_goods_only | shared ap | 20 | 25 | +0.008 | [+0.000, +0.023] |
| a6_stoplist | shared ap | 10 | 25 | +0.027 | [-0.017, +0.083] |
| a6_stoplist | shared ap | 20 | 25 | +0.056 | [+0.000, +0.126] |
