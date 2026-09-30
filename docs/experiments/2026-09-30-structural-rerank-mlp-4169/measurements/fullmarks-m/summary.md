## All classes

36 classes.

### Closed loop: positives found after v votes (mean over classes)

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.00 | 2.81 | 4.67 | 8.56 | 15.50 |
| a1_max | 0.00 | 2.83 | 4.69 | 8.78 | 16.19 |
| a5_mlp | 0.00 | 2.83 | 4.58 | 8.53 | 15.75 |

### Shared sequence: AP on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.87 | 0.85 | 0.81 (33) | 0.76 (31) | 0.67 (31) |
| a1_max | 0.87 | 0.91 | 0.90 (33) | 0.87 (31) | 0.86 (31) |
| a5_mlp | 0.87 | 0.86 | 0.79 (33) | 0.70 (31) | 0.59 (31) |

### Shared sequence: P@10 on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.86 | 0.79 | 0.82 (33) | 0.81 (31) | 0.70 (31) |
| a1_max | 0.86 | 0.82 | 0.86 (33) | 0.86 (31) | 0.81 (31) |
| a5_mlp | 0.86 | 0.77 | 0.78 (33) | 0.72 (31) | 0.58 (31) |

### Shared sequence: F1 of the accept decision on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.33 | 0.31 | 0.31 (33) | 0.29 (31) | 0.22 (31) |
| a1_max | 0.33 | 0.23 | 0.23 (33) | 0.20 (31) | 0.14 (31) |
| a5_mlp | 0.33 | 0.20 | 0.20 (33) | 0.17 (31) | 0.11 (31) |

### Paired against `a1_max` (mean difference, bootstrap 95% interval over classes)

| arm | readout | v | classes | mean diff | 95% interval |
|---|---|---:|---:|---:|---|
| a0_exemplar | shared ap | 10 | 31 | -0.116 | [-0.191, -0.056] |
| a0_exemplar | shared ap | 20 | 31 | -0.184 | [-0.265, -0.108] |
| a0_exemplar | shared f1 | 10 | 31 | +0.093 | [+0.034, +0.161] |
| a0_exemplar | shared f1 | 20 | 31 | +0.080 | [+0.026, +0.141] |
| a0_exemplar | closed found | 10 | 36 | -0.222 | [-0.528, +0.028] |
| a5_mlp | shared ap | 10 | 31 | -0.176 | [-0.300, -0.066] |
| a5_mlp | shared ap | 20 | 31 | -0.271 | [-0.409, -0.142] |
| a5_mlp | shared f1 | 10 | 31 | -0.030 | [-0.082, -0.001] |
| a5_mlp | shared f1 | 20 | 31 | -0.036 | [-0.089, -0.005] |
| a5_mlp | closed found | 10 | 36 | -0.250 | [-0.639, +0.000] |


## The 27 classes v4.3 had (the nine v5.0 added are easy for SIFT)

27 classes.

### Closed loop: positives found after v votes (mean over classes)

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.00 | 2.74 | 4.63 | 8.74 | 15.85 |
| a1_max | 0.00 | 2.78 | 4.63 | 8.96 | 16.63 |
| a5_mlp | 0.00 | 2.78 | 4.48 | 8.63 | 16.04 |

### Shared sequence: AP on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.83 | 0.81 | 0.79 | 0.72 (25) | 0.64 (25) |
| a1_max | 0.83 | 0.88 | 0.88 | 0.85 (25) | 0.84 (25) |
| a5_mlp | 0.83 | 0.81 | 0.79 | 0.69 (25) | 0.57 (25) |

### Shared sequence: P@10 on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.87 | 0.82 | 0.78 | 0.77 (25) | 0.68 (25) |
| a1_max | 0.87 | 0.86 | 0.83 | 0.83 (25) | 0.79 (25) |
| a5_mlp | 0.87 | 0.78 | 0.77 | 0.70 (25) | 0.57 (25) |

### Shared sequence: F1 of the accept decision on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 |
|---|---:|---:|---:|---:|---:|
| a0_exemplar | 0.40 | 0.39 | 0.36 | 0.34 (25) | 0.27 (25) |
| a1_max | 0.40 | 0.30 | 0.28 | 0.24 (25) | 0.18 (25) |
| a5_mlp | 0.40 | 0.26 | 0.24 | 0.20 (25) | 0.13 (25) |

### Paired against `a1_max` (mean difference, bootstrap 95% interval over classes)

| arm | readout | v | classes | mean diff | 95% interval |
|---|---|---:|---:|---:|---|
| a0_exemplar | shared ap | 10 | 25 | -0.133 | [-0.222, -0.058] |
| a0_exemplar | shared ap | 20 | 25 | -0.199 | [-0.298, -0.111] |
| a0_exemplar | shared f1 | 10 | 25 | +0.100 | [+0.027, +0.181] |
| a0_exemplar | shared f1 | 20 | 25 | +0.089 | [+0.024, +0.162] |
| a0_exemplar | closed found | 10 | 27 | -0.222 | [-0.593, +0.111] |
| a5_mlp | shared ap | 10 | 25 | -0.166 | [-0.303, -0.044] |
| a5_mlp | shared ap | 20 | 25 | -0.267 | [-0.420, -0.127] |
| a5_mlp | shared f1 | 10 | 25 | -0.036 | [-0.099, -0.001] |
| a5_mlp | shared f1 | 20 | 25 | -0.045 | [-0.108, -0.006] |
| a5_mlp | closed found | 10 | 27 | -0.333 | [-0.852, +0.000] |
