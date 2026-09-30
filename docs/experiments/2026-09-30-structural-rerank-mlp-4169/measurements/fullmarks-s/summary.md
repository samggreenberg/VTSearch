## All classes

35 classes.

### Closed loop: positives found after v votes (mean over classes)

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.00 | 2.74 | 4.43 | 7.97 | 14.49 | 22.11 |
| a1_max | 0.00 | 2.83 | 4.57 | 8.17 | 15.17 | 23.14 |
| a5_mlp | 0.00 | 2.83 | 4.51 | 8.03 | 14.60 | 22.20 |
| a3_vlad_svm | 0.00 | 0.00 | 0.00 | 0.03 | 0.17 | 0.71 |
| a3s_production | 0.00 | 0.00 | 0.00 | 0.03 | 0.17 | 1.09 |
| a3s_cold | 0.00 | 0.00 | 0.00 | 0.03 | 0.17 | 1.14 |
| a3s_inliers | 0.00 | 0.00 | 0.00 | 0.03 | 0.17 | 1.11 |

### Shared sequence: AP on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.86 | 0.82 (34) | 0.79 (31) | 0.74 (29) | 0.65 (29) | 0.32 (22) |
| a1_max | 0.86 | 0.90 (34) | 0.88 (31) | 0.87 (29) | 0.85 (29) | 0.71 (22) |
| a5_mlp | 0.86 | 0.80 (34) | 0.77 (31) | 0.69 (29) | 0.59 (29) | 0.24 (22) |
| a3_vlad_svm | 0.02 | 0.02 (34) | 0.03 (31) | 0.08 (29) | 0.12 (29) | 0.14 (22) |
| a3s_production | 0.02 | 0.07 (34) | 0.09 (31) | 0.17 (29) | 0.19 (29) | 0.27 (22) |
| a3s_cold | 0.02 | 0.07 (34) | 0.09 (31) | 0.20 (29) | 0.22 (29) | 0.37 (22) |
| a3s_inliers | 0.02 | 0.07 (34) | 0.09 (31) | 0.20 (29) | 0.22 (29) | 0.41 (22) |

### Shared sequence: P@10 on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.80 | 0.75 (34) | 0.78 (31) | 0.79 (29) | 0.67 (29) | 0.28 (22) |
| a1_max | 0.80 | 0.78 (34) | 0.82 (31) | 0.85 (29) | 0.80 (29) | 0.44 (22) |
| a5_mlp | 0.80 | 0.71 (34) | 0.73 (31) | 0.69 (29) | 0.57 (29) | 0.23 (22) |
| a3_vlad_svm | 0.02 | 0.01 (34) | 0.01 (31) | 0.06 (29) | 0.11 (29) | 0.10 (22) |
| a3s_production | 0.02 | 0.07 (34) | 0.12 (31) | 0.19 (29) | 0.18 (29) | 0.13 (22) |
| a3s_cold | 0.02 | 0.07 (34) | 0.12 (31) | 0.20 (29) | 0.19 (29) | 0.17 (22) |
| a3s_inliers | 0.02 | 0.07 (34) | 0.12 (31) | 0.21 (29) | 0.20 (29) | 0.17 (22) |

### Shared sequence: F1 of the accept decision on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.44 | 0.36 (34) | 0.37 (31) | 0.33 (29) | 0.24 (29) | 0.13 (22) |
| a1_max | 0.44 | 0.26 (34) | 0.26 (31) | 0.25 (29) | 0.20 (29) | 0.26 (22) |
| a5_mlp | 0.44 | 0.21 (34) | 0.20 (31) | 0.18 (29) | 0.13 (29) | 0.10 (22) |
| a3s_production | nan (0) | 0.08 (34) | 0.07 (31) | 0.04 (29) | 0.01 (29) | 0.24 (22) |
| a3s_cold | nan (0) | 0.08 (34) | 0.09 (31) | 0.19 (29) | 0.21 (29) | 0.31 (22) |
| a3s_inliers | nan (0) | 0.08 (34) | 0.09 (31) | 0.19 (29) | 0.21 (29) | 0.31 (22) |

### Paired against `a1_max` (mean difference, bootstrap 95% interval over classes)

| arm | readout | v | classes | mean diff | 95% interval |
|---|---|---:|---:|---:|---|
| a0_exemplar | shared ap | 10 | 29 | -0.128 | [-0.216, -0.040] |
| a0_exemplar | shared ap | 20 | 29 | -0.200 | [-0.302, -0.103] |
| a0_exemplar | shared f1 | 10 | 29 | +0.085 | [-0.005, +0.166] |
| a0_exemplar | shared f1 | 20 | 29 | +0.040 | [-0.046, +0.116] |
| a0_exemplar | closed found | 10 | 35 | -0.200 | [-0.600, +0.171] |
| a0_exemplar | closed found | 40 | 35 | -1.029 | [-2.114, +0.057] |
| a5_mlp | shared ap | 10 | 29 | -0.182 | [-0.292, -0.082] |
| a5_mlp | shared ap | 20 | 29 | -0.264 | [-0.396, -0.139] |
| a5_mlp | shared f1 | 10 | 29 | -0.065 | [-0.134, -0.013] |
| a5_mlp | shared f1 | 20 | 29 | -0.072 | [-0.140, -0.015] |
| a5_mlp | closed found | 10 | 35 | -0.143 | [-0.429, +0.000] |
| a5_mlp | closed found | 40 | 35 | -0.943 | [-2.086, -0.229] |
| a3_vlad_svm | shared ap | 10 | 29 | -0.794 | [-0.890, -0.681] |
| a3_vlad_svm | shared ap | 20 | 29 | -0.730 | [-0.847, -0.598] |
| a3_vlad_svm | closed found | 10 | 35 | -8.143 | [-9.057, -7.114] |
| a3_vlad_svm | closed found | 40 | 35 | -22.429 | [-26.429, -18.257] |
| a3s_production | shared ap | 10 | 29 | -0.702 | [-0.831, -0.560] |
| a3s_production | shared ap | 20 | 29 | -0.667 | [-0.804, -0.514] |
| a3s_production | shared f1 | 10 | 29 | -0.208 | [-0.330, -0.094] |
| a3s_production | shared f1 | 20 | 29 | -0.189 | [-0.295, -0.097] |
| a3s_production | closed found | 10 | 35 | -8.143 | [-9.057, -7.114] |
| a3s_production | closed found | 40 | 35 | -22.057 | [-26.001, -17.943] |
| a3s_cold | shared ap | 10 | 29 | -0.673 | [-0.821, -0.505] |
| a3s_cold | shared ap | 20 | 29 | -0.638 | [-0.786, -0.474] |
| a3s_cold | shared f1 | 10 | 29 | -0.059 | [-0.233, +0.119] |
| a3s_cold | shared f1 | 20 | 29 | +0.009 | [-0.167, +0.195] |
| a3s_cold | closed found | 10 | 35 | -8.143 | [-9.057, -7.114] |
| a3s_cold | closed found | 40 | 35 | -22.000 | [-25.914, -17.914] |
| a3s_inliers | shared ap | 10 | 29 | -0.673 | [-0.819, -0.509] |
| a3s_inliers | shared ap | 20 | 29 | -0.631 | [-0.783, -0.464] |
| a3s_inliers | shared f1 | 10 | 29 | -0.059 | [-0.233, +0.119] |
| a3s_inliers | shared f1 | 20 | 29 | +0.009 | [-0.167, +0.195] |
| a3s_inliers | closed found | 10 | 35 | -8.143 | [-9.057, -7.114] |
| a3s_inliers | closed found | 40 | 35 | -22.029 | [-25.971, -17.943] |

### Paired against `a3s_production` (mean difference, bootstrap 95% interval over classes)

| arm | readout | v | classes | mean diff | 95% interval |
|---|---|---:|---:|---:|---|
| a3_vlad_svm | shared ap | 10 | 29 | -0.091 | [-0.144, -0.045] |
| a3_vlad_svm | shared ap | 20 | 29 | -0.063 | [-0.121, -0.018] |
| a3_vlad_svm | closed found | 10 | 35 | +0.000 | [+0.000, +0.000] |
| a3_vlad_svm | closed found | 40 | 35 | -0.371 | [-0.857, -0.029] |
| a3s_cold | shared ap | 10 | 29 | +0.029 | [-0.002, +0.071] |
| a3s_cold | shared ap | 20 | 29 | +0.029 | [-0.009, +0.082] |
| a3s_cold | shared f1 | 10 | 29 | +0.149 | [+0.053, +0.261] |
| a3s_cold | shared f1 | 20 | 29 | +0.198 | [+0.078, +0.336] |
| a3s_cold | closed found | 10 | 35 | +0.000 | [+0.000, +0.000] |
| a3s_cold | closed found | 40 | 35 | +0.057 | [-0.057, +0.200] |
| a3s_inliers | shared ap | 10 | 29 | +0.029 | [+0.001, +0.064] |
| a3s_inliers | shared ap | 20 | 29 | +0.036 | [+0.000, +0.088] |
| a3s_inliers | shared f1 | 10 | 29 | +0.149 | [+0.053, +0.261] |
| a3s_inliers | shared f1 | 20 | 29 | +0.198 | [+0.078, +0.336] |
| a3s_inliers | closed found | 10 | 35 | +0.000 | [+0.000, +0.000] |
| a3s_inliers | closed found | 40 | 35 | +0.029 | [-0.057, +0.114] |


## The 27 classes v4.3 had (the nine v5.0 added are easy for SIFT)

26 classes.

### Closed loop: positives found after v votes (mean over classes)

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.00 | 2.65 | 4.31 | 7.96 | 14.50 | 22.27 |
| a1_max | 0.00 | 2.77 | 4.46 | 8.15 | 15.27 | 23.38 |
| a5_mlp | 0.00 | 2.77 | 4.38 | 7.96 | 14.50 | 22.35 |
| a3_vlad_svm | 0.00 | 0.00 | 0.00 | 0.04 | 0.23 | 0.96 |
| a3s_production | 0.00 | 0.00 | 0.00 | 0.04 | 0.23 | 1.46 |
| a3s_cold | 0.00 | 0.00 | 0.00 | 0.04 | 0.23 | 1.54 |
| a3s_inliers | 0.00 | 0.00 | 0.00 | 0.04 | 0.23 | 1.50 |

### Shared sequence: AP on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.82 | 0.77 (25) | 0.75 (25) | 0.69 (23) | 0.61 (23) | 0.27 (17) |
| a1_max | 0.82 | 0.87 (25) | 0.86 (25) | 0.85 (23) | 0.83 (23) | 0.72 (17) |
| a5_mlp | 0.82 | 0.74 (25) | 0.76 (25) | 0.66 (23) | 0.57 (23) | 0.25 (17) |
| a3_vlad_svm | 0.02 | 0.03 (25) | 0.02 (25) | 0.07 (23) | 0.11 (23) | 0.14 (17) |
| a3s_production | 0.02 | 0.09 (25) | 0.09 (25) | 0.15 (23) | 0.15 (23) | 0.24 (17) |
| a3s_cold | 0.02 | 0.09 (25) | 0.10 (25) | 0.19 (23) | 0.19 (23) | 0.31 (17) |
| a3s_inliers | 0.02 | 0.09 (25) | 0.10 (25) | 0.19 (23) | 0.19 (23) | 0.32 (17) |

### Shared sequence: P@10 on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.80 | 0.76 (25) | 0.73 (25) | 0.74 (23) | 0.63 (23) | 0.27 (17) |
| a1_max | 0.80 | 0.80 (25) | 0.77 (25) | 0.81 (23) | 0.77 (23) | 0.46 (17) |
| a5_mlp | 0.80 | 0.71 (25) | 0.70 (25) | 0.66 (23) | 0.55 (23) | 0.24 (17) |
| a3_vlad_svm | 0.02 | 0.01 (25) | 0.00 (25) | 0.04 (23) | 0.09 (23) | 0.11 (17) |
| a3s_production | 0.02 | 0.10 (25) | 0.11 (25) | 0.15 (23) | 0.14 (23) | 0.12 (17) |
| a3s_cold | 0.02 | 0.10 (25) | 0.11 (25) | 0.17 (23) | 0.16 (23) | 0.18 (17) |
| a3s_inliers | 0.02 | 0.10 (25) | 0.11 (25) | 0.18 (23) | 0.16 (23) | 0.18 (17) |

### Shared sequence: F1 of the accept decision on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.50 | 0.43 (25) | 0.41 (25) | 0.37 (23) | 0.27 (23) | 0.16 (17) |
| a1_max | 0.50 | 0.33 (25) | 0.31 (25) | 0.30 (23) | 0.25 (23) | 0.33 (17) |
| a5_mlp | 0.50 | 0.26 (25) | 0.23 (25) | 0.22 (23) | 0.16 (23) | 0.13 (17) |
| a3s_production | nan (0) | 0.10 (25) | 0.07 (25) | 0.05 (23) | 0.02 (23) | 0.23 (17) |
| a3s_cold | nan (0) | 0.11 (25) | 0.09 (25) | 0.18 (23) | 0.19 (23) | 0.26 (17) |
| a3s_inliers | nan (0) | 0.11 (25) | 0.09 (25) | 0.18 (23) | 0.19 (23) | 0.26 (17) |

### Paired against `a1_max` (mean difference, bootstrap 95% interval over classes)

| arm | readout | v | classes | mean diff | 95% interval |
|---|---|---:|---:|---:|---|
| a0_exemplar | shared ap | 10 | 23 | -0.150 | [-0.260, -0.039] |
| a0_exemplar | shared ap | 20 | 23 | -0.221 | [-0.342, -0.104] |
| a0_exemplar | shared f1 | 10 | 23 | +0.076 | [-0.039, +0.180] |
| a0_exemplar | shared f1 | 20 | 23 | +0.028 | [-0.079, +0.126] |
| a0_exemplar | closed found | 10 | 26 | -0.192 | [-0.692, +0.308] |
| a0_exemplar | closed found | 40 | 26 | -1.115 | [-2.577, +0.346] |
| a5_mlp | shared ap | 10 | 23 | -0.181 | [-0.304, -0.071] |
| a5_mlp | shared ap | 20 | 23 | -0.264 | [-0.413, -0.127] |
| a5_mlp | shared f1 | 10 | 23 | -0.077 | [-0.165, -0.012] |
| a5_mlp | shared f1 | 20 | 23 | -0.088 | [-0.171, -0.019] |
| a5_mlp | closed found | 10 | 26 | -0.192 | [-0.577, +0.000] |
| a5_mlp | closed found | 40 | 26 | -1.038 | [-2.500, -0.115] |
| a3_vlad_svm | shared ap | 10 | 23 | -0.776 | [-0.891, -0.633] |
| a3_vlad_svm | shared ap | 20 | 23 | -0.727 | [-0.863, -0.569] |
| a3_vlad_svm | closed found | 10 | 26 | -8.115 | [-9.231, -6.846] |
| a3_vlad_svm | closed found | 40 | 26 | -22.423 | [-27.269, -17.423] |
| a3s_production | shared ap | 10 | 23 | -0.695 | [-0.842, -0.528] |
| a3s_production | shared ap | 20 | 23 | -0.685 | [-0.829, -0.522] |
| a3s_production | shared f1 | 10 | 23 | -0.249 | [-0.390, -0.106] |
| a3s_production | shared f1 | 20 | 23 | -0.231 | [-0.352, -0.119] |
| a3s_production | closed found | 10 | 26 | -8.115 | [-9.231, -6.846] |
| a3s_production | closed found | 40 | 26 | -21.923 | [-26.655, -17.000] |
| a3s_cold | shared ap | 10 | 23 | -0.654 | [-0.827, -0.449] |
| a3s_cold | shared ap | 20 | 23 | -0.641 | [-0.807, -0.455] |
| a3s_cold | shared f1 | 10 | 23 | -0.123 | [-0.316, +0.079] |
| a3s_cold | shared f1 | 20 | 23 | -0.062 | [-0.253, +0.143] |
| a3s_cold | closed found | 10 | 26 | -8.115 | [-9.231, -6.846] |
| a3s_cold | closed found | 40 | 26 | -21.846 | [-26.538, -16.962] |
| a3s_inliers | shared ap | 10 | 23 | -0.658 | [-0.827, -0.457] |
| a3s_inliers | shared ap | 20 | 23 | -0.640 | [-0.807, -0.454] |
| a3s_inliers | shared f1 | 10 | 23 | -0.123 | [-0.316, +0.079] |
| a3s_inliers | shared f1 | 20 | 23 | -0.062 | [-0.253, +0.143] |
| a3s_inliers | closed found | 10 | 26 | -8.115 | [-9.231, -6.846] |
| a3s_inliers | closed found | 40 | 26 | -21.885 | [-26.577, -17.000] |

### Paired against `a3s_production` (mean difference, bootstrap 95% interval over classes)

| arm | readout | v | classes | mean diff | 95% interval |
|---|---|---:|---:|---:|---|
| a3_vlad_svm | shared ap | 10 | 23 | -0.081 | [-0.132, -0.038] |
| a3_vlad_svm | shared ap | 20 | 23 | -0.042 | [-0.082, -0.010] |
| a3_vlad_svm | closed found | 10 | 26 | +0.000 | [+0.000, +0.000] |
| a3_vlad_svm | closed found | 40 | 26 | -0.500 | [-1.154, -0.038] |
| a3s_cold | shared ap | 10 | 23 | +0.041 | [+0.001, +0.093] |
| a3s_cold | shared ap | 20 | 23 | +0.044 | [+0.000, +0.113] |
| a3s_cold | shared f1 | 10 | 23 | +0.125 | [+0.028, +0.243] |
| a3s_cold | shared f1 | 20 | 23 | +0.169 | [+0.053, +0.311] |
| a3s_cold | closed found | 10 | 26 | +0.000 | [+0.000, +0.000] |
| a3s_cold | closed found | 40 | 26 | +0.077 | [-0.077, +0.269] |
| a3s_inliers | shared ap | 10 | 23 | +0.037 | [+0.001, +0.082] |
| a3s_inliers | shared ap | 20 | 23 | +0.045 | [+0.000, +0.113] |
| a3s_inliers | shared f1 | 10 | 23 | +0.125 | [+0.028, +0.243] |
| a3s_inliers | shared f1 | 20 | 23 | +0.169 | [+0.053, +0.311] |
| a3s_inliers | closed found | 10 | 26 | +0.000 | [+0.000, +0.000] |
| a3s_inliers | closed found | 40 | 26 | +0.038 | [-0.077, +0.154] |
