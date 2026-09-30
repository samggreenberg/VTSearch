## All classes

18 classes.

### Closed loop: positives found after v votes (mean over classes)

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.00 | 2.67 | 3.94 | 7.00 | 10.44 | 14.00 |
| a1_max | 0.00 | 2.67 | 4.17 | 7.89 | 13.00 | 20.72 |
| a5_mlp | 0.00 | 2.67 | 4.11 | 7.61 | 12.00 | 16.89 |
| a3_vlad_svm | 0.00 | 0.33 | 0.50 | 0.78 | 1.78 | 2.94 |
| a3s_production | 0.00 | 0.33 | 0.56 | 1.00 | 1.89 | 3.11 |
| a3s_cold | 0.00 | 0.33 | 0.56 | 1.00 | 1.89 | 3.56 |
| a3s_inliers | 0.00 | 0.33 | 0.56 | 1.00 | 1.78 | 3.17 |

### Shared sequence: AP on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.30 | 0.24 | 0.20 | 0.11 | 0.07 | 0.03 |
| a1_max | 0.30 | 0.36 | 0.37 | 0.31 | 0.29 | 0.28 |
| a5_mlp | 0.30 | 0.34 | 0.31 | 0.24 | 0.22 | 0.22 |
| a3_vlad_svm | 0.05 | 0.05 | 0.04 | 0.02 | 0.02 | 0.03 |
| a3s_production | 0.05 | 0.08 | 0.06 | 0.02 | 0.04 | 0.07 |
| a3s_cold | 0.05 | 0.08 | 0.06 | 0.02 | 0.05 | 0.08 |
| a3s_inliers | 0.05 | 0.08 | 0.06 | 0.02 | 0.05 | 0.07 |

### Shared sequence: P@10 on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.70 | 0.56 | 0.49 | 0.34 | 0.20 | 0.08 |
| a1_max | 0.70 | 0.70 | 0.69 | 0.61 | 0.58 | 0.51 |
| a5_mlp | 0.70 | 0.63 | 0.58 | 0.51 | 0.46 | 0.36 |
| a3_vlad_svm | 0.11 | 0.09 | 0.06 | 0.03 | 0.05 | 0.05 |
| a3s_production | 0.11 | 0.14 | 0.09 | 0.04 | 0.09 | 0.11 |
| a3s_cold | 0.11 | 0.14 | 0.09 | 0.04 | 0.09 | 0.11 |
| a3s_inliers | 0.11 | 0.14 | 0.09 | 0.04 | 0.08 | 0.12 |

### Shared sequence: F1 of the accept decision on the unlabelled remainder

| arm | v=0 | v=3 | v=5 | v=10 | v=20 | v=40 |
|---|---:|---:|---:|---:|---:|---:|
| a0_exemplar | 0.15 | 0.09 | 0.06 | 0.01 | 0.00 | 0.00 |
| a1_max | 0.15 | 0.24 | 0.27 | 0.26 | 0.24 | 0.24 |
| a5_mlp | 0.15 | 0.24 | 0.17 | 0.13 | 0.06 | 0.15 |
| a3s_production | nan (0) | 0.07 (17) | 0.07 (17) | 0.01 | 0.04 | 0.07 |
| a3s_cold | nan (0) | 0.07 (17) | 0.06 (17) | 0.01 | 0.05 | 0.08 |
| a3s_inliers | nan (0) | 0.07 (17) | 0.06 (17) | 0.01 | 0.05 | 0.08 |

### Paired against `a1_max` (mean difference, bootstrap 95% interval over classes)

| arm | readout | v | classes | mean diff | 95% interval |
|---|---|---:|---:|---:|---|
| a0_exemplar | shared ap | 10 | 18 | -0.202 | [-0.312, -0.104] |
| a0_exemplar | shared ap | 20 | 18 | -0.224 | [-0.340, -0.118] |
| a0_exemplar | shared f1 | 10 | 18 | -0.254 | [-0.378, -0.143] |
| a0_exemplar | shared f1 | 20 | 18 | -0.243 | [-0.354, -0.142] |
| a0_exemplar | closed found | 10 | 18 | -0.889 | [-2.056, +0.333] |
| a0_exemplar | closed found | 40 | 18 | -6.722 | [-11.168, -2.389] |
| a5_mlp | shared ap | 10 | 18 | -0.068 | [-0.105, -0.033] |
| a5_mlp | shared ap | 20 | 18 | -0.068 | [-0.100, -0.039] |
| a5_mlp | shared f1 | 10 | 18 | -0.131 | [-0.231, -0.047] |
| a5_mlp | shared f1 | 20 | 18 | -0.187 | [-0.281, -0.099] |
| a5_mlp | closed found | 10 | 18 | -0.278 | [-0.778, +0.000] |
| a5_mlp | closed found | 40 | 18 | -3.833 | [-5.944, -2.111] |
| a3_vlad_svm | shared ap | 10 | 18 | -0.293 | [-0.413, -0.183] |
| a3_vlad_svm | shared ap | 20 | 18 | -0.268 | [-0.391, -0.155] |
| a3_vlad_svm | closed found | 10 | 18 | -7.111 | [-8.722, -5.444] |
| a3_vlad_svm | closed found | 40 | 18 | -17.778 | [-23.444, -12.389] |
| a3s_production | shared ap | 10 | 18 | -0.292 | [-0.412, -0.182] |
| a3s_production | shared ap | 20 | 18 | -0.248 | [-0.372, -0.132] |
| a3s_production | shared f1 | 10 | 18 | -0.253 | [-0.375, -0.143] |
| a3s_production | shared f1 | 20 | 18 | -0.199 | [-0.324, -0.079] |
| a3s_production | closed found | 10 | 18 | -6.889 | [-8.611, -5.056] |
| a3s_production | closed found | 40 | 18 | -17.611 | [-23.222, -12.278] |
| a3s_cold | shared ap | 10 | 18 | -0.292 | [-0.412, -0.182] |
| a3s_cold | shared ap | 20 | 18 | -0.246 | [-0.372, -0.129] |
| a3s_cold | shared f1 | 10 | 18 | -0.253 | [-0.375, -0.143] |
| a3s_cold | shared f1 | 20 | 18 | -0.195 | [-0.319, -0.077] |
| a3s_cold | closed found | 10 | 18 | -6.889 | [-8.611, -5.056] |
| a3s_cold | closed found | 40 | 18 | -17.167 | [-22.724, -12.000] |
| a3s_inliers | shared ap | 10 | 18 | -0.293 | [-0.412, -0.182] |
| a3s_inliers | shared ap | 20 | 18 | -0.247 | [-0.372, -0.130] |
| a3s_inliers | shared f1 | 10 | 18 | -0.253 | [-0.375, -0.143] |
| a3s_inliers | shared f1 | 20 | 18 | -0.195 | [-0.319, -0.077] |
| a3s_inliers | closed found | 10 | 18 | -6.889 | [-8.611, -5.056] |
| a3s_inliers | closed found | 40 | 18 | -17.556 | [-23.167, -12.222] |

### Paired against `a3s_production` (mean difference, bootstrap 95% interval over classes)

| arm | readout | v | classes | mean diff | 95% interval |
|---|---|---:|---:|---:|---|
| a3_vlad_svm | shared ap | 10 | 18 | -0.001 | [-0.002, -0.000] |
| a3_vlad_svm | shared ap | 20 | 18 | -0.021 | [-0.049, -0.004] |
| a3_vlad_svm | closed found | 10 | 18 | -0.222 | [-0.667, +0.000] |
| a3_vlad_svm | closed found | 40 | 18 | -0.167 | [-1.000, +0.500] |
| a3s_cold | shared ap | 10 | 18 | +0.000 | [+0.000, +0.000] |
| a3s_cold | shared ap | 20 | 18 | +0.002 | [-0.001, +0.006] |
| a3s_cold | shared f1 | 10 | 18 | +0.000 | [+0.000, +0.000] |
| a3s_cold | shared f1 | 20 | 18 | +0.003 | [-0.004, +0.010] |
| a3s_cold | closed found | 10 | 18 | +0.000 | [+0.000, +0.000] |
| a3s_cold | closed found | 40 | 18 | +0.444 | [+0.000, +1.056] |
| a3s_inliers | shared ap | 10 | 18 | -0.000 | [-0.001, +0.000] |
| a3s_inliers | shared ap | 20 | 18 | +0.001 | [-0.002, +0.005] |
| a3s_inliers | shared f1 | 10 | 18 | +0.000 | [+0.000, +0.000] |
| a3s_inliers | shared f1 | 20 | 18 | +0.003 | [-0.004, +0.010] |
| a3s_inliers | closed found | 10 | 18 | +0.000 | [+0.000, +0.000] |
| a3s_inliers | closed found | 40 | 18 | +0.056 | [-0.667, +0.833] |
