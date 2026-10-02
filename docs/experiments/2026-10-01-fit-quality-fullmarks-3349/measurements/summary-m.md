36 classes, matrix `/expscratch/sgreenberg/fullmarks/votes-4162/matrix-m`.

## Templates: the query crop only

2,092 true pairs, 1,743,275 false (negatives capped at 200,000 a class).
A sane model (`model_ok`): **0.95** of true pairs, **0.12** of false.
Past the 8-inlier gate: 1,864 true, 13,594 false.

| statistic | median, true (model_ok) | median, false (model_ok) | AUC, all fits | AUC, model_ok | **AUC, past the gate** |
|---|---:|---:|---:|---:|---:|
| `inlier_count` | 28 | 4 | 0.92 | 0.98 | **0.93** |
| `inlier_ratio` | 0.604 | 0.3076 | 0.71 | 0.84 | **0.80** |
| `tentative_count` | 56.89 | 15.01 | 0.89 | 0.88 | **0.79** |
| `mean_reproj_error` | 0.007015 | 0.00391 | 0.07 | 0.24 | **0.48** |
| `median_reproj_error` | 0.006264 | 0.003544 | 0.09 | 0.27 | **0.49** |
| `scale` | 0.1182 | 0.2397 | 0.44 | 0.33 | **0.75** |
| `reflection` | 0 | 0 | 0.50 | 0.50 | **0.50** |
| `inlier_spread` | 0.03513 | 0.06052 | 0.63 | 0.36 | **0.77** |

## Templates: every template (crop + Good boxes)

41,306 true pairs, 7,086,498 false (negatives capped at 200,000 a class).
A sane model (`model_ok`): **0.97** of true pairs, **0.13** of false.
Past the 8-inlier gate: 37,858 true, 82,174 false.

| statistic | median, true (model_ok) | median, false (model_ok) | AUC, all fits | AUC, model_ok | **AUC, past the gate** |
|---|---:|---:|---:|---:|---:|
| `inlier_count` | 41.03 | 4 | 0.97 | 0.98 | **0.95** |
| `inlier_ratio` | 0.8867 | 0.3635 | 0.86 | 0.95 | **0.93** |
| `tentative_count` | 49.99 | 12.99 | 0.94 | 0.88 | **0.81** |
| `mean_reproj_error` | 0.001045 | 0.003557 | 0.15 | 0.67 | **0.86** |
| `median_reproj_error` | 0.0007281 | 0.003168 | 0.16 | 0.69 | **0.88** |
| `scale` | 0.9741 | 0.8149 | 0.38 | 0.51 | **0.85** |
| `reflection` | 0 | 0 | 0.50 | 0.50 | **0.50** |
| `inlier_spread` | 0.03793 | 0.03485 | 0.61 | 0.51 | **0.86** |

## Past the gate, per class (every template; classes with >= 20 false and >= 5 true gate-passing fits)

| statistic | classes | median AUC | min | max |
|---|---:|---:|---:|---:|
| `mean_reproj_error` | 31 | 0.87 | 0.37 | 0.98 |
| `inlier_ratio` | 31 | 0.90 | 0.19 | 1.00 |
| `inlier_spread` | 31 | 0.95 | 0.61 | 1.00 |
| `scale` | 31 | 0.90 | 0.51 | 0.99 |
