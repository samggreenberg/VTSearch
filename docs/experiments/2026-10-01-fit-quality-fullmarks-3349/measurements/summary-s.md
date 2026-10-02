36 classes, matrix `/expscratch/sgreenberg/fullmarks/votes-4162/matrix-s`.

## Templates: the query crop only

989 true pairs, 171,558 false (negatives capped at 200,000 a class).
A sane model (`model_ok`): **0.97** of true pairs, **0.15** of false.
Past the 8-inlier gate: 918 true, 3,697 false.

| statistic | median, true (model_ok) | median, false (model_ok) | AUC, all fits | AUC, model_ok | **AUC, past the gate** |
|---|---:|---:|---:|---:|---:|
| `inlier_count` | 46.99 | 4 | 0.94 | 0.98 | **0.92** |
| `inlier_ratio` | 0.6228 | 0.3157 | 0.74 | 0.85 | **0.84** |
| `tentative_count` | 77.52 | 16.01 | 0.92 | 0.92 | **0.82** |
| `mean_reproj_error` | 0.006681 | 0.004662 | 0.11 | 0.32 | **0.55** |
| `median_reproj_error` | 0.006054 | 0.004166 | 0.12 | 0.34 | **0.55** |
| `scale` | 0.1234 | 0.1536 | 0.50 | 0.42 | **0.71** |
| `reflection` | 0 | 0 | 0.50 | 0.50 | **0.50** |
| `inlier_spread` | 0.03783 | 0.03897 | 0.66 | 0.46 | **0.78** |

## Templates: every template (crop + Good boxes)

41,859 true pairs, 4,342,853 false (negatives capped at 200,000 a class).
A sane model (`model_ok`): **0.93** of true pairs, **0.16** of false.
Past the 8-inlier gate: 36,404 true, 123,400 false.

| statistic | median, true (model_ok) | median, false (model_ok) | AUC, all fits | AUC, model_ok | **AUC, past the gate** |
|---|---:|---:|---:|---:|---:|
| `inlier_count` | 46.99 | 5.001 | 0.95 | 0.96 | **0.92** |
| `inlier_ratio` | 0.8945 | 0.3999 | 0.86 | 0.96 | **0.95** |
| `tentative_count` | 56.89 | 14.01 | 0.91 | 0.85 | **0.78** |
| `mean_reproj_error` | 0.001185 | 0.004185 | 0.19 | 0.70 | **0.84** |
| `median_reproj_error` | 0.0008035 | 0.0037 | 0.20 | 0.72 | **0.85** |
| `scale` | 0.9966 | 0.7139 | 0.43 | 0.58 | **0.80** |
| `reflection` | 0 | 0 | 0.50 | 0.50 | **0.50** |
| `inlier_spread` | 0.04102 | 0.0276 | 0.60 | 0.59 | **0.83** |

## Past the gate, per class (every template; classes with >= 20 false and >= 5 true gate-passing fits)

| statistic | classes | median AUC | min | max |
|---|---:|---:|---:|---:|
| `mean_reproj_error` | 30 | 0.85 | 0.48 | 0.98 |
| `inlier_ratio` | 30 | 0.90 | 0.21 | 1.00 |
| `inlier_spread` | 30 | 0.94 | 0.56 | 1.00 |
| `scale` | 30 | 0.90 | 0.55 | 0.99 |
