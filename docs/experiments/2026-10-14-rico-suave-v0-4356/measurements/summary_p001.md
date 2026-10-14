# Rico Suave v0 (#4356)

F-beta at the session's preset of what Find returns on the withheld half, every run at every vote (split by app). Runs per preset: {'b025': 110, 'b1': 110, 'b4': 110}. Empty cells: {'b025': 0, 'b1': 0, 'b4': 0}.

## Per preset, every class

| preset | dataset | runs | session mean F-beta (1-150) | short (1-50) | at vote 1 | at 150 | after the check | AP at 150 | ceiling AP | return-everything F-beta | first detector (median vote) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b025 | rico_suave | 110 | 0.61 | 0.52 | 0.46 | 0.63 | 0.63 | 0.61 | 0.63 | 0.01 | 30 |
| b025 | coco_better (ref) | 720 | 0.56 | 0.49 | 0.46 | 0.64 | 0.65 | 0.56 |  | 0.00 | 30 |
| b1 | rico_suave | 110 | 0.42 | 0.28 | 0.18 | 0.51 | 0.50 | 0.61 | 0.63 | 0.02 | 30 |
| b1 | coco_better (ref) | 720 | 0.46 | 0.39 | 0.36 | 0.53 | 0.54 | 0.56 |  | 0.01 | 30 |
| b4 | rico_suave | 110 | 0.52 | 0.37 | 0.29 | 0.64 | 0.64 | 0.60 | 0.63 | 0.14 | 30 |
| b4 | coco_better (ref) | 720 | 0.54 | 0.48 | 0.47 | 0.61 | 0.62 | 0.56 |  | 0.07 | 30 |

## The typed query's hold (#4736), priced on the same sessions

Find shows the typed query's set until 30 labels; the head's own rows at those votes say what it would have shown. Difference = the session as run minus the session with the head shown from the label quota (positive: the hold helps).

| preset | dataset | runs | hold - no hold (1-150) | hold - no hold (1-50) | runs the hold helps (1-50) |
|---|---|---|---|---|---|
| b025 | rico_suave | 110 | -0.014 ± 0.003 | -0.043 ± 0.009 | 25% |
| b025 | coco_better (ref) | 720 | -0.003 ± 0.001 | -0.010 ± 0.002 | 31% |
| b1 | rico_suave | 110 | -0.026 ± 0.003 | -0.079 ± 0.008 | 12% |
| b1 | coco_better (ref) | 720 | -0.001 ± 0.000 | -0.002 ± 0.001 | 43% |
| b4 | rico_suave | 110 | -0.021 ± 0.004 | -0.064 ± 0.012 | 25% |
| b4 | coco_better (ref) | 720 | +0.001 ± 0.001 | +0.003 ± 0.002 | 52% |

Per class (hold - no hold, votes 1-50, mean over seeds):

| class | b025 | b1 | b4 |
|---|---|---|---|
| browser | +0.044 | +0.008 | +0.022 |
| cart | -0.065 | -0.131 | -0.129 |
| chat | -0.065 | -0.122 | -0.155 |
| gallery | -0.031 | -0.019 | -0.018 |
| login | -0.015 | -0.053 | +0.012 |
| map | -0.010 | -0.093 | -0.043 |
| onboarding | -0.108 | -0.044 | -0.049 |
| player | -0.045 | -0.064 | +0.004 |
| search | -0.074 | -0.094 | -0.063 |
| settings | -0.085 | -0.235 | -0.354 |
| signup | -0.014 | -0.019 | +0.064 |

## Per class (session mean F-beta over votes 1-150, the floor of returning everything, AP at 150 and the full-label ceiling's AP; mean over seeds)

| class | prevalence | withheld positives | b025 mean | b025 everything | b1 mean | b1 everything | b4 mean | b4 everything | AP at 150 | ceiling AP |
|---|---|---|---|---|---|---|---|---|---|---|
| cart | 1.1% | 17 [14-21] | 0.81 | 0.01 | 0.71 | 0.02 | 0.78 | 0.16 | 0.93 | 0.95 |
| map | 1.1% | 16 [9-21] | 0.81 | 0.01 | 0.64 | 0.02 | 0.81 | 0.16 | 0.82 | 0.82 |
| login | 1.0% | 12 [9-17] | 0.57 | 0.01 | 0.31 | 0.02 | 0.44 | 0.15 | 0.47 | 0.48 |
| signup | 1.0% | 14 [10-17] | 0.32 | 0.01 | 0.18 | 0.02 | 0.29 | 0.15 | 0.36 | 0.43 |
| chat | 1.0% | 15 [11-19] | 0.63 | 0.01 | 0.48 | 0.02 | 0.62 | 0.14 | 0.70 | 0.71 |
| search | 1.0% | 14 [11-18] | 0.62 | 0.01 | 0.41 | 0.02 | 0.54 | 0.14 | 0.60 | 0.63 |
| player | 1.0% | 15 [11-22] | 0.78 | 0.01 | 0.56 | 0.02 | 0.66 | 0.14 | 0.70 | 0.71 |
| browser | 0.9% | 14 [1-29] | 0.19 | 0.01 | 0.07 | 0.02 | 0.10 | 0.13 | 0.25 | 0.30 |
| onboarding | 0.9% | 12 [9-14] | 0.31 | 0.01 | 0.16 | 0.02 | 0.25 | 0.13 | 0.29 | 0.32 |
| settings | 0.8% | 8 [3-11] | 0.87 | 0.01 | 0.67 | 0.02 | 0.69 | 0.13 | 0.91 | 0.90 |
| gallery | 0.8% | 13 [6-25] | 0.73 | 0.01 | 0.46 | 0.02 | 0.59 | 0.12 | 0.64 | 0.64 |
