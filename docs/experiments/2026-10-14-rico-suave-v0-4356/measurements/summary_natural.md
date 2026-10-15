# Rico Suave v0 (#4356)

F-beta at the session's preset of what Find returns on the withheld half, every run at every vote (split by app). Runs per preset: {'b025': 110, 'b1': 110, 'b4': 110}. Empty cells: {'b025': 0, 'b1': 0, 'b4': 0}.

## Per preset, every class

| preset | dataset | runs | session mean F-beta (1-150) | short (1-50) | at vote 1 | at 150 | after the check | AP at 150 | ceiling AP | return-everything F-beta | first detector (median vote) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b025 | rico_suave | 110 | 0.67 | 0.44 | 0.30 | 0.85 | 0.85 | 0.84 | 0.86 | 0.10 | 30 |
| b025 | coco_better (ref) | 720 | 0.56 | 0.49 | 0.46 | 0.64 | 0.65 | 0.56 |  | 0.00 | 30 |
| b1 | rico_suave | 110 | 0.54 | 0.29 | 0.14 | 0.73 | 0.75 | 0.84 | 0.86 | 0.15 | 30 |
| b1 | coco_better (ref) | 720 | 0.46 | 0.39 | 0.36 | 0.53 | 0.54 | 0.56 |  | 0.01 | 30 |
| b4 | rico_suave | 110 | 0.67 | 0.49 | 0.39 | 0.82 | 0.83 | 0.83 | 0.86 | 0.49 | 30 |
| b4 | coco_better (ref) | 720 | 0.54 | 0.48 | 0.47 | 0.61 | 0.62 | 0.56 |  | 0.07 | 30 |

## The typed query's hold (#4736), priced on the same sessions

Find shows the typed query's set until 30 labels; the head's own rows at those votes say what it would have shown. Difference = the session as run minus the session with the head shown from the label quota (positive: the hold helps).

| preset | dataset | runs | hold - no hold (1-150) | hold - no hold (1-50) | runs the hold helps (1-50) |
|---|---|---|---|---|---|
| b025 | rico_suave | 110 | -0.031 ± 0.003 | -0.094 ± 0.008 | 7% |
| b025 | coco_better (ref) | 720 | -0.003 ± 0.001 | -0.010 ± 0.002 | 31% |
| b1 | rico_suave | 110 | -0.032 ± 0.003 | -0.097 ± 0.008 | 6% |
| b1 | coco_better (ref) | 720 | -0.001 ± 0.000 | -0.002 ± 0.001 | 43% |
| b4 | rico_suave | 110 | -0.005 ± 0.006 | -0.014 ± 0.018 | 33% |
| b4 | coco_better (ref) | 720 | +0.001 ± 0.001 | +0.003 ± 0.002 | 52% |

Per class (hold - no hold, votes 1-50, mean over seeds):

| class | b025 | b1 | b4 |
|---|---|---|---|
| browser | +0.019 | -0.009 | +0.030 |
| cart | -0.087 | -0.145 | -0.141 |
| chat | -0.159 | -0.166 | -0.186 |
| gallery | -0.025 | -0.024 | -0.024 |
| login | -0.039 | -0.033 | +0.365 |
| map | -0.151 | -0.203 | -0.068 |
| onboarding | -0.095 | -0.066 | -0.109 |
| player | -0.058 | -0.068 | -0.015 |
| search | -0.174 | -0.128 | -0.084 |
| settings | -0.210 | -0.180 | -0.222 |
| signup | -0.050 | -0.040 | +0.305 |

## Per class (session mean F-beta over votes 1-150, the floor of returning everything, AP at 150 and the full-label ceiling's AP; mean over seeds)

| class | prevalence | withheld positives | b025 mean | b025 everything | b1 mean | b1 everything | b4 mean | b4 everything | AP at 150 | ceiling AP |
|---|---|---|---|---|---|---|---|---|---|---|
| settings | 36.4% | 564 [532-611] | 0.75 | 0.38 | 0.70 | 0.53 | 0.74 | 0.91 | 1.00 | 1.00 |
| login | 20.7% | 314 [290-352] | 0.58 | 0.22 | 0.50 | 0.34 | 0.79 | 0.82 | 0.92 | 0.95 |
| onboarding | 11.7% | 178 [164-195] | 0.51 | 0.12 | 0.36 | 0.21 | 0.51 | 0.69 | 0.78 | 0.85 |
| signup | 7.9% | 120 [102-143] | 0.50 | 0.08 | 0.42 | 0.15 | 0.73 | 0.59 | 0.84 | 0.87 |
| map | 6.1% | 93 [74-113] | 0.86 | 0.06 | 0.74 | 0.12 | 0.88 | 0.52 | 0.96 | 0.96 |
| search | 5.7% | 87 [71-99] | 0.72 | 0.06 | 0.54 | 0.11 | 0.62 | 0.51 | 0.79 | 0.81 |
| chat | 4.2% | 65 [46-81] | 0.72 | 0.04 | 0.62 | 0.08 | 0.73 | 0.42 | 0.91 | 0.92 |
| gallery | 2.6% | 40 [23-73] | 0.81 | 0.03 | 0.60 | 0.05 | 0.70 | 0.30 | 0.83 | 0.81 |
| player | 2.5% | 39 [25-46] | 0.81 | 0.03 | 0.65 | 0.05 | 0.77 | 0.30 | 0.88 | 0.89 |
| browser | 1.5% | 24 [3-43] | 0.24 | 0.02 | 0.09 | 0.03 | 0.14 | 0.19 | 0.37 | 0.43 |
| cart | 1.1% | 17 [13-22] | 0.84 | 0.01 | 0.73 | 0.02 | 0.80 | 0.16 | 0.94 | 0.96 |
