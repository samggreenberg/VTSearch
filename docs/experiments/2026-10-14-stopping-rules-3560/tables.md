## Where the app said stop

| arm | runs | fired | stop click (KM) | stop click (median of fired) | fbeta at stop | fbeta at budget | Δfbeta (paired) | clicks after stop | short of run's best | clicks past best |
|---|---|---|---|---|---|---|---|---|---|---|
| β 1/4 | 720 | 640 (89%) | 71 | 67 [57–83] | 0.72 | 0.85 | 0.03 | 83 | 0.10 | -21 |
| β 1 | 720 | 640 (89%) | 75 | 71 [59–86] | 0.47 | 0.64 | 0.05 | 79 | 0.09 | -39 |
| β 4 | 720 | 629 (87%) | 87 | 82 [67–100] | 0.64 | 0.72 | 0.04 | 68 | 0.08 | -28 |

| arm | band | runs | fired | stop click (KM) | stop click (median of fired) | fbeta at stop | fbeta at budget | Δfbeta (paired) | clicks after stop | short of run's best | clicks past best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| β 1/4 | large | 245 | 245 (100%) | 67 | 67 [57–79] | 0.93 | 0.95 | 0.02 | 83 | 0.04 | -33 |
| β 1/4 | medium | 245 | 238 (97%) | 63 | 62 [54–78] | 0.57 | 0.77 | 0.07 | 88 | 0.15 | -29 |
| β 1/4 | small | 230 | 157 (68%) | 111 | 85 [63–112] | 0.37 | 0.43 | 0.02 | 65 | 0.12 | -7 |
| β 1 | large | 245 | 245 (100%) | 68 | 68 [59–82] | 0.81 | 0.87 | 0.05 | 82 | 0.08 | -40 |
| β 1 | medium | 245 | 238 (97%) | 66 | 66 [55–82] | 0.36 | 0.54 | 0.07 | 84 | 0.12 | -48 |
| β 1 | small | 230 | 157 (68%) | 112 | 87 [68–117] | 0.23 | 0.26 | 0.03 | 63 | 0.07 | -22 |
| β 4 | large | 245 | 245 (100%) | 78 | 78 [65–98] | 0.84 | 0.89 | 0.03 | 72 | 0.06 | -27 |
| β 4 | medium | 245 | 235 (96%) | 77 | 76 [65–98] | 0.52 | 0.62 | 0.06 | 74 | 0.10 | -42 |
| β 4 | small | 230 | 149 (65%) | 130 | 100 [79–127] | 0.37 | 0.41 | 0.02 | 50 | 0.07 | -9 |

## Where the clicks before the stop went

| arm | fired | stop click (mean of fired) | opening | Hard | New | mid-session check | check before Done | fbeta at Hard's end | fbeta at stop | Δ (paired) |
|---|---|---|---|---|---|---|---|---|---|---|
| β 1/4 | 640 (89%) | 74 | 14 | 13 | 19 | 18 | 10 | 0.66 | 0.72 | 0.01 |
| β 1 | 640 (89%) | 76 | 14 | 14 | 19 | 19 | 10 | 0.44 | 0.47 | 0.01 |
| β 4 | 629 (87%) | 87 | 13 | 15 | 18 | 27 | 13 | 0.59 | 0.64 | 0.02 |

| arm | band | fired | stop click (mean of fired) | opening | Hard | New | mid-session check | check before Done | fbeta at Hard's end | fbeta at stop | Δ (paired) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| β 1/4 | large | 245 (100%) | 68 | 7 | 14 | 20 | 15 | 13 | 0.89 | 0.93 | 0.01 |
| β 1/4 | medium | 238 (97%) | 69 | 12 | 11 | 19 | 18 | 8 | 0.54 | 0.57 | 0.01 |
| β 1/4 | small | 157 (68%) | 89 | 27 | 16 | 19 | 21 | 7 | 0.31 | 0.37 | 0.04 |
| β 1 | large | 245 (100%) | 70 | 7 | 14 | 19 | 17 | 13 | 0.76 | 0.81 | 0.01 |
| β 1 | medium | 238 (97%) | 71 | 12 | 12 | 19 | 19 | 9 | 0.36 | 0.36 | 0.01 |
| β 1 | small | 157 (68%) | 92 | 27 | 17 | 18 | 22 | 7 | 0.18 | 0.23 | 0.02 |
| β 4 | large | 245 (100%) | 81 | 7 | 15 | 19 | 23 | 17 | 0.80 | 0.84 | 0.02 |
| β 4 | medium | 235 (96%) | 83 | 12 | 13 | 18 | 29 | 11 | 0.49 | 0.52 | 0.01 |
| β 4 | small | 149 (65%) | 102 | 26 | 18 | 18 | 29 | 11 | 0.32 | 0.37 | 0.01 |

β 1/4: **smart** is the binding rule in Hard: Smart was not green on 99% of Hard's clicks and Stable on 65% (both on 65%). Hard last ended waiting on Smart in 63% of the sessions that stopped, on Stable in 0%, on both in 36%. New waits on Span alone and is the atlas walk that turns it green, so Span is not a gate that fails: it is 19 clicks of walk, and the check before Done 10 more. Before the stop a light went from green to not green 0.9 times a session for Smart, 0.0 for Stable and 0.0 for Span; after it, `done` fell back 0.7 / 0.0 / 0.0 times a session on Smart / Stable / Span.

β 1: **smart** is the binding rule in Hard: Smart was not green on 94% of Hard's clicks and Stable on 69% (both on 64%). Hard last ended waiting on Smart in 59% of the sessions that stopped, on Stable in 5%, on both in 35%. New waits on Span alone and is the atlas walk that turns it green, so Span is not a gate that fails: it is 19 clicks of walk, and the check before Done 10 more. Before the stop a light went from green to not green 0.9 times a session for Smart, 0.1 for Stable and 0.0 for Span; after it, `done` fell back 0.6 / 0.0 / 0.0 times a session on Smart / Stable / Span.

β 4: **smart** is the binding rule in Hard: Smart was not green on 85% of Hard's clicks and Stable on 76% (both on 61%). Hard last ended waiting on Smart in 53% of the sessions that stopped, on Stable in 21%, on both in 25%. New waits on Span alone and is the atlas walk that turns it green, so Span is not a gate that fails: it is 18 clicks of walk, and the check before Done 13 more. Before the stop a light went from green to not green 1.0 times a session for Smart, 0.3 for Stable and 0.0 for Span; after it, `done` fell back 0.7 / 0.1 / 0.0 times a session on Smart / Stable / Span.

## Paired gains, mean ± SE over class means

| preset | after the stop | clicks after the stop | Hard's end to the stop | clicks |
|---|---|---|---|---|
| β 1/4 | +0.077 ± 0.009 | 83 | +0.036 ± 0.008 | 31 |
| β 1 | +0.077 ± 0.007 | 79 | +0.031 ± 0.005 | 30 |
| β 4 | +0.059 ± 0.005 | 68 | +0.032 ± 0.005 | 34 |

## Smart's lapses after the stop

| preset | fired | Done broken by Smart | gain after stop, broken | kept | within class, broken − kept |
|---|---|---|---|---|---|
| β 1/4 | 640 | 30% | +0.072 | +0.079 | +0.019 ± 0.014 (41 classes) |
| β 1 | 640 | 28% | +0.076 | +0.077 | +0.020 ± 0.013 (41 classes) |
| β 4 | 629 | 27% | +0.057 | +0.060 | +0.009 ± 0.012 (39 classes) |

## How close each gate came

| arm | runs | Hard steps | New steps | Smart slope | Smart t | Stable avg | Stable max | Span nodes | Stable blocked by |
|---|---|---|---|---|---|---|---|---|---|
| β 1/4 | 720 | 12 | 19 | -0.046 [1%] | -1.942 [1%] | 0.005 [100%] | 0.009 [99%] | -17 [0%] | warm-up 99%, avg 0%, max 1%, falling 0% |
| β 1 | 720 | 12 | 19 | -0.041 [13%] | -1.760 [14%] | 0.005 [97%] | 0.009 [80%] | -17 [0%] | warm-up 90%, avg 1%, max 9%, falling 0% |
| β 4 | 720 | 13 | 18 | -0.022 [30%] | -1.155 [32%] | 0.004 [84%] | 0.003 [55%] | -17 [0%] | warm-up 75%, avg 9%, max 16%, falling 0% |

| arm | band | runs | Hard steps | New steps | Smart slope | Smart t | Stable avg | Stable max | Span nodes | Stable blocked by |
|---|---|---|---|---|---|---|---|---|---|---|
| β 1/4 | large | 245 | 12 | 20 | -0.054 [0%] | -1.970 [0%] | 0.005 [100%] | 0.009 [100%] | -17 [0%] | warm-up 100%, avg 0%, max 0%, falling 0% |
| β 1/4 | medium | 245 | 9 | 19 | -0.042 [3%] | -1.930 [3%] | 0.005 [100%] | 0.009 [96%] | -17 [0%] | warm-up 98%, avg 0%, max 2%, falling 0% |
| β 1/4 | small | 230 | 12 | 16 | -0.032 [0%] | -1.740 [0%] | 0.005 [100%] | 0.009 [100%] | -18 [0%] | warm-up 100%, avg 0%, max 0%, falling 0% |
| β 1 | large | 245 | 12 | 20 | -0.055 [9%] | -1.950 [10%] | 0.005 [99%] | 0.009 [86%] | -17 [0%] | warm-up 78%, avg 2%, max 20%, falling 0% |
| β 1 | medium | 245 | 10 | 19 | -0.031 [17%] | -1.738 [18%] | 0.005 [96%] | 0.008 [76%] | -17 [0%] | warm-up 86%, avg 2%, max 12%, falling 0% |
| β 1 | small | 230 | 13 | 16 | -0.021 [18%] | -1.362 [19%] | 0.004 [96%] | 0.008 [71%] | -18 [0%] | warm-up 96%, avg 1%, max 4%, falling 0% |
| β 4 | large | 245 | 13 | 19 | -0.046 [18%] | -1.540 [21%] | 0.004 [88%] | 0.007 [69%] | -16 [0%] | warm-up 59%, avg 15%, max 26%, falling 1% |
| β 4 | medium | 245 | 11 | 18 | -0.013 [37%] | -0.985 [39%] | 0.003 [79%] | -0.000 [46%] | -17 [0%] | warm-up 67%, avg 12%, max 20%, falling 1% |
| β 4 | small | 230 | 16 | 15 | -0.002 [46%] | -0.320 [47%] | 0.003 [79%] | -0.002 [36%] | -18 [0%] | warm-up 86%, avg 4%, max 9%, falling 0% |
