# #4816 literal examples, Region

## Region, beta 1: the judge keeps the query past the quota (calib, 1 pseudo-vote, unvoted media)

| run | vote | withheld F head/query | today shows | |D| |Q| |DnQ| (unvoted) | votes D only (Good) | votes Q only (Good) | true Goods D only / Q only | judge's F head/query | exact on collection head/query |
|---|---|---|---|---|---|---|---|---|---|
| skateboard@small s1 | 10 | 0.09 / 0.69 | head | 520 55 37 | 0 (0) | 0 (0) | 4 / 6 | 0.38 / 0.42 | 0.13 / 0.69 |
| skateboard@large s1 | 10 | 0.23 / 0.82 | head | 320 69 57 | 2 (0) | 0 (0) | 2 / 3 | 0.45 / 0.70 | 0.27 / 0.82 |
| surfboard@medium s1 | 10 | 0.22 / 0.81 | head | 393 71 63 | 2 (0) | 0 (0) | 4 / 0 | 0.22 / 0.77 | 0.25 / 0.80 |
| baseball bat@medium s0 | 10 | 0.28 / 0.85 | head | 271 57 50 | 1 (0) | 0 (0) | 2 / 1 | 0.35 / 0.63 | 0.29 / 0.84 |
| skateboard@large s0 | 20 | 0.32 / 0.82 | head | 247 73 59 | 4 (0) | 0 (0) | 0 / 3 | 0.43 / 0.82 | 0.32 / 0.81 |
| skis@large s0 | 20 | 0.29 / 0.74 | head | 319 98 90 | 12 (0) | 0 (0) | 2 / 0 | 0.47 / 0.87 | 0.25 / 0.60 |
| baseball bat@small s0 | 20 | 0.40 / 0.82 | head | 195 44 41 | 1 (0) | 0 (0) | 2 / 0 | 0.39 / 0.88 | 0.32 / 0.82 |
| skis@medium s1 | 20 | 0.18 / 0.54 | head | 480 89 82 | 6 (0) | 1 (0) | 6 / 0 | 0.20 / 0.70 | 0.19 / 0.62 |
| skateboard@small s0 | 40 | 0.14 / 0.64 | head | 501 62 46 | 5 (0) | 1 (1) | 4 / 2 | 0.34 / 0.59 | 0.17 / 0.73 |
| kite@small s1 | 40 | 0.80 / 0.54 | head | 54 22 22 | 11 (2) | 0 (0) | 14 / 0 | 0.67 / 0.71 | 0.78 / 0.72 |
| snowboard@medium s0 | 40 | 0.24 / 0.47 | head | 287 120 102 | 9 (1) | 0 (0) | 3 / 1 | 0.47 / 0.52 | 0.30 / 0.54 |
| toothbrush@large s0 | 40 | 0.70 / 0.87 | head | 61 44 33 | 4 (0) | 0 (0) | 2 / 1 | 0.66 / 0.73 | 0.73 / 0.85 |


## Region, beta 1: and where that is wrong

| run | vote | withheld F head/query | today shows | |D| |Q| |DnQ| (unvoted) | votes D only (Good) | votes Q only (Good) | true Goods D only / Q only | judge's F head/query | exact on collection head/query |
|---|---|---|---|---|---|---|---|---|---|
| skateboard@medium s0 | 10 | 0.24 / 0.87 | head | 292 78 54 | 0 (0) | 0 (0) | 1 / 5 | 0.87 / 0.35 | 0.27 / 0.76 |
| cell phone@large s1 | 10 | 0.18 / 0.77 | head | 130 71 24 | 1 (0) | 0 (0) | 0 / 18 | 0.51 / 0.41 | 0.28 / 0.70 |
| skateboard@small s0 | 10 | 0.11 / 0.64 | head | 576 63 43 | 0 (0) | 1 (1) | 4 / 7 | 0.87 / 0.16 | 0.13 / 0.73 |
| skateboard@medium s0 | 20 | 0.22 / 0.87 | head | 5 73 5 | 3 (0) | 0 (0) | 0 / 37 | 0.86 / 0.17 | 0.40 / 0.76 |
| skateboard@small s0 | 20 | 0.10 / 0.64 | head | 589 63 48 | 1 (0) | 1 (1) | 4 / 4 | 0.65 / 0.28 | 0.14 / 0.73 |
| skateboard@small s1 | 20 | 0.18 / 0.69 | head | 3 48 3 | 3 (0) | 0 (0) | 0 / 25 | 0.05 / 0.00 | 0.41 / 0.69 |
| skateboard@small s1 | 40 | 0.18 / 0.69 | head | 9 42 2 | 4 (0) | 1 (0) | 1 / 22 | 0.00 / 0.00 | 0.45 / 0.69 |
| cell phone@large s1 | 40 | 0.47 / 0.77 | head | 7 59 7 | 7 (0) | 0 (0) | 0 / 21 | 0.01 / 0.00 | 0.63 / 0.70 |
| kite@small s1 | 40 | 0.80 / 0.54 | head | 54 22 22 | 11 (2) | 0 (0) | 14 / 0 | 0.67 / 0.71 | 0.78 / 0.72 |
