# #4816 literal examples, Binary

The `rate` judge (3 pseudo-votes toward the head's posterior, P from the class model, unvoted media). The first two tables are the runs with the largest gap between the sets at each vote; the last two, the runs where the judge shows the worse set.

== b1
| run | vote | withheld F head/query | today shows | |D| |Q| |DnQ| (unvoted) | votes D only (Good) | votes Q only (Good) | true Goods D only / Q only | rate judge F head/query | exact on collection head/query |
|---|---|---|---|---|---|---|---|---|---|
| tennis racket@small s2 | 20 | 0.19 / 0.93 | query | 458 61 50 | 0 (0) | 1 (0) | 2 / 0 | 0.18 / 0.80 | 0.19 / 0.80 |
| surfboard@large s2 | 20 | 0.20 / 0.83 | query | 10 52 10 | 3 (0) | 0 (0) | 0 / 23 | 0.65 / 0.22 | 0.58 / 0.79 |
| stop sign@large s4 | 20 | 0.89 / 0.28 | query | 26 249 26 | 2 (0) | 1 (1) | 0 / 12 | 0.46 / 0.50 | 0.84 / 0.32 |
| person@large s4 | 40 | 0.79 / 0.00 | head | 16 3 2 | 21 (15) | 0 (0) | 12 / 0 | 0.62 / 0.15 | 0.77 / 0.21 |
| person@large s3 | 40 | 0.82 / 0.06 | head | 14 3 2 | 9 (8) | 1 (1) | 11 / 0 | 0.65 / 0.13 | 0.74 / 0.20 |
| person@large s0 | 40 | 0.89 / 0.15 | head | 37 0 0 | 12 (8) | 3 (1) | 26 / 0 | 0.66 / 0.00 | 0.87 / 0.07 |
| person@large s4 | 100 | 0.95 / 0.00 | head | 1 1 0 | 57 (38) | 0 (0) | 0 / 0 | 0.78 / 0.00 | 0.96 / 0.21 |
| person@large s3 | 100 | 0.88 / 0.06 | head | 5 2 1 | 29 (24) | 1 (1) | 4 / 0 | 0.81 / 0.32 | 1.00 / 0.20 |
| person@large s1 | 100 | 0.95 / 0.15 | head | 7 0 0 | 17 (14) | 4 (2) | 7 / 0 | 0.82 / 0.00 | 0.98 / 0.11 |

== b4
| run | vote | withheld F head/query | today shows | |D| |Q| |DnQ| (unvoted) | votes D only (Good) | votes Q only (Good) | true Goods D only / Q only | rate judge F head/query | exact on collection head/query |
|---|---|---|---|---|---|---|---|---|---|
| keyboard@medium s2 | 20 | 0.02 / 0.61 | query | 36 273 16 | 0 (0) | 5 (1) | 1 / 24 | 0.14 / 0.72 | 0.20 / 0.56 |
| dog@medium s1 | 20 | 0.00 / 0.51 | query | 5 197 1 | 2 (0) | 7 (2) | 0 / 34 | 0.00 / 0.81 | 0.05 / 0.54 |
| person@large s2 | 20 | 0.75 / 0.24 | query | 54 23 7 | 7 (5) | 1 (1) | 23 / 2 | 0.95 / 0.28 | 0.71 / 0.25 |
| person@large s0 | 40 | 0.90 / 0.27 | head | 55 13 3 | 10 (7) | 4 (2) | 25 / 0 | 0.97 / 0.16 | 0.86 / 0.23 |
| traffic light@large s3 | 40 | 0.87 / 0.25 | head | 24 773 24 | 0 (0) | 2 (0) | 0 / 14 | 0.97 / 0.31 | 0.73 / 0.54 |
| traffic light@large s1 | 40 | 0.88 / 0.27 | head | 37 773 37 | 0 (0) | 0 (0) | 0 / 10 | 0.96 / 0.41 | 0.77 / 0.52 |
| person@large s3 | 100 | 0.91 / 0.21 | head | 38 16 3 | 42 (17) | 1 (1) | 13 / 0 | 0.71 / 0.26 | 0.96 / 0.29 |
| person@large s4 | 100 | 0.94 / 0.25 | head | 11 12 2 | 43 (30) | 1 (0) | 4 / 0 | 0.86 / 0.18 | 0.96 / 0.24 |
| person@large s0 | 100 | 0.94 / 0.27 | head | 12 10 0 | 49 (28) | 4 (2) | 8 / 0 | 0.71 / 0.29 | 0.96 / 0.23 |

== b1 wrong
| run | vote | withheld F head/query | today shows | |D| |Q| |DnQ| (unvoted) | votes D only (Good) | votes Q only (Good) | true Goods D only / Q only | rate judge F head/query | exact on collection head/query |
|---|---|---|---|---|---|---|---|---|---|
| surfboard@large s2 | 20 | 0.20 / 0.83 | query | 10 52 10 | 3 (0) | 0 (0) | 0 / 23 | 0.65 / 0.22 | 0.58 / 0.79 |
| stop sign@large s4 | 20 | 0.89 / 0.28 | query | 26 249 26 | 2 (0) | 1 (1) | 0 / 12 | 0.46 / 0.50 | 0.84 / 0.32 |
| kite@large s1 | 20 | 0.21 / 0.71 | query | 6 20 5 | 4 (2) | 0 (0) | 1 / 13 | 0.66 / 0.26 | 0.48 / 0.67 |
| traffic light@large s3 | 40 | 0.87 / 0.23 | head | 14 285 14 | 0 (0) | 7 (4) | 0 / 16 | 0.18 / 0.60 | 0.80 / 0.28 |
| stop sign@large s4 | 40 | 0.91 / 0.28 | head | 24 233 24 | 2 (0) | 6 (4) | 0 / 4 | 0.28 / 0.65 | 0.94 / 0.32 |
| stop sign@large s3 | 40 | 0.85 / 0.28 | head | 12 240 12 | 0 (0) | 2 (2) | 0 / 12 | 0.18 / 0.60 | 0.79 / 0.31 |
| traffic light@large s3 | 100 | 0.90 / 0.23 | head | 1 243 1 | 7 (0) | 9 (5) | 0 / 7 | 0.01 / 0.59 | 0.91 / 0.28 |
| stop sign@large s3 | 100 | 0.91 / 0.28 | head | 1 196 1 | 1 (0) | 15 (4) | 0 / 3 | 0.03 / 0.37 | 0.88 / 0.31 |
| traffic light@large s4 | 100 | 0.88 / 0.25 | head | 9 272 9 | 0 (0) | 29 (5) | 0 / 6 | 0.26 / 0.30 | 0.91 / 0.27 |

== b4 wrong
| run | vote | withheld F head/query | today shows | |D| |Q| |DnQ| (unvoted) | votes D only (Good) | votes Q only (Good) | true Goods D only / Q only | rate judge F head/query | exact on collection head/query |
|---|---|---|---|---|---|---|---|---|---|
| frisbee@small s3 | 20 | 0.21 / 0.71 | query | 13 266 13 | 1 (0) | 0 (0) | 0 / 25 | 0.97 / 0.36 | 0.45 / 0.72 |
| bird@large s1 | 20 | 0.40 / 0.84 | query | 12 54 12 | 0 (0) | 0 (0) | 0 / 17 | 0.97 / 0.77 | 0.53 / 0.87 |
| book@large s2 | 20 | 0.22 / 0.65 | query | 16 130 12 | 0 (0) | 1 (0) | 0 / 22 | 0.94 / 0.51 | 0.40 / 0.73 |
| laptop@medium s4 | 40 | 0.06 / 0.48 | head | 87 283 25 | 2 (0) | 16 (0) | 1 / 28 | 0.32 / 0.17 | 0.20 / 0.59 |
| fork@large s4 | 40 | 0.18 / 0.56 | head | 39 156 19 | 6 (0) | 3 (0) | 1 / 18 | 0.20 / 0.11 | 0.25 / 0.52 |
| parking meter@large s2 | 40 | 0.90 / 0.54 | head | 25 742 25 | 1 (0) | 13 (2) | 0 / 7 | 0.20 / 0.75 | 0.86 / 0.53 |
| person@large s1 | 100 | 0.94 / 0.29 | head | 4 10 0 | 42 (30) | 5 (3) | 4 / 0 | 0.68 / 0.80 | 0.96 / 0.22 |
| parking meter@medium s2 | 100 | 0.62 / 0.18 | head | 2 3933 2 | 0 (0) | 42 (4) | 0 / 22 | 0.00 / 0.62 | 0.57 / 0.18 |
| parking meter@large s4 | 100 | 0.94 / 0.52 | head | 10 715 10 | 0 (0) | 31 (2) | 0 / 5 | 0.17 / 0.56 | 0.91 / 0.55 |
