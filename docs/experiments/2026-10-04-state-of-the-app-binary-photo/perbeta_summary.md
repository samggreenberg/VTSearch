# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 702 | 0.415 | 0.508 | 0.585 | 0.623 | +0.037 ± 0.004 | 0.697 | 0.402 | 35 / 26 | 6% / 1% | 20.826 |
| 1 | 702 | 0.292 | 0.418 | 0.488 | 0.522 | +0.034 ± 0.003 | 0.57 | 0.537 | 49 / 47 | 19% / 6% | 21.51 |
| 4 | 702 | 0.352 | 0.486 | 0.563 | 0.599 | +0.036 ± 0.003 | 0.417 | 0.677 | 75 / 80 | 31% / 26% | 30.556 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the full-label model's threshold. `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | SigLIP binary | app line | 0.25 | text | 4507.602 | 0.011 | 0.913 | 0.011 | 0.556 | 0.077 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 25 | 971.272 | 0.546 | 0.332 | 0.362 | 0.526 | 0.514 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 50 | 754.232 | 0.543 | 0.46 | 0.466 | 0.582 | 0.616 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | app line | 0.25 | final | 185.024 | 0.618 | 0.449 | 0.571 | 0.668 | 0.724 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | app line | 0.25 | ceiling | 841.765 | 0.204 | 0.931 | 0.209 | 0.713 | 0.246 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | text | 32.0 | 0.487 | 0.313 | 0.471 | 0.556 | 0.696 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 25 | 32.0 | 0.43 | 0.277 | 0.416 | 0.526 | 0.616 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 50 | 32.0 | 0.506 | 0.326 | 0.489 | 0.582 | 0.682 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | final | 32.0 | 0.593 | 0.382 | 0.574 | 0.668 | 0.754 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | ceiling | 32.0 | 0.635 | 0.409 | 0.615 | 0.713 | 0.81 | 0.0 | 720 |
| beta 1 | SigLIP binary | app line | 1.0 | text | 4507.602 | 0.011 | 0.913 | 0.021 | 0.45 | 0.128 | 0.0 | 720 |
| beta 1 | SigLIP binary | app line | 1.0 | 25 | 1426.617 | 0.486 | 0.444 | 0.256 | 0.386 | 0.473 | 0.0 | 720 |
| beta 1 | SigLIP binary | app line | 1.0 | 50 | 1198.9 | 0.46 | 0.566 | 0.384 | 0.457 | 0.632 | 0.0 | 720 |
| beta 1 | SigLIP binary | app line | 1.0 | final | 374.806 | 0.51 | 0.576 | 0.476 | 0.535 | 0.747 | 0.0 | 720 |
| beta 1 | SigLIP binary | app line | 1.0 | ceiling | 841.765 | 0.204 | 0.931 | 0.278 | 0.578 | 0.401 | 0.0 | 720 |
| beta 1 | SigLIP binary | top-K | 1.0 | text | 32.0 | 0.487 | 0.313 | 0.38 | 0.45 | 0.714 | 0.0 | 720 |
| beta 1 | SigLIP binary | top-K | 1.0 | 25 | 32.0 | 0.43 | 0.276 | 0.336 | 0.386 | 0.705 | 0.0 | 720 |
| beta 1 | SigLIP binary | top-K | 1.0 | 50 | 32.0 | 0.504 | 0.325 | 0.394 | 0.457 | 0.724 | 0.0 | 720 |
| beta 1 | SigLIP binary | top-K | 1.0 | final | 32.0 | 0.593 | 0.382 | 0.463 | 0.535 | 0.794 | 0.0 | 720 |
| beta 1 | SigLIP binary | top-K | 1.0 | ceiling | 32.0 | 0.635 | 0.409 | 0.496 | 0.578 | 0.838 | 0.0 | 720 |
| beta 4 | SigLIP binary | app line | 4.0 | text | 4507.602 | 0.011 | 0.913 | 0.15 | 0.566 | 0.362 | 0.0 | 720 |
| beta 4 | SigLIP binary | app line | 4.0 | 25 | 1788.8 | 0.364 | 0.54 | 0.322 | 0.479 | 0.547 | 0.0 | 720 |
| beta 4 | SigLIP binary | app line | 4.0 | 50 | 1768.949 | 0.35 | 0.669 | 0.455 | 0.545 | 0.706 | 0.0 | 720 |
| beta 4 | SigLIP binary | app line | 4.0 | final | 825.131 | 0.395 | 0.697 | 0.551 | 0.625 | 0.796 | 0.0 | 720 |
| beta 4 | SigLIP binary | app line | 4.0 | ceiling | 841.765 | 0.204 | 0.931 | 0.594 | 0.692 | 0.829 | 0.0 | 720 |
| beta 4 | SigLIP binary | top-K | 4.0 | text | 128.0 | 0.207 | 0.531 | 0.486 | 0.566 | 0.739 | 0.0 | 720 |
| beta 4 | SigLIP binary | top-K | 4.0 | 25 | 128.0 | 0.172 | 0.442 | 0.404 | 0.479 | 0.7 | 0.0 | 720 |
| beta 4 | SigLIP binary | top-K | 4.0 | 50 | 128.0 | 0.202 | 0.517 | 0.473 | 0.545 | 0.74 | 0.0 | 720 |
| beta 4 | SigLIP binary | top-K | 4.0 | final | 128.0 | 0.239 | 0.613 | 0.561 | 0.625 | 0.815 | 0.0 | 720 |
| beta 4 | SigLIP binary | top-K | 4.0 | ceiling | 128.0 | 0.266 | 0.681 | 0.623 | 0.692 | 0.864 | 0.0 | 720 |

## The early dip

The detector's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's blind GMM cut against the detector's labels line; under `top-K` the old cap on both. F-beta, what each returned, and the click by which the detector's F-beta reaches the text sort's; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | detector: F at 5 / 10 / 25 | detector: returned at 5 / 10 / 25 | detector F >= text sort's by click | lowest share | at click | share >= text sort's by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.011 | 4515.496 | 0.077 | 0.19 / 0.23 / 0.42 | 1 / 1 / 12 | 2 | 0.191 | 2 | 2 |
| 0.25 | top-K | 0.471 | 32.0 | 0.696 | 0.45 / 0.49 / 0.48 | 32 / 32 / 32 | 6 | 0.311 | 3 | 40 |
| 1 | app line | 0.021 | 4515.496 | 0.128 | 0.03 / 0.08 / 0.29 | 1 / 1 / 27 | 4 | 0.088 | 7 | 10 |
| 1 | top-K | 0.38 | 32.0 | 0.714 | 0.36 / 0.40 / 0.38 | 32 / 32 / 32 | 6 | 0.434 | 3 | 4 |
| 4 | app line | 0.15 | 4515.496 | 0.362 | 0.03 / 0.11 / 0.35 | 1 / 1 / 56 | 15 | 0.073 | 7 | 15 |
| 4 | top-K | 0.486 | 128.0 | 0.739 | 0.45 / 0.48 / 0.46 | 128 / 128 / 128 | 40 | 0.418 | 2 | 4 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 702 | 20.826 | 0.175 | 0.822 | 0.647 | 0.453 | 1.0 |
| 1 | 702 | 21.51 | 0.117 | 0.806 | 0.688 | 0.38 | 1.0 |
| 4 | 702 | 30.556 | 0.075 | 0.787 | 0.711 | 0.29 | 1.0 |
