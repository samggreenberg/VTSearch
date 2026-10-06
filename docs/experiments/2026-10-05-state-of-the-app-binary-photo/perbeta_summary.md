# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1403 | 0.431 | 0.509 | 0.612 | 0.636 | +0.024 ± 0.003 | 0.731 | 0.383 | 27 / 21 | 5% / 0% | 20.741 |
| 1 | 1403 | 0.335 | 0.418 | 0.504 | 0.526 | +0.022 ± 0.002 | 0.597 | 0.523 | 44 / 44 | 11% / 5% | 21.219 |
| 4 | 1403 | 0.42 | 0.477 | 0.569 | 0.602 | +0.033 ± 0.002 | 0.428 | 0.678 | 72 / 80 | 29% / 28% | 30.445 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set at 25, 50, 100 and 150 clicks and after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over the trained runs with a line by that click; `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | 25 | 0.522 | 0.385 | 0.431 | 30.999 | 1256 |
| 0.25 | 50 | 0.586 | 0.424 | 0.509 | 33.0 | 1327 |
| 0.25 | 100 | 0.688 | 0.39 | 0.594 | 25.999 | 1356 |
| 0.25 | 150 | 0.688 | 0.408 | 0.612 | 26.997 | 1403 |
| 0.25 | after the check | 0.731 | 0.383 | 0.636 | 21.005 | 1403 |
| 1 | 25 | 0.406 | 0.51 | 0.335 | 53.5 | 1256 |
| 1 | 50 | 0.496 | 0.554 | 0.418 | 45.999 | 1327 |
| 1 | 100 | 0.582 | 0.527 | 0.486 | 41.998 | 1356 |
| 1 | 150 | 0.565 | 0.536 | 0.504 | 44.004 | 1403 |
| 1 | after the check | 0.597 | 0.523 | 0.526 | 43.995 | 1403 |
| 4 | 25 | 0.266 | 0.634 | 0.42 | 129.499 | 1256 |
| 4 | 50 | 0.372 | 0.677 | 0.477 | 76.999 | 1327 |
| 4 | 100 | 0.436 | 0.668 | 0.55 | 65.995 | 1356 |
| 4 | 150 | 0.424 | 0.677 | 0.569 | 72.001 | 1403 |
| 4 | after the check | 0.428 | 0.678 | 0.602 | 80.005 | 1403 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | SigLIP binary | app line | 0.25 | text | 4508.876 | 0.011 | 0.913 | 0.011 | 0.559 | 0.079 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 25 | 1095.127 | 0.458 | 0.432 | 0.379 | 0.524 | 0.539 | 0.0 | 1434 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 50 | 854.538 | 0.52 | 0.474 | 0.461 | 0.571 | 0.603 | 0.0 | 1274 |
| beta 0.25 | SigLIP binary | app line | 0.25 | final | 220.917 | 0.671 | 0.412 | 0.597 | 0.679 | 0.753 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | app line | 0.25 | ceiling | 117.692 | 0.466 | 0.66 | 0.465 | 0.711 | 0.585 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | text | 32.0 | 0.487 | 0.314 | 0.472 | 0.559 | 0.694 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 25 | 32.0 | 0.427 | 0.275 | 0.414 | 0.524 | 0.611 | 0.0 | 1434 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 50 | 32.0 | 0.501 | 0.323 | 0.485 | 0.571 | 0.664 | 0.0 | 1274 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | final | 32.0 | 0.603 | 0.389 | 0.584 | 0.679 | 0.756 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | ceiling | 32.0 | 0.631 | 0.407 | 0.611 | 0.711 | 0.806 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | text | 4508.876 | 0.011 | 0.913 | 0.021 | 0.45 | 0.129 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | 25 | 1460.769 | 0.357 | 0.541 | 0.295 | 0.384 | 0.57 | 0.0 | 1435 |
| beta 1 | SigLIP binary | app line | 1.0 | 50 | 1332.273 | 0.449 | 0.58 | 0.385 | 0.455 | 0.623 | 0.0 | 1283 |
| beta 1 | SigLIP binary | app line | 1.0 | final | 357.1 | 0.55 | 0.537 | 0.491 | 0.543 | 0.78 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | ceiling | 296.793 | 0.369 | 0.779 | 0.438 | 0.576 | 0.671 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | text | 32.0 | 0.487 | 0.314 | 0.381 | 0.45 | 0.713 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | 25 | 32.0 | 0.428 | 0.276 | 0.334 | 0.384 | 0.703 | 0.0 | 1435 |
| beta 1 | SigLIP binary | top-K | 1.0 | 50 | 32.0 | 0.5 | 0.322 | 0.391 | 0.455 | 0.71 | 0.0 | 1283 |
| beta 1 | SigLIP binary | top-K | 1.0 | final | 32.0 | 0.602 | 0.389 | 0.471 | 0.543 | 0.798 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | ceiling | 32.0 | 0.631 | 0.407 | 0.494 | 0.576 | 0.834 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | text | 4508.876 | 0.011 | 0.913 | 0.15 | 0.565 | 0.364 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | 25 | 1978.06 | 0.234 | 0.649 | 0.383 | 0.477 | 0.679 | 0.0 | 1434 |
| beta 4 | SigLIP binary | app line | 4.0 | 50 | 2066.864 | 0.352 | 0.686 | 0.442 | 0.538 | 0.69 | 0.0 | 1251 |
| beta 4 | SigLIP binary | app line | 4.0 | final | 688.356 | 0.413 | 0.674 | 0.557 | 0.629 | 0.808 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | ceiling | 854.619 | 0.278 | 0.885 | 0.609 | 0.691 | 0.834 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | text | 128.0 | 0.207 | 0.53 | 0.485 | 0.565 | 0.739 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | 25 | 128.0 | 0.171 | 0.44 | 0.402 | 0.477 | 0.7 | 0.0 | 1434 |
| beta 4 | SigLIP binary | top-K | 4.0 | 50 | 128.0 | 0.199 | 0.51 | 0.467 | 0.538 | 0.727 | 0.0 | 1251 |
| beta 4 | SigLIP binary | top-K | 4.0 | final | 128.0 | 0.242 | 0.621 | 0.568 | 0.629 | 0.823 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | ceiling | 128.0 | 0.265 | 0.68 | 0.622 | 0.691 | 0.866 | 0.0 | 1440 |

## The early dip

The detector's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's blind GMM cut against the detector's labels line; under `top-K` the old cap on both. F-beta, what each returned, and the click by which the detector's F-beta reaches the text sort's; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | detector: F at 5 / 10 / 25 | detector: returned at 5 / 10 / 25 | detector F >= text sort's by click | lowest share | at click | share >= text sort's by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.011 | 4516.999 | 0.079 | 0.40 / 0.45 / 0.43 | 16 / 24 / 31 | 2 | 0.141 | 2 | 2 |
| 0.25 | top-K | 0.472 | 32.0 | 0.694 | 0.45 / 0.49 / 0.47 | 32 / 32 / 32 | 7 | 0.324 | 3 | 31 |
| 1 | app line | 0.021 | 4516.999 | 0.129 | 0.28 / 0.34 / 0.34 | 29 / 39 / 53 | 4 | 0.076 | 2 | 4 |
| 1 | top-K | 0.381 | 32.0 | 0.713 | 0.36 / 0.39 / 0.38 | 32 / 32 / 32 | 7 | 0.469 | 3 | 4 |
| 4 | app line | 0.15 | 4516.999 | 0.364 | 0.34 / 0.41 / 0.42 | 48 / 75 / 129 | 5 | 0.029 | 2 | 5 |
| 4 | top-K | 0.485 | 128.0 | 0.739 | 0.44 / 0.47 / 0.45 | 128 / 128 / 128 | 34 | 0.429 | 2 | 4 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 1403 | 20.741 | 0.179 | 0.82 | 0.641 | 0.453 | 1.0 |
| 1 | 1403 | 21.219 | 0.119 | 0.803 | 0.683 | 0.378 | 1.0 |
| 4 | 1403 | 30.445 | 0.079 | 0.782 | 0.703 | 0.293 | 1.0 |
