# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1403 | 0.459 | 0.509 | 0.612 | 0.636 | +0.025 ± 0.003 | 0.731 | 0.383 | 25 / 21 | 1% / 0% | 20.741 |
| 1 | 1403 | 0.366 | 0.42 | 0.504 | 0.526 | +0.022 ± 0.002 | 0.597 | 0.523 | 43 / 44 | 7% / 5% | 21.219 |
| 4 | 1403 | 0.473 | 0.496 | 0.57 | 0.602 | +0.032 ± 0.002 | 0.428 | 0.678 | 83 / 80 | 29% / 28% | 30.445 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set from the typed query (the text sort at its own blind GMM cut, before any vote), through 25, 50, 100 and 150 clicks, to after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over the trained runs with a line by that click; `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | typed query | 0.538 | 0.263 | 0.478 | 18.999 | 1403 |
| 0.25 | 25 | 0.517 | 0.292 | 0.459 | 19.003 | 1403 |
| 0.25 | 50 | 0.587 | 0.321 | 0.509 | 20.0 | 1403 |
| 0.25 | 100 | 0.666 | 0.349 | 0.575 | 21.004 | 1403 |
| 0.25 | 150 | 0.69 | 0.391 | 0.612 | 24.998 | 1403 |
| 0.25 | after the check | 0.731 | 0.383 | 0.636 | 21.005 | 1403 |
| 1 | typed query | 0.374 | 0.447 | 0.373 | 50.995 | 1403 |
| 1 | 25 | 0.398 | 0.409 | 0.366 | 44.003 | 1403 |
| 1 | 50 | 0.499 | 0.426 | 0.42 | 40.001 | 1403 |
| 1 | 100 | 0.563 | 0.474 | 0.473 | 40.005 | 1403 |
| 1 | 150 | 0.563 | 0.513 | 0.504 | 43.005 | 1403 |
| 1 | after the check | 0.597 | 0.523 | 0.526 | 43.995 | 1403 |
| 4 | typed query | 0.169 | 0.597 | 0.48 | 203.0 | 1403 |
| 4 | 25 | 0.239 | 0.563 | 0.473 | 181.003 | 1403 |
| 4 | 50 | 0.362 | 0.564 | 0.496 | 100.996 | 1403 |
| 4 | 100 | 0.42 | 0.618 | 0.543 | 81.001 | 1403 |
| 4 | 150 | 0.423 | 0.653 | 0.57 | 82.996 | 1403 |
| 4 | after the check | 0.428 | 0.678 | 0.602 | 80.005 | 1403 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | SigLIP binary | app line | 0.25 | text | 23.914 | 0.525 | 0.257 | 0.466 | 0.559 | 0.643 | 0.009 | 1440 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 25 | 42.073 | 0.505 | 0.285 | 0.449 | 0.555 | 0.625 | 0.009 | 1434 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 50 | 41.089 | 0.555 | 0.323 | 0.491 | 0.573 | 0.641 | 0.009 | 1274 |
| beta 0.25 | SigLIP binary | app line | 0.25 | final | 39.6 | 0.673 | 0.381 | 0.596 | 0.672 | 0.743 | 0.005 | 1440 |
| beta 0.25 | SigLIP binary | app line | 0.25 | ceiling | 117.692 | 0.466 | 0.66 | 0.465 | 0.711 | 0.585 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | text | 32.0 | 0.487 | 0.314 | 0.472 | 0.559 | 0.694 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 25 | 32.0 | 0.474 | 0.305 | 0.459 | 0.555 | 0.683 | 0.0 | 1434 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 50 | 32.0 | 0.509 | 0.328 | 0.493 | 0.573 | 0.695 | 0.0 | 1274 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | final | 32.0 | 0.598 | 0.386 | 0.579 | 0.672 | 0.743 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | ceiling | 32.0 | 0.631 | 0.407 | 0.611 | 0.711 | 0.806 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | text | 63.415 | 0.365 | 0.436 | 0.363 | 0.45 | 0.676 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | 25 | 78.238 | 0.389 | 0.399 | 0.357 | 0.431 | 0.683 | 0.0 | 1435 |
| beta 1 | SigLIP binary | app line | 1.0 | 50 | 92.573 | 0.481 | 0.412 | 0.411 | 0.466 | 0.7 | 0.0 | 1283 |
| beta 1 | SigLIP binary | app line | 1.0 | final | 101.834 | 0.549 | 0.5 | 0.491 | 0.539 | 0.776 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | ceiling | 296.793 | 0.369 | 0.779 | 0.438 | 0.576 | 0.671 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | text | 32.0 | 0.487 | 0.314 | 0.381 | 0.45 | 0.713 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | 25 | 32.0 | 0.474 | 0.305 | 0.371 | 0.431 | 0.721 | 0.0 | 1435 |
| beta 1 | SigLIP binary | top-K | 1.0 | 50 | 32.0 | 0.509 | 0.327 | 0.398 | 0.466 | 0.707 | 0.0 | 1283 |
| beta 1 | SigLIP binary | top-K | 1.0 | final | 32.0 | 0.597 | 0.386 | 0.468 | 0.539 | 0.777 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | ceiling | 32.0 | 0.631 | 0.407 | 0.494 | 0.576 | 0.834 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | text | 288.108 | 0.165 | 0.583 | 0.469 | 0.565 | 0.739 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | 25 | 271.319 | 0.234 | 0.549 | 0.463 | 0.545 | 0.751 | 0.0 | 1434 |
| beta 4 | SigLIP binary | app line | 4.0 | 50 | 265.018 | 0.363 | 0.538 | 0.483 | 0.565 | 0.741 | 0.0 | 1251 |
| beta 4 | SigLIP binary | app line | 4.0 | final | 327.001 | 0.412 | 0.637 | 0.556 | 0.627 | 0.799 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | ceiling | 854.619 | 0.278 | 0.885 | 0.609 | 0.691 | 0.834 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | text | 128.0 | 0.207 | 0.53 | 0.485 | 0.565 | 0.739 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | 25 | 128.0 | 0.199 | 0.509 | 0.466 | 0.545 | 0.739 | 0.0 | 1434 |
| beta 4 | SigLIP binary | top-K | 4.0 | 50 | 128.0 | 0.206 | 0.528 | 0.483 | 0.565 | 0.725 | 0.0 | 1251 |
| beta 4 | SigLIP binary | top-K | 4.0 | final | 128.0 | 0.24 | 0.615 | 0.563 | 0.627 | 0.804 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | ceiling | 128.0 | 0.265 | 0.68 | 0.622 | 0.691 | 0.866 | 0.0 | 1440 |

## The early dip

The detector's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's blind GMM cut against the detector's labels line; under `top-K` the old cap on both. F-beta, what each returned, and the click by which the detector's F-beta reaches the text sort's; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | detector: F at 5 / 10 / 25 | detector: returned at 5 / 10 / 25 | detector F >= text sort's by click | lowest share | at click | share >= text sort's by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.466 | 18.004 | 0.643 | 0.40 / 0.45 / 0.43 | 16 / 24 / 31 | 33 | 0.141 | 2 | 33 |
| 0.25 | top-K | 0.472 | 32.0 | 0.694 | 0.45 / 0.49 / 0.47 | 32 / 32 / 32 | 7 | 0.324 | 3 | 31 |
| 1 | app line | 0.363 | 49.003 | 0.676 | 0.28 / 0.34 / 0.34 | 29 / 39 / 53 | 30 | 0.076 | 2 | 29 |
| 1 | top-K | 0.381 | 32.0 | 0.713 | 0.36 / 0.39 / 0.38 | 32 / 32 / 32 | 7 | 0.469 | 3 | 4 |
| 4 | app line | 0.469 | 203.001 | 0.739 | 0.34 / 0.41 / 0.42 | 48 / 75 / 129 | 34 | 0.029 | 2 | 29 |
| 4 | top-K | 0.485 | 128.0 | 0.739 | 0.44 / 0.47 / 0.45 | 128 / 128 / 128 | 34 | 0.429 | 2 | 4 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 1403 | 20.741 | 0.179 | 0.82 | 0.641 | 0.453 | 1.0 |
| 1 | 1403 | 21.219 | 0.119 | 0.803 | 0.683 | 0.378 | 1.0 |
| 4 | 1403 | 30.445 | 0.079 | 0.782 | 0.703 | 0.293 | 1.0 |
