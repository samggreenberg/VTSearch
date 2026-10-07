# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1440 | 0.447 | 0.496 | 0.596 | 0.62 | +0.024 ± 0.003 | 0.712 | 0.373 | 24 / 20 | 1% / 0% | 20.208 |
| 1 | 1440 | 0.356 | 0.41 | 0.491 | 0.513 | +0.022 ± 0.002 | 0.582 | 0.51 | 43 / 43 | 7% / 5% | 20.674 |
| 4 | 1440 | 0.462 | 0.484 | 0.556 | 0.587 | +0.031 ± 0.002 | 0.417 | 0.661 | 85 / 83 | 30% / 29% | 29.663 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set from the typed query (the text sort at its own blind GMM cut, before any vote), through 25, 50, 100 and 150 clicks, to after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over every run, a session still in the opening or one that never trained at the typed query's set (#4605, #4631); `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | typed query | 0.52 | 0.257 | 0.466 | 18.004 | 1440 |
| 0.25 | 25 | 0.499 | 0.284 | 0.447 | 18.997 | 1440 |
| 0.25 | 50 | 0.567 | 0.313 | 0.496 | 19.004 | 1440 |
| 0.25 | 100 | 0.646 | 0.34 | 0.56 | 20.005 | 1440 |
| 0.25 | 150 | 0.669 | 0.381 | 0.596 | 24.0 | 1440 |
| 0.25 | after the check | 0.712 | 0.373 | 0.62 | 20.499 | 1440 |
| 1 | typed query | 0.365 | 0.436 | 0.363 | 49.003 | 1440 |
| 1 | 25 | 0.388 | 0.399 | 0.356 | 43.995 | 1440 |
| 1 | 50 | 0.487 | 0.415 | 0.41 | 39.996 | 1440 |
| 1 | 100 | 0.549 | 0.462 | 0.461 | 39.998 | 1440 |
| 1 | 150 | 0.549 | 0.5 | 0.491 | 43.0 | 1440 |
| 1 | after the check | 0.582 | 0.51 | 0.513 | 43.003 | 1440 |
| 4 | typed query | 0.165 | 0.583 | 0.469 | 203.001 | 1440 |
| 4 | 25 | 0.234 | 0.549 | 0.462 | 181.996 | 1440 |
| 4 | 50 | 0.353 | 0.551 | 0.484 | 101.999 | 1440 |
| 4 | 100 | 0.41 | 0.603 | 0.529 | 84.003 | 1440 |
| 4 | 150 | 0.412 | 0.637 | 0.556 | 84.998 | 1440 |
| 4 | after the check | 0.417 | 0.661 | 0.587 | 83.5 | 1440 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | SigLIP binary | app line | 0.25 | text | 23.914 | 0.52 | 0.257 | 0.466 | 0.559 | 0.643 | 0.009 | 1440 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 25 | 42.861 | 0.499 | 0.285 | 0.447 | 0.553 | 0.624 | 0.009 | 1440 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 50 | 41.982 | 0.567 | 0.31 | 0.496 | 0.584 | 0.648 | 0.009 | 1440 |
| beta 0.25 | SigLIP binary | app line | 0.25 | final | 39.6 | 0.669 | 0.381 | 0.596 | 0.672 | 0.743 | 0.005 | 1440 |
| beta 0.25 | SigLIP binary | app line | 0.25 | ceiling | 117.692 | 0.466 | 0.66 | 0.465 | 0.711 | 0.585 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | text | 32.0 | 0.487 | 0.314 | 0.472 | 0.559 | 0.694 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 25 | 32.0 | 0.473 | 0.304 | 0.457 | 0.553 | 0.682 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 50 | 32.0 | 0.513 | 0.33 | 0.496 | 0.584 | 0.696 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | final | 32.0 | 0.598 | 0.386 | 0.579 | 0.672 | 0.743 | 0.0 | 1440 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | ceiling | 32.0 | 0.631 | 0.407 | 0.611 | 0.711 | 0.806 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | text | 63.415 | 0.365 | 0.436 | 0.363 | 0.45 | 0.676 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | 25 | 78.42 | 0.388 | 0.398 | 0.356 | 0.43 | 0.683 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | 50 | 104.066 | 0.487 | 0.413 | 0.41 | 0.467 | 0.704 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | final | 101.834 | 0.549 | 0.5 | 0.491 | 0.539 | 0.776 | 0.0 | 1440 |
| beta 1 | SigLIP binary | app line | 1.0 | ceiling | 296.793 | 0.369 | 0.779 | 0.438 | 0.576 | 0.671 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | text | 32.0 | 0.487 | 0.314 | 0.381 | 0.45 | 0.713 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | 25 | 32.0 | 0.473 | 0.305 | 0.37 | 0.43 | 0.72 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | 50 | 32.0 | 0.513 | 0.331 | 0.401 | 0.467 | 0.723 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | final | 32.0 | 0.597 | 0.386 | 0.468 | 0.539 | 0.777 | 0.0 | 1440 |
| beta 1 | SigLIP binary | top-K | 1.0 | ceiling | 32.0 | 0.631 | 0.407 | 0.494 | 0.576 | 0.834 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | text | 288.108 | 0.165 | 0.583 | 0.469 | 0.565 | 0.739 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | 25 | 272.819 | 0.234 | 0.548 | 0.462 | 0.543 | 0.752 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | 50 | 283.6 | 0.353 | 0.547 | 0.484 | 0.565 | 0.749 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | final | 327.001 | 0.412 | 0.637 | 0.556 | 0.627 | 0.799 | 0.0 | 1440 |
| beta 4 | SigLIP binary | app line | 4.0 | ceiling | 854.619 | 0.278 | 0.885 | 0.609 | 0.691 | 0.834 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | text | 128.0 | 0.207 | 0.53 | 0.485 | 0.565 | 0.739 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | 25 | 128.0 | 0.198 | 0.508 | 0.464 | 0.543 | 0.737 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | 50 | 128.0 | 0.207 | 0.532 | 0.486 | 0.565 | 0.742 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | final | 128.0 | 0.24 | 0.615 | 0.563 | 0.627 | 0.804 | 0.0 | 1440 |
| beta 4 | SigLIP binary | top-K | 4.0 | ceiling | 128.0 | 0.265 | 0.68 | 0.622 | 0.691 | 0.866 | 0.0 | 1440 |

## The early dip

The session's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's own line in the app against the detector's labels line; under `top-K` the old cap on both. Every run at every click as its session shows it: the typed query's set until the app shows the run's detector (#4605), and throughout a run that never trained (#4631). F-beta, what each returned, and the click by which the session's F-beta is back at the text sort's after its lowest point; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | session: F at 25 / 50 / 100 | session: returned at 25 / 50 / 100 | session F >= text sort's again by click | lowest share | at click | share >= text sort's again by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.466 | 18.004 | 0.643 | 0.45 / 0.50 / 0.56 | 19 / 19 / 20 | 35 | 0.624 | 29 | 40 |
| 0.25 | top-K | 0.472 | 32.0 | 0.694 | 0.46 / 0.50 / 0.55 | 32 / 32 / 32 | 35 | 0.682 | 25 | 46 |
| 1 | app line | 0.363 | 49.003 | 0.676 | 0.36 / 0.41 / 0.46 | 44 / 39 / 40 | 30 | 0.676 | 2 | never below |
| 1 | top-K | 0.381 | 32.0 | 0.713 | 0.37 / 0.40 / 0.45 | 32 / 32 / 32 | 35 | 0.713 | 2 | never below |
| 4 | app line | 0.469 | 203.001 | 0.739 | 0.46 / 0.48 / 0.53 | 181 / 102 / 84 | 38 | 0.739 | 2 | never below |
| 4 | top-K | 0.485 | 128.0 | 0.739 | 0.46 / 0.49 / 0.54 | 128 / 128 / 128 | 50 | 0.734 | 30 | 50 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 1403 | 20.741 | 0.179 | 0.82 | 0.641 | 0.453 | 1.0 |
| 1 | 1403 | 21.219 | 0.119 | 0.803 | 0.683 | 0.378 | 1.0 |
| 4 | 1403 | 30.445 | 0.079 | 0.782 | 0.703 | 0.293 | 1.0 |
