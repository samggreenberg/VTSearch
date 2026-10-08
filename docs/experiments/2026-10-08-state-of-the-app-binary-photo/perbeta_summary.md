# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 3456 | 0.449 | 0.493 | 0.611 | 0.625 | +0.014 ± 0.001 | 0.717 | 0.359 | 18 / 20 | 1% / 0% | 20.284 |
| 1 | 3456 | 0.358 | 0.405 | 0.511 | 0.52 | +0.008 ± 0.001 | 0.572 | 0.519 | 43 / 44 | 4% / 2% | 20.326 |
| 4 | 3456 | 0.461 | 0.484 | 0.584 | 0.594 | +0.010 ± 0.001 | 0.388 | 0.658 | 101 / 101 | 28% / 29% | 29.612 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set from the typed query (the text sort at its own blind GMM cut, before any vote), through 25, 50, 100 and 150 clicks, to after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over every run, a session still in the opening or one that never trained at the typed query's set (#4605, #4631); `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | typed query | 0.516 | 0.255 | 0.464 | 18.995 | 3456 |
| 0.25 | 25 | 0.502 | 0.28 | 0.449 | 18.996 | 3456 |
| 0.25 | 50 | 0.579 | 0.269 | 0.493 | 16.002 | 3456 |
| 0.25 | 100 | 0.678 | 0.283 | 0.56 | 15.0 | 3456 |
| 0.25 | 150 | 0.716 | 0.343 | 0.611 | 17.998 | 3456 |
| 0.25 | after the check | 0.717 | 0.359 | 0.625 | 19.998 | 3456 |
| 1 | typed query | 0.364 | 0.435 | 0.363 | 49.004 | 3456 |
| 1 | 25 | 0.395 | 0.396 | 0.358 | 43.001 | 3456 |
| 1 | 50 | 0.5 | 0.398 | 0.405 | 36.995 | 3456 |
| 1 | 100 | 0.579 | 0.453 | 0.468 | 38.998 | 3456 |
| 1 | 150 | 0.587 | 0.502 | 0.511 | 42.996 | 3456 |
| 1 | after the check | 0.572 | 0.519 | 0.52 | 44.004 | 3456 |
| 4 | typed query | 0.164 | 0.581 | 0.467 | 204.995 | 3456 |
| 4 | 25 | 0.241 | 0.545 | 0.461 | 178.005 | 3456 |
| 4 | 50 | 0.336 | 0.552 | 0.484 | 104.005 | 3456 |
| 4 | 100 | 0.361 | 0.626 | 0.551 | 97.996 | 3456 |
| 4 | 150 | 0.377 | 0.65 | 0.584 | 100.999 | 3456 |
| 4 | after the check | 0.388 | 0.658 | 0.594 | 100.996 | 3456 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | SigLIP binary | app line | 0.25 | text | 23.854 | 0.516 | 0.255 | 0.464 | 0.557 | 0.639 | 0.007 | 3456 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 25 | 30.201 | 0.502 | 0.279 | 0.449 | 0.553 | 0.624 | 0.007 | 3456 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 50 | 40.373 | 0.58 | 0.268 | 0.494 | 0.585 | 0.642 | 0.007 | 3456 |
| beta 0.25 | SigLIP binary | app line | 0.25 | final | 29.839 | 0.716 | 0.343 | 0.611 | 0.69 | 0.75 | 0.003 | 3456 |
| beta 0.25 | SigLIP binary | app line | 0.25 | ceiling | 30.754 | 0.667 | 0.418 | 0.618 | 0.709 | 0.811 | 0.0 | 3456 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | text | 32.0 | 0.485 | 0.313 | 0.469 | 0.557 | 0.69 | 0.0 | 3456 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 25 | 32.0 | 0.472 | 0.304 | 0.457 | 0.553 | 0.68 | 0.0 | 3456 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 50 | 32.0 | 0.514 | 0.332 | 0.497 | 0.585 | 0.696 | 0.0 | 3456 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | final | 32.0 | 0.619 | 0.4 | 0.599 | 0.69 | 0.756 | 0.0 | 3456 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | ceiling | 32.0 | 0.63 | 0.407 | 0.61 | 0.709 | 0.807 | 0.0 | 3456 |
| beta 1 | SigLIP binary | app line | 1.0 | text | 63.469 | 0.364 | 0.435 | 0.363 | 0.449 | 0.676 | 0.0 | 3456 |
| beta 1 | SigLIP binary | app line | 1.0 | 25 | 66.996 | 0.395 | 0.395 | 0.358 | 0.431 | 0.685 | 0.0 | 3456 |
| beta 1 | SigLIP binary | app line | 1.0 | 50 | 94.623 | 0.501 | 0.396 | 0.405 | 0.469 | 0.696 | 0.0 | 3456 |
| beta 1 | SigLIP binary | app line | 1.0 | final | 76.378 | 0.587 | 0.502 | 0.511 | 0.56 | 0.79 | 0.0 | 3456 |
| beta 1 | SigLIP binary | app line | 1.0 | ceiling | 104.5 | 0.497 | 0.623 | 0.521 | 0.576 | 0.852 | 0.0 | 3456 |
| beta 1 | SigLIP binary | top-K | 1.0 | text | 32.0 | 0.485 | 0.313 | 0.38 | 0.449 | 0.71 | 0.0 | 3456 |
| beta 1 | SigLIP binary | top-K | 1.0 | 25 | 32.0 | 0.472 | 0.304 | 0.369 | 0.431 | 0.717 | 0.0 | 3456 |
| beta 1 | SigLIP binary | top-K | 1.0 | 50 | 32.0 | 0.513 | 0.332 | 0.402 | 0.469 | 0.723 | 0.0 | 3456 |
| beta 1 | SigLIP binary | top-K | 1.0 | final | 32.0 | 0.618 | 0.4 | 0.484 | 0.56 | 0.786 | 0.0 | 3456 |
| beta 1 | SigLIP binary | top-K | 1.0 | ceiling | 32.0 | 0.63 | 0.407 | 0.493 | 0.576 | 0.833 | 0.0 | 3456 |
| beta 4 | SigLIP binary | app line | 4.0 | text | 287.998 | 0.164 | 0.581 | 0.467 | 0.564 | 0.737 | 0.0 | 3456 |
| beta 4 | SigLIP binary | app line | 4.0 | 25 | 264.225 | 0.241 | 0.544 | 0.461 | 0.544 | 0.747 | 0.0 | 3456 |
| beta 4 | SigLIP binary | app line | 4.0 | 50 | 284.368 | 0.336 | 0.548 | 0.484 | 0.562 | 0.75 | 0.0 | 3456 |
| beta 4 | SigLIP binary | app line | 4.0 | final | 238.902 | 0.377 | 0.65 | 0.584 | 0.637 | 0.83 | 0.0 | 3456 |
| beta 4 | SigLIP binary | app line | 4.0 | ceiling | 404.916 | 0.349 | 0.805 | 0.646 | 0.691 | 0.916 | 0.0 | 3456 |
| beta 4 | SigLIP binary | top-K | 4.0 | text | 128.0 | 0.206 | 0.529 | 0.484 | 0.564 | 0.738 | 0.0 | 3456 |
| beta 4 | SigLIP binary | top-K | 4.0 | 25 | 128.0 | 0.198 | 0.509 | 0.465 | 0.544 | 0.737 | 0.0 | 3456 |
| beta 4 | SigLIP binary | top-K | 4.0 | 50 | 128.0 | 0.205 | 0.526 | 0.481 | 0.562 | 0.739 | 0.0 | 3456 |
| beta 4 | SigLIP binary | top-K | 4.0 | final | 128.0 | 0.244 | 0.627 | 0.573 | 0.637 | 0.812 | 0.0 | 3456 |
| beta 4 | SigLIP binary | top-K | 4.0 | ceiling | 128.0 | 0.265 | 0.68 | 0.622 | 0.691 | 0.866 | 0.0 | 3456 |

## The early dip

The session's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's own line in the app against the detector's labels line; under `top-K` the old cap on both. Every run at every click as its session shows it: the typed query's set until the app shows the run's detector (#4605), and throughout a run that never trained (#4631). F-beta, what each returned, and the click by which the session's F-beta is back at the text sort's after its lowest point; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | session: F at 25 / 50 / 100 | session: returned at 25 / 50 / 100 | session F >= text sort's again by click | lowest share | at click | share >= text sort's again by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.464 | 18.995 | 0.639 | 0.45 / 0.49 / 0.56 | 19 / 16 / 15 | 30 | 0.624 | 29 | 43 |
| 0.25 | top-K | 0.469 | 32.0 | 0.69 | 0.46 / 0.50 / 0.56 | 32 / 32 / 32 | 35 | 0.679 | 25 | 45 |
| 1 | app line | 0.363 | 49.004 | 0.676 | 0.36 / 0.41 / 0.47 | 43 / 37 / 39 | 30 | 0.676 | 7 | never below |
| 1 | top-K | 0.38 | 32.0 | 0.71 | 0.37 / 0.40 / 0.46 | 32 / 32 / 32 | 35 | 0.71 | 7 | never below |
| 4 | app line | 0.467 | 204.995 | 0.737 | 0.46 / 0.48 / 0.55 | 178 / 104 / 96 | 40 | 0.737 | 7 | never below |
| 4 | top-K | 0.484 | 128.0 | 0.738 | 0.47 / 0.48 / 0.54 | 128 / 128 / 128 | 53 | 0.732 | 35 | 49 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 3369 | 20.807 | 0.054 | 0.634 | 0.58 | 0.168 | 1.0 |
| 1 | 3369 | 20.85 | 0.045 | 0.633 | 0.588 | 0.16 | 1.0 |
| 4 | 3369 | 30.377 | 0.038 | 0.649 | 0.61 | 0.146 | 1.0 |
