# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 288 | 0.423 | 0.526 | 0.665 | 0.711 | +0.046 ± 0.007 | 0.788 | 0.461 | 34 / 29 | 0% / 1% | 20.087 |
| 1 | 288 | 0.361 | 0.453 | 0.58 | 0.614 | +0.034 ± 0.005 | 0.651 | 0.626 | 49 / 47 | 4% / 2% | 20.573 |
| 4 | 288 | 0.489 | 0.55 | 0.659 | 0.703 | +0.044 ± 0.006 | 0.439 | 0.798 | 92 / 94 | 27% / 26% | 29.392 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set from the typed query (the text sort at its own blind GMM cut, before any vote), through 25, 50, 100 and 150 clicks, to after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over every run, a session still in the opening or one that never trained at the typed query's set (#4605, #4631); `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | typed query | 0.523 | 0.255 | 0.468 | 17.998 | 288 |
| 0.25 | 25 | 0.46 | 0.354 | 0.423 | 20.997 | 288 |
| 0.25 | 50 | 0.558 | 0.409 | 0.526 | 29.0 | 288 |
| 0.25 | 100 | 0.691 | 0.451 | 0.629 | 30.503 | 288 |
| 0.25 | 150 | 0.728 | 0.473 | 0.665 | 34.002 | 288 |
| 0.25 | after the check | 0.788 | 0.461 | 0.711 | 29.005 | 288 |
| 1 | typed query | 0.366 | 0.438 | 0.365 | 49.996 | 288 |
| 1 | 25 | 0.361 | 0.453 | 0.361 | 48.997 | 288 |
| 1 | 50 | 0.464 | 0.498 | 0.453 | 46.996 | 288 |
| 1 | 100 | 0.588 | 0.577 | 0.548 | 46.003 | 288 |
| 1 | 150 | 0.606 | 0.611 | 0.58 | 48.501 | 288 |
| 1 | after the check | 0.651 | 0.626 | 0.614 | 47.003 | 288 |
| 4 | typed query | 0.166 | 0.583 | 0.469 | 203.501 | 288 |
| 4 | 25 | 0.229 | 0.585 | 0.489 | 186.499 | 288 |
| 4 | 50 | 0.344 | 0.62 | 0.55 | 109.499 | 288 |
| 4 | 100 | 0.42 | 0.717 | 0.63 | 92.999 | 288 |
| 4 | 150 | 0.427 | 0.757 | 0.659 | 92.499 | 288 |
| 4 | after the check | 0.439 | 0.798 | 0.703 | 93.997 | 288 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | DINOv3 region | app line | 0.25 | text | 23.708 | 0.523 | 0.255 | 0.468 | 0.561 | 0.654 | 0.003 | 288 |
| beta 0.25 | DINOv3 region | app line | 0.25 | 25 | 42.649 | 0.46 | 0.354 | 0.423 | 0.567 | 0.597 | 0.003 | 288 |
| beta 0.25 | DINOv3 region | app line | 0.25 | 50 | 40.413 | 0.561 | 0.408 | 0.528 | 0.622 | 0.674 | 0.003 | 288 |
| beta 0.25 | DINOv3 region | app line | 0.25 | final | 32.233 | 0.728 | 0.473 | 0.665 | 0.756 | 0.782 | 0.003 | 288 |
| beta 0.25 | DINOv3 region | app line | 0.25 | ceiling | 180.771 | 0.433 | 0.859 | 0.442 | 0.815 | 0.496 | 0.0 | 144 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | text | 32.0 | 0.486 | 0.314 | 0.471 | 0.561 | 0.69 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | 25 | 32.0 | 0.488 | 0.315 | 0.472 | 0.567 | 0.688 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | 50 | 32.0 | 0.563 | 0.363 | 0.545 | 0.622 | 0.726 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | final | 32.0 | 0.705 | 0.456 | 0.682 | 0.756 | 0.816 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | ceiling | 32.0 | 0.759 | 0.489 | 0.735 | 0.815 | 0.876 | 0.0 | 144 |
| beta 1 | DINOv3 region | app line | 1.0 | text | 63.42 | 0.366 | 0.438 | 0.365 | 0.45 | 0.694 | 0.0 | 288 |
| beta 1 | DINOv3 region | app line | 1.0 | 25 | 74.188 | 0.363 | 0.452 | 0.361 | 0.451 | 0.682 | 0.0 | 288 |
| beta 1 | DINOv3 region | app line | 1.0 | 50 | 81.594 | 0.465 | 0.497 | 0.453 | 0.515 | 0.741 | 0.0 | 288 |
| beta 1 | DINOv3 region | app line | 1.0 | final | 73.118 | 0.606 | 0.611 | 0.58 | 0.639 | 0.819 | 0.0 | 288 |
| beta 1 | DINOv3 region | app line | 1.0 | ceiling | 319.354 | 0.37 | 0.911 | 0.468 | 0.687 | 0.61 | 0.0 | 144 |
| beta 1 | DINOv3 region | top-K | 1.0 | text | 32.0 | 0.486 | 0.314 | 0.381 | 0.45 | 0.72 | 0.0 | 288 |
| beta 1 | DINOv3 region | top-K | 1.0 | 25 | 32.0 | 0.488 | 0.315 | 0.382 | 0.451 | 0.723 | 0.0 | 288 |
| beta 1 | DINOv3 region | top-K | 1.0 | 50 | 32.0 | 0.562 | 0.363 | 0.44 | 0.515 | 0.733 | 0.0 | 288 |
| beta 1 | DINOv3 region | top-K | 1.0 | final | 32.0 | 0.705 | 0.456 | 0.553 | 0.639 | 0.797 | 0.0 | 288 |
| beta 1 | DINOv3 region | top-K | 1.0 | ceiling | 32.0 | 0.759 | 0.489 | 0.593 | 0.687 | 0.863 | 0.0 | 144 |
| beta 4 | DINOv3 region | app line | 4.0 | text | 288.312 | 0.166 | 0.583 | 0.469 | 0.566 | 0.74 | 0.0 | 288 |
| beta 4 | DINOv3 region | app line | 4.0 | 25 | 269.726 | 0.231 | 0.584 | 0.489 | 0.567 | 0.759 | 0.0 | 288 |
| beta 4 | DINOv3 region | app line | 4.0 | 50 | 233.233 | 0.343 | 0.619 | 0.549 | 0.616 | 0.789 | 0.0 | 288 |
| beta 4 | DINOv3 region | app line | 4.0 | final | 295.701 | 0.427 | 0.757 | 0.659 | 0.722 | 0.842 | 0.0 | 288 |
| beta 4 | DINOv3 region | app line | 4.0 | ceiling | 605.431 | 0.308 | 0.948 | 0.701 | 0.796 | 0.852 | 0.0 | 144 |
| beta 4 | DINOv3 region | top-K | 4.0 | text | 128.0 | 0.207 | 0.531 | 0.485 | 0.566 | 0.741 | 0.0 | 288 |
| beta 4 | DINOv3 region | top-K | 4.0 | 25 | 128.0 | 0.206 | 0.528 | 0.483 | 0.567 | 0.738 | 0.0 | 288 |
| beta 4 | DINOv3 region | top-K | 4.0 | 50 | 128.0 | 0.228 | 0.586 | 0.536 | 0.616 | 0.754 | 0.0 | 288 |
| beta 4 | DINOv3 region | top-K | 4.0 | final | 128.0 | 0.281 | 0.721 | 0.659 | 0.722 | 0.841 | 0.0 | 288 |
| beta 4 | DINOv3 region | top-K | 4.0 | ceiling | 128.0 | 0.315 | 0.805 | 0.737 | 0.796 | 0.914 | 0.0 | 144 |

## The early dip

The session's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's own line in the app against the detector's labels line; under `top-K` the old cap on both. Every run at every click as its session shows it: the typed query's set until the app shows the run's detector (#4605), and throughout a run that never trained (#4631). F-beta, what each returned, and the click by which the session's F-beta is back at the text sort's after its lowest point; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | session: F at 25 / 50 / 100 | session: returned at 25 / 50 / 100 | session F >= text sort's again by click | lowest share | at click | share >= text sort's again by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.468 | 17.998 | 0.654 | 0.42 / 0.52 / 0.62 | 21 / 29 / 32 | 50 | 0.597 | 25 | 50 |
| 0.25 | top-K | 0.471 | 32.0 | 0.69 | 0.47 / 0.54 / 0.64 | 32 / 32 / 32 | never below | 0.688 | 25 | 50 |
| 1 | app line | 0.365 | 49.996 | 0.694 | 0.36 / 0.45 / 0.53 | 49 / 47 / 46 | 50 | 0.682 | 29 | 50 |
| 1 | top-K | 0.381 | 32.0 | 0.72 | 0.38 / 0.44 / 0.51 | 32 / 32 / 32 | never below | 0.72 | 10 | never below |
| 4 | app line | 0.469 | 203.501 | 0.74 | 0.49 / 0.55 / 0.62 | 186 / 112 / 92 | never below | 0.74 | 10 | never below |
| 4 | top-K | 0.485 | 128.0 | 0.741 | 0.48 / 0.53 / 0.62 | 128 / 128 / 128 | 50 | 0.736 | 25 | 50 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 279 | 20.735 | 0.197 | 0.824 | 0.627 | 0.476 | 1.0 |
| 1 | 279 | 21.237 | 0.129 | 0.805 | 0.676 | 0.4 | 1.0 |
| 4 | 279 | 30.341 | 0.09 | 0.789 | 0.699 | 0.326 | 1.0 |
