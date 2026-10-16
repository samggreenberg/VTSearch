# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 6480 | 0.463 | 0.539 | 0.64 | 0.65 | +0.010 ± 0.001 | 0.758 | 0.357 | 15 / 17 | 0% / 0% | 20.137 |
| 1 | 6480 | 0.363 | 0.445 | 0.535 | 0.541 | +0.006 ± 0.001 | 0.61 | 0.52 | 40 / 42 | 0% / 0% | 20.2 |
| 4 | 6480 | 0.468 | 0.52 | 0.606 | 0.613 | +0.007 ± 0.001 | 0.422 | 0.659 | 82 / 80 | 18% / 19% | 29.549 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set from the typed query (the text sort at its own blind GMM cut, before any vote), through 25, 50, 100 and 150 clicks, to after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over every run, a session still in the opening or one that never trained at the typed query's set (#4605, #4631); `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | typed query | 0.515 | 0.254 | 0.463 | 18.995 | 6480 |
| 0.25 | 25 | 0.515 | 0.254 | 0.463 | 18.995 | 6480 |
| 0.25 | 50 | 0.656 | 0.308 | 0.539 | 16.0 | 6480 |
| 0.25 | 100 | 0.75 | 0.28 | 0.592 | 11.994 | 6480 |
| 0.25 | 150 | 0.759 | 0.342 | 0.64 | 15.001 | 6480 |
| 0.25 | after the check | 0.758 | 0.357 | 0.65 | 17.0 | 6480 |
| 1 | typed query | 0.363 | 0.436 | 0.363 | 49.995 | 6480 |
| 1 | 25 | 0.363 | 0.436 | 0.363 | 49.995 | 6480 |
| 1 | 50 | 0.579 | 0.417 | 0.445 | 35.0 | 6480 |
| 1 | 100 | 0.647 | 0.443 | 0.499 | 32.998 | 6480 |
| 1 | 150 | 0.623 | 0.503 | 0.535 | 39.999 | 6480 |
| 1 | after the check | 0.61 | 0.52 | 0.541 | 42.003 | 6480 |
| 4 | typed query | 0.164 | 0.582 | 0.468 | 204.996 | 6480 |
| 4 | 25 | 0.164 | 0.582 | 0.468 | 204.996 | 6480 |
| 4 | 50 | 0.434 | 0.543 | 0.52 | 64.003 | 6480 |
| 4 | 100 | 0.423 | 0.602 | 0.574 | 73.004 | 6480 |
| 4 | 150 | 0.402 | 0.649 | 0.606 | 82.004 | 6480 |
| 4 | after the check | 0.422 | 0.659 | 0.613 | 80.5 | 6480 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | SigLIP binary | app line | 0.25 | text | 23.835 | 0.515 | 0.254 | 0.463 | 0.557 | 0.639 | 0.009 | 6480 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 25 | 23.835 | 0.515 | 0.254 | 0.463 | 0.557 | 0.639 | 0.009 | 6480 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 50 | 22.031 | 0.656 | 0.308 | 0.539 | 0.658 | 0.677 | 0.004 | 6480 |
| beta 0.25 | SigLIP binary | app line | 0.25 | final | 20.102 | 0.759 | 0.342 | 0.64 | 0.724 | 0.788 | 0.001 | 6480 |
| beta 0.25 | SigLIP binary | app line | 0.25 | ceiling | 30.348 | 0.683 | 0.42 | 0.634 | 0.72 | 0.821 | 0.0 | 6480 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | text | 32.0 | 0.485 | 0.313 | 0.47 | 0.557 | 0.69 | 0.0 | 6480 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 25 | 32.0 | 0.485 | 0.313 | 0.47 | 0.557 | 0.69 | 0.0 | 6480 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 50 | 32.0 | 0.579 | 0.374 | 0.561 | 0.658 | 0.729 | 0.0 | 6480 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | final | 32.0 | 0.648 | 0.418 | 0.627 | 0.724 | 0.781 | 0.0 | 6480 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | ceiling | 32.0 | 0.641 | 0.414 | 0.621 | 0.72 | 0.809 | 0.0 | 6480 |
| beta 1 | SigLIP binary | app line | 1.0 | text | 63.492 | 0.363 | 0.436 | 0.363 | 0.449 | 0.674 | 0.0 | 6480 |
| beta 1 | SigLIP binary | app line | 1.0 | 25 | 63.492 | 0.363 | 0.436 | 0.363 | 0.449 | 0.674 | 0.0 | 6480 |
| beta 1 | SigLIP binary | app line | 1.0 | 50 | 40.531 | 0.579 | 0.417 | 0.445 | 0.523 | 0.722 | 0.0 | 6480 |
| beta 1 | SigLIP binary | app line | 1.0 | final | 41.725 | 0.623 | 0.503 | 0.535 | 0.584 | 0.828 | 0.0 | 6480 |
| beta 1 | SigLIP binary | app line | 1.0 | ceiling | 102.665 | 0.508 | 0.627 | 0.531 | 0.585 | 0.855 | 0.0 | 6480 |
| beta 1 | SigLIP binary | top-K | 1.0 | text | 32.0 | 0.485 | 0.313 | 0.38 | 0.449 | 0.71 | 0.0 | 6480 |
| beta 1 | SigLIP binary | top-K | 1.0 | 25 | 32.0 | 0.485 | 0.313 | 0.38 | 0.449 | 0.71 | 0.0 | 6480 |
| beta 1 | SigLIP binary | top-K | 1.0 | 50 | 32.0 | 0.579 | 0.374 | 0.453 | 0.523 | 0.774 | 0.0 | 6480 |
| beta 1 | SigLIP binary | top-K | 1.0 | final | 32.0 | 0.648 | 0.418 | 0.507 | 0.584 | 0.819 | 0.0 | 6480 |
| beta 1 | SigLIP binary | top-K | 1.0 | ceiling | 32.0 | 0.641 | 0.414 | 0.502 | 0.585 | 0.834 | 0.0 | 6480 |
| beta 4 | SigLIP binary | app line | 4.0 | text | 288.214 | 0.164 | 0.582 | 0.468 | 0.564 | 0.738 | 0.0 | 6480 |
| beta 4 | SigLIP binary | app line | 4.0 | 25 | 288.214 | 0.164 | 0.582 | 0.468 | 0.564 | 0.738 | 0.0 | 6480 |
| beta 4 | SigLIP binary | app line | 4.0 | 50 | 98.173 | 0.435 | 0.543 | 0.52 | 0.61 | 0.743 | 0.0 | 6480 |
| beta 4 | SigLIP binary | app line | 4.0 | final | 121.273 | 0.402 | 0.649 | 0.606 | 0.66 | 0.841 | 0.0 | 6480 |
| beta 4 | SigLIP binary | app line | 4.0 | ceiling | 413.022 | 0.332 | 0.816 | 0.653 | 0.697 | 0.918 | 0.0 | 6480 |
| beta 4 | SigLIP binary | top-K | 4.0 | text | 128.0 | 0.206 | 0.529 | 0.484 | 0.564 | 0.737 | 0.0 | 6480 |
| beta 4 | SigLIP binary | top-K | 4.0 | 25 | 128.0 | 0.206 | 0.529 | 0.484 | 0.564 | 0.737 | 0.0 | 6480 |
| beta 4 | SigLIP binary | top-K | 4.0 | 50 | 128.0 | 0.231 | 0.592 | 0.541 | 0.61 | 0.787 | 0.0 | 6480 |
| beta 4 | SigLIP binary | top-K | 4.0 | final | 128.0 | 0.256 | 0.657 | 0.601 | 0.66 | 0.847 | 0.0 | 6480 |
| beta 4 | SigLIP binary | top-K | 4.0 | ceiling | 128.0 | 0.268 | 0.686 | 0.628 | 0.697 | 0.869 | 0.0 | 6480 |

## The early dip

The session's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's own line in the app against the detector's labels line; under `top-K` the old cap on both. Every run at every click as its session shows it: the typed query's set until the app shows the run's detector (#4605), and throughout a run that never trained (#4631). F-beta, what each returned, and the click by which the session's F-beta is back at the text sort's after its lowest point; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | session: F at 25 / 50 / 100 | session: returned at 25 / 50 / 100 | session F >= text sort's again by click | lowest share | at click | share >= text sort's again by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.463 | 18.995 | 0.639 | 0.46 / 0.54 / 0.59 | 19 / 16 / 12 | never below | 0.631 | 30 | 33 |
| 0.25 | top-K | 0.47 | 32.0 | 0.69 | 0.47 / 0.56 / 0.61 | 32 / 32 / 32 | never below | 0.69 | 7 | never below |
| 1 | app line | 0.363 | 49.995 | 0.674 | 0.36 / 0.44 / 0.50 | 50 / 35 / 33 | never below | 0.651 | 30 | 37 |
| 1 | top-K | 0.38 | 32.0 | 0.71 | 0.38 / 0.45 / 0.49 | 32 / 32 / 32 | never below | 0.71 | 7 | never below |
| 4 | app line | 0.468 | 204.996 | 0.738 | 0.47 / 0.52 / 0.57 | 205 / 65 / 73 | never below | 0.718 | 35 | 41 |
| 4 | top-K | 0.484 | 128.0 | 0.737 | 0.48 / 0.54 / 0.58 | 128 / 128 / 128 | never below | 0.737 | 7 | never below |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 6309 | 20.682 | 0.203 | 0.56 | 0.357 | 0.306 | 0.986 |
| 1 | 6309 | 20.747 | 0.205 | 0.562 | 0.357 | 0.308 | 0.975 |
| 4 | 6309 | 30.35 | 0.215 | 0.586 | 0.371 | 0.321 | 0.979 |
