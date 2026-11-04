# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 11520 | 0.462 | 0.604 | 0.66 | 0.66 | +0.000 ± 0.001 | 0.711 | 0.441 | 25 / 27 | 0% / 0% | 20.019 |
| 1 | 11520 | 0.363 | 0.494 | 0.548 | 0.551 | +0.003 ± 0.000 | 0.57 | 0.561 | 47 / 48 | 0% / 0% | 20.053 |
| 4 | 11520 | 0.468 | 0.547 | 0.617 | 0.623 | +0.006 ± 0.000 | 0.419 | 0.668 | 91 / 87 | 16% / 19% | 29.49 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set from the typed query (the text sort at its own blind GMM cut, before any vote), through 25, 50, 100 and 150 clicks, to after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over every run, a session still in the opening or one that never trained at the typed query's set (#4605, #4631); `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | typed query | 0.514 | 0.255 | 0.462 | 18.995 | 11520 |
| 0.25 | 25 | 0.514 | 0.255 | 0.462 | 18.995 | 11520 |
| 0.25 | 50 | 0.695 | 0.347 | 0.604 | 18.0 | 11520 |
| 0.25 | 100 | 0.724 | 0.389 | 0.646 | 20.999 | 11520 |
| 0.25 | 150 | 0.719 | 0.43 | 0.66 | 25.003 | 11520 |
| 0.25 | after the check | 0.711 | 0.441 | 0.66 | 27.0 | 11520 |
| 1 | typed query | 0.363 | 0.436 | 0.363 | 50.0 | 11520 |
| 1 | 25 | 0.363 | 0.436 | 0.363 | 50.0 | 11520 |
| 1 | 50 | 0.587 | 0.464 | 0.494 | 38.005 | 11520 |
| 1 | 100 | 0.598 | 0.513 | 0.534 | 42.001 | 11520 |
| 1 | 150 | 0.579 | 0.549 | 0.548 | 46.998 | 11520 |
| 1 | after the check | 0.57 | 0.561 | 0.551 | 48.006 | 11520 |
| 4 | typed query | 0.165 | 0.582 | 0.468 | 204.003 | 11520 |
| 4 | 25 | 0.165 | 0.582 | 0.468 | 204.003 | 11520 |
| 4 | 50 | 0.399 | 0.575 | 0.547 | 74.003 | 11520 |
| 4 | 100 | 0.394 | 0.629 | 0.593 | 85.0 | 11520 |
| 4 | 150 | 0.39 | 0.66 | 0.617 | 91.006 | 11520 |
| 4 | after the check | 0.419 | 0.668 | 0.623 | 86.996 | 11520 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | SigLIP binary | app line | 0.25 | text | 23.936 | 0.514 | 0.255 | 0.462 | 0.557 | 0.636 | 0.01 | 11520 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 25 | 23.936 | 0.514 | 0.255 | 0.462 | 0.557 | 0.636 | 0.01 | 11520 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 50 | 21.125 | 0.695 | 0.347 | 0.604 | 0.67 | 0.742 | 0.004 | 11520 |
| beta 0.25 | SigLIP binary | app line | 0.25 | final | 25.974 | 0.719 | 0.43 | 0.66 | 0.726 | 0.812 | 0.002 | 11520 |
| beta 0.25 | SigLIP binary | app line | 0.25 | ceiling | 30.51 | 0.684 | 0.422 | 0.634 | 0.72 | 0.821 | 0.0 | 11520 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | text | 32.0 | 0.486 | 0.313 | 0.47 | 0.557 | 0.688 | 0.0 | 11520 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 25 | 32.0 | 0.486 | 0.313 | 0.47 | 0.557 | 0.688 | 0.0 | 11520 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 50 | 32.0 | 0.595 | 0.383 | 0.576 | 0.67 | 0.734 | 0.0 | 11520 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | final | 32.0 | 0.651 | 0.42 | 0.63 | 0.726 | 0.786 | 0.0 | 11520 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | ceiling | 32.0 | 0.643 | 0.414 | 0.622 | 0.72 | 0.811 | 0.0 | 11520 |
| beta 1 | SigLIP binary | app line | 1.0 | text | 63.678 | 0.363 | 0.436 | 0.363 | 0.449 | 0.672 | 0.001 | 11520 |
| beta 1 | SigLIP binary | app line | 1.0 | 25 | 63.678 | 0.363 | 0.436 | 0.363 | 0.449 | 0.672 | 0.001 | 11520 |
| beta 1 | SigLIP binary | app line | 1.0 | 50 | 38.005 | 0.586 | 0.463 | 0.494 | 0.539 | 0.768 | 0.0 | 11520 |
| beta 1 | SigLIP binary | app line | 1.0 | final | 47.127 | 0.579 | 0.549 | 0.548 | 0.587 | 0.843 | 0.0 | 11520 |
| beta 1 | SigLIP binary | app line | 1.0 | ceiling | 103.967 | 0.507 | 0.629 | 0.531 | 0.586 | 0.853 | 0.0 | 11520 |
| beta 1 | SigLIP binary | top-K | 1.0 | text | 32.0 | 0.486 | 0.313 | 0.38 | 0.449 | 0.708 | 0.0 | 11520 |
| beta 1 | SigLIP binary | top-K | 1.0 | 25 | 32.0 | 0.486 | 0.313 | 0.38 | 0.449 | 0.708 | 0.0 | 11520 |
| beta 1 | SigLIP binary | top-K | 1.0 | 50 | 32.0 | 0.595 | 0.383 | 0.465 | 0.539 | 0.771 | 0.0 | 11520 |
| beta 1 | SigLIP binary | top-K | 1.0 | final | 32.0 | 0.651 | 0.42 | 0.509 | 0.587 | 0.817 | 0.0 | 11520 |
| beta 1 | SigLIP binary | top-K | 1.0 | ceiling | 32.0 | 0.643 | 0.414 | 0.502 | 0.586 | 0.835 | 0.0 | 11520 |
| beta 4 | SigLIP binary | app line | 4.0 | text | 288.323 | 0.165 | 0.582 | 0.468 | 0.564 | 0.737 | 0.0 | 11520 |
| beta 4 | SigLIP binary | app line | 4.0 | 25 | 288.323 | 0.165 | 0.582 | 0.468 | 0.564 | 0.737 | 0.0 | 11520 |
| beta 4 | SigLIP binary | app line | 4.0 | 50 | 105.359 | 0.399 | 0.574 | 0.547 | 0.616 | 0.774 | 0.0 | 11520 |
| beta 4 | SigLIP binary | app line | 4.0 | final | 124.196 | 0.39 | 0.66 | 0.617 | 0.663 | 0.86 | 0.0 | 11520 |
| beta 4 | SigLIP binary | app line | 4.0 | ceiling | 414.21 | 0.334 | 0.816 | 0.653 | 0.696 | 0.918 | 0.0 | 11520 |
| beta 4 | SigLIP binary | top-K | 4.0 | text | 128.0 | 0.206 | 0.528 | 0.483 | 0.564 | 0.735 | 0.0 | 11520 |
| beta 4 | SigLIP binary | top-K | 4.0 | 25 | 128.0 | 0.206 | 0.528 | 0.483 | 0.564 | 0.735 | 0.0 | 11520 |
| beta 4 | SigLIP binary | top-K | 4.0 | 50 | 128.0 | 0.234 | 0.598 | 0.547 | 0.616 | 0.789 | 0.0 | 11520 |
| beta 4 | SigLIP binary | top-K | 4.0 | final | 128.0 | 0.258 | 0.661 | 0.604 | 0.663 | 0.85 | 0.0 | 11520 |
| beta 4 | SigLIP binary | top-K | 4.0 | ceiling | 128.0 | 0.268 | 0.686 | 0.627 | 0.696 | 0.868 | 0.0 | 11520 |

## The early dip

The session's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's own line in the app against the detector's labels line; under `top-K` the old cap on both. Every run at every click as its session shows it: the typed query's set until the app shows the run's detector (#4605), and throughout a run that never trained (#4631). F-beta, what each returned, and the click by which the session's F-beta is back at the text sort's after its lowest point; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | session: F at 25 / 50 / 100 | session: returned at 25 / 50 / 100 | session F >= text sort's again by click | lowest share | at click | share >= text sort's again by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.462 | 18.995 | 0.636 | 0.46 / 0.60 / 0.65 | 19 / 18 / 21 | never below | 0.636 | 7 | never below |
| 0.25 | top-K | 0.47 | 32.0 | 0.688 | 0.47 / 0.58 / 0.62 | 32 / 32 / 32 | never below | 0.688 | 7 | never below |
| 1 | app line | 0.363 | 50.0 | 0.672 | 0.36 / 0.49 / 0.53 | 50 / 38 / 42 | never below | 0.672 | 7 | never below |
| 1 | top-K | 0.38 | 32.0 | 0.708 | 0.38 / 0.46 / 0.50 | 32 / 32 / 32 | never below | 0.708 | 7 | never below |
| 4 | app line | 0.468 | 204.003 | 0.737 | 0.47 / 0.55 / 0.59 | 204 / 74 / 85 | never below | 0.737 | 7 | never below |
| 4 | top-K | 0.483 | 128.0 | 0.735 | 0.48 / 0.55 / 0.59 | 128 / 128 / 128 | never below | 0.735 | 7 | never below |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 11202 | 20.587 | 0.304 | 0.659 | 0.355 | 0.453 | 0.978 |
| 1 | 11202 | 20.623 | 0.31 | 0.664 | 0.354 | 0.463 | 0.97 |
| 4 | 11202 | 30.327 | 0.312 | 0.678 | 0.366 | 0.47 | 0.97 |
