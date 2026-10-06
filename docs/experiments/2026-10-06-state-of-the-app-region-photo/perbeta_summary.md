# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 133 | 0.573 | 0.638 | 0.689 | 0.735 | +0.046 ± 0.009 | 0.814 | 0.482 | 40 / 31 | 6% / 1% | 20.677 |
| 1 | 133 | 0.501 | 0.548 | 0.594 | 0.634 | +0.040 ± 0.007 | 0.671 | 0.649 | 50 / 48 | 9% / 4% | 21.09 |
| 4 | 133 | 0.629 | 0.639 | 0.679 | 0.725 | +0.046 ± 0.009 | 0.454 | 0.828 | 82 / 95 | 28% / 26% | 30.338 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set at 25, 50, 100 and 150 clicks and after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over the trained runs with a line by that click; `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | 25 | 0.635 | 0.581 | 0.573 | 44.001 | 116 |
| 0.25 | 50 | 0.682 | 0.559 | 0.638 | 39.503 | 124 |
| 0.25 | 100 | 0.776 | 0.524 | 0.698 | 36.5 | 126 |
| 0.25 | 150 | 0.75 | 0.538 | 0.689 | 40.0 | 133 |
| 0.25 | after the check | 0.814 | 0.482 | 0.735 | 30.998 | 133 |
| 1 | 25 | 0.516 | 0.676 | 0.501 | 58.998 | 116 |
| 1 | 50 | 0.559 | 0.67 | 0.548 | 53.0 | 124 |
| 1 | 100 | 0.665 | 0.667 | 0.607 | 47.999 | 126 |
| 1 | 150 | 0.622 | 0.665 | 0.594 | 50.006 | 133 |
| 1 | after the check | 0.671 | 0.649 | 0.634 | 48.0 | 133 |
| 4 | 25 | 0.377 | 0.751 | 0.629 | 104.002 | 116 |
| 4 | 50 | 0.428 | 0.766 | 0.639 | 71.999 | 124 |
| 4 | 100 | 0.485 | 0.807 | 0.69 | 78.003 | 126 |
| 4 | 150 | 0.449 | 0.82 | 0.679 | 81.998 | 133 |
| 4 | after the check | 0.454 | 0.828 | 0.725 | 95.001 | 133 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | DINOv3 region | app line | 0.25 | text | 4525.326 | 0.011 | 0.914 | 0.011 | 0.562 | 0.072 | 0.0 | 138 |
| beta 0.25 | DINOv3 region | app line | 0.25 | 25 | 791.275 | 0.535 | 0.607 | 0.483 | 0.673 | 0.599 | 0.0 | 138 |
| beta 0.25 | DINOv3 region | app line | 0.25 | 50 | 555.386 | 0.617 | 0.586 | 0.582 | 0.705 | 0.701 | 0.0 | 132 |
| beta 0.25 | DINOv3 region | app line | 0.25 | final | 270.464 | 0.723 | 0.541 | 0.664 | 0.774 | 0.781 | 0.0 | 138 |
| beta 0.25 | DINOv3 region | app line | 0.25 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | text | 32.0 | 0.493 | 0.318 | 0.477 | 0.562 | 0.704 | 0.0 | 138 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | 25 | 32.0 | 0.609 | 0.391 | 0.589 | 0.673 | 0.753 | 0.0 | 138 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | 50 | 32.0 | 0.651 | 0.42 | 0.63 | 0.705 | 0.76 | 0.0 | 132 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | final | 32.0 | 0.716 | 0.463 | 0.693 | 0.774 | 0.825 | 0.0 | 138 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 1 | DINOv3 region | app line | 1.0 | text | 4525.326 | 0.011 | 0.914 | 0.021 | 0.455 | 0.125 | 0.0 | 138 |
| beta 1 | DINOv3 region | app line | 1.0 | 25 | 862.949 | 0.438 | 0.689 | 0.426 | 0.548 | 0.648 | 0.0 | 137 |
| beta 1 | DINOv3 region | app line | 1.0 | 50 | 653.836 | 0.507 | 0.676 | 0.497 | 0.588 | 0.721 | 0.0 | 134 |
| beta 1 | DINOv3 region | app line | 1.0 | final | 370.159 | 0.6 | 0.663 | 0.573 | 0.645 | 0.813 | 0.0 | 138 |
| beta 1 | DINOv3 region | app line | 1.0 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 1 | DINOv3 region | top-K | 1.0 | text | 32.0 | 0.493 | 0.318 | 0.385 | 0.455 | 0.723 | 0.0 | 138 |
| beta 1 | DINOv3 region | top-K | 1.0 | 25 | 32.0 | 0.61 | 0.392 | 0.476 | 0.548 | 0.764 | 0.0 | 137 |
| beta 1 | DINOv3 region | top-K | 1.0 | 50 | 32.0 | 0.65 | 0.419 | 0.508 | 0.588 | 0.749 | 0.0 | 134 |
| beta 1 | DINOv3 region | top-K | 1.0 | final | 32.0 | 0.71 | 0.459 | 0.556 | 0.645 | 0.818 | 0.0 | 138 |
| beta 1 | DINOv3 region | top-K | 1.0 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 4 | DINOv3 region | app line | 4.0 | text | 4525.326 | 0.011 | 0.914 | 0.15 | 0.571 | 0.361 | 0.0 | 138 |
| beta 4 | DINOv3 region | app line | 4.0 | 25 | 1045.869 | 0.32 | 0.75 | 0.551 | 0.64 | 0.789 | 0.0 | 137 |
| beta 4 | DINOv3 region | app line | 4.0 | 50 | 1002.023 | 0.397 | 0.756 | 0.591 | 0.679 | 0.794 | 0.0 | 130 |
| beta 4 | DINOv3 region | app line | 4.0 | final | 780.181 | 0.433 | 0.813 | 0.658 | 0.736 | 0.843 | 0.0 | 138 |
| beta 4 | DINOv3 region | app line | 4.0 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 4 | DINOv3 region | top-K | 4.0 | text | 128.0 | 0.209 | 0.536 | 0.49 | 0.571 | 0.739 | 0.0 | 138 |
| beta 4 | DINOv3 region | top-K | 4.0 | 25 | 128.0 | 0.244 | 0.624 | 0.571 | 0.64 | 0.787 | 0.0 | 137 |
| beta 4 | DINOv3 region | top-K | 4.0 | 50 | 128.0 | 0.261 | 0.669 | 0.612 | 0.679 | 0.802 | 0.0 | 130 |
| beta 4 | DINOv3 region | top-K | 4.0 | final | 128.0 | 0.287 | 0.735 | 0.672 | 0.736 | 0.865 | 0.0 | 138 |
| beta 4 | DINOv3 region | top-K | 4.0 | ceiling |  |  |  |  |  |  | 0.0 | 0 |

## The early dip

The detector's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's blind GMM cut against the detector's labels line; under `top-K` the old cap on both. F-beta, what each returned, and the click by which the detector's F-beta reaches the text sort's; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | detector: F at 5 / 10 / 25 | detector: returned at 5 / 10 / 25 | detector F >= text sort's by click | lowest share | at click | share >= text sort's by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.011 | 4517.001 | 0.072 | nan / 0.55 / 0.57 | nan / 50 / 44 | 10 | 0.082 | 54 | 54 |
| 0.25 | top-K | 0.477 | 32.0 | 0.704 | nan / 0.69 / 0.70 | nan / 32 / 32 | 10 | 0.621 | 54 | 100 |
| 1 | app line | 0.021 | 4517.001 | 0.125 | nan / 0.50 / 0.50 | nan / 56 / 59 | 10 | 0.13 | 54 | 54 |
| 1 | top-K | 0.385 | 32.0 | 0.723 | nan / 0.56 / 0.56 | nan / 32 / 32 | 10 | 0.735 | 54 | 54 |
| 4 | app line | 0.15 | 4517.001 | 0.361 | nan / 0.61 / 0.63 | nan / 80 / 104 | 10 | 0.48 | 54 | 54 |
| 4 | top-K | 0.49 | 128.0 | 0.739 | nan / 0.67 / 0.67 | nan / 128 / 128 | 10 | 0.745 | 54 | 54 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 133 | 20.677 | 0.202 | 0.824 | 0.622 | 0.487 | 1.0 |
| 1 | 133 | 21.09 | 0.132 | 0.8 | 0.667 | 0.404 | 1.0 |
| 4 | 133 | 30.338 | 0.093 | 0.788 | 0.695 | 0.334 | 1.0 |
