# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 139 | 0.578 | 0.641 | 0.693 | 0.736 | +0.043 ± 0.009 | 0.82 | 0.474 | 38 / 30 | 6% / 1% | 20.647 |
| 1 | 139 | 0.499 | 0.547 | 0.594 | 0.633 | +0.039 ± 0.007 | 0.671 | 0.646 | 51 / 48 | 9% / 4% | 21.043 |
| 4 | 139 | 0.624 | 0.638 | 0.68 | 0.723 | +0.043 ± 0.009 | 0.448 | 0.826 | 84 / 96 | 29% / 27% | 30.324 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set at 25, 50, 100 and 150 clicks and after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over the trained runs with a line by that click; `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | 25 | 0.642 | 0.57 | 0.578 | 43.998 | 122 |
| 0.25 | 50 | 0.687 | 0.553 | 0.641 | 39.002 | 130 |
| 0.25 | 100 | 0.777 | 0.516 | 0.698 | 35.503 | 132 |
| 0.25 | 150 | 0.756 | 0.531 | 0.693 | 38.005 | 139 |
| 0.25 | after the check | 0.82 | 0.474 | 0.736 | 29.999 | 139 |
| 1 | 25 | 0.517 | 0.667 | 0.499 | 58.998 | 122 |
| 1 | 50 | 0.563 | 0.665 | 0.547 | 52.002 | 130 |
| 1 | 100 | 0.663 | 0.666 | 0.607 | 48.0 | 132 |
| 1 | 150 | 0.62 | 0.664 | 0.594 | 50.998 | 139 |
| 1 | after the check | 0.671 | 0.646 | 0.633 | 47.997 | 139 |
| 4 | 25 | 0.376 | 0.743 | 0.624 | 104.002 | 122 |
| 4 | 50 | 0.425 | 0.762 | 0.638 | 71.999 | 130 |
| 4 | 100 | 0.477 | 0.806 | 0.689 | 79.502 | 132 |
| 4 | 150 | 0.441 | 0.82 | 0.68 | 83.997 | 139 |
| 4 | after the check | 0.448 | 0.826 | 0.723 | 96.0 | 139 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | DINOv3 region | app line | 0.25 | text | 4508.493 | 0.011 | 0.914 | 0.011 | 0.56 | 0.071 | 0.0 | 144 |
| beta 0.25 | DINOv3 region | app line | 0.25 | 25 | 759.326 | 0.546 | 0.597 | 0.491 | 0.678 | 0.61 | 0.0 | 144 |
| beta 0.25 | DINOv3 region | app line | 0.25 | 50 | 532.746 | 0.625 | 0.579 | 0.586 | 0.711 | 0.705 | 0.0 | 138 |
| beta 0.25 | DINOv3 region | app line | 0.25 | final | 260.083 | 0.73 | 0.534 | 0.669 | 0.778 | 0.786 | 0.0 | 144 |
| beta 0.25 | DINOv3 region | app line | 0.25 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | text | 32.0 | 0.488 | 0.315 | 0.472 | 0.56 | 0.702 | 0.0 | 144 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | 25 | 32.0 | 0.61 | 0.391 | 0.59 | 0.678 | 0.754 | 0.0 | 144 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | 50 | 32.0 | 0.654 | 0.422 | 0.632 | 0.711 | 0.762 | 0.0 | 138 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | final | 32.0 | 0.716 | 0.463 | 0.693 | 0.778 | 0.824 | 0.0 | 144 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 1 | DINOv3 region | app line | 1.0 | text | 4508.493 | 0.011 | 0.914 | 0.021 | 0.451 | 0.125 | 0.0 | 144 |
| beta 1 | DINOv3 region | app line | 1.0 | 25 | 828.937 | 0.442 | 0.68 | 0.427 | 0.548 | 0.655 | 0.0 | 143 |
| beta 1 | DINOv3 region | app line | 1.0 | 50 | 628.121 | 0.513 | 0.671 | 0.499 | 0.589 | 0.726 | 0.0 | 140 |
| beta 1 | DINOv3 region | app line | 1.0 | final | 357.521 | 0.599 | 0.663 | 0.574 | 0.646 | 0.816 | 0.0 | 144 |
| beta 1 | DINOv3 region | app line | 1.0 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 1 | DINOv3 region | top-K | 1.0 | text | 32.0 | 0.488 | 0.315 | 0.382 | 0.451 | 0.725 | 0.0 | 144 |
| beta 1 | DINOv3 region | top-K | 1.0 | 25 | 32.0 | 0.611 | 0.393 | 0.477 | 0.548 | 0.77 | 0.0 | 143 |
| beta 1 | DINOv3 region | top-K | 1.0 | 50 | 32.0 | 0.653 | 0.421 | 0.51 | 0.589 | 0.757 | 0.0 | 140 |
| beta 1 | DINOv3 region | top-K | 1.0 | final | 32.0 | 0.713 | 0.461 | 0.558 | 0.646 | 0.821 | 0.0 | 144 |
| beta 1 | DINOv3 region | top-K | 1.0 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 4 | DINOv3 region | app line | 4.0 | text | 4508.493 | 0.011 | 0.914 | 0.151 | 0.568 | 0.363 | 0.0 | 144 |
| beta 4 | DINOv3 region | app line | 4.0 | 25 | 1008.231 | 0.322 | 0.744 | 0.549 | 0.64 | 0.789 | 0.0 | 143 |
| beta 4 | DINOv3 region | app line | 4.0 | 50 | 963.765 | 0.396 | 0.753 | 0.592 | 0.68 | 0.797 | 0.0 | 136 |
| beta 4 | DINOv3 region | app line | 4.0 | final | 756.694 | 0.426 | 0.813 | 0.66 | 0.735 | 0.847 | 0.0 | 144 |
| beta 4 | DINOv3 region | app line | 4.0 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 4 | DINOv3 region | top-K | 4.0 | text | 128.0 | 0.208 | 0.531 | 0.486 | 0.568 | 0.739 | 0.0 | 144 |
| beta 4 | DINOv3 region | top-K | 4.0 | 25 | 128.0 | 0.245 | 0.624 | 0.571 | 0.64 | 0.793 | 0.0 | 143 |
| beta 4 | DINOv3 region | top-K | 4.0 | 50 | 128.0 | 0.263 | 0.672 | 0.615 | 0.68 | 0.809 | 0.0 | 136 |
| beta 4 | DINOv3 region | top-K | 4.0 | final | 128.0 | 0.288 | 0.737 | 0.674 | 0.735 | 0.869 | 0.0 | 144 |
| beta 4 | DINOv3 region | top-K | 4.0 | ceiling |  |  |  |  |  |  | 0.0 | 0 |

## The early dip

The detector's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's blind GMM cut against the detector's labels line; under `top-K` the old cap on both. F-beta, what each returned, and the click by which the detector's F-beta reaches the text sort's; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | detector: F at 5 / 10 / 25 | detector: returned at 5 / 10 / 25 | detector F >= text sort's by click | lowest share | at click | share >= text sort's by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.011 | 4518.501 | 0.071 | nan / 0.54 / 0.58 | nan / 50 / 44 | 10 | 0.082 | 54 | 54 |
| 0.25 | top-K | 0.472 | 32.0 | 0.702 | nan / 0.69 / 0.69 | nan / 32 / 32 | 10 | 0.621 | 54 | 100 |
| 1 | app line | 0.021 | 4518.501 | 0.125 | nan / 0.49 / 0.50 | nan / 56 / 59 | 10 | 0.13 | 54 | 54 |
| 1 | top-K | 0.382 | 32.0 | 0.725 | nan / 0.56 / 0.56 | nan / 32 / 32 | 10 | 0.735 | 54 | 54 |
| 4 | app line | 0.151 | 4518.501 | 0.363 | nan / 0.61 / 0.63 | nan / 99 / 104 | 10 | 0.48 | 54 | 54 |
| 4 | top-K | 0.486 | 128.0 | 0.739 | nan / 0.67 / 0.67 | nan / 128 / 128 | 10 | 0.745 | 54 | 54 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 139 | 20.647 | 0.199 | 0.824 | 0.625 | 0.485 | 1.0 |
| 1 | 139 | 21.043 | 0.134 | 0.802 | 0.668 | 0.408 | 1.0 |
| 4 | 139 | 30.324 | 0.092 | 0.789 | 0.696 | 0.333 | 1.0 |
