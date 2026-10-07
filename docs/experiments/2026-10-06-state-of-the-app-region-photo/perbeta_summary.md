# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 279 | 0.565 | 0.638 | 0.69 | 0.733 | +0.043 ± 0.006 | 0.812 | 0.475 | 36 / 30 | 4% / 1% | 20.735 |
| 1 | 279 | 0.485 | 0.54 | 0.601 | 0.633 | +0.032 ± 0.005 | 0.671 | 0.646 | 50 / 48 | 8% / 3% | 21.237 |
| 4 | 279 | 0.608 | 0.634 | 0.683 | 0.725 | +0.042 ± 0.006 | 0.453 | 0.822 | 85 / 92 | 27% / 25% | 30.341 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set from the typed query (the text sort at its own blind GMM cut, before any vote), through 25, 50, 100 and 150 clicks, to after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over the trained runs with a line by that click; `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | typed query | 0.011 | 0.922 | 0.011 | 4519.003 | 279 |
| 0.25 | 25 | 0.631 | 0.566 | 0.565 | 43.004 | 247 |
| 0.25 | 50 | 0.689 | 0.562 | 0.638 | 39.0 | 261 |
| 0.25 | 100 | 0.773 | 0.518 | 0.692 | 34.999 | 267 |
| 0.25 | 150 | 0.757 | 0.516 | 0.69 | 35.998 | 279 |
| 0.25 | after the check | 0.812 | 0.475 | 0.733 | 30.003 | 279 |
| 1 | typed query | 0.011 | 0.922 | 0.021 | 4519.003 | 279 |
| 1 | 25 | 0.511 | 0.661 | 0.485 | 56.998 | 247 |
| 1 | 50 | 0.566 | 0.673 | 0.54 | 51.999 | 261 |
| 1 | 100 | 0.657 | 0.662 | 0.599 | 47.996 | 267 |
| 1 | 150 | 0.63 | 0.664 | 0.601 | 49.996 | 279 |
| 1 | after the check | 0.671 | 0.646 | 0.633 | 47.997 | 279 |
| 4 | typed query | 0.011 | 0.922 | 0.152 | 4519.003 | 279 |
| 4 | 25 | 0.379 | 0.737 | 0.608 | 100.004 | 247 |
| 4 | 50 | 0.421 | 0.777 | 0.634 | 86.997 | 261 |
| 4 | 100 | 0.473 | 0.804 | 0.684 | 80.004 | 267 |
| 4 | 150 | 0.445 | 0.813 | 0.683 | 84.997 | 279 |
| 4 | after the check | 0.453 | 0.822 | 0.725 | 92.005 | 279 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | DINOv3 region | app line | 0.25 | text | 4508.031 | 0.011 | 0.912 | 0.011 | 0.561 | 0.074 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | app line | 0.25 | 25 | 708.305 | 0.542 | 0.592 | 0.486 | 0.678 | 0.602 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | app line | 0.25 | 50 | 527.263 | 0.628 | 0.586 | 0.585 | 0.72 | 0.7 | 0.0 | 278 |
| beta 0.25 | DINOv3 region | app line | 0.25 | final | 209.024 | 0.734 | 0.518 | 0.669 | 0.778 | 0.787 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | app line | 0.25 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | text | 32.0 | 0.486 | 0.314 | 0.471 | 0.561 | 0.69 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | 25 | 32.0 | 0.607 | 0.392 | 0.588 | 0.678 | 0.748 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | 50 | 32.0 | 0.658 | 0.425 | 0.637 | 0.72 | 0.769 | 0.0 | 278 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | final | 32.0 | 0.72 | 0.466 | 0.697 | 0.778 | 0.837 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 1 | DINOv3 region | app line | 1.0 | text | 4508.031 | 0.011 | 0.912 | 0.021 | 0.45 | 0.126 | 0.0 | 288 |
| beta 1 | DINOv3 region | app line | 1.0 | 25 | 774.383 | 0.441 | 0.674 | 0.419 | 0.547 | 0.637 | 0.0 | 287 |
| beta 1 | DINOv3 region | app line | 1.0 | 50 | 669.832 | 0.518 | 0.681 | 0.496 | 0.593 | 0.72 | 0.0 | 280 |
| beta 1 | DINOv3 region | app line | 1.0 | final | 290.816 | 0.61 | 0.661 | 0.583 | 0.652 | 0.822 | 0.0 | 288 |
| beta 1 | DINOv3 region | app line | 1.0 | ceiling | 147.0 | 0.627 | 0.924 | 0.695 | 0.824 | 0.784 | 0.0 | 5 |
| beta 1 | DINOv3 region | top-K | 1.0 | text | 32.0 | 0.486 | 0.314 | 0.381 | 0.45 | 0.72 | 0.0 | 288 |
| beta 1 | DINOv3 region | top-K | 1.0 | 25 | 32.0 | 0.608 | 0.393 | 0.476 | 0.547 | 0.768 | 0.0 | 287 |
| beta 1 | DINOv3 region | top-K | 1.0 | 50 | 32.0 | 0.658 | 0.425 | 0.515 | 0.593 | 0.771 | 0.0 | 280 |
| beta 1 | DINOv3 region | top-K | 1.0 | final | 32.0 | 0.72 | 0.466 | 0.564 | 0.652 | 0.829 | 0.0 | 288 |
| beta 1 | DINOv3 region | top-K | 1.0 | ceiling | 32.0 | 0.869 | 0.547 | 0.67 | 0.824 | 0.813 | 0.0 | 5 |
| beta 4 | DINOv3 region | app line | 4.0 | text | 4508.031 | 0.011 | 0.912 | 0.15 | 0.566 | 0.363 | 0.0 | 288 |
| beta 4 | DINOv3 region | app line | 4.0 | 25 | 954.056 | 0.327 | 0.738 | 0.54 | 0.641 | 0.767 | 0.0 | 287 |
| beta 4 | DINOv3 region | app line | 4.0 | 50 | 1018.124 | 0.393 | 0.769 | 0.589 | 0.682 | 0.796 | 0.0 | 273 |
| beta 4 | DINOv3 region | app line | 4.0 | final | 601.677 | 0.431 | 0.806 | 0.665 | 0.739 | 0.853 | 0.0 | 288 |
| beta 4 | DINOv3 region | app line | 4.0 | ceiling |  |  |  |  |  |  | 0.0 | 0 |
| beta 4 | DINOv3 region | top-K | 4.0 | text | 128.0 | 0.207 | 0.531 | 0.485 | 0.566 | 0.741 | 0.0 | 288 |
| beta 4 | DINOv3 region | top-K | 4.0 | 25 | 128.0 | 0.242 | 0.623 | 0.569 | 0.641 | 0.791 | 0.0 | 287 |
| beta 4 | DINOv3 region | top-K | 4.0 | 50 | 128.0 | 0.262 | 0.673 | 0.615 | 0.682 | 0.816 | 0.0 | 273 |
| beta 4 | DINOv3 region | top-K | 4.0 | final | 128.0 | 0.289 | 0.741 | 0.678 | 0.739 | 0.875 | 0.0 | 288 |
| beta 4 | DINOv3 region | top-K | 4.0 | ceiling |  |  |  |  |  |  | 0.0 | 0 |

## The early dip

The detector's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's blind GMM cut against the detector's labels line; under `top-K` the old cap on both. F-beta, what each returned, and the click by which the detector's F-beta reaches the text sort's; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | detector: F at 5 / 10 / 25 | detector: returned at 5 / 10 / 25 | detector F >= text sort's by click | lowest share | at click | share >= text sort's by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.011 | 4516.0 | 0.074 | nan / 0.55 / 0.57 | nan / 48 / 43 | 10 | 0.566 | 103 | 103 |
| 0.25 | top-K | 0.471 | 32.0 | 0.69 | nan / 0.69 / 0.68 | nan / 32 / 32 | 10 | 0.629 | 103 | 104 |
| 1 | app line | 0.021 | 4516.0 | 0.126 | nan / 0.48 / 0.49 | nan / 58 / 56 | 10 | 0.547 | 29 | 29 |
| 1 | top-K | 0.381 | 32.0 | 0.72 | nan / 0.56 / 0.55 | nan / 32 / 32 | 10 | 0.776 | 54 | 54 |
| 4 | app line | 0.15 | 4516.0 | 0.363 | nan / 0.60 / 0.61 | nan / 99 / 100 | 10 | 0.606 | 54 | 54 |
| 4 | top-K | 0.485 | 128.0 | 0.741 | nan / 0.67 / 0.65 | nan / 128 / 128 | 10 | 0.721 | 54 | 100 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 279 | 20.735 | 0.197 | 0.824 | 0.627 | 0.476 | 1.0 |
| 1 | 279 | 21.237 | 0.129 | 0.805 | 0.676 | 0.4 | 1.0 |
| 4 | 279 | 30.341 | 0.09 | 0.789 | 0.699 | 0.326 | 1.0 |
