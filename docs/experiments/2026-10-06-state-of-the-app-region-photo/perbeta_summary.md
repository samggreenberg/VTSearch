# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 279 | 0.435 | 0.542 | 0.686 | 0.733 | +0.048 ± 0.007 | 0.812 | 0.475 | 35 / 30 | 0% / 1% | 20.735 |
| 1 | 279 | 0.372 | 0.467 | 0.599 | 0.633 | +0.035 ± 0.005 | 0.671 | 0.646 | 49 / 48 | 4% / 3% | 21.237 |
| 4 | 279 | 0.504 | 0.567 | 0.679 | 0.725 | +0.046 ± 0.006 | 0.453 | 0.822 | 91 / 92 | 27% / 25% | 30.341 |

## The returned set through the session

Per beta, the precision and recall of the same withheld set from the typed query (the text sort at its own blind GMM cut, before any vote), through 25, 50, 100 and 150 clicks, to after the check (#4519): the path a preset's returned set takes as the user clicks (`precision_recall_path.png`, `precision_recall_path.csv`). Means over the trained runs with a line by that click; `returned` is the median size.

| beta | point | precision | recall | fbeta | returned, median | runs |
|---|---|---|---|---|---|---|
| 0.25 | typed query | 0.541 | 0.262 | 0.483 | 18.0 | 279 |
| 0.25 | 25 | 0.476 | 0.365 | 0.435 | 21.998 | 279 |
| 0.25 | 50 | 0.577 | 0.422 | 0.542 | 29.997 | 279 |
| 0.25 | 100 | 0.715 | 0.466 | 0.648 | 31.999 | 279 |
| 0.25 | 150 | 0.753 | 0.488 | 0.686 | 35.0 | 279 |
| 0.25 | after the check | 0.812 | 0.475 | 0.733 | 30.003 | 279 |
| 1 | typed query | 0.378 | 0.452 | 0.376 | 51.001 | 279 |
| 1 | 25 | 0.372 | 0.467 | 0.372 | 49.996 | 279 |
| 1 | 50 | 0.479 | 0.513 | 0.467 | 47.003 | 279 |
| 1 | 100 | 0.607 | 0.596 | 0.566 | 46.994 | 279 |
| 1 | 150 | 0.625 | 0.63 | 0.599 | 49.0 | 279 |
| 1 | after the check | 0.671 | 0.646 | 0.633 | 47.997 | 279 |
| 4 | typed query | 0.17 | 0.601 | 0.483 | 203.997 | 279 |
| 4 | 25 | 0.236 | 0.603 | 0.504 | 186.001 | 279 |
| 4 | 50 | 0.354 | 0.639 | 0.567 | 109.002 | 279 |
| 4 | 100 | 0.433 | 0.739 | 0.649 | 92.996 | 279 |
| 4 | 150 | 0.44 | 0.781 | 0.679 | 91.005 | 279 |
| 4 | after the check | 0.453 | 0.822 | 0.725 | 92.005 | 279 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | DINOv3 region | app line | 0.25 | text | 23.708 | 0.525 | 0.255 | 0.468 | 0.561 | 0.654 | 0.003 | 288 |
| beta 0.25 | DINOv3 region | app line | 0.25 | 25 | 42.649 | 0.462 | 0.354 | 0.423 | 0.567 | 0.597 | 0.003 | 288 |
| beta 0.25 | DINOv3 region | app line | 0.25 | 50 | 34.709 | 0.56 | 0.414 | 0.53 | 0.623 | 0.671 | 0.003 | 278 |
| beta 0.25 | DINOv3 region | app line | 0.25 | final | 32.233 | 0.731 | 0.473 | 0.665 | 0.756 | 0.782 | 0.003 | 288 |
| beta 0.25 | DINOv3 region | app line | 0.25 | ceiling | 180.771 | 0.433 | 0.859 | 0.442 | 0.815 | 0.496 | 0.0 | 144 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | text | 32.0 | 0.486 | 0.314 | 0.471 | 0.561 | 0.69 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | 25 | 32.0 | 0.488 | 0.315 | 0.472 | 0.567 | 0.688 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | 50 | 32.0 | 0.566 | 0.366 | 0.548 | 0.623 | 0.726 | 0.0 | 278 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | final | 32.0 | 0.705 | 0.456 | 0.682 | 0.756 | 0.816 | 0.0 | 288 |
| beta 0.25 | DINOv3 region | top-K | 0.25 | ceiling | 32.0 | 0.759 | 0.489 | 0.735 | 0.815 | 0.876 | 0.0 | 144 |
| beta 1 | DINOv3 region | app line | 1.0 | text | 63.42 | 0.366 | 0.438 | 0.365 | 0.45 | 0.694 | 0.0 | 288 |
| beta 1 | DINOv3 region | app line | 1.0 | 25 | 74.422 | 0.362 | 0.453 | 0.362 | 0.451 | 0.683 | 0.0 | 287 |
| beta 1 | DINOv3 region | app line | 1.0 | 50 | 65.111 | 0.468 | 0.498 | 0.457 | 0.519 | 0.739 | 0.0 | 280 |
| beta 1 | DINOv3 region | app line | 1.0 | final | 73.118 | 0.606 | 0.611 | 0.58 | 0.639 | 0.819 | 0.0 | 288 |
| beta 1 | DINOv3 region | app line | 1.0 | ceiling | 319.354 | 0.37 | 0.911 | 0.468 | 0.687 | 0.61 | 0.0 | 144 |
| beta 1 | DINOv3 region | top-K | 1.0 | text | 32.0 | 0.486 | 0.314 | 0.381 | 0.45 | 0.72 | 0.0 | 288 |
| beta 1 | DINOv3 region | top-K | 1.0 | 25 | 32.0 | 0.488 | 0.315 | 0.382 | 0.451 | 0.722 | 0.0 | 287 |
| beta 1 | DINOv3 region | top-K | 1.0 | 50 | 32.0 | 0.567 | 0.366 | 0.444 | 0.519 | 0.729 | 0.0 | 280 |
| beta 1 | DINOv3 region | top-K | 1.0 | final | 32.0 | 0.705 | 0.456 | 0.553 | 0.639 | 0.797 | 0.0 | 288 |
| beta 1 | DINOv3 region | top-K | 1.0 | ceiling | 32.0 | 0.759 | 0.489 | 0.593 | 0.687 | 0.863 | 0.0 | 144 |
| beta 4 | DINOv3 region | app line | 4.0 | text | 288.312 | 0.166 | 0.583 | 0.469 | 0.566 | 0.74 | 0.0 | 288 |
| beta 4 | DINOv3 region | app line | 4.0 | 25 | 270.547 | 0.23 | 0.585 | 0.489 | 0.567 | 0.76 | 0.0 | 287 |
| beta 4 | DINOv3 region | app line | 4.0 | 50 | 198.941 | 0.353 | 0.611 | 0.551 | 0.617 | 0.787 | 0.0 | 273 |
| beta 4 | DINOv3 region | app line | 4.0 | final | 295.701 | 0.427 | 0.757 | 0.659 | 0.722 | 0.842 | 0.0 | 288 |
| beta 4 | DINOv3 region | app line | 4.0 | ceiling | 605.431 | 0.308 | 0.948 | 0.701 | 0.796 | 0.852 | 0.0 | 144 |
| beta 4 | DINOv3 region | top-K | 4.0 | text | 128.0 | 0.207 | 0.531 | 0.485 | 0.566 | 0.741 | 0.0 | 288 |
| beta 4 | DINOv3 region | top-K | 4.0 | 25 | 128.0 | 0.206 | 0.529 | 0.484 | 0.567 | 0.738 | 0.0 | 287 |
| beta 4 | DINOv3 region | top-K | 4.0 | 50 | 128.0 | 0.229 | 0.588 | 0.538 | 0.617 | 0.751 | 0.0 | 273 |
| beta 4 | DINOv3 region | top-K | 4.0 | final | 128.0 | 0.281 | 0.721 | 0.659 | 0.722 | 0.841 | 0.0 | 288 |
| beta 4 | DINOv3 region | top-K | 4.0 | ceiling | 128.0 | 0.315 | 0.805 | 0.737 | 0.796 | 0.914 | 0.0 | 144 |

## The early dip

The detector's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's blind GMM cut against the detector's labels line; under `top-K` the old cap on both. F-beta, what each returned, and the click by which the detector's F-beta reaches the text sort's; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | detector: F at 5 / 10 / 25 | detector: returned at 5 / 10 / 25 | detector F >= text sort's by click | lowest share | at click | share >= text sort's by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.468 | 17.998 | 0.654 | nan / 0.55 / 0.57 | nan / 48 / 43 | 10 | 0.566 | 103 | 104 |
| 0.25 | top-K | 0.471 | 32.0 | 0.69 | nan / 0.69 / 0.68 | nan / 32 / 32 | 10 | 0.629 | 103 | 104 |
| 1 | app line | 0.365 | 49.996 | 0.694 | nan / 0.48 / 0.49 | nan / 58 / 56 | 10 | 0.547 | 29 | 50 |
| 1 | top-K | 0.381 | 32.0 | 0.72 | nan / 0.56 / 0.55 | nan / 32 / 32 | 10 | 0.776 | 54 | 54 |
| 4 | app line | 0.469 | 203.501 | 0.74 | nan / 0.60 / 0.61 | nan / 99 / 100 | 10 | 0.606 | 54 | 100 |
| 4 | top-K | 0.485 | 128.0 | 0.741 | nan / 0.67 / 0.65 | nan / 128 / 128 | 10 | 0.721 | 54 | 100 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 279 | 20.735 | 0.197 | 0.824 | 0.627 | 0.476 | 1.0 |
| 1 | 279 | 21.237 | 0.129 | 0.805 | 0.676 | 0.4 | 1.0 |
| 4 | 279 | 30.341 | 0.09 | 0.789 | 0.699 | 0.326 | 1.0 |
