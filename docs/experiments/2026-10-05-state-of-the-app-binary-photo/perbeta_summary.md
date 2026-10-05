# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 563 | 0.435 | 0.514 | 0.611 | 0.632 | +0.022 ± 0.004 | 0.723 | 0.381 | 27 / 20 | 6% / 0% | 20.631 |
| 1 | 563 | 0.34 | 0.421 | 0.499 | 0.524 | +0.025 ± 0.002 | 0.593 | 0.522 | 45 / 44 | 12% / 5% | 21.146 |
| 4 | 563 | 0.425 | 0.484 | 0.564 | 0.597 | +0.033 ± 0.004 | 0.43 | 0.671 | 69 / 75 | 30% / 28% | 30.409 |

## The returned set at each balance (rank frames)

One rule per row (owner, 2026-10-04: apples to apples). `app line`: each sort's own line in the app - the text sort's blind GMM cut, the detector's labels line (#4452), the ceiling's Find line from its full labels (#4486). `top-K`: set-constant at the old cap (32 at beta <= 1, 128 above) on every sort. Compare the text sort and the detector within a rule, never across. `fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | rule | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | SigLIP binary | app line | 0.25 | text | 4507.481 | 0.011 | 0.912 | 0.011 | 0.557 | 0.077 | 0.0 | 576 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 25 | 1172.22 | 0.459 | 0.443 | 0.38 | 0.53 | 0.538 | 0.0 | 573 |
| beta 0.25 | SigLIP binary | app line | 0.25 | 50 | 904.218 | 0.515 | 0.479 | 0.457 | 0.566 | 0.601 | 0.0 | 510 |
| beta 0.25 | SigLIP binary | app line | 0.25 | final | 194.427 | 0.67 | 0.414 | 0.597 | 0.678 | 0.747 | 0.0 | 576 |
| beta 0.25 | SigLIP binary | app line | 0.25 | ceiling | 117.059 | 0.469 | 0.662 | 0.468 | 0.713 | 0.586 | 0.0 | 576 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | text | 32.0 | 0.487 | 0.312 | 0.471 | 0.557 | 0.696 | 0.0 | 576 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 25 | 32.0 | 0.433 | 0.278 | 0.419 | 0.53 | 0.62 | 0.0 | 573 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | 50 | 32.0 | 0.498 | 0.32 | 0.482 | 0.566 | 0.678 | 0.0 | 510 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | final | 32.0 | 0.602 | 0.387 | 0.583 | 0.678 | 0.754 | 0.0 | 576 |
| beta 0.25 | SigLIP binary | top-K | 0.25 | ceiling | 32.0 | 0.634 | 0.408 | 0.613 | 0.713 | 0.809 | 0.0 | 576 |
| beta 1 | SigLIP binary | app line | 1.0 | text | 4507.481 | 0.011 | 0.912 | 0.021 | 0.45 | 0.127 | 0.0 | 576 |
| beta 1 | SigLIP binary | app line | 1.0 | 25 | 1504.157 | 0.36 | 0.552 | 0.298 | 0.388 | 0.572 | 0.0 | 573 |
| beta 1 | SigLIP binary | app line | 1.0 | 50 | 1369.314 | 0.445 | 0.587 | 0.382 | 0.453 | 0.618 | 0.0 | 510 |
| beta 1 | SigLIP binary | app line | 1.0 | final | 350.031 | 0.546 | 0.54 | 0.488 | 0.541 | 0.767 | 0.0 | 576 |
| beta 1 | SigLIP binary | app line | 1.0 | ceiling | 295.767 | 0.369 | 0.777 | 0.438 | 0.578 | 0.665 | 0.0 | 576 |
| beta 1 | SigLIP binary | top-K | 1.0 | text | 32.0 | 0.487 | 0.312 | 0.38 | 0.45 | 0.717 | 0.0 | 576 |
| beta 1 | SigLIP binary | top-K | 1.0 | 25 | 32.0 | 0.434 | 0.278 | 0.338 | 0.388 | 0.711 | 0.0 | 573 |
| beta 1 | SigLIP binary | top-K | 1.0 | 50 | 32.0 | 0.499 | 0.32 | 0.389 | 0.453 | 0.717 | 0.0 | 510 |
| beta 1 | SigLIP binary | top-K | 1.0 | final | 32.0 | 0.601 | 0.387 | 0.469 | 0.541 | 0.794 | 0.0 | 576 |
| beta 1 | SigLIP binary | top-K | 1.0 | ceiling | 32.0 | 0.634 | 0.408 | 0.495 | 0.578 | 0.837 | 0.0 | 576 |
| beta 4 | SigLIP binary | app line | 4.0 | text | 4507.481 | 0.011 | 0.912 | 0.15 | 0.566 | 0.362 | 0.0 | 576 |
| beta 4 | SigLIP binary | app line | 4.0 | 25 | 1981.894 | 0.233 | 0.655 | 0.386 | 0.481 | 0.683 | 0.0 | 573 |
| beta 4 | SigLIP binary | app line | 4.0 | 50 | 2084.085 | 0.351 | 0.694 | 0.446 | 0.54 | 0.701 | 0.0 | 508 |
| beta 4 | SigLIP binary | app line | 4.0 | final | 779.646 | 0.417 | 0.679 | 0.553 | 0.627 | 0.801 | 0.0 | 576 |
| beta 4 | SigLIP binary | app line | 4.0 | ceiling | 860.738 | 0.278 | 0.885 | 0.607 | 0.692 | 0.829 | 0.0 | 576 |
| beta 4 | SigLIP binary | top-K | 4.0 | text | 128.0 | 0.208 | 0.531 | 0.486 | 0.566 | 0.741 | 0.0 | 576 |
| beta 4 | SigLIP binary | top-K | 4.0 | 25 | 128.0 | 0.173 | 0.442 | 0.405 | 0.481 | 0.696 | 0.0 | 573 |
| beta 4 | SigLIP binary | top-K | 4.0 | 50 | 128.0 | 0.2 | 0.511 | 0.467 | 0.54 | 0.728 | 0.0 | 508 |
| beta 4 | SigLIP binary | top-K | 4.0 | final | 128.0 | 0.242 | 0.62 | 0.567 | 0.627 | 0.826 | 0.0 | 576 |
| beta 4 | SigLIP binary | top-K | 4.0 | ceiling | 128.0 | 0.266 | 0.681 | 0.623 | 0.692 | 0.865 | 0.0 | 576 |

## The early dip

The detector's returned set against the text sort's, the same rule on both sides (#4384): under `app line` the text sort's blind GMM cut against the detector's labels line; under `top-K` the old cap on both. F-beta, what each returned, and the click by which the detector's F-beta reaches the text sort's; then the share of the best cut, which rises with the clicks.

| beta | rule | text sort: F | text sort: returned | text sort: share of best | detector: F at 5 / 10 / 25 | detector: returned at 5 / 10 / 25 | detector F >= text sort's by click | lowest share | at click | share >= text sort's by click |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | app line | 0.011 | 4513.999 | 0.077 | 0.41 / 0.46 / 0.44 | 15 / 25 / 31 | 2 | 0.118 | 2 | 2 |
| 0.25 | top-K | 0.471 | 32.0 | 0.696 | 0.46 / 0.50 / 0.48 | 32 / 32 / 32 | 6 | 0.324 | 3 | 34 |
| 1 | app line | 0.021 | 4513.999 | 0.127 | 0.28 / 0.34 / 0.34 | 29 / 40 / 54 | 4 | 0.063 | 2 | 4 |
| 1 | top-K | 0.38 | 32.0 | 0.717 | 0.37 / 0.40 / 0.39 | 32 / 32 / 32 | 6 | 0.439 | 29 | 30 |
| 4 | app line | 0.15 | 4513.999 | 0.362 | 0.35 / 0.42 / 0.43 | 50 / 76 / 133 | 5 | 0.023 | 2 | 5 |
| 4 | top-K | 0.486 | 128.0 | 0.741 | 0.45 / 0.48 / 0.46 | 128 / 128 / 128 | 37 | 0.278 | 29 | 30 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 563 | 20.631 | 0.177 | 0.818 | 0.64 | 0.451 | 1.0 |
| 1 | 563 | 21.146 | 0.116 | 0.8 | 0.684 | 0.374 | 1.0 |
| 4 | 563 | 30.409 | 0.076 | 0.781 | 0.705 | 0.286 | 1.0 |
