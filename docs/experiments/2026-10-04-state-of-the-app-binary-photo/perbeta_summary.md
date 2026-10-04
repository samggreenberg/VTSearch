# Each balance read off its own sessions (#4413)

## The objective

F-beta at the sessions' own beta of the withheld images above the app's threshold (#4427): at 25 and 50 clicks, at the last click before the spot check (`unchecked`), and after the check and its votes; the check's paired effect; the returned set's size.

| beta | runs | F at 25 | F at 50 | F unchecked | F after the check | check's effect | precision after | recall after | returned median, unchecked / after | over 200, unchecked / after | check votes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 702 | 0.415 | 0.508 | 0.585 | 0.623 | +0.037 ± 0.004 | 0.697 | 0.402 | 35 / 26 | 6% / 1% | 20.826 |
| 1 | 702 | 0.292 | 0.418 | 0.488 | 0.522 | +0.034 ± 0.003 | 0.57 | 0.537 | 49 / 47 | 19% / 6% | 21.51 |
| 4 | 702 | 0.352 | 0.486 | 0.563 | 0.599 | +0.036 ± 0.003 | 0.417 | 0.677 | 75 / 80 | 31% / 26% | 30.556 |

## The returned set at each balance (rank frames)

`fbeta` against `oracle_fbeta` (the best any cut of the same ranking reaches; `fb_share` = fbeta / oracle), with `k`, `precision` and `recall`. Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | beta | point | k | precision | recall | fbeta | oracle_fbeta | fb_share | empty | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beta 0.25 | SigLIP binary | 0.25 | text | 32.0 | 0.487 | 0.313 | 0.471 | 0.556 | 0.696 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | 0.25 | 25 | 396.944 | 0.548 | 0.235 | 0.364 | 0.526 | 0.512 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | 0.25 | 50 | 382.047 | 0.543 | 0.4 | 0.466 | 0.582 | 0.6 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | 0.25 | final | 74.394 | 0.618 | 0.434 | 0.571 | 0.668 | 0.717 | 0.0 | 720 |
| beta 0.25 | SigLIP binary | 0.25 | ceiling | 32.0 | 0.635 | 0.409 | 0.615 | 0.713 | 0.81 | 0.0 | 720 |
| beta 1 | SigLIP binary | 1.0 | text | 32.0 | 0.487 | 0.313 | 0.38 | 0.45 | 0.714 | 0.0 | 720 |
| beta 1 | SigLIP binary | 1.0 | 25 | 852.289 | 0.489 | 0.347 | 0.257 | 0.386 | 0.454 | 0.0 | 720 |
| beta 1 | SigLIP binary | 1.0 | 50 | 826.715 | 0.461 | 0.507 | 0.383 | 0.457 | 0.606 | 0.0 | 720 |
| beta 1 | SigLIP binary | 1.0 | final | 264.176 | 0.511 | 0.562 | 0.476 | 0.535 | 0.739 | 0.0 | 720 |
| beta 1 | SigLIP binary | 1.0 | ceiling | 32.0 | 0.635 | 0.409 | 0.496 | 0.578 | 0.838 | 0.0 | 720 |
| beta 4 | SigLIP binary | 4.0 | text | 128.0 | 0.207 | 0.531 | 0.486 | 0.566 | 0.739 | 0.0 | 720 |
| beta 4 | SigLIP binary | 4.0 | 25 | 1227.139 | 0.366 | 0.45 | 0.314 | 0.479 | 0.49 | 0.0 | 720 |
| beta 4 | SigLIP binary | 4.0 | 50 | 1404.897 | 0.35 | 0.611 | 0.447 | 0.545 | 0.657 | 0.0 | 720 |
| beta 4 | SigLIP binary | 4.0 | final | 716.901 | 0.395 | 0.683 | 0.549 | 0.625 | 0.781 | 0.0 | 720 |
| beta 4 | SigLIP binary | 4.0 | ceiling | 128.0 | 0.266 | 0.681 | 0.623 | 0.692 | 0.864 | 0.0 | 720 |

## The early dip

The returned set's share of the best cut, mean over runs: the typed query's set at click 0, the detector's lowest, and the click by which it is back at the typed query's level.

| beta | typed query | lowest | at click | at 10 | at 25 | back at the typed query's level by click |
|---|---|---|---|---|---|---|
| 0.25 | 0.696 | 0.266 | 4 | 0.333 | 0.512 | 100 |
| 1 | 0.714 | 0.144 | 7 | 0.192 | 0.454 | 95 |
| 4 | 0.739 | 0.133 | 7 | 0.203 | 0.49 | 85 |

## The spot check

`truth` is the share right of the set the check audited (#4474); `covered` is how often the range held it.

| beta | checks | votes | range lo | range hi | width | truth | covered |
|---|---|---|---|---|---|---|---|
| 0.25 | 702 | 20.826 | 0.175 | 0.822 | 0.647 | 0.453 | 1.0 |
| 1 | 702 | 21.51 | 0.117 | 0.806 | 0.688 | 0.38 | 1.0 |
| 4 | 702 | 30.556 | 0.075 | 0.787 | 0.711 | 0.29 | 1.0 |
