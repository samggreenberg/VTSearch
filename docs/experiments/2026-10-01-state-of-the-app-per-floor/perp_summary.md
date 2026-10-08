# The returned set at each P, each read off its own sessions (#4408)

`precision` against the target P (`gap` = precision - P), `meets` the share of runs at or above P, `recall` against `oracle_recall` (the most any cut of the same ranking returns at or above P; `share` = recall / oracle recall). Points: text sort, 25 and 50 clicks, the end, full labels.

| sessions_at | arm | floor | point | k | precision | gap | meets | recall | oracle_recall | share | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | SigLIP binary | 0.1 | text | 128.0 | 0.207 | 0.107 | 0.683 | 0.53 | 0.58 | 0.915 | 1440 |
| 10% | SigLIP binary | 0.1 | 25 | 122.01 | 0.182 | 0.082 | 0.604 | 0.435 | 0.482 | 0.903 | 1440 |
| 10% | SigLIP binary | 0.1 | 50 | 123.84 | 0.205 | 0.105 | 0.651 | 0.516 | 0.556 | 0.928 | 1440 |
| 10% | SigLIP binary | 0.1 | final | 124.545 | 0.247 | 0.147 | 0.792 | 0.614 | 0.673 | 0.912 | 1440 |
| 10% | SigLIP binary | 0.1 | ceiling | 128.0 | 0.265 | 0.165 | 0.915 | 0.68 | 0.764 | 0.89 | 1440 |
| 50% (read at 10%) | SigLIP binary | 0.1 | final | 124.269 | 0.248 | 0.148 | 0.793 | 0.613 | 0.673 | 0.911 | 1440 |
| 50% | SigLIP binary | 0.5 | text | 32.0 | 0.487 | -0.013 | 0.506 | 0.314 | 0.416 | 0.754 | 1440 |
| 50% | SigLIP binary | 0.5 | 25 | 30.326 | 0.439 | -0.061 | 0.432 | 0.272 | 0.334 | 0.816 | 1440 |
| 50% | SigLIP binary | 0.5 | 50 | 30.833 | 0.508 | 0.008 | 0.501 | 0.324 | 0.422 | 0.769 | 1440 |
| 50% | SigLIP binary | 0.5 | final | 30.99 | 0.6 | 0.1 | 0.609 | 0.379 | 0.504 | 0.753 | 1440 |
| 50% | SigLIP binary | 0.5 | ceiling | 32.0 | 0.631 | 0.131 | 0.623 | 0.407 | 0.535 | 0.761 | 1440 |
| 90% | SigLIP binary | 0.9 | text | 32.0 | 0.487 | -0.413 | 0.226 | 0.314 | 0.27 | 1.16 | 1440 |
| 90% | SigLIP binary | 0.9 | 25 | 28.637 | 0.454 | -0.446 | 0.234 | 0.266 | 0.24 | 1.109 | 1440 |
| 90% | SigLIP binary | 0.9 | 50 | 29.595 | 0.512 | -0.388 | 0.32 | 0.322 | 0.325 | 0.99 | 1440 |
| 90% | SigLIP binary | 0.9 | final | 30.078 | 0.608 | -0.292 | 0.377 | 0.375 | 0.379 | 0.991 | 1440 |
| 90% | SigLIP binary | 0.9 | ceiling | 32.0 | 0.631 | -0.269 | 0.358 | 0.407 | 0.388 | 1.049 | 1440 |
| 50% (read at 90%) | SigLIP binary | 0.9 | final | 30.122 | 0.608 | -0.292 | 0.378 | 0.376 | 0.378 | 0.994 | 1440 |

## Each P's sessions at the end

| sessions_at | runs | final_ap | ceiling_ap | goods_found | check_confirmed |
|---|---|---|---|---|---|
| 10% | 1440 | 0.513 | 0.55 | 19.301 | 0.911 |
| 50% | 1440 | 0.514 | 0.55 | 19.736 | 0.476 |
| 90% | 1440 | 0.514 | 0.55 | 19.802 | 0.246 |
