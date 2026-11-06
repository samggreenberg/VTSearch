# The typed-query hold's count, where the app says stop and on the way there (#5556)

Every cell: the count minus 30 (the shipped hold), mean ± SE clustered on the class. Positive means the count beats 30. `at_stop` is the objective where the app says stop (the last click if it never does); `path` is each session's mean over votes 1 to there.

## The stop click (the same at every count: the stop rule reads only the head)

| dataset | preset | scope | runs | fired | KM median | quartiles (fired) | labels at stop (median, min) | stop below 20 | stop below 25 | stop below 30 | stop below 35 | stop below 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| screens | b025 | all | 110 | 97% | 52 | 47 / 51 / 58 | 51, 42 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| screens | b1 | all | 110 | 97% | 52 | 48 / 52 / 59 | 52, 42 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| screens | b4 | all | 110 | 98% | 60 | 55 / 60 / 68 | 60, 50 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## The objective where the app says stop (`at_stop`)

| dataset | preset | scope | at 30 | h=0 | h=20 | h=25 | h=35 | h=40 | h=50 | h=60 | best h (gain) | within 2 SE of best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| screens | b025 | all | 0.6171 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | -0.0641 ± 0.0152 | -0.1164 ± 0.0227 | 0 (+0.0000) | 0-44 |
| screens | b1 | all | 0.4330 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | -0.1105 ± 0.0311 | -0.1954 ± 0.0433 | 0 (+0.0000) | 0-46 |
| screens | b4 | all | 0.5504 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | -0.1564 ± 0.0569 | 0 (+0.0000) | 0-53 |

## The mean objective on the way to the stop (`path`)

| dataset | preset | scope | at 30 | h=0 | h=20 | h=25 | h=35 | h=40 | h=50 | h=60 | best h (gain) | within 2 SE of best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| screens | b025 | all | 0.5263 | +0.0444 ± 0.0116 | +0.0194 ± 0.0060 | +0.0096 ± 0.0030 | -0.0100 ± 0.0029 | -0.0220 ± 0.0051 | -0.0443 ± 0.0086 | -0.0535 ± 0.0099 | 0 (+0.0444) | 0-10 |
| screens | b1 | all | 0.2940 | +0.0762 ± 0.0195 | +0.0384 ± 0.0098 | +0.0203 ± 0.0054 | -0.0194 ± 0.0042 | -0.0417 ± 0.0093 | -0.0819 ± 0.0173 | -0.0962 ± 0.0188 | 0 (+0.0762) | 0-7 |
| screens | b4 | all | 0.4044 | +0.0562 ± 0.0275 | +0.0304 ± 0.0130 | +0.0168 ± 0.0067 | -0.0165 ± 0.0065 | -0.0353 ± 0.0131 | -0.0718 ± 0.0251 | -0.1026 ± 0.0348 | 0 (+0.0562) | 0-25 |

## The sessions a count moves at the stop (stop between the count and 30 labels)

| dataset | preset | scope | h=20: share, gain there | h=25: share, gain there | h=35: share, gain there | h=40: share, gain there |
|---|---|---|---|---|---|---|
| screens | b025 | all | 0.0%, +nan | 0.0%, +nan | 0.0%, +nan | 0.0%, +nan |
| screens | b1 | all | 0.0%, +nan | 0.0%, +nan | 0.0%, +nan | 0.0%, +nan |
| screens | b4 | all | 0.0%, +nan | 0.0%, +nan | 0.0%, +nan | 0.0%, +nan |

## Head's set minus the query's, by label count

| dataset | preset | scope | positive for good from | 2 SE covers zero (labels) | counts near zero |
|---|---|---|---|---|---|
| screens | b025 | all | 1 | 1-8 | 8 |
| screens | b1 | all | 7 | 1-6 | 6 |
| screens | b4 | all | 7 | 1-24 | 22 |

## One count for every preset, against one a beta

| read | best single count | its gain over 30 | per-beta bests minus it | h=20 | h=25 | h=30 | h=35 | h=40 |
|---|---|---|---|---|---|---|---|---|
| at_stop | 0 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| path | 0 | +0.0589 ± 0.0180 | +0.0000 ± 0.0000 | +0.0294 | +0.0156 | +0.0000 | -0.0153 | -0.0330 |

## The decision rule, applied

| dataset | preset | scope | count | why |
|---|---|---|---|---|
| screens | b025 | all | 0 | the path: ties 30 at the stop, and its path beats 30's by 2 SE (+0.0444 ± 0.0116) |
| screens | b1 | all | 0 | the path: ties 30 at the stop, and its path beats 30's by 2 SE (+0.0762 ± 0.0195) |
| screens | b4 | all | 0 | the path: ties 30 at the stop, and its path beats 30's by 2 SE (+0.0562 ± 0.0275) |
