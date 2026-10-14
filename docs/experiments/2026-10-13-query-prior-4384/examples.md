# #4384 examples: classes where the head started from the typed query helps or hurts most


## The ten classes where it loses most (all presets pooled)

| class | typed query | runs | objective 1-150 | head's set 1-25 | Goods @150 (of control's) | query AP |
|---|---|---|---|---|---|---|
| stop sign | "a stop sign" | 45 | -0.030 | -0.073 | +0.3 (30) | 0.53 |
| traffic light | "a traffic light" | 45 | -0.021 | -0.059 | +2.1 (31) | 0.47 |
| snowboard | "a snowboard" | 45 | -0.018 | +0.048 | +0.2 (46) | 0.73 |
| boat | "a boat on the water" | 45 | -0.013 | +0.087 | +1.2 (36) | 0.49 |
| mouse | "a computer mouse" | 45 | -0.011 | +0.021 | +0.1 (41) | 0.45 |
| person | "a person" | 45 | -0.007 | +0.001 | -1.0 (26) | 0.10 |
| orange | "an orange fruit" | 30 | -0.003 | +0.032 | -4.5 (44) | 0.71 |
| skis | "a pair of skis" | 45 | +0.001 | +0.018 | +0.2 (48) | 0.60 |
| parking meter | "a parking meter" | 45 | +0.001 | -0.004 | +2.1 (29) | 0.50 |
| tv | "a television screen" | 45 | +0.001 | +0.023 | +0.5 (24) | 0.19 |

## The ten classes where it gains most

| class | typed query | runs | objective 1-150 | head's set 1-25 | Goods @150 (of control's) | query AP |
|---|---|---|---|---|---|---|
| chair | "a chair" | 45 | +0.071 | +0.032 | +3.6 (6) | 0.06 |
| fork | "a fork" | 45 | +0.047 | +0.043 | +3.2 (24) | 0.15 |
| sink | "a sink" | 45 | +0.043 | +0.039 | +2.8 (27) | 0.19 |
| single serving drinking vessel | "a cup or glass" | 45 | +0.040 | +0.040 | +1.8 (25) | 0.11 |
| dining table | "a dining table" | 45 | +0.040 | +0.058 | +2.4 (16) | 0.11 |
| book | "a book" | 45 | +0.039 | +0.063 | +0.4 (20) | 0.24 |
| toothbrush | "a toothbrush" | 45 | +0.035 | +0.040 | -4.4 (38) | 0.59 |
| bench | "a bench" | 45 | +0.034 | +0.091 | +0.8 (18) | 0.23 |
| vase or potted plant | "a vase or potted plant" | 45 | +0.033 | +0.066 | +1.1 (17) | 0.27 |
| knife | "a knife" | 45 | +0.033 | +0.022 | +2.7 (10) | 0.06 |

42 of 49 classes gain on the session mean; 18 find fewer Goods by vote 150.


## The eight runs where it loses most

| preset | cell | seed | objective 1-150 | Goods @150 | first Done (control, arm) |
|---|---|---|---|---|---|
| b025 | stop sign@medium | 3 | -0.199 | +0 of 36 | 64, 61 |
| b025 | laptop@large | 0 | -0.159 | +0 of 53 | 82, 59 |
| b025 | knife@medium | 0 | -0.150 | +11 of 5 | inf, 86 |
| b025 | boat@small | 3 | -0.146 | +1 of 19 | 49, 51 |
| b025 | bird@small | 1 | -0.145 | +5 of 14 | 92, 68 |
| b025 | mouse@medium | 2 | -0.144 | +2 of 49 | 56, 56 |
| b025 | bowl@large | 3 | -0.135 | -1 of 21 | 60, 59 |
| b025 | stop sign@medium | 1 | -0.134 | +2 of 29 | 83, 60 |
