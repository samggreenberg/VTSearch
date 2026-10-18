# Keypoint cuts vs `all`: /expscratch/sgreenberg/strongk-4934/pilot-s

## `rows2048`: 66 paired sessions, 33 classes

| readout | 0 clicks | 10 clicks | 25 clicks | 50 clicks |
|---|---|---|---|---|
| ap | -0.462 [-0.562, -0.361] | -0.474 [-0.581, -0.365] | -0.479 [-0.585, -0.372] | -0.481 [-0.588, -0.374] |
| gate_f1 | -0.264 [-0.344, -0.188] | -0.405 [-0.514, -0.297] | -0.470 [-0.589, -0.347] | -0.477 [-0.596, -0.356] |
| gate_fb025 | -0.189 [-0.278, -0.108] | -0.246 [-0.381, -0.116] | -0.317 [-0.459, -0.176] | -0.327 [-0.468, -0.189] |
| gate_fb4 | -0.450 [-0.543, -0.357] | -0.489 [-0.605, -0.369] | -0.524 [-0.636, -0.408] | -0.526 [-0.639, -0.410] |

Retrain, median after a Good 1.99 s (control 6.52), after a Bad 0.39 s (0.44), p90 2.49 s (7.81).
**Pre-registered rule at 25 clicks (lower bounds >= -0.01):** AP -0.585, F1 -0.589 -> **FAIL**
Largest AP losses at 25 clicks: staver/stamp_stampds-00230_0 -0.97, tobacco800/logo_bqz95d00_1 -0.92, spods/stamp_00737_1 -0.91, spods/stamp_00129_1 -0.88, spods/stamp_00514_1 -0.88

## `scale2048`: 66 paired sessions, 33 classes

| readout | 0 clicks | 10 clicks | 25 clicks | 50 clicks |
|---|---|---|---|---|
| ap | -0.060 [-0.125, -0.007] | -0.050 [-0.096, -0.015] | -0.053 [-0.106, -0.010] | -0.048 [-0.099, -0.007] |
| gate_f1 | +0.004 [-0.052, +0.055] | -0.106 [-0.169, -0.052] | -0.098 [-0.181, -0.028] | -0.096 [-0.176, -0.031] |
| gate_fb025 | +0.001 [-0.054, +0.051] | -0.109 [-0.180, -0.044] | -0.094 [-0.171, -0.027] | -0.091 [-0.166, -0.030] |
| gate_fb4 | -0.032 [-0.084, +0.006] | -0.086 [-0.148, -0.035] | -0.094 [-0.179, -0.026] | -0.091 [-0.174, -0.024] |

Retrain, median after a Good 2.13 s (control 6.52), after a Bad 0.38 s (0.44), p90 2.49 s (7.81).
**Pre-registered rule at 25 clicks (lower bounds >= -0.01):** AP -0.106, F1 -0.181 -> **FAIL**
Largest AP losses at 25 clicks: ucsf/logo_bat_leaf -0.53, spods/stamp_00737_1 -0.51, staver/stamp_stampds-00213_1 -0.33, spods/stamp_00514_1 -0.31, tobacco800/logo_asg54f00_1 -0.09

## `strong1024`: 66 paired sessions, 33 classes

| readout | 0 clicks | 10 clicks | 25 clicks | 50 clicks |
|---|---|---|---|---|
| ap | -0.503 [-0.601, -0.404] | -0.480 [-0.593, -0.367] | -0.496 [-0.608, -0.385] | -0.488 [-0.599, -0.374] |
| gate_f1 | -0.333 [-0.449, -0.209] | -0.520 [-0.628, -0.413] | -0.600 [-0.712, -0.487] | -0.616 [-0.726, -0.503] |
| gate_fb025 | -0.247 [-0.375, -0.117] | -0.427 [-0.558, -0.300] | -0.503 [-0.650, -0.357] | -0.496 [-0.638, -0.355] |
| gate_fb4 | -0.548 [-0.659, -0.432] | -0.578 [-0.693, -0.461] | -0.634 [-0.737, -0.527] | -0.647 [-0.753, -0.536] |

Retrain, median after a Good 1.34 s (control 6.52), after a Bad 0.42 s (0.44), p90 1.80 s (7.81).
**Pre-registered rule at 25 clicks (lower bounds >= -0.01):** AP -0.608, F1 -0.712 -> **FAIL**
Largest AP losses at 25 clicks: staver/stamp_stampds-00230_0 -0.96, spods/stamp_00129_1 -0.92, spods/stamp_00577_1 -0.91, spods/stamp_00641_1 -0.90, spods/stamp_00514_1 -0.89

## `strong2048`: 66 paired sessions, 33 classes

| readout | 0 clicks | 10 clicks | 25 clicks | 50 clicks |
|---|---|---|---|---|
| ap | -0.395 [-0.499, -0.294] | -0.358 [-0.471, -0.251] | -0.367 [-0.481, -0.257] | -0.365 [-0.478, -0.254] |
| gate_f1 | -0.219 [-0.333, -0.106] | -0.375 [-0.492, -0.262] | -0.449 [-0.577, -0.322] | -0.450 [-0.579, -0.325] |
| gate_fb025 | -0.137 [-0.261, -0.019] | -0.273 [-0.411, -0.138] | -0.369 [-0.519, -0.223] | -0.367 [-0.515, -0.226] |
| gate_fb4 | -0.417 [-0.530, -0.306] | -0.450 [-0.572, -0.330] | -0.487 [-0.604, -0.371] | -0.485 [-0.606, -0.366] |

Retrain, median after a Good 2.12 s (control 6.52), after a Bad 0.42 s (0.44), p90 2.66 s (7.81).
**Pre-registered rule at 25 clicks (lower bounds >= -0.01):** AP -0.481, F1 -0.577 -> **FAIL**
Largest AP losses at 25 clicks: staver/stamp_stampds-00230_0 -0.97, spods/stamp_00641_1 -0.93, spods/stamp_00612_1 -0.88, spods/stamp_00769_1 -0.85, spods/stamp_00931_1 -0.81

## `strongfrac035`: 66 paired sessions, 33 classes

| readout | 0 clicks | 10 clicks | 25 clicks | 50 clicks |
|---|---|---|---|---|
| ap | -0.382 [-0.488, -0.280] | -0.349 [-0.463, -0.241] | -0.341 [-0.458, -0.230] | -0.342 [-0.459, -0.230] |
| gate_f1 | -0.205 [-0.321, -0.089] | -0.349 [-0.460, -0.242] | -0.420 [-0.543, -0.299] | -0.420 [-0.542, -0.302] |
| gate_fb025 | -0.121 [-0.240, -0.006] | -0.247 [-0.377, -0.122] | -0.333 [-0.478, -0.193] | -0.323 [-0.464, -0.189] |
| gate_fb4 | -0.401 [-0.519, -0.286] | -0.419 [-0.538, -0.305] | -0.447 [-0.565, -0.332] | -0.452 [-0.571, -0.336] |

Retrain, median after a Good 2.46 s (control 6.52), after a Bad 0.41 s (0.44), p90 3.58 s (7.81).
**Pre-registered rule at 25 clicks (lower bounds >= -0.01):** AP -0.458, F1 -0.543 -> **FAIL**
Largest AP losses at 25 clicks: staver/stamp_stampds-00230_0 -0.96, spods/stamp_00641_1 -0.92, spods/stamp_00577_1 -0.90, spods/stamp_00612_1 -0.88, spods/stamp_00769_1 -0.87

## Verdicts

- `rows2048`: FAIL
- `scale2048`: FAIL
- `strong1024`: FAIL
- `strong2048`: FAIL
- `strongfrac035`: FAIL
