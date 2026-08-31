# Selection-bias sweep: summary tables

224 cells (arm x seed x votes x policy); budget violation rate at inclusion >= 0: uniform 0.632, autopilot 0.558.

## Conformal budget compliance (inclusion >= 0)

| arm group | policy | cells | FNR (mean) | alpha cap (mean) | violation rate | FNR excess (mean) | FNR excess (p90) |
|---|---|---:|---:|---:|---:|---:|---:|
| agnews | autopilot | 384 | 0.152 | 0.069 | 0.562 | 0.118 | 0.326 |
| agnews | uniform | 384 | 0.118 | 0.069 | 0.646 | 0.079 | 0.245 |
| synth:easy | autopilot | 96 | 0.038 | 0.069 | 0.260 | 0.022 | 0.135 |
| synth:easy | uniform | 96 | 0.074 | 0.069 | 0.458 | 0.055 | 0.159 |
| synth:hard | autopilot | 96 | 0.311 | 0.069 | 0.802 | 0.251 | 0.692 |
| synth:hard | uniform | 96 | 0.256 | 0.069 | 0.698 | 0.206 | 0.606 |
| synth:medium | autopilot | 96 | 0.189 | 0.069 | 0.594 | 0.150 | 0.555 |
| synth:medium | uniform | 96 | 0.204 | 0.069 | 0.688 | 0.165 | 0.526 |

## Violation rate by inclusion value

| inclusion | alpha cap | violation (uniform) | violation (autopilot) | FNR excess (uniform) | FNR excess (autopilot) |
|---:|---:|---:|---:|---:|---:|
| +0 | 0.2500 | 0.214 | 0.286 | 0.047 | 0.082 |
| +1 | 0.1250 | 0.411 | 0.357 | 0.081 | 0.109 |
| +3 | 0.0312 | 0.607 | 0.554 | 0.116 | 0.136 |
| +5 | 0.0078 | 0.759 | 0.652 | 0.127 | 0.145 |
| +7 | 0.0020 | 0.902 | 0.759 | 0.131 | 0.148 |
| +10 | 0.0002 | 0.902 | 0.741 | 0.133 | 0.149 |

## Operating point at the default (inclusion 0)

| arm group | policy | recall | precision | FPR | n included / pool |
|---|---|---:|---:|---:|---:|
| agnews | autopilot | 0.821 | 0.644 | 0.237 | 0.383 |
| agnews | uniform | 0.858 | 0.664 | 0.197 | 0.362 |
| synth:easy | autopilot | 0.953 | 0.954 | 0.014 | 0.108 |
| synth:easy | uniform | 0.926 | 0.999 | 0.000 | 0.093 |
| synth:hard | autopilot | 0.590 | 0.556 | 0.115 | 0.163 |
| synth:hard | uniform | 0.702 | 0.359 | 0.167 | 0.220 |
| synth:medium | autopilot | 0.761 | 0.686 | 0.050 | 0.121 |
| synth:medium | uniform | 0.781 | 0.587 | 0.073 | 0.144 |

## Threshold placement vs oracle (inclusion 0)

| arm group | policy | threshold - oracle (mean) | (p90) |
|---|---|---:|---:|
| agnews | autopilot | 0.032 | 0.136 |
| agnews | uniform | 0.032 | 0.129 |
| synth:easy | autopilot | -0.063 | 0.030 |
| synth:easy | uniform | 0.041 | 0.090 |
| synth:hard | autopilot | 0.060 | 0.142 |
| synth:hard | uniform | 0.026 | 0.066 |
| synth:medium | autopilot | 0.066 | 0.109 |
| synth:medium | uniform | 0.049 | 0.102 |

### Signed threshold error by vote count

| votes | threshold - oracle (uniform) | (autopilot) |
|---:|---:|---:|
| 12 | 0.019 | 0.017 |
| 24 | 0.018 | 0.015 |
| 50 | 0.056 | 0.049 |
| 100 | 0.046 | 0.029 |

## Safe-blend mitigation reach (autopilot policy)

| votes | violation (conformal) | violation (blend) | FNR excess (conformal) | FNR excess (blend) |
|---:|---:|---:|---:|---:|
| 12 | 0.798 | 0.804 | 0.336 | 0.167 |
| 24 | 0.577 | 0.577 | 0.099 | 0.099 |
| 50 | 0.554 | 0.554 | 0.069 | 0.069 |
| 100 | 0.304 | 0.304 | 0.007 | 0.007 |

## Vote-set composition drift

| arm group | policy | vote pos frac (mean) | cal q25 - pool q25 (mean) | (p90) |
|---|---|---:|---:|---:|
| agnews | autopilot | 0.29 | -0.054 | 0.011 |
| agnews | uniform | 0.33 | -0.027 | 0.016 |
| synth:easy | autopilot | 0.33 | -0.136 | -0.003 |
| synth:easy | uniform | 0.33 | -0.049 | -0.009 |
| synth:hard | autopilot | 0.20 | 0.000 | 0.050 |
| synth:hard | uniform | 0.33 | -0.016 | 0.030 |
| synth:medium | autopilot | 0.28 | -0.084 | 0.027 |
| synth:medium | uniform | 0.33 | -0.022 | 0.023 |
