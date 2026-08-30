# Adversarial-bound sweep (toplist): summary tables

224 cells (arm x seed x votes x policy); budget violation rate at inclusion >= 0: uniform 0.619, toplist 0.796.

## Conformal budget compliance (inclusion >= 0)

| arm group | policy | cells | FNR (mean) | alpha cap (mean) | violation rate | FNR excess (mean) | FNR excess (p90) |
|---|---|---:|---:|---:|---:|---:|---:|
| agnews | toplist | 384 | 0.431 | 0.069 | 0.797 | 0.373 | 0.874 |
| agnews | uniform | 384 | 0.104 | 0.069 | 0.612 | 0.068 | 0.198 |
| synth:easy | toplist | 96 | 0.237 | 0.069 | 0.615 | 0.198 | 0.812 |
| synth:easy | uniform | 96 | 0.064 | 0.069 | 0.479 | 0.043 | 0.131 |
| synth:hard | toplist | 96 | 0.341 | 0.069 | 0.958 | 0.272 | 0.455 |
| synth:hard | uniform | 96 | 0.269 | 0.069 | 0.698 | 0.218 | 0.600 |
| synth:medium | toplist | 96 | 0.242 | 0.069 | 0.812 | 0.189 | 0.630 |
| synth:medium | uniform | 96 | 0.160 | 0.069 | 0.708 | 0.120 | 0.379 |

## Violation rate by inclusion value

| inclusion | alpha cap | violation (uniform) | violation (toplist) | FNR excess (uniform) | FNR excess (toplist) |
|---:|---:|---:|---:|---:|---:|
| +0 | 0.2500 | 0.223 | 0.679 | 0.049 | 0.328 |
| +1 | 0.1250 | 0.384 | 0.741 | 0.067 | 0.366 |
| +3 | 0.0312 | 0.598 | 0.777 | 0.101 | 0.330 |
| +5 | 0.0078 | 0.768 | 0.848 | 0.111 | 0.279 |
| +7 | 0.0020 | 0.866 | 0.857 | 0.115 | 0.271 |
| +10 | 0.0002 | 0.875 | 0.875 | 0.116 | 0.269 |

## Threshold inflation vs oracle (inclusion 0)

| arm group | policy | threshold - oracle (mean) | (p90) |
|---|---|---:|---:|
| agnews | toplist | 0.066 | 0.250 |
| agnews | uniform | 0.022 | 0.085 |
| synth:easy | toplist | 0.030 | 0.077 |
| synth:easy | uniform | 0.025 | 0.075 |
| synth:hard | toplist | 0.099 | 0.239 |
| synth:hard | uniform | 0.034 | 0.069 |
| synth:medium | toplist | 0.047 | 0.107 |
| synth:medium | uniform | 0.051 | 0.100 |

## Safe-blend mitigation reach (toplist policy)

| votes | violation (conformal) | violation (blend) | FNR excess (conformal) | FNR excess (blend) |
|---:|---:|---:|---:|---:|
| 12 | 0.744 | 0.839 | 0.247 | 0.200 |
| 24 | 0.804 | 0.804 | 0.360 | 0.360 |
| 50 | 0.833 | 0.833 | 0.317 | 0.317 |
| 100 | 0.804 | 0.804 | 0.305 | 0.305 |

## Vote-set composition drift

| arm group | policy | vote pos frac (mean) | cal q25 - pool q25 (mean) | (p90) |
|---|---|---:|---:|---:|
| agnews | toplist | 0.86 | 0.076 | 0.256 |
| agnews | uniform | 0.33 | -0.028 | 0.012 |
| synth:easy | toplist | 0.92 | 0.002 | 0.074 |
| synth:easy | uniform | 0.33 | -0.051 | -0.005 |
| synth:hard | toplist | 0.72 | 0.067 | 0.136 |
| synth:hard | uniform | 0.33 | -0.006 | 0.041 |
| synth:medium | toplist | 0.82 | 0.042 | 0.100 |
| synth:medium | uniform | 0.33 | -0.025 | 0.023 |
