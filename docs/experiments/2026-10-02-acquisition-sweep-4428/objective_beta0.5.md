## beta 0.5: click 150 (control: line-4)

**The objective: the withheld set above the app's threshold** (F-beta at the session's beta; paired d vs the control):

| arm | unchecked | after the check | walk's effect | precision after | recall after | returned (unchecked → checked) | d after the check | d unchecked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| line-4 | 0.529 | **0.467** | -0.060 ± 0.005 | 0.462 | 0.573 | 40.3 → 66.8 | – | – |
| x1.0 | 0.573 | **0.379** | -0.186 ± 0.008 | 0.359 | 0.601 | 31.5 → 85.3 | -0.089 ± 0.007 | +0.032 ± 0.004 |
| x0.5 | 0.329 | **0.317** | -0.009 ± 0.007 | 0.291 | 0.624 | 12.4 → 102.3 | -0.150 ± 0.007 | -0.199 ± 0.011 |
| x0.25 | 0.344 | **0.307** | -0.035 ± 0.008 | 0.278 | 0.635 | 13.7 → 109.3 | -0.159 ± 0.007 | -0.185 ± 0.011 |

The rank-count reading (the balance's rule re-drawn on the fresh ranking; secondary):

| arm | cells | **F-beta at the line** | share of best | kept | precision | recall | final AP | Goods @150 | check votes | own-corpus kept precision | left unvoted | in hand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| line-4 | 702 | **0.534** | 0.813 | 30.2 | 0.612 | 0.375 | 0.527 | 20.2 | 20.9 | 0.423 | 29.8 | 28.7 |
| x1.0 | 702 | **0.543** | 0.787 | 27.2 | 0.654 | 0.372 | 0.542 | 20.7 | 20.6 | 0.522 | 29.4 | 30.2 |
| x0.5 | 702 | **0.551** | 0.773 | 26.0 | 0.676 | 0.375 | 0.552 | 30.1 | 20.4 | 0.148 | 20.0 | 32.6 |
| x0.25 | 702 | **0.555** | 0.783 | 26.4 | 0.679 | 0.376 | 0.555 | 30.4 | 20.8 | 0.133 | 19.7 | 32.8 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | **d F-beta** | d share | d kept | d precision | d recall | d AP | d Goods | d check votes | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x1.0 | **+0.009 ± 0.003** | -0.026 ± 0.007 | -3.1 ± 0.3 | +0.042 ± 0.007 | -0.004 ± 0.002 | +0.016 ± 0.002 | +0.4 ± 0.4 | -0.3 ± 0.1 | 720 |
| x0.5 | **+0.017 ± 0.003** | -0.040 ± 0.008 | -4.2 ± 0.4 | +0.064 ± 0.007 | +0.000 ± 0.002 | +0.026 ± 0.002 | +9.8 ± 0.3 | -0.5 ± 0.1 | 720 |
| x0.25 | **+0.021 ± 0.003** | -0.031 ± 0.008 | -3.8 ± 0.4 | +0.066 ± 0.007 | +0.001 ± 0.002 | +0.029 ± 0.002 | +10.1 ± 0.3 | -0.1 ± 0.1 | 720 |

By size band, F-beta at the line (paired d vs the control):

| band | line-4 | x1.0 (d) | x0.5 (d) | x0.25 (d) |
|---|---:|---:|---:|---:|
| small | 0.278 | 0.280 (+0.002) | 0.289 (+0.011) | 0.288 (+0.010) |
| medium | 0.528 | 0.551 (+0.023) | 0.555 (+0.027) | 0.564 (+0.036) |
| large | 0.779 | 0.782 (+0.003) | 0.793 (+0.014) | 0.795 (+0.016) |
