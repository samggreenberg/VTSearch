## beta 1: click 150 (control: x0.5-shipped@300)

**The objective: the withheld set above the app's threshold** (F-beta at the session's beta; paired d vs the control):

| arm | unchecked | after the check | walk's effect | precision after | recall after | returned (unchecked → checked) | d after the check | d unchecked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x0.5-shipped@300 | 0.218 | **0.237** | +0.018 ± 0.008 | 0.149 | 0.702 | 13.1 → 227.5 | – | – |
| x0.25@300 | 0.231 | **0.237** | +0.006 ± 0.008 | 0.148 | 0.711 | 13.0 → 233.1 | +0.000 ± 0.002 | +0.015 ± 0.007 |
| x0.75@300 | 0.261 | **0.247** | -0.011 ± 0.008 | 0.156 | 0.708 | 13.6 → 223.7 | +0.011 ± 0.002 | +0.049 ± 0.008 |

The rank-count reading (the balance's rule re-drawn on the fresh ranking; secondary):

| arm | cells | **F-beta at the line** | share of best | kept | precision | recall | final AP | Goods @150 | check votes | own-corpus kept precision | left unvoted | in hand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x0.5-shipped@300 | 430 | **0.480** | 0.731 | 25.3 | 0.709 | 0.386 | 0.564 | 29.2 | 20.3 | 0.113 | 15.1 | 36.5 |
| x0.25@300 | 430 | **0.480** | 0.729 | 24.9 | 0.713 | 0.385 | 0.565 | 29.8 | 20.4 | 0.112 | 14.8 | 36.7 |
| x0.75@300 | 430 | **0.479** | 0.721 | 24.8 | 0.711 | 0.385 | 0.565 | 27.3 | 20.6 | 0.179 | 15.6 | 36.6 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | **d F-beta** | d share | d kept | d precision | d recall | d AP | d Goods | d check votes | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x0.25@300 | **-0.001 ± 0.001** | -0.002 ± 0.006 | -0.4 ± 0.3 | +0.004 ± 0.008 | -0.001 ± 0.001 | +0.001 ± 0.001 | +0.7 ± 0.1 | +0.1 ± 0.1 | 432 |
| x0.75@300 | **-0.001 ± 0.001** | -0.010 ± 0.006 | -0.5 ± 0.2 | +0.002 ± 0.008 | -0.001 ± 0.001 | +0.001 ± 0.001 | -1.9 ± 0.2 | +0.2 ± 0.1 | 432 |

By size band, F-beta at the line (paired d vs the control):

| band | x0.5-shipped@300 | x0.25@300 (d) | x0.75@300 (d) |
|---|---:|---:|---:|
| small | 0.258 | 0.257 (-0.000) | 0.261 (+0.004) |
| medium | 0.474 | 0.472 (-0.002) | 0.469 (-0.005) |
| large | 0.695 | 0.696 (+0.000) | 0.694 (-0.002) |
