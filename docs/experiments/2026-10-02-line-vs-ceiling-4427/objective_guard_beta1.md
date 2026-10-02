## beta 1: click 150 (control: app-walk)

**The objective: the withheld set above the app's threshold** (F-beta at the session's beta; paired d vs the control):

| arm | unchecked | after the check | walk's effect | precision after | recall after | returned (unchecked → checked) | d after the check | d unchecked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| app-walk | 0.502 | **0.468** | -0.032 ± 0.005 | 0.421 | 0.586 | 41.4 → 74.1 | – | – |
| guard-1.0 | 0.502 | **0.441** | -0.059 ± 0.006 | 0.366 | 0.594 | 41.4 → 79.1 | -0.027 ± 0.004 | +0.000 ± 0.000 |
| guard-0.5 | 0.502 | **0.437** | -0.063 ± 0.006 | 0.361 | 0.600 | 41.4 → 83.7 | -0.031 ± 0.003 | +0.000 ± 0.000 |
| advisory | 0.502 | **0.511** | +0.002 ± 0.004 | 0.656 | 0.458 | 41.4 → 34.3 | +0.035 ± 0.006 | +0.000 ± 0.000 |

The rank-count reading (the balance's rule re-drawn on the fresh ranking; secondary):

| arm | cells | **F-beta at the line** | share of best | kept | precision | recall | final AP | Goods @150 | check votes | own-corpus kept precision | left unvoted | in hand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| app-walk | 421 | **0.460** | 0.783 | 30.8 | 0.607 | 0.377 | 0.524 | 20.3 | 21.4 | 0.381 | 29.9 | 29.8 |
| guard-1.0 | 421 | **0.460** | 0.783 | 30.8 | 0.607 | 0.377 | 0.524 | 20.3 | 20.8 | 0.288 | 29.9 | 30.2 |
| guard-0.5 | 421 | **0.460** | 0.783 | 30.8 | 0.607 | 0.377 | 0.524 | 20.3 | 21.3 | 0.286 | 29.9 | 30.5 |
| advisory | 421 | **0.460** | 0.783 | 30.8 | 0.607 | 0.377 | 0.524 | 20.3 | 21.4 | 0.497 | 29.9 | 24.7 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | **d F-beta** | d share | d kept | d precision | d recall | d AP | d Goods | d check votes | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| guard-1.0 | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | -0.6 ± 0.1 | 432 |
| guard-0.5 | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | -0.2 ± 0.0 | 432 |
| advisory | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.0 ± 0.0 | 432 |

By size band, F-beta at the line (paired d vs the control):

| band | app-walk | guard-1.0 (d) | guard-0.5 (d) | advisory (d) |
|---|---:|---:|---:|---:|
| small | 0.238 | 0.238 (+0.000) | 0.238 (+0.000) | 0.238 (+0.000) |
| medium | 0.452 | 0.452 (+0.000) | 0.452 (+0.000) | 0.452 (+0.000) |
| large | 0.676 | 0.676 (+0.000) | 0.676 (+0.000) | 0.676 (+0.000) |
