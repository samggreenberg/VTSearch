## beta 0.5: click 150 (control: app-walk)

**The objective: the withheld set above the app's threshold** (F-beta at the session's beta; paired d vs the control):

| arm | unchecked | after the check | walk's effect | precision after | recall after | returned (unchecked → checked) | d after the check | d unchecked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| app-walk | 0.526 | **0.467** | -0.057 ± 0.006 | 0.461 | 0.568 | 40.5 → 65.0 | – | – |
| guard-1.0 | 0.526 | **0.390** | -0.135 ± 0.008 | 0.365 | 0.595 | 40.5 → 79.3 | -0.077 ± 0.005 | +0.000 ± 0.000 |
| guard-0.5 | 0.526 | **0.395** | -0.129 ± 0.008 | 0.373 | 0.594 | 40.5 → 79.6 | -0.072 ± 0.005 | +0.000 ± 0.000 |
| advisory | 0.526 | **0.561** | +0.027 ± 0.004 | 0.664 | 0.453 | 40.5 → 33.3 | +0.085 ± 0.007 | +0.000 ± 0.000 |

The rank-count reading (the balance's rule re-drawn on the fresh ranking; secondary):

| arm | cells | **F-beta at the line** | share of best | kept | precision | recall | final AP | Goods @150 | check votes | own-corpus kept precision | left unvoted | in hand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| app-walk | 421 | **0.533** | 0.815 | 30.4 | 0.611 | 0.376 | 0.524 | 20.4 | 21.0 | 0.425 | 29.8 | 28.6 |
| guard-1.0 | 421 | **0.533** | 0.815 | 30.4 | 0.611 | 0.376 | 0.524 | 20.4 | 20.8 | 0.285 | 29.8 | 30.2 |
| guard-0.5 | 421 | **0.533** | 0.815 | 30.4 | 0.611 | 0.376 | 0.524 | 20.4 | 21.0 | 0.296 | 29.8 | 30.1 |
| advisory | 421 | **0.533** | 0.815 | 30.4 | 0.611 | 0.376 | 0.524 | 20.4 | 21.0 | 0.498 | 29.8 | 24.6 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | **d F-beta** | d share | d kept | d precision | d recall | d AP | d Goods | d check votes | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| guard-1.0 | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | -0.2 ± 0.1 | 432 |
| guard-0.5 | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.0 ± 0.0 | 432 |
| advisory | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.0 ± 0.0 | 432 |

By size band, F-beta at the line (paired d vs the control):

| band | app-walk | guard-1.0 (d) | guard-0.5 (d) | advisory (d) |
|---|---:|---:|---:|---:|
| small | 0.277 | 0.277 (+0.000) | 0.277 (+0.000) | 0.277 (+0.000) |
| medium | 0.529 | 0.529 (+0.000) | 0.529 (+0.000) | 0.529 (+0.000) |
| large | 0.778 | 0.778 (+0.000) | 0.778 (+0.000) | 0.778 (+0.000) |
