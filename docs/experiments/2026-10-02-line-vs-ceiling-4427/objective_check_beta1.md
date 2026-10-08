## beta 1: click 150 (control: app-walk)

**The objective: the withheld set above the app's threshold** (F-beta at the session's beta; paired d vs the control):

| arm | unchecked | after the check | walk's effect | precision after | recall after | returned (unchecked → checked) | d after the check | d unchecked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| app-walk | 0.502 | **0.468** | -0.032 ± 0.004 | 0.418 | 0.589 | 41.6 → 75.2 | – | – |
| advisory | 0.502 | **0.513** | +0.004 ± 0.003 | 0.652 | 0.461 | 41.6 → 34.5 | +0.037 ± 0.005 | +0.000 ± 0.000 |
| shallow | 0.502 | **0.487** | -0.013 ± 0.003 | 0.461 | 0.552 | 41.6 → 54.7 | +0.019 ± 0.002 | +0.000 ± 0.000 |

The rank-count reading (the balance's rule re-drawn on the fresh ranking; secondary):

| arm | cells | **F-beta at the line** | share of best | kept | precision | recall | final AP | Goods @150 | check votes | own-corpus kept precision | left unvoted | in hand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| app-walk | 702 | **0.464** | 0.784 | 30.7 | 0.615 | 0.380 | 0.526 | 20.2 | 21.5 | 0.379 | 29.8 | 29.9 |
| advisory | 702 | **0.464** | 0.784 | 30.7 | 0.615 | 0.380 | 0.526 | 20.2 | 21.5 | 0.504 | 29.8 | 24.6 |
| shallow | 702 | **0.460** | 0.778 | 30.7 | 0.609 | 0.376 | 0.526 | 20.2 | 15.0 | 0.416 | 29.8 | 28.4 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | **d F-beta** | d share | d kept | d precision | d recall | d AP | d Goods | d check votes | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| advisory | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.0 ± 0.0 | 713 |
| shallow | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | -6.5 ± 0.1 | 713 |

By size band, F-beta at the line (paired d vs the control):

| band | app-walk | advisory (d) | shallow (d) |
|---|---:|---:|---:|
| small | 0.247 | 0.247 (+0.000) | 0.240 (+0.000) |
| medium | 0.452 | 0.452 (+0.000) | 0.452 (+0.000) |
| large | 0.674 | 0.674 (+0.000) | 0.674 (+0.000) |
