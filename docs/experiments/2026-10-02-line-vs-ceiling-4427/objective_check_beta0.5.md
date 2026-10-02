## beta 0.5: click 150 (control: app-walk)

**The objective: the withheld set above the app's threshold** (F-beta at the session's beta; paired d vs the control):

| arm | unchecked | after the check | walk's effect | precision after | recall after | returned (unchecked → checked) | d after the check | d unchecked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| app-walk | 0.526 | **0.466** | -0.059 ± 0.005 | 0.459 | 0.573 | 40.7 → 67.0 | – | – |
| advisory | 0.526 | **0.564** | +0.029 ± 0.003 | 0.662 | 0.455 | 40.7 → 33.2 | +0.090 ± 0.005 | +0.000 ± 0.000 |
| shallow | 0.526 | **0.488** | -0.037 ± 0.004 | 0.489 | 0.542 | 40.7 → 51.2 | +0.022 ± 0.003 | +0.000 ± 0.000 |

The rank-count reading (the balance's rule re-drawn on the fresh ranking; secondary):

| arm | cells | **F-beta at the line** | share of best | kept | precision | recall | final AP | Goods @150 | check votes | own-corpus kept precision | left unvoted | in hand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| app-walk | 702 | **0.539** | 0.820 | 30.4 | 0.618 | 0.379 | 0.526 | 20.2 | 21.2 | 0.423 | 29.8 | 28.7 |
| advisory | 702 | **0.539** | 0.820 | 30.4 | 0.618 | 0.379 | 0.526 | 20.2 | 21.2 | 0.504 | 29.8 | 24.4 |
| shallow | 702 | **0.534** | 0.815 | 30.4 | 0.612 | 0.375 | 0.526 | 20.2 | 15.0 | 0.447 | 29.8 | 27.5 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | **d F-beta** | d share | d kept | d precision | d recall | d AP | d Goods | d check votes | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| advisory | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.0 ± 0.0 | 713 |
| shallow | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | -6.2 ± 0.1 | 713 |

By size band, F-beta at the line (paired d vs the control):

| band | app-walk | advisory (d) | shallow (d) |
|---|---:|---:|---:|
| small | 0.287 | 0.287 (+0.000) | 0.279 (+0.000) |
| medium | 0.529 | 0.529 (+0.000) | 0.529 (+0.000) |
| large | 0.778 | 0.778 (+0.000) | 0.778 (+0.000) |
