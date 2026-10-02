## beta 2: click 150 (control: app-walk)

**The objective: the withheld set above the app's threshold** (F-beta at the session's beta; paired d vs the control):

| arm | unchecked | after the check | walk's effect | precision after | recall after | returned (unchecked → checked) | d after the check | d unchecked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| app-walk | 0.507 | **0.509** | +0.005 ± 0.004 | 0.347 | 0.632 | 82.0 → 114.5 | – | – |
| advisory | 0.507 | **0.501** | -0.014 ± 0.004 | 0.637 | 0.484 | 82.0 → 47.6 | -0.019 ± 0.005 | +0.000 ± 0.000 |
| shallow | 0.507 | **0.517** | +0.013 ± 0.004 | 0.364 | 0.615 | 82.0 → 92.7 | +0.008 ± 0.002 | +0.000 ± 0.000 |

The rank-count reading (the balance's rule re-drawn on the fresh ranking; secondary):

| arm | cells | **F-beta at the line** | share of best | kept | precision | recall | final AP | Goods @150 | check votes | own-corpus kept precision | left unvoted | in hand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| app-walk | 702 | **0.508** | 0.841 | 85.0 | 0.441 | 0.558 | 0.527 | 20.0 | 30.4 | 0.298 | 30.1 | 31.8 |
| advisory | 702 | **0.508** | 0.841 | 85.0 | 0.443 | 0.558 | 0.527 | 20.0 | 30.4 | 0.521 | 30.1 | 25.3 |
| shallow | 702 | **0.508** | 0.841 | 85.0 | 0.441 | 0.558 | 0.527 | 20.0 | 25.0 | 0.311 | 30.1 | 31.1 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | **d F-beta** | d share | d kept | d precision | d recall | d AP | d Goods | d check votes | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| advisory | **+0.000 ± 0.000** | +0.000 ± 0.000 | -0.0 ± 0.0 | +0.001 ± 0.001 | +0.000 ± 0.000 | -0.000 ± 0.000 | -0.0 ± 0.0 | +0.0 ± 0.0 | 720 |
| shallow | **+0.000 ± 0.000** | +0.000 ± 0.000 | +0.0 ± 0.0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.0 ± 0.0 | -5.4 ± 0.1 | 720 |

By size band, F-beta at the line (paired d vs the control):

| band | app-walk | advisory (d) | shallow (d) |
|---|---:|---:|---:|
| small | 0.285 | 0.285 (+0.000) | 0.285 (+0.000) |
| medium | 0.483 | 0.483 (+0.000) | 0.483 (+0.000) |
| large | 0.741 | 0.741 (+0.000) | 0.741 (+0.000) |
