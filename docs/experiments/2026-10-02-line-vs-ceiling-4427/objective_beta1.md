## beta 1: click 150 (control: line-4)

**The objective: the withheld set above the app's threshold** (F-beta at the session's beta; paired d vs the control):

| arm | unchecked | after the check | walk's effect | precision after | recall after | returned (unchecked → checked) | d after the check | d unchecked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| line-4 | 0.504 | **0.470** | -0.032 ± 0.004 | 0.422 | 0.587 | 41.1 → 73.3 | – | – |
| x1.0 | 0.508 | **0.415** | -0.087 ± 0.006 | 0.342 | 0.602 | 34.1 → 89.5 | -0.056 ± 0.005 | -0.005 ± 0.003 |
| x0.5 | 0.218 | **0.385** | +0.165 ± 0.007 | 0.295 | 0.624 | 13.4 → 101.3 | -0.085 ± 0.005 | -0.281 ± 0.012 |
| x0.25 | 0.223 | **0.377** | +0.154 ± 0.007 | 0.282 | 0.635 | 13.4 → 107.6 | -0.092 ± 0.005 | -0.280 ± 0.012 |

The rank-count reading (the balance's rule re-drawn on the fresh ranking; secondary):

| arm | cells | **F-beta at the line** | share of best | kept | precision | recall | final AP | Goods @150 | check votes | own-corpus kept precision | left unvoted | in hand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| line-4 | 702 | **0.459** | 0.777 | 30.5 | 0.611 | 0.376 | 0.526 | 20.2 | 21.2 | 0.379 | 29.8 | 29.8 |
| x1.0 | 702 | **0.457** | 0.745 | 28.7 | 0.629 | 0.372 | 0.532 | 19.2 | 21.0 | 0.472 | 30.9 | 30.7 |
| x0.5 | 702 | **0.468** | 0.725 | 27.0 | 0.665 | 0.379 | 0.552 | 29.7 | 20.5 | 0.149 | 20.4 | 32.5 |
| x0.25 | 702 | **0.469** | 0.728 | 26.8 | 0.676 | 0.380 | 0.553 | 30.2 | 20.8 | 0.138 | 19.8 | 32.8 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | **d F-beta** | d share | d kept | d precision | d recall | d AP | d Goods | d check votes | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x1.0 | **-0.002 ± 0.002** | -0.032 ± 0.007 | -1.8 ± 0.3 | +0.017 ± 0.006 | -0.004 ± 0.002 | +0.005 ± 0.002 | -1.0 ± 0.3 | -0.2 ± 0.2 | 720 |
| x0.5 | **+0.009 ± 0.002** | -0.052 ± 0.008 | -3.5 ± 0.4 | +0.054 ± 0.007 | +0.003 ± 0.002 | +0.025 ± 0.002 | +9.5 ± 0.3 | -0.7 ± 0.1 | 720 |
| x0.25 | **+0.010 ± 0.002** | -0.049 ± 0.008 | -3.7 ± 0.4 | +0.065 ± 0.007 | +0.004 ± 0.002 | +0.027 ± 0.002 | +10.0 ± 0.3 | -0.4 ± 0.2 | 720 |

By size band, F-beta at the line (paired d vs the control):

| band | line-4 | x1.0 (d) | x0.5 (d) | x0.25 (d) |
|---|---:|---:|---:|---:|
| small | 0.239 | 0.233 (-0.005) | 0.237 (-0.002) | 0.237 (-0.001) |
| medium | 0.452 | 0.453 (+0.001) | 0.469 (+0.017) | 0.470 (+0.018) |
| large | 0.674 | 0.671 (-0.003) | 0.684 (+0.011) | 0.686 (+0.013) |
