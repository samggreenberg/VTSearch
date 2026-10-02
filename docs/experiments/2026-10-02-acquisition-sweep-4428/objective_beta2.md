## beta 2: click 150 (control: line-4)

**The objective: the withheld set above the app's threshold** (F-beta at the session's beta; paired d vs the control):

| arm | unchecked | after the check | walk's effect | precision after | recall after | returned (unchecked → checked) | d after the check | d unchecked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| line-4 | 0.508 | **0.510** | +0.005 ± 0.004 | 0.351 | 0.629 | 80.4 → 114.1 | – | – |
| x1.0 | 0.479 | **0.488** | +0.014 ± 0.005 | 0.303 | 0.635 | 91.7 → 126.9 | -0.022 ± 0.004 | -0.030 ± 0.004 |
| x0.5 | 0.198 | **0.491** | +0.285 ± 0.011 | 0.284 | 0.648 | 35.2 → 127.7 | -0.019 ± 0.004 | -0.298 ± 0.013 |
| x0.25 | 0.189 | **0.487** | +0.294 ± 0.010 | 0.263 | 0.658 | 29.6 → 136.3 | -0.022 ± 0.004 | -0.315 ± 0.013 |

The rank-count reading (the balance's rule re-drawn on the fresh ranking; secondary):

| arm | cells | **F-beta at the line** | share of best | kept | precision | recall | final AP | Goods @150 | check votes | own-corpus kept precision | left unvoted | in hand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| line-4 | 702 | **0.507** | 0.839 | 84.4 | 0.443 | 0.557 | 0.527 | 20.0 | 30.1 | 0.299 | 30.1 | 31.8 |
| x1.0 | 702 | **0.482** | 0.804 | 89.2 | 0.426 | 0.548 | 0.514 | 17.1 | 30.3 | 0.339 | 32.9 | 31.7 |
| x0.5 | 702 | **0.514** | 0.784 | 62.2 | 0.550 | 0.530 | 0.547 | 28.8 | 30.0 | 0.140 | 21.2 | 33.3 |
| x0.25 | 702 | **0.513** | 0.773 | 58.0 | 0.574 | 0.524 | 0.552 | 30.0 | 30.4 | 0.117 | 20.0 | 33.7 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | **d F-beta** | d share | d kept | d precision | d recall | d AP | d Goods | d check votes | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x1.0 | **-0.025 ± 0.003** | -0.035 ± 0.007 | +4.8 ± 1.7 | -0.017 ± 0.010 | -0.010 ± 0.003 | -0.012 ± 0.002 | -2.8 ± 0.3 | +0.2 ± 0.1 | 720 |
| x0.5 | **+0.007 ± 0.003** | -0.055 ± 0.009 | -22.2 ± 1.5 | +0.107 ± 0.008 | -0.027 ± 0.004 | +0.020 ± 0.002 | +8.8 ± 0.3 | -0.1 ± 0.1 | 720 |
| x0.25 | **+0.006 ± 0.003** | -0.067 ± 0.009 | -26.3 ± 1.6 | +0.130 ± 0.009 | -0.033 ± 0.004 | +0.026 ± 0.002 | +10.1 ± 0.3 | +0.3 ± 0.1 | 720 |

By size band, F-beta at the line (paired d vs the control):

| band | line-4 | x1.0 (d) | x0.5 (d) | x0.25 (d) |
|---|---:|---:|---:|---:|
| small | 0.284 | 0.266 (-0.018) | 0.271 (-0.013) | 0.271 (-0.013) |
| medium | 0.483 | 0.454 (-0.029) | 0.484 (+0.000) | 0.482 (-0.001) |
| large | 0.740 | 0.714 (-0.026) | 0.771 (+0.031) | 0.770 (+0.030) |
