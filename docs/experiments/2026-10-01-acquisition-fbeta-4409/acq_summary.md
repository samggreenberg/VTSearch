# Acquisition at the F-beta argmax vs the shipped line - 4, per beta (#4409 / #4413 step 5)

## beta 0.5: the returned set at its own balance, click 150 (Binary, 5 seeds)

| arm | cells | final AP | Goods @25 / 50 / 100 / 150 | check votes | kept | precision | recall | F-beta | best F-beta | share | checked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| line-4 (shipped) | 702 | 0.527 | 11.3 / 14.8 / 17.9 / 20.2 | 20.9 | 30.2 | 0.612 | 0.375 | 0.534 | 0.587 | **0.813** | 0.99 |
| argmax x1.0 | 702 | 0.542 | 11.3 / 14.0 / 17.0 / 20.7 | 20.6 | 27.2 | 0.654 | 0.372 | 0.543 | 0.602 | **0.787** | 0.99 |
| argmax x0.5 | 702 | 0.552 | 11.3 / 15.7 / 24.7 / 30.1 | 20.4 | 26.0 | 0.676 | 0.375 | 0.551 | 0.614 | **0.773** | 0.99 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | d share | d F-beta | d precision | d recall | d kept | d Goods @150 | d final AP | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax x1.0 | -0.026 ± 0.007 | 0.009 ± 0.003 | 0.042 ± 0.007 | -0.004 ± 0.002 | -3.054 ± 0.343 | 0.444 ± 0.352 | 0.016 ± 0.002 | 720 |
| argmax x0.5 | -0.040 ± 0.008 | 0.017 ± 0.003 | 0.064 ± 0.007 | 0.000 ± 0.002 | -4.243 ± 0.370 | 9.823 ± 0.340 | 0.026 ± 0.002 | 720 |

By size band at click 150 (share of the best F-beta, then F-beta / Goods / final AP; paired d vs the control):

| band | metric | line-4 | x1.0 (d) | x0.5 (d) |
|---|---|---:|---:|---:|
| small | share | 0.667 | 0.612 (-0.055) | 0.608 (-0.059) |
| small | F-beta | 0.278 | 0.280 (+0.002) | 0.289 (+0.011) |
| small | Goods @150 | 10.5 | 10.9 (+0.4) | 16.3 (+5.8) |
| small | final AP | 0.263 | 0.274 (+0.012) | 0.282 (+0.019) |
| medium | share | 0.845 | 0.828 (-0.017) | 0.794 (-0.051) |
| medium | F-beta | 0.528 | 0.551 (+0.023) | 0.555 (+0.027) |
| medium | Goods @150 | 20.0 | 22.1 (+2.1) | 29.9 (+9.9) |
| medium | final AP | 0.485 | 0.510 (+0.024) | 0.519 (+0.034) |
| large | share | 0.919 | 0.910 (-0.009) | 0.908 (-0.011) |
| large | F-beta | 0.779 | 0.782 (+0.003) | 0.793 (+0.014) |
| large | Goods @150 | 28.9 | 27.7 (-1.2) | 42.1 (+13.3) |
| large | final AP | 0.796 | 0.807 (+0.011) | 0.819 (+0.023) |

## beta 1: the returned set at its own balance, click 150 (Binary, 5 seeds)

| arm | cells | final AP | Goods @25 / 50 / 100 / 150 | check votes | kept | precision | recall | F-beta | best F-beta | share | checked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| line-4 (shipped) | 702 | 0.526 | 11.3 / 14.8 / 17.9 / 20.2 | 21.2 | 30.5 | 0.611 | 0.376 | 0.459 | 0.536 | **0.777** | 0.99 |
| argmax x1.0 | 702 | 0.532 | 11.3 / 13.7 / 16.0 / 19.2 | 21.0 | 28.7 | 0.629 | 0.372 | 0.457 | 0.543 | **0.745** | 0.99 |
| argmax x0.5 | 702 | 0.552 | 11.3 / 15.4 / 23.7 / 29.7 | 20.5 | 27.0 | 0.665 | 0.379 | 0.468 | 0.561 | **0.725** | 0.99 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | d share | d F-beta | d precision | d recall | d kept | d Goods @150 | d final AP | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax x1.0 | -0.032 ± 0.007 | -0.002 ± 0.002 | 0.017 ± 0.006 | -0.004 ± 0.002 | -1.825 ± 0.292 | -1.030 ± 0.347 | 0.005 ± 0.002 | 720 |
| argmax x0.5 | -0.052 ± 0.008 | 0.009 ± 0.002 | 0.054 ± 0.007 | 0.003 ± 0.002 | -3.483 ± 0.351 | 9.481 ± 0.341 | 0.025 ± 0.002 | 720 |

By size band at click 150 (share of the best F-beta, then F-beta / Goods / final AP; paired d vs the control):

| band | metric | line-4 | x1.0 (d) | x0.5 (d) |
|---|---|---:|---:|---:|
| small | share | 0.645 | 0.598 (-0.046) | 0.561 (-0.083) |
| small | F-beta | 0.239 | 0.233 (-0.005) | 0.237 (-0.002) |
| small | Goods @150 | 10.5 | 10.1 (-0.4) | 15.9 (+5.4) |
| small | final AP | 0.263 | 0.267 (+0.004) | 0.281 (+0.018) |
| medium | share | 0.827 | 0.782 (-0.045) | 0.772 (-0.055) |
| medium | F-beta | 0.452 | 0.453 (+0.001) | 0.469 (+0.017) |
| medium | Goods @150 | 20.0 | 20.4 (+0.5) | 29.5 (+9.5) |
| medium | final AP | 0.486 | 0.496 (+0.011) | 0.520 (+0.034) |
| large | share | 0.851 | 0.846 (-0.005) | 0.833 (-0.018) |
| large | F-beta | 0.674 | 0.671 (-0.003) | 0.684 (+0.011) |
| large | Goods @150 | 28.9 | 25.8 (-3.1) | 41.8 (+13.0) |
| large | final AP | 0.796 | 0.796 (+0.001) | 0.818 (+0.022) |

## beta 2: the returned set at its own balance, click 150 (Binary, 5 seeds)

| arm | cells | final AP | Goods @25 / 50 / 100 / 150 | check votes | kept | precision | recall | F-beta | best F-beta | share | checked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| line-4 (shipped) | 702 | 0.527 | 11.3 / 14.6 / 17.4 / 20.0 | 30.1 | 84.4 | 0.443 | 0.557 | 0.507 | 0.554 | **0.839** | 0.99 |
| argmax x1.0 | 702 | 0.514 | 11.3 / 13.4 / 15.0 / 17.1 | 30.3 | 89.2 | 0.426 | 0.548 | 0.482 | 0.545 | **0.804** | 0.99 |
| argmax x0.5 | 702 | 0.547 | 11.3 / 14.9 / 22.1 / 28.8 | 30.0 | 62.2 | 0.550 | 0.530 | 0.514 | 0.571 | **0.784** | 0.99 |

Paired against the control (cell = category x seed), mean difference ± SE:

| arm | d share | d F-beta | d precision | d recall | d kept | d Goods @150 | d final AP | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax x1.0 | -0.035 ± 0.007 | -0.025 ± 0.003 | -0.017 ± 0.010 | -0.010 ± 0.003 | 4.825 ± 1.740 | -2.829 ± 0.329 | -0.012 ± 0.002 | 720 |
| argmax x0.5 | -0.055 ± 0.009 | 0.007 ± 0.003 | 0.107 ± 0.008 | -0.027 ± 0.004 | -22.193 ± 1.541 | 8.819 ± 0.346 | 0.020 ± 0.002 | 720 |

By size band at click 150 (share of the best F-beta, then F-beta / Goods / final AP; paired d vs the control):

| band | metric | line-4 | x1.0 (d) | x0.5 (d) |
|---|---|---:|---:|---:|
| small | share | 0.731 | 0.681 (-0.049) | 0.638 (-0.093) |
| small | F-beta | 0.284 | 0.266 (-0.018) | 0.271 (-0.013) |
| small | Goods @150 | 10.3 | 9.0 (-1.3) | 15.0 (+4.7) |
| small | final AP | 0.264 | 0.260 (-0.004) | 0.276 (+0.012) |
| medium | share | 0.863 | 0.813 (-0.050) | 0.784 (-0.079) |
| medium | F-beta | 0.483 | 0.454 (-0.029) | 0.484 (+0.000) |
| medium | Goods @150 | 19.9 | 18.3 (-1.6) | 28.4 (+8.5) |
| medium | final AP | 0.486 | 0.479 (-0.008) | 0.514 (+0.028) |
| large | share | 0.917 | 0.910 (-0.007) | 0.920 (+0.003) |
| large | F-beta | 0.740 | 0.714 (-0.026) | 0.771 (+0.031) |
| large | Goods @150 | 28.4 | 23.0 (-5.3) | 41.1 (+12.8) |
| large | final AP | 0.794 | 0.771 (-0.024) | 0.814 (+0.020) |
