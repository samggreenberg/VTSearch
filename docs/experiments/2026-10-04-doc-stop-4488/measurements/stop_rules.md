72 sessions (36 classes).

- chosen on fold 1, applied to fold 0: e=0, c=0.01
- chosen on fold 0, applied to fold 1: e=0, c=0.01

| rule | stops by 50 | stop click, median (range) | unfound click-half positives | AP at 50 − at stop | returned-set F1 at 50 − at stop |
|---|---:|---|---:|---:|---:|
| **stop (held-out e, c)** | 83% | 31 (17–50) | 5.2% | +0.018 | +0.017 |
| Smart alone (held-out e) | 99% | 10 (10–44) | 37.0% | +0.030 | +0.111 |
| Stable alone (held-out c) | 96% | 24 (11–50) | 13.0% | +0.016 | -0.001 |
| dry run alone (Span) | 83% | 31 (17–50) | 5.2% | +0.018 | +0.017 |
