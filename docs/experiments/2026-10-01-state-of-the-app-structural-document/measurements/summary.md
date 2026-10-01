8 classes.

### Headline (mean over classes)

| | AP | Goods found | gate precision | gate recall | gate F1 | best-cut F1 |
|---|---:|---:|---:|---:|---:|---:|
| click 0 (example sort) | 0.58 | 0.0 | 0.73 | 0.57 | 0.49 | 0.64 |
| 10 clicks | 0.82 | 8.2 | 0.54 | 0.85 | 0.58 | 0.87 |
| 25 clicks | 0.88 | 17.9 | 0.49 | 0.91 | 0.53 | 0.91 |
| final | 0.88 | 27.6 | 0.48 | 0.91 | 0.53 | 0.92 |

### Floor-style cuts at P (oracle: the deepest cut of the ranking with precision >= P)

| | recall at P=10% | recall at P=50% | recall at P=90% | gate recall |
|---|---:|---:|---:|---:|
| click 0 (example sort) | 0.63 | 0.60 | 0.47 | 0.57 |
| 10 clicks | 0.90 | 0.83 | 0.78 | 0.85 |
| 25 clicks | 0.91 | 0.90 | 0.81 | 0.91 |
| final | 0.92 | 0.89 | 0.82 | 0.91 |

### Per class, final click

| class | positives | pool | click-0 AP | final AP | found | gate P / R / F1 | recall at P=50% | retrain median |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `ucsf/logo_bat_leaf` | 200 | 47907 | 0.40 | 0.71 | 50 | 0.98 / 0.69 / 0.81 | 0.70 | 1.8 s |
| `tobacco800/logo_cgr96c00_1` | 9 | 46659 | 0.25 | 0.75 | 5 | 0.14 / 0.75 / 0.23 | 0.75 | 0.1 s |
| `staver/stamp_stampds-00213_1` | 19 | 49968 | 0.78 | 0.80 | 7 | 0.12 / 1.00 / 0.21 | 0.82 | 0.1 s |
| `tobacco800/logo_aeq93a00_1` | 116 | 46755 | 0.42 | 0.86 | 50 | 0.98 / 0.89 / 0.93 | 0.90 | 2.1 s |
| `spods/stamp_00612_1` | 31 | 49968 | 0.67 | 0.96 | 14 | 0.12 / 1.00 / 0.22 | 1.00 | 0.1 s |
| `tobacco800/logo_aah97e00-page02_1_0` | 354 | 46957 | 0.18 | 0.99 | 50 | 1.00 / 0.99 / 0.99 | 0.99 | 2.5 s |
| `spods/logo_00003_0` | 30 | 49968 | 1.00 | 1.00 | 20 | 0.20 / 1.00 / 0.34 | 1.00 | 0.1 s |
| `ucsf/logo_rjr_script` | 52 | 47044 | 0.97 | 1.00 | 25 | 0.31 / 1.00 / 0.48 | 1.00 | 2.0 s |

### Most helpful and most harmful clicks (credit = that click's change in AP; one observation each)

| | class | page | label | click | credit |
|---|---|---|---|---:|---:|
| harmful | `staver/stamp_stampds-00213_1` | `staver/stampds-00213` | good | 4 | -0.407 |
| harmful | `tobacco800/logo_aeq93a00_1` | `ucsf/fhnd0003#0` | good | 5 | -0.060 |
| harmful | `ucsf/logo_bat_leaf` | `ucsf/hfyl0214#0` | good | 16 | -0.041 |
| harmful | `ucsf/logo_bat_leaf` | `ucsf/frcm0193#0` | good | 12 | -0.031 |
| harmful | `ucsf/logo_bat_leaf` | `ucsf/fkpc0214#0` | good | 8 | -0.031 |
| harmful | `spods/stamp_00612_1` | `spods/00636` | good | 4 | -0.026 |
| helpful | `tobacco800/logo_cgr96c00_1` | `tobacco800/sia26d00` | good | 1 | +0.501 |
| helpful | `tobacco800/logo_aah97e00-page02_1_0` | `ucsf/fmgk0164#0` | good | 1 | +0.465 |
| helpful | `tobacco800/logo_aeq93a00_1` | `ucsf/ffbf0003#0` | good | 1 | +0.342 |
| helpful | `staver/stamp_stampds-00213_1` | `staver/stampds-00223` | good | 12 | +0.338 |
| helpful | `ucsf/logo_bat_leaf` | `ucsf/hywy0206#0` | good | 1 | +0.331 |
| helpful | `spods/stamp_00612_1` | `spods/00637` | good | 1 | +0.288 |
