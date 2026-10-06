36 classes.

### Headline (mean over classes)

| | AP | Goods found | gate precision | gate recall | gate F1 | best-cut F1 |
|---|---:|---:|---:|---:|---:|---:|
| click 0 (example sort) | 0.79 | 0.0 | 0.53 | 0.81 | 0.53 | 0.84 |
| 10 clicks | 0.92 | 8.1 | 0.79 | 0.85 | 0.79 | 0.93 |
| 25 clicks | 0.93 | 13.9 | 0.91 | 0.89 | 0.88 | 0.94 |
| final | 0.93 | 18.2 | 0.93 | 0.88 | 0.88 | 0.94 |

### Floor-style cuts at P (oracle: the deepest cut of the ranking with precision >= P)

| | recall at P=10% | recall at P=50% | recall at P=90% | gate recall |
|---|---:|---:|---:|---:|
| click 0 (example sort) | 0.82 | 0.81 | 0.73 | 0.81 |
| 10 clicks | 0.96 | 0.93 | 0.86 | 0.85 |
| 25 clicks | 0.96 | 0.94 | 0.90 | 0.89 |
| final | 0.97 | 0.94 | 0.90 | 0.88 |

### Per class, final click

| class | positives | pool | click-0 AP | final AP | found | gate P / R / F1 | recall at P=50% | retrain median |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `tobacco800/logo_asg54f00_1` | 9 | 46659 | 0.67 | 0.40 | 2 | 1.00 / 0.33 / 0.50 | 0.33 | 0.2 s |
| `ucsf/logo_bw_oval_emblem` | 16 | 46798 | 0.40 | 0.70 | 4 | 1.00 / 0.70 / 0.82 | 0.70 | 0.2 s |
| `staver/stamp_stampds-00213_1` | 19 | 49968 | 0.80 | 0.72 | 7 | nan / 0.00 / 0.00 | 0.82 | 0.2 s |
| `tobacco800/logo_cgr96c00_1` | 9 | 46659 | 0.50 | 0.75 | 5 | 0.43 / 0.75 / 0.55 | 0.75 | 0.2 s |
| `ucsf/logo_p_lorillard_crest` | 10 | 46746 | 0.63 | 0.79 | 6 | 1.00 / 0.75 / 0.86 | 0.75 | 0.2 s |
| `ucsf/logo_bat_leaf` | 200 | 47907 | 0.50 | 0.86 | 50 | 1.00 / 0.85 / 0.92 | 0.86 | 4.1 s |
| `tobacco800/logo_aeq93a00_1` | 116 | 46755 | 0.50 | 0.89 | 49 | 0.97 / 0.94 / 0.95 | 0.94 | 4.9 s |
| `tobacco800/logo_afm90c00-first_1_0` | 97 | 46673 | 0.68 | 0.91 | 50 | 0.95 / 0.91 / 0.93 | 0.91 | 4.2 s |
| `spods/stamp_00716_1` | 29 | 49968 | 0.89 | 0.92 | 18 | 0.75 / 0.82 / 0.78 | 1.00 | 0.3 s |
| `tobacco800/logo_ajj10e00_1` | 383 | 47018 | 0.66 | 0.92 | 50 | 0.94 / 0.76 / 0.84 | 0.94 | 5.3 s |
| `spods/stamp_00769_1` | 31 | 49968 | 0.92 | 0.94 | 14 | 1.00 / 0.88 / 0.94 | 0.94 | 0.3 s |
| `spods/stamp_00931_1` | 31 | 49968 | 0.68 | 0.95 | 14 | 1.00 / 0.82 / 0.90 | 1.00 | 0.2 s |
| `spods/stamp_00612_1` | 31 | 49968 | 0.64 | 0.96 | 14 | 0.93 / 0.81 / 0.87 | 1.00 | 0.3 s |
| `tobacco800/logo_ald41a00-ernest_1` | 191 | 46805 | 0.47 | 0.96 | 50 | 1.00 / 0.95 / 0.98 | 0.96 | 5.8 s |
| `spods/stamp_00577_1` | 31 | 49968 | 1.00 | 0.97 | 17 | 0.93 / 0.93 / 0.93 | 1.00 | 0.3 s |
| `spods/stamp_00737_1` | 29 | 49968 | 0.98 | 0.97 | 12 | 0.83 / 0.88 / 0.86 | 1.00 | 0.3 s |
| `spods/stamp_00514_1` | 31 | 49968 | 0.75 | 0.99 | 18 | 0.92 / 0.92 / 0.92 | 1.00 | 0.3 s |
| `spods/stamp_00641_1` | 31 | 49968 | 0.84 | 0.99 | 13 | 1.00 / 0.94 / 0.97 | 1.00 | 0.3 s |
| `tobacco800/logo_aah97e00-page02_1_0` | 354 | 46957 | 0.21 | 0.99 | 50 | 1.00 / 0.99 / 0.99 | 0.99 | 5.1 s |
| `tobacco800/logo_ciy01a00-page02_1_0` | 23 | 46659 | 0.75 | 1.00 | 7 | 1.00 / 0.88 / 0.93 | 1.00 | 0.2 s |
| `spods/logo_00003_0` | 30 | 49968 | 1.00 | 1.00 | 20 | 0.83 / 1.00 / 0.91 | 1.00 | 0.3 s |
| `spods/logo_00011_0` | 30 | 49968 | 1.00 | 1.00 | 15 | 0.94 / 1.00 / 0.97 | 1.00 | 0.3 s |
| `spods/logo_00014_0` | 30 | 49968 | 1.00 | 1.00 | 13 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `spods/logo_00023_0` | 30 | 49968 | 1.00 | 1.00 | 13 | 0.89 / 1.00 / 0.94 | 1.00 | 0.3 s |
| `spods/logo_00029_0` | 30 | 49968 | 1.00 | 1.00 | 15 | 0.94 / 1.00 / 0.97 | 1.00 | 0.3 s |
| `spods/stamp_00129_1` | 30 | 49968 | 0.89 | 1.00 | 12 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `spods/stamp_00293_1` | 33 | 49968 | 0.90 | 1.00 | 13 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `spods/stamp_00546_1` | 31 | 49968 | 0.79 | 1.00 | 16 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `spods/stamp_00996_1` | 63 | 49968 | 0.82 | 1.00 | 30 | 0.94 / 1.00 / 0.97 | 1.00 | 5.6 s |
| `staver/stamp_stampds-00230_0` | 7 | 49968 | 1.00 | 1.00 | 2 | 1.00 / 1.00 / 1.00 | 1.00 | 0.2 s |
| `tobacco800/logo_azb11c00_1` | 32 | 46659 | 0.81 | 1.00 | 16 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `tobacco800/logo_bea6aa00_1` | 5 | 46659 | 1.00 | 1.00 | 3 | 1.00 / 1.00 / 1.00 | 1.00 | 0.2 s |
| `tobacco800/logo_bqz95d00_1` | 9 | 46659 | 0.92 | 1.00 | 6 | 0.38 / 1.00 / 0.55 | 1.00 | 0.2 s |
| `tobacco800/logo_drm00d00_1` | 4 | 46659 | 1.00 | 1.00 | 3 | 1.00 / 1.00 / 1.00 | 1.00 | 0.2 s |
| `tobacco800/logo_kan00d00_1` | 5 | 46659 | 1.00 | 1.00 | 4 | 1.00 / 1.00 / 1.00 | 1.00 | 0.2 s |
| `ucsf/logo_rjr_script` | 52 | 47044 | 0.97 | 1.00 | 25 | 1.00 / 1.00 / 1.00 | 1.00 | 4.8 s |

### The returned set at each balance (F-beta as a share of the best cut's; one set serves every beta)

| | beta 0.5: F-beta / best = share | beta 1: F-beta / best = share | beta 2: F-beta / best = share | precision | recall |
|---|---|---|---|---:|---:|
| click 0 (example sort) | 0.51 / 0.90 = 0.58 | 0.53 / 0.84 = 0.66 | 0.62 / 0.81 = 0.78 | 0.53 | 0.81 |
| 10 clicks | 0.78 / 0.96 = 0.81 | 0.79 / 0.93 = 0.85 | 0.82 / 0.92 = 0.88 | 0.79 | 0.85 |
| 25 clicks | 0.89 / 0.97 = 0.93 | 0.88 / 0.94 = 0.94 | 0.88 / 0.93 = 0.94 | 0.91 | 0.89 |
| final | 0.89 / 0.97 = 0.92 | 0.88 / 0.94 = 0.93 | 0.88 / 0.93 = 0.94 | 0.93 | 0.88 |

Retrain wall clock: median 0.3 s, p90 6.6 s, 24% of steps over 5 s (after click 0: p90 6.4 s).

### Most helpful and most harmful clicks (credit = that click's change in AP; one observation each)

| | class | page | label | click | credit |
|---|---|---|---|---:|---:|
| harmful | `staver/stamp_stampds-00213_1` | `staver/stampds-00213` | good | 3 | -0.557 |
| harmful | `ucsf/logo_p_lorillard_crest` | `ucsf/gnwh0113#0` | good | 1 | -0.502 |
| harmful | `tobacco800/logo_asg54f00_1` | `tobacco800/bjn43c00-page02_2` | good | 1 | -0.306 |
| harmful | `tobacco800/logo_cgr96c00_1` | `tobacco800/sia26d00` | good | 1 | -0.125 |
| harmful | `spods/stamp_00577_1` | `spods/00604` | good | 10 | -0.071 |
| harmful | `spods/stamp_00293_1` | `spods/00337` | good | 2 | -0.068 |
| helpful | `ucsf/logo_p_lorillard_crest` | `ucsf/fydf0107#0` | good | 2 | +0.689 |
| helpful | `tobacco800/logo_aah97e00-page02_1_0` | `ucsf/fmgk0164#0` | good | 1 | +0.591 |
| helpful | `tobacco800/logo_ald41a00-ernest_1` | `tobacco800/vbd23f00` | good | 1 | +0.437 |
| helpful | `staver/stamp_stampds-00213_1` | `staver/stampds-00223` | good | 10 | +0.423 |
| helpful | `tobacco800/logo_cgr96c00_1` | `tobacco800/zjy03e00-page02_2` | bad | 3 | +0.417 |
| helpful | `tobacco800/logo_aeq93a00_1` | `ucsf/ffbf0003#0` | good | 1 | +0.314 |

### Where the test positives sit at the final click

| class | final AP | test positives | beyond the shortlist | inside, failed the gate |
|---|---:|---:|---:|---:|
| `tobacco800/logo_asg54f00_1` | 0.40 | 3 | 1 | 0 |
| `ucsf/logo_bw_oval_emblem` | 0.70 | 10 | 3 | 0 |
| `tobacco800/logo_cgr96c00_1` | 0.75 | 4 | 1 | 0 |
| `ucsf/logo_p_lorillard_crest` | 0.79 | 4 | 0 | 1 |
| `ucsf/logo_bat_leaf` | 0.86 | 93 | 13 | 1 |
| `tobacco800/logo_aeq93a00_1` | 0.89 | 62 | 4 | 0 |
| `tobacco800/logo_afm90c00-first_1_0` | 0.91 | 43 | 3 | 1 |
| `tobacco800/logo_ajj10e00_1` | 0.92 | 201 | 13 | 35 |
| `tobacco800/logo_ald41a00-ernest_1` | 0.96 | 84 | 3 | 1 |
| `tobacco800/logo_aah97e00-page02_1_0` | 0.99 | 171 | 1 | 1 |
| `tobacco800/logo_ciy01a00-page02_1_0` | 1.00 | 16 | 0 | 2 |

| class | page | rank in pool | box (fraction of page) | thumbnail |
|---|---|---:|---|---|
| `tobacco800/logo_asg54f00_1` | `tobacco800/jdc94f00` | 23314 | 0.10 x 0.07 | `misses/tobacco800__logo_asg54f00_1-tobacco800__jdc94f00.png` |
| `ucsf/logo_bw_oval_emblem` | `ucsf/flly0212#0` | 3111 | 0.11 x 0.03 | `misses/ucsf__logo_bw_oval_emblem-ucsf__flly0212_0.png` |
| `ucsf/logo_bw_oval_emblem` | `ucsf/jnww0193#0` | 7390 | 0.11 x 0.03 | `misses/ucsf__logo_bw_oval_emblem-ucsf__jnww0193_0.png` |
| `tobacco800/logo_cgr96c00_1` | `tobacco800/alk3aa00` | 8476 | 0.09 x 0.06 | `misses/tobacco800__logo_cgr96c00_1-tobacco800__alk3aa00.png` |
| `ucsf/logo_bat_leaf` | `ucsf/ffmg0199#0` | 2162 | 0.07 x 0.05 | `misses/ucsf__logo_bat_leaf-ucsf__ffmg0199_0.png` |
| `ucsf/logo_bat_leaf` | `ucsf/fsyj0214#0` | 36523 | 0.06 x 0.04 | `misses/ucsf__logo_bat_leaf-ucsf__fsyj0214_0.png` |
