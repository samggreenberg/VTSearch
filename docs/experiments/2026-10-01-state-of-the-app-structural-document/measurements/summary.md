36 classes.

### Headline (mean over classes)

| | AP | Goods found | gate precision | gate recall | gate F1 | best-cut F1 |
|---|---:|---:|---:|---:|---:|---:|
| click 0 (example sort) | 0.78 | 0.0 | 0.55 | 0.79 | 0.53 | 0.82 |
| 10 clicks | 0.90 | 8.1 | 0.39 | 0.92 | 0.44 | 0.92 |
| 25 clicks | 0.91 | 13.8 | 0.38 | 0.93 | 0.43 | 0.93 |
| final | 0.91 | 17.9 | 0.37 | 0.93 | 0.42 | 0.93 |

### Floor-style cuts at P (oracle: the deepest cut of the ranking with precision >= P)

| | recall at P=10% | recall at P=50% | recall at P=90% | gate recall |
|---|---:|---:|---:|---:|
| click 0 (example sort) | 0.81 | 0.80 | 0.70 | 0.79 |
| 10 clicks | 0.94 | 0.91 | 0.87 | 0.92 |
| 25 clicks | 0.94 | 0.92 | 0.88 | 0.93 |
| final | 0.94 | 0.92 | 0.88 | 0.93 |

### Per class, final click

| class | positives | pool | click-0 AP | final AP | found | gate P / R / F1 | recall at P=50% | retrain median |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `tobacco800/logo_asg54f00_1` | 9 | 46659 | 0.67 | 0.41 | 2 | 0.09 / 0.67 / 0.15 | 0.33 | 0.1 s |
| `ucsf/logo_bw_oval_emblem` | 16 | 46798 | 0.30 | 0.50 | 3 | 1.00 / 0.50 / 0.67 | 0.50 | 0.1 s |
| `tobacco800/logo_afm90c00-first_1_0` | 97 | 46673 | 0.59 | 0.70 | 42 | 0.88 / 0.70 / 0.78 | 0.70 | 1.9 s |
| `ucsf/logo_bat_leaf` | 200 | 47907 | 0.40 | 0.71 | 50 | 0.98 / 0.69 / 0.81 | 0.70 | 1.9 s |
| `tobacco800/logo_cgr96c00_1` | 9 | 46659 | 0.25 | 0.75 | 5 | 0.14 / 0.75 / 0.23 | 0.75 | 0.1 s |
| `ucsf/logo_p_lorillard_crest` | 10 | 46746 | 0.65 | 0.78 | 6 | 0.25 / 0.75 / 0.38 | 0.75 | 0.1 s |
| `staver/stamp_stampds-00213_1` | 19 | 49968 | 0.78 | 0.80 | 7 | 0.12 / 1.00 / 0.21 | 0.82 | 0.1 s |
| `tobacco800/logo_aeq93a00_1` | 116 | 46755 | 0.42 | 0.86 | 50 | 0.98 / 0.89 / 0.93 | 0.90 | 2.2 s |
| `tobacco800/logo_azb11c00_1` | 32 | 46659 | 0.69 | 0.88 | 14 | 1.00 / 0.88 / 0.93 | 0.88 | 0.1 s |
| `tobacco800/logo_ald41a00-ernest_1` | 191 | 46805 | 0.49 | 0.88 | 50 | 0.99 / 0.88 / 0.93 | 0.88 | 2.6 s |
| `spods/stamp_00931_1` | 31 | 49968 | 0.71 | 0.91 | 14 | 0.25 / 0.94 / 0.40 | 0.94 | 0.1 s |
| `spods/stamp_00716_1` | 29 | 49968 | 0.93 | 0.92 | 18 | 0.04 / 1.00 / 0.08 | 1.00 | 0.2 s |
| `tobacco800/logo_ciy01a00-page02_1_0` | 23 | 46659 | 0.75 | 0.94 | 7 | 1.00 / 0.88 / 0.93 | 0.94 | 0.1 s |
| `spods/stamp_00769_1` | 31 | 49968 | 0.92 | 0.95 | 14 | 0.11 / 1.00 / 0.20 | 0.94 | 0.1 s |
| `spods/stamp_00612_1` | 31 | 49968 | 0.67 | 0.96 | 14 | 0.12 / 1.00 / 0.22 | 1.00 | 0.1 s |
| `spods/stamp_00577_1` | 31 | 49968 | 1.00 | 0.97 | 17 | 0.03 / 1.00 / 0.07 | 1.00 | 0.2 s |
| `spods/stamp_00737_1` | 29 | 49968 | 0.98 | 0.97 | 12 | 0.06 / 1.00 / 0.11 | 1.00 | 0.1 s |
| `tobacco800/logo_ajj10e00_1` | 383 | 47018 | 0.69 | 0.97 | 50 | 0.47 / 0.97 / 0.63 | 0.97 | 2.4 s |
| `spods/stamp_00514_1` | 31 | 49968 | 0.76 | 0.99 | 18 | 0.07 / 1.00 / 0.13 | 1.00 | 0.2 s |
| `spods/stamp_00641_1` | 31 | 49968 | 0.84 | 0.99 | 13 | 0.08 / 1.00 / 0.14 | 1.00 | 0.1 s |
| `tobacco800/logo_aah97e00-page02_1_0` | 354 | 46957 | 0.18 | 0.99 | 50 | 1.00 / 0.99 / 0.99 | 0.99 | 2.5 s |
| `spods/logo_00003_0` | 30 | 49968 | 1.00 | 1.00 | 20 | 0.20 / 1.00 / 0.34 | 1.00 | 0.1 s |
| `spods/logo_00011_0` | 30 | 49968 | 1.00 | 1.00 | 15 | 0.18 / 1.00 / 0.30 | 1.00 | 0.1 s |
| `spods/logo_00014_0` | 30 | 49968 | 1.00 | 1.00 | 13 | 0.35 / 1.00 / 0.52 | 1.00 | 0.1 s |
| `spods/logo_00023_0` | 30 | 49968 | 1.00 | 1.00 | 13 | 0.08 / 1.00 / 0.15 | 1.00 | 0.1 s |
| `spods/logo_00029_0` | 30 | 49968 | 1.00 | 1.00 | 15 | 0.36 / 1.00 / 0.53 | 1.00 | 0.1 s |
| `spods/stamp_00129_1` | 30 | 49968 | 0.89 | 1.00 | 12 | 0.11 / 1.00 / 0.20 | 1.00 | 0.1 s |
| `spods/stamp_00293_1` | 33 | 49968 | 0.95 | 1.00 | 13 | 0.47 / 1.00 / 0.63 | 1.00 | 0.1 s |
| `spods/stamp_00546_1` | 31 | 49968 | 0.84 | 1.00 | 16 | 0.06 / 1.00 / 0.11 | 1.00 | 0.1 s |
| `spods/stamp_00996_1` | 63 | 49968 | 0.88 | 1.00 | 30 | 0.10 / 1.00 / 0.19 | 1.00 | 3.0 s |
| `staver/stamp_stampds-00230_0` | 7 | 49968 | 1.00 | 1.00 | 2 | 0.38 / 1.00 / 0.56 | 1.00 | 0.1 s |
| `tobacco800/logo_bea6aa00_1` | 5 | 46659 | 1.00 | 1.00 | 3 | 1.00 / 1.00 / 1.00 | 1.00 | 0.1 s |
| `tobacco800/logo_bqz95d00_1` | 9 | 46659 | 0.92 | 1.00 | 6 | 0.04 / 1.00 / 0.08 | 1.00 | 0.1 s |
| `tobacco800/logo_drm00d00_1` | 4 | 46659 | 1.00 | 1.00 | 3 | 0.02 / 1.00 / 0.03 | 1.00 | 0.1 s |
| `tobacco800/logo_kan00d00_1` | 5 | 46659 | 1.00 | 1.00 | 4 | 0.08 / 1.00 / 0.15 | 1.00 | 0.1 s |
| `ucsf/logo_rjr_script` | 52 | 47044 | 0.97 | 1.00 | 25 | 0.31 / 1.00 / 0.48 | 1.00 | 1.9 s |

### Most helpful and most harmful clicks (credit = that click's change in AP; one observation each)

| | class | page | label | click | credit |
|---|---|---|---|---:|---:|
| harmful | `staver/stamp_stampds-00213_1` | `staver/stampds-00213` | good | 4 | -0.407 |
| harmful | `ucsf/logo_p_lorillard_crest` | `ucsf/gnwh0113#0` | good | 1 | -0.297 |
| harmful | `tobacco800/logo_asg54f00_1` | `tobacco800/bjn43c00-page02_2` | good | 1 | -0.250 |
| harmful | `tobacco800/logo_afm90c00-first_1_0` | `tobacco800/egg15f00-page2-full_1` | good | 12 | -0.093 |
| harmful | `tobacco800/logo_afm90c00-first_1_0` | `tobacco800/mma35f00` | good | 34 | -0.069 |
| harmful | `spods/stamp_00769_1` | `spods/00791` | good | 1 | -0.067 |
| helpful | `ucsf/logo_p_lorillard_crest` | `ucsf/fydf0107#0` | good | 2 | +0.508 |
| helpful | `tobacco800/logo_cgr96c00_1` | `tobacco800/sia26d00` | good | 1 | +0.501 |
| helpful | `tobacco800/logo_aah97e00-page02_1_0` | `ucsf/fmgk0164#0` | good | 1 | +0.465 |
| helpful | `tobacco800/logo_aeq93a00_1` | `ucsf/ffbf0003#0` | good | 1 | +0.342 |
| helpful | `staver/stamp_stampds-00213_1` | `staver/stampds-00223` | good | 12 | +0.338 |
| helpful | `ucsf/logo_bat_leaf` | `ucsf/hywy0206#0` | good | 1 | +0.331 |

### Where the test positives sit at the final click

| class | final AP | test positives | beyond the shortlist | inside, failed the gate |
|---|---:|---:|---:|---:|
| `tobacco800/logo_asg54f00_1` | 0.41 | 3 | 1 | 0 |
| `ucsf/logo_bw_oval_emblem` | 0.50 | 10 | 5 | 0 |
| `tobacco800/logo_afm90c00-first_1_0` | 0.70 | 43 | 13 | 0 |
| `ucsf/logo_bat_leaf` | 0.71 | 93 | 28 | 1 |
| `tobacco800/logo_cgr96c00_1` | 0.75 | 4 | 1 | 0 |
| `ucsf/logo_p_lorillard_crest` | 0.78 | 4 | 0 | 1 |
| `tobacco800/logo_aeq93a00_1` | 0.86 | 62 | 6 | 1 |
| `tobacco800/logo_azb11c00_1` | 0.88 | 16 | 2 | 0 |
| `tobacco800/logo_ald41a00-ernest_1` | 0.88 | 84 | 10 | 0 |
| `spods/stamp_00931_1` | 0.91 | 17 | 1 | 0 |
| `tobacco800/logo_ciy01a00-page02_1_0` | 0.94 | 16 | 1 | 1 |
| `tobacco800/logo_ajj10e00_1` | 0.97 | 201 | 6 | 0 |
| `tobacco800/logo_aah97e00-page02_1_0` | 0.99 | 171 | 2 | 0 |

| class | page | rank in pool | box (fraction of page) | thumbnail |
|---|---|---:|---|---|
| `tobacco800/logo_asg54f00_1` | `tobacco800/jdc94f00` | 14857 | 0.10 x 0.07 | `misses/tobacco800__logo_asg54f00_1-tobacco800__jdc94f00.png` |
| `ucsf/logo_bw_oval_emblem` | `ucsf/frvw0195#0` | 4225 | 0.10 x 0.02 | `misses/ucsf__logo_bw_oval_emblem-ucsf__frvw0195_0.png` |
| `ucsf/logo_bw_oval_emblem` | `ucsf/fmly0212#0` | 20100 | 0.09 x 0.02 | `misses/ucsf__logo_bw_oval_emblem-ucsf__fmly0212_0.png` |
| `tobacco800/logo_afm90c00-first_1_0` | `tobacco800/xik90c00_1` | 2007 | 0.10 x 0.08 | `misses/tobacco800__logo_afm90c00-first_1_0-tobacco800__xik90c00_1.png` |
| `tobacco800/logo_afm90c00-first_1_0` | `tobacco800/btt85f00-page2_3` | 6186 | 0.08 x 0.06 | `misses/tobacco800__logo_afm90c00-first_1_0-tobacco800__btt85f00-page2_3.png` |
| `ucsf/logo_bat_leaf` | `ucsf/fspc0214#0` | 2275 | 0.09 x 0.06 | `misses/ucsf__logo_bat_leaf-ucsf__fspc0214_0.png` |
| `ucsf/logo_bat_leaf` | `ucsf/fsyj0214#0` | 24495 | 0.06 x 0.04 | `misses/ucsf__logo_bat_leaf-ucsf__fsyj0214_0.png` |
