36 classes.

### Headline (mean over classes)

| | AP | Goods found | gate precision | gate recall | gate F1 | best-cut F1 |
|---|---:|---:|---:|---:|---:|---:|
| click 0 (example sort) | 0.78 | 0.0 | 0.54 | 0.80 | 0.53 | 0.84 |
| 10 clicks | 0.90 | 8.1 | 0.76 | 0.83 | 0.73 | 0.91 |
| 25 clicks | 0.91 | 14.3 | 0.89 | 0.84 | 0.82 | 0.91 |
| final | 0.95 | 18.7 | 0.93 | 0.86 | 0.86 | 0.95 |

### Floor-style cuts at P (oracle: the deepest cut of the ranking with precision >= P)

| | recall at P=10% | recall at P=50% | recall at P=90% | gate recall |
|---|---:|---:|---:|---:|
| click 0 (example sort) | 0.83 | 0.80 | 0.72 | 0.80 |
| 10 clicks | 0.93 | 0.91 | 0.86 | 0.83 |
| 25 clicks | 0.94 | 0.92 | 0.88 | 0.84 |
| final | 0.98 | 0.96 | 0.92 | 0.86 |

### Per class, final click

| class | positives | pool | click-0 AP | final AP | found | gate P / R / F1 | recall at P=50% | retrain median |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `staver/stamp_stampds-00213_1` | 19 | 49968 | 0.42 | 0.41 | 10 | 0.50 / 0.38 / 0.43 | 0.62 | 0.3 s |
| `tobacco800/logo_asg54f00_1` | 9 | 46659 | 0.31 | 0.50 | 2 | 1.00 / 0.33 / 0.50 | 0.50 | 0.2 s |
| `ucsf/logo_p_lorillard_crest` | 10 | 46746 | 0.67 | 0.81 | 4 | 1.00 / 0.67 / 0.80 | 0.67 | 0.2 s |
| `spods/stamp_00612_1` | 31 | 49968 | 0.53 | 0.89 | 16 | 1.00 / 0.60 / 0.75 | 0.93 | 0.3 s |
| `ucsf/logo_bat_leaf` | 200 | 47907 | 0.60 | 0.90 | 50 | 0.98 / 0.90 / 0.94 | 0.90 | 4.6 s |
| `tobacco800/logo_ajj10e00_1` | 383 | 47018 | 0.68 | 0.93 | 50 | 0.94 / 0.66 / 0.78 | 0.95 | 5.3 s |
| `tobacco800/logo_afm90c00-first_1_0` | 97 | 46673 | 0.72 | 0.94 | 41 | 0.96 / 0.93 / 0.94 | 0.93 | 3.9 s |
| `tobacco800/logo_aeq93a00_1` | 116 | 46755 | 0.59 | 0.95 | 49 | nan / 0.00 / 0.00 | 0.94 | 4.2 s |
| `spods/stamp_00769_1` | 31 | 49968 | 0.72 | 0.96 | 17 | 0.92 / 0.86 / 0.89 | 1.00 | 0.3 s |
| `spods/stamp_00931_1` | 31 | 49968 | 0.85 | 0.96 | 17 | 1.00 / 0.86 / 0.92 | 1.00 | 0.3 s |
| `tobacco800/logo_ald41a00-ernest_1` | 191 | 46805 | 0.39 | 0.96 | 50 | 0.97 / 0.36 / 0.52 | 0.99 | 6.4 s |
| `tobacco800/logo_aah97e00-page02_1_0` | 354 | 46957 | 0.28 | 0.97 | 50 | 1.00 / 0.97 / 0.98 | 0.97 | 5.5 s |
| `spods/stamp_00641_1` | 31 | 49968 | 1.00 | 0.97 | 18 | 0.92 / 0.85 / 0.88 | 1.00 | 0.3 s |
| `spods/stamp_00514_1` | 31 | 49968 | 0.86 | 0.99 | 13 | 1.00 / 0.83 / 0.91 | 1.00 | 0.3 s |
| `spods/logo_00003_0` | 30 | 49968 | 1.00 | 1.00 | 10 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `spods/logo_00011_0` | 30 | 49968 | 1.00 | 1.00 | 15 | 0.94 / 1.00 / 0.97 | 1.00 | 0.3 s |
| `spods/logo_00014_0` | 30 | 49968 | 1.00 | 1.00 | 17 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `spods/logo_00023_0` | 30 | 49968 | 1.00 | 1.00 | 17 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `spods/logo_00029_0` | 30 | 49968 | 1.00 | 1.00 | 15 | 0.94 / 1.00 / 0.97 | 1.00 | 0.3 s |
| `spods/stamp_00129_1` | 30 | 49968 | 0.75 | 1.00 | 18 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `spods/stamp_00293_1` | 33 | 49968 | 0.85 | 1.00 | 20 | 0.81 / 1.00 / 0.90 | 1.00 | 0.3 s |
| `spods/stamp_00546_1` | 31 | 49968 | 0.88 | 1.00 | 15 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `spods/stamp_00577_1` | 31 | 49968 | 0.94 | 1.00 | 14 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `spods/stamp_00716_1` | 29 | 49968 | 0.85 | 1.00 | 11 | 1.00 / 0.89 / 0.94 | 1.00 | 0.3 s |
| `spods/stamp_00737_1` | 29 | 49968 | 0.96 | 1.00 | 17 | 0.92 / 1.00 / 0.96 | 1.00 | 0.3 s |
| `spods/stamp_00996_1` | 63 | 49968 | 0.90 | 1.00 | 33 | 1.00 / 1.00 / 1.00 | 1.00 | 6.0 s |
| `staver/stamp_stampds-00230_0` | 7 | 49968 | 1.00 | 1.00 | 5 | 0.40 / 1.00 / 0.57 | 1.00 | 0.2 s |
| `tobacco800/logo_azb11c00_1` | 32 | 46659 | 0.87 | 1.00 | 16 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `tobacco800/logo_bea6aa00_1` | 5 | 46659 | 1.00 | 1.00 | 2 | 1.00 / 1.00 / 1.00 | 1.00 | 0.2 s |
| `tobacco800/logo_bqz95d00_1` | 9 | 46659 | 1.00 | 1.00 | 3 | 1.00 / 0.83 / 0.91 | 1.00 | 0.2 s |
| `tobacco800/logo_cgr96c00_1` | 9 | 46659 | 0.40 | 1.00 | 3 | 1.00 / 1.00 / 1.00 | 1.00 | 0.2 s |
| `tobacco800/logo_ciy01a00-page02_1_0` | 23 | 46659 | 0.82 | 1.00 | 16 | 1.00 / 1.00 / 1.00 | 1.00 | 0.3 s |
| `tobacco800/logo_drm00d00_1` | 4 | 46659 | 1.00 | 1.00 | 1 | 0.60 / 1.00 / 0.75 | 1.00 | 0.2 s |
| `tobacco800/logo_kan00d00_1` | 5 | 46659 | 1.00 | 1.00 | 1 | 0.57 / 1.00 / 0.73 | 1.00 | 0.2 s |
| `ucsf/logo_bw_oval_emblem` | 16 | 46798 | 0.50 | 1.00 | 9 | 1.00 / 1.00 / 1.00 | 1.00 | 0.2 s |
| `ucsf/logo_rjr_script` | 52 | 47044 | 0.89 | 1.00 | 27 | 1.00 / 1.00 / 1.00 | 1.00 | 4.7 s |

### The returned set at each balance (F-beta as a share of the best cut's; one set serves every beta)

| | beta 0.5: F-beta / best = share | beta 1: F-beta / best = share | beta 2: F-beta / best = share | precision | recall |
|---|---|---|---|---:|---:|
| click 0 (example sort) | 0.51 / 0.89 = 0.59 | 0.53 / 0.84 = 0.66 | 0.62 / 0.81 = 0.78 | 0.54 | 0.80 |
| 10 clicks | 0.71 / 0.93 = 0.74 | 0.73 / 0.91 = 0.78 | 0.77 / 0.90 = 0.82 | 0.76 | 0.83 |
| 25 clicks | 0.82 / 0.94 = 0.86 | 0.82 / 0.91 = 0.87 | 0.82 / 0.92 = 0.87 | 0.89 | 0.84 |
| final | 0.88 / 0.97 = 0.90 | 0.86 / 0.95 = 0.90 | 0.85 / 0.95 = 0.89 | 0.93 | 0.86 |

Retrain wall clock: median 0.3 s, p90 6.8 s, 26% of steps over 5 s (after click 0: p90 6.6 s).

### Most helpful and most harmful clicks (credit = that click's change in AP; one observation each)

| | class | page | label | click | credit |
|---|---|---|---|---:|---:|
| harmful | `ucsf/logo_p_lorillard_crest` | `ucsf/gmkg0055#0` | good | 1 | -0.666 |
| harmful | `tobacco800/logo_asg54f00_1` | `tobacco800/asg54f00` | good | 1 | -0.261 |
| harmful | `spods/stamp_00129_1` | `spods/00136` | good | 4 | -0.175 |
| harmful | `spods/stamp_00737_1` | `spods/00751` | good | 1 | -0.121 |
| harmful | `staver/stamp_stampds-00213_1` | `staver/stampds-00225` | good | 2 | -0.119 |
| harmful | `staver/stamp_stampds-00213_1` | `staver/stampds-00220` | good | 4 | -0.098 |
| helpful | `tobacco800/logo_aah97e00-page02_1_0` | `tobacco800/uiw28e00` | good | 1 | +0.604 |
| helpful | `tobacco800/logo_ald41a00-ernest_1` | `tobacco800/avd23f00-first_1` | good | 1 | +0.420 |
| helpful | `ucsf/logo_p_lorillard_crest` | `ucsf/fggb0129#0` | good | 41 | +0.419 |
| helpful | `spods/stamp_00612_1` | `spods/00616` | good | 1 | +0.403 |
| helpful | `ucsf/logo_p_lorillard_crest` | `ucsf/fywh0113#0` | good | 40 | +0.388 |
| helpful | `tobacco800/logo_cgr96c00_1` | `ucsf/lqpg0227#0` | bad | 2 | +0.375 |

### Where the test positives sit at the final click

| class | final AP | test positives | beyond the shortlist | inside, failed the gate |
|---|---:|---:|---:|---:|
| `staver/stamp_stampds-00213_1` | 0.41 | 8 | 0 | 1 |
| `tobacco800/logo_asg54f00_1` | 0.50 | 6 | 3 | 0 |
| `ucsf/logo_bat_leaf` | 0.90 | 107 | 11 | 0 |
| `tobacco800/logo_ajj10e00_1` | 0.93 | 182 | 8 | 53 |
| `tobacco800/logo_afm90c00-first_1_0` | 0.94 | 54 | 3 | 1 |
| `tobacco800/logo_aeq93a00_1` | 0.95 | 54 | 3 | 0 |
| `tobacco800/logo_ald41a00-ernest_1` | 0.96 | 107 | 1 | 68 |
| `tobacco800/logo_aah97e00-page02_1_0` | 0.97 | 183 | 6 | 0 |

| class | page | rank in pool | box (fraction of page) | thumbnail |
|---|---|---:|---|---|
| `tobacco800/logo_asg54f00_1` | `tobacco800/kjj24f00` | 2693 | 0.10 x 0.07 | `misses/tobacco800__logo_asg54f00_1-tobacco800__kjj24f00.png` |
| `tobacco800/logo_asg54f00_1` | `tobacco800/cjb54c00` | 23169 | 0.12 x 0.09 | `misses/tobacco800__logo_asg54f00_1-tobacco800__cjb54c00.png` |
| `ucsf/logo_bat_leaf` | `ucsf/gxwl0213#0` | 2654 | 0.04 x 0.03 | `misses/ucsf__logo_bat_leaf-ucsf__gxwl0213_0.png` |
| `ucsf/logo_bat_leaf` | `ucsf/hgwd0212#0` | 16565 | 0.08 x 0.05 | `misses/ucsf__logo_bat_leaf-ucsf__hgwd0212_0.png` |
| `tobacco800/logo_ajj10e00_1` | `ucsf/fsvh0071#0` | 2262 | 0.14 x 0.04 | `misses/tobacco800__logo_ajj10e00_1-ucsf__fsvh0071_0.png` |
| `tobacco800/logo_ajj10e00_1` | `ucsf/gfxc0121#0` | 18197 | 1.00 x 0.25 | `misses/tobacco800__logo_ajj10e00_1-ucsf__gfxc0121_0.png` |
| `tobacco800/logo_afm90c00-first_1_0` | `tobacco800/aww54f00_1` | 2448 | 0.08 x 0.06 | `misses/tobacco800__logo_afm90c00-first_1_0-tobacco800__aww54f00_1.png` |
| `tobacco800/logo_afm90c00-first_1_0` | `tobacco800/cgx54f00_1` | 19768 | 0.07 x 0.06 | `misses/tobacco800__logo_afm90c00-first_1_0-tobacco800__cgx54f00_1.png` |
