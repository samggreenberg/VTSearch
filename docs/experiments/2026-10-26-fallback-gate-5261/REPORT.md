# How often the fused-cut fallback decides (issue #5261)

**Status (2026-10-26): a count over recorded cells, no new run, read at 10:41 (the #5241 SAM3 runs were still
writing). Nothing in the app changes.**

**Never, on the shipped path.** Across every recorded State of the App and acquisition run since #4878 added the
column (110 runs, 124,421 cells, 14.06M trained-head clicks under a balance), **no** click had a trained head that drew
no labels line. Every trained-head base row reads `line_head_count` 0 or 1 with `threshold_provenance = balance`, in
both modes, at every beta and every vote count. So the fused cut (`_fused_threshold`) never drew a trained head's line
there, and Autopilot never left the labels line's even-odds target for the line − 4 re-cut. With no click to price,
the F-beta gap against `oracle_fbeta_cut` is not measured: there is nothing to measure it on.

**Consequence.** The gate item of `population-anchored-calibration` retires that plan (its remaining items could only
move the fallback). `region-vs-binary-kappa-mechanism` drops its two real-data items; its synthetic-bench theory items
cost no GPU and stand on their own (owner, 2026-10-10), so they stay.

## What was counted

A calibration base row (`gmm_variant` empty) is the arm's own operating point at that click. Under a balance it
records `line_head_count` 1 where the head's count line served, 0 where the labels' class model cut, and −1 where no
labels line was drawn (#4878); its provenance is `balance` only when the labels line drew it. `fit_labels_line`
returns nothing only when even the head's in-sample scores of the labels put the Goods no higher than the Bads, so a
trained head in the fallback state shows up as `detector_tier = trained` with `line_head_count = −1` or a provenance
other than `balance`. The scan counted those rows, and kept each with its F-beta and `oracle_fbeta` for pricing.

Rows with `line_head_count = −1` do exist, 4.09M of them, but every one is a `query` (or `centroid`) row: a click where
the app still shows the typed query, before any head is trained (`app_trained = 0`). Those clicks carry no trained row
at all, in the held clicks checked by hand (binary to t = 29, region t = 4–6), so no fused cut is in play there either.

## Rate by mode and vote count

Trained-head clicks under a balance, and those where the fallback decided. Binary is `whole_image` (SigLIP); region
pools `max_patch` (175,151), `max_patch_hac` (16,596) and `max_patch_pca_hac` (16,589). Datasets: `coco_better`
13.57M, `o365_better` 0.49M. A binary head first trains at click 30, a region head at click 7.

| votes | binary clicks | fallback | region clicks | fallback |
|---|---:|---:|---:|---:|
| 1–10 | — | — | 7,147 | 0 |
| 11–25 | — | — | 18,470 | 0 |
| 26–50 | 1,251,313 | 0 | 32,369 | 0 |
| 51–100 | 4,876,107 | 0 | 80,806 | 0 |
| 101–150 | 5,387,859 | 0 | 64,017 | 0 |
| >150 | 2,340,141 | 0 | 5,527 | 0 |
| **all** | **13,855,420** (95 runs) | **0** | **208,336** (15 runs) | **0** |

By beta: binary 4.53M / 4.95M / 4.38M clicks at beta 1/4, 1, 4; region 21K / 187K / 132 (region at beta 4 is
barely sampled). Zero fallback clicks in every cell.

Zero out of n bounds the rate at about 3/n (95%): 2e-7 per binary click and 1.4e-5 per region click. Clicks in one
trajectory are not independent, so read the bound per cell instead: under 3e-5 over 124,421 cells. Either way the
fallback is not a regime the shipped path visits.

## What this does not cover

- **A library caller with no balance** (`beta=None`) still gets the fused cut on every click. The app always sets
  one, so that path has no user; the regression `docs/ML.md` records (fusion vs `cap50` on binary voting) bears only
  there.
- **Runs before #4878** carry no `line_head_count` and are not counted.

## Reproducing

```bash
# runs.txt: one run directory per line (the 110 used are in /expscratch/sgreenberg/fallback-gate-5261/runs.txt)
sbatch -c 16 --mem=24G --wrap "python scripts/experiments/state_of_app/fallback_gate_5261.py scan --runs runs.txt --out ."
python scripts/experiments/state_of_app/fallback_gate_5261.py tables --out .
```

`scan` takes ~12 minutes at 16 CPUs and writes `agg.csv` (base-row counts) and `odd.csv` (the fallback rows; empty
here).
