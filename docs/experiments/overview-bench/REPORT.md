# VTSearch overview benchmark: what a current user actually gets

**Run:** 2026-08-12 · branch `claude/vts-benchmark` · array `496044`
**Data:** `/expscratch/sgreenberg/bench-overview/results` · tables in `ANALYSIS_TABLES.txt`

Every *behavioural* knob is left at its shipped default — head `linear`,
`safe_thresholds=False`, `calibrate_count=2`, acquisition inclusion offset
`-1`, production `max_patch` geometry. Only sizing knobs (which datasets,
embedders, categories, seeds) were set. That is what makes these numbers
readable as a baseline rather than as an arm.

## The grid

3 datasets × 3 embedders × 6–8 categories × 3 seeds × 150 votes = **189 cells,
26,538 steps**. `siglip` (shipped default) and `siglip2_l` (premium) are
whole-image; `dinov3_patch` carries the patch geometry and is the only
embedder that can region-vote — and only on a boxed dataset.

| dataset | medias | categories | region-voting |
|---|---:|---|---|
| `visual_genome_m` | 4,193 | 8, scale-banded | `dinov3_patch` only |
| `coco_val` | 4,952 | 7, scale-banded | `dinov3_patch` only |
| `caltech101_m` | 838 | 6, prevalence-spread | none (boxless) |

## Coverage — and what was dropped

**189/189 cells ran, 0 failed, 0 zero-byte. But only 182 carry data.**

Seven cells (3.7%) emitted a header and no rows. This is not a harness fault
and not a disk incident: **no row is ever emitted with `n_good == 0`** (verified
across all 26,538 rows — minimum `n_good` is 1). A cell emits its first row only
once the first true positive has been found. Those seven runs **never surfaced a
single positive in 150 votes.**

| cell | category | prevalence |
|---|---|---|
| `coco_val` × all 3 embedders, seed 0 | `refrigerator` | 101 / 4,952 |
| `coco_val` × all 3 embedders, seed 1 | `sports ball` | 169 / 4,952 |
| `visual_genome_m` × `siglip`, seed 1 | `ball` | 51 / 4,193 |

That the *same* (category, seed) fails across all three embedders says this is
the sim/test draw and the acquisition loop, not the representation.

**This is a finding, not an exclusion.** Read as a user-facing rate: on the
rarest categories, roughly one run in twenty-seven is a total loss — 150 clicks,
nothing found, no model. All averages below are over the 182 cells that produced
data, which means they are conditioned on the run having worked at all and are
therefore **optimistic**.

## Headline (deep regime, t ≥ 100)

cost = fpr + fnr; regret is against the oracle threshold on the same scores.

| arm | cost | fpr | fnr | regret | oracle cost | AP | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `caltech101_m` × `siglip2_l` | 0.0013 | 0.0013 | 0.0000 | 0.0013 | 0.0000 | 1.000 | 1.000 |
| `caltech101_m` × `siglip` | 0.0039 | 0.0039 | 0.0000 | 0.0039 | 0.0000 | 1.000 | 1.000 |
| `caltech101_m` × `dinov3_patch` | 0.0047 | 0.0038 | 0.0009 | 0.0031 | 0.0016 | 1.000 | 1.000 |
| `coco_val` × **`dinov3_patch`** | **0.1524** | 0.0920 | 0.0604 | 0.0496 | 0.1029 | 0.787 | 0.976 |
| `coco_val` × `siglip2_l` | 0.2019 | 0.1154 | 0.0865 | 0.0618 | 0.1401 | 0.711 | 0.955 |
| `coco_val` × `siglip` | 0.2177 | 0.0711 | 0.1466 | 0.0448 | 0.1729 | 0.695 | 0.942 |
| `visual_genome_m` × **`dinov3_patch`** | **0.3242** | 0.1819 | 0.1423 | 0.1100 | 0.2142 | 0.525 | 0.913 |
| `visual_genome_m` × `siglip2_l` | 0.3666 | 0.2438 | 0.1228 | 0.0960 | 0.2706 | 0.457 | 0.899 |
| `visual_genome_m` × `siglip` | 0.3918 | 0.3006 | 0.0912 | 0.1113 | 0.2805 | 0.428 | 0.898 |

### What works

**Region voting wins wherever it is available.** `dinov3_patch` is the best arm
on both boxed datasets — cost 0.152 vs 0.202/0.218 on COCO, 0.324 vs 0.367/0.392
on VG — and it wins on ranking too (AP 0.787 vs 0.711/0.695; 0.525 vs
0.457/0.428). It is the only arm whose Good votes carry box supervision, and it
converts that into both a better ranking and a better operating point. This is
now measured in **two** region-voting environments, not one.

**The threshold machinery is clean.** Across 26,538 steps: `cut_fallback` fires
**0** times, `degenerate` on **0.04%**, and 93–99% of steps take the `conformal`
path. Whatever is wrong below is not the cut rule failing to compute.

### What needs work

**1. Positives are desperately scarce — this is the binding constraint.**
Median positives found after 150 votes: **4** (caltech × siglip) to **11**
(coco × siglip). The traces are worse than the medians suggest:

```
visual_genome_m x dinov3_patch, category=bed, prevalence=0.020
  t=14  n_good=3 ... t=122 n_good=3 ... t=146 n_good=4
```

Three positives held for **120 consecutive votes**. The slowest successful cell
(`visual_genome_m × siglip2_l`, `ball`) took **80 votes to find its first
positive**. Seven cells never found one at all. Everything else in this report is
downstream of that: a linear head fit on 3–7 positives is what produces the
regret below.

**2. Regret is calibration transfer, not the cut rule.** Decomposing regret into
`rule_inefficiency + calibration_shift` on the two hard datasets:

| arm | regret | rule_ineff | cal_shift | cal share |
|---|---:|---:|---:|---:|
| `visual_genome_m` × `dinov3_patch` | 0.1119 | −0.0041 | 0.1160 | 1.04 |
| `visual_genome_m` × `siglip` | 0.1113 | −0.0220 | 0.1333 | 1.20 |
| `coco_val` × `dinov3_patch` | 0.0496 | −0.0179 | 0.0675 | 1.36 |
| `coco_val` × `siglip2_l` | 0.0618 | −0.0040 | 0.0658 | 1.07 |

`rule_inefficiency` is **negative everywhere**: the shipped rule already picks a
better cut than the in-sample calibration optimum. The entire regret — and
slightly more — is `calibration_shift`, i.e. the sim→test move. This
independently reproduces #2836's finding that the residual gap is transfer no
cut rule can close. *Effort spent on smarter cut rules is effort spent on the
term that is already negative.*

(The caltech `cal_share` values of 9.8–77.6 in the raw tables are artefacts of a
near-zero denominator on a saturated dataset — ignore them.)

**3. `caltech101_m` is saturated and should be retired from this benchmark.**
Cost 0.001–0.005, AP 1.000, AUROC 1.000 on all three embedders. It cannot
distinguish anything and it drags every cross-dataset average toward zero.

**4. Cold start is visible but small.** `too_few_default` covers 1–7% of steps
(worst: VG × `dinov3_patch` at 7.2%), concentrated at low `t`. The first emitted
step routinely shows a wild cost (e.g. 1.05, 0.95) before conformal takes over
by t≈14.

**5. Region voting costs ~10× per run.** Cumulative wall-clock per cell: ~110 s
whole-image vs **17–19 min** patch (1,121 s max on COCO). Whether the cost
table above justifies that is a product call, but it should be a conscious one.

## The vote-count axis

Cost falls steeply then flattens; regret flattens earlier and stays put.

| arm | cost 1–20 | 21–50 | 51–100 | 101–150 | regret 1–20 | 101–150 |
|---|---:|---:|---:|---:|---:|---:|
| `visual_genome_m` × `dinov3_patch` | 0.484 | 0.383 | 0.345 | 0.324 | 0.169 | 0.110 |
| `visual_genome_m` × `siglip` | 0.672 | 0.503 | 0.434 | 0.392 | 0.206 | 0.112 |
| `coco_val` × `dinov3_patch` | 0.345 | 0.221 | 0.191 | 0.152 | 0.151 | 0.050 |
| `coco_val` × `siglip` | 0.586 | 0.356 | 0.256 | 0.218 | 0.193 | 0.045 |

Between 50 and 150 votes — two thirds of the clicks — VG × `dinov3_patch`
improves cost by 0.021. **The second hundred votes buys almost nothing**, which
is consistent with finding (1): those votes are nearly all Bad, and Bad votes
past saturation add little.

## Caveats

- `caltech101_m × dinov3_patch` is a **new pairing** (not in `dev`). It is
  DINOv3 as a whole-image embedder — no boxes, whole-image Good pile, whole-image
  Bad pile, whole-image haystack.
- **COCO's `sub_patch` band is under-populated**: 1 candidate (`sports ball`)
  against a target of 2, so COCO's smallest-box band rests on one category —
  and that category is one of the two that starved. `vg_box_small` (wave 2)
  is the proper cover.
- Averages are over the 182 cells that produced data, so they are conditioned on
  the run having worked (see Coverage).
- 3 seeds. Differences smaller than roughly 0.02 in cost should not be read as
  real without a paired test.

## Follow-ups worth filing

1. **A zero-positive run is silent.** A cell that never finds a positive writes a
   header and exits 0. It should say so — this is exactly the shape that hid
   #2877. A one-line warning plus a `starved` column would make the 3.7% visible
   without an analyst noticing a row-count discrepancy.
2. **Acquisition, not calibration, is the frontier.** `rule_inefficiency` is
   already negative; positives are the scarce resource. This is the same
   direction #2876 found when decoupling the selector threshold.
3. **Retire `caltech101_m`** from overview benchmarks; substitute a dataset that
   discriminates.
