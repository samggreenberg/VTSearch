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

---

# Wave 2 — the box-size banding axis

**Run:** array `496454`, drained 18:36:41 · `/expscratch/sgreenberg/bench-vgbox/results`

99 cells / 14,042 steps over `vg_box_{small,medium,large}` — 12,000 images each,
banded on voted-box area at the DINOv3 patch geometry (small = under one patch,
1/196; medium → the smallest HAC leaf, 1/12; large → the 80 % cap), drawn from
the **whole** VG source rather than the demo pipeline's curated vocabulary.

**99/99 cells carry data. Zero starved** — against wave 1's seven. The
purpose-built banded sets are better populated than the rare tail of
`visual_genome_m` and `coco_val`, so no run failed to find a positive.

## Headline (t ≥ 100)

| arm | cost | fpr | fnr | regret | AP | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `vg_box_large` × `siglip2_l` | **0.3234** | 0.1591 | 0.1643 | 0.0709 | 0.277 | 0.906 |
| `vg_box_large` × `dinov3_patch` | **0.3243** | 0.1554 | 0.1689 | 0.1241 | 0.300 | 0.940 |
| `vg_box_large` × `siglip` | 0.3677 | 0.2050 | 0.1628 | 0.0974 | 0.283 | 0.901 |
| `vg_box_medium` × **`dinov3_patch`** | **0.5902** | 0.1725 | 0.4176 | 0.0885 | 0.143 | 0.789 |
| `vg_box_medium` × `siglip2_l` | 0.7365 | 0.3830 | 0.3535 | 0.0945 | 0.105 | 0.708 |
| `vg_box_medium` × `siglip` | 0.7471 | 0.4062 | 0.3409 | 0.0800 | 0.097 | 0.689 |
| `vg_box_small` × **`dinov3_patch`** | **0.6463** | 0.4429 | 0.2035 | 0.1964 | 0.136 | 0.823 |
| `vg_box_small` × `siglip2_l` | 0.7391 | 0.4363 | 0.3028 | 0.0902 | 0.086 | 0.683 |
| `vg_box_small` × `siglip` | 0.7774 | 0.4131 | 0.3642 | 0.0934 | 0.093 | 0.668 |

## The finding: region voting earns its cost only when the target is small

Difference in cost, whole-image best vs `dinov3_patch`:

| band | best whole-image | `dinov3_patch` | dinov3 advantage |
|---|---:|---:|---:|
| large (> 33 % of image) | 0.3234 | 0.3243 | **−0.001 (tie)** |
| medium (8–33 %) | 0.7365 | 0.5902 | **+0.146** |
| small (< 0.5 %) | 0.7391 | 0.6463 | **+0.093** |

On **large** boxes region voting buys nothing at the operating point: a box that
covers a third of the image *is* roughly the whole image, so the whole-image
vector already carries the signal — and `siglip2_l` matches it for ~1/10th the
compute (401 s median per cell vs ~40 s).

This is not "the patch model is worse there". Its **ranking is better**
(AP 0.300 vs 0.277, AUROC 0.940 vs 0.906) and it is the only arm in either wave
with a **positive `rule_inefficiency`** (+0.0208 vs −0.021 for `siglip2_l`), with
regret 0.124 vs 0.071. So on large boxes `dinov3_patch` **ranks better and cuts
worse, and the two cancel.** The ranking advantage is real and currently
unrealised — a calibration problem, not a representation problem.

Combined with wave 1 (where `dinov3_patch` won on both boxed datasets), the rule
is: **region voting pays when the target is small relative to the frame, and is
a wash when it is large.** That is a concrete product criterion for when to spend
the ~10× — and it is exactly the question `vg_box_*` was built to answer.

## Everything degrades with box size

Cost rises monotonically as boxes shrink (best arm per band): **0.323 → 0.590 →
0.646**, and ranking collapses: AP **0.30 → 0.14 → 0.14**, AUROC 0.94 → 0.79 →
0.82. The sub-patch band is genuinely hard — and with 643 sub-patch categories in
the full VG vocabulary (against 5 in the demo vocabulary), this is the first time
that band has been measured on a real sample rather than an artefact.

## Cold start gets much worse as boxes shrink

`too_few_default` share of steps:

| band | dinov3 | siglip | siglip2_l |
|---|---:|---:|---:|
| large | 4.4 % | 8.1 % | 6.5 % |
| medium | 11.2 % | 8.3 % | 5.4 % |
| small | **18.0 %** | **17.3 %** | **20.4 %** |

On the small band roughly **one step in five** never reaches the conformal path.
`degenerate` steps are also 50× wave 1's rate (**1.96 %** vs 0.04 %). The
threshold machinery still never falls back (`cut_fallback` 0/14,042), but it is
visibly working harder.

## One wrong-way trend

`vg_box_small × dinov3_patch` is the only arm whose **regret grows with votes**:
0.129 (1–20) → 0.114 (21–50) → 0.209 (51–100) → **0.196 (101–150)**. Every other
arm in both waves improves or flattens. More labels making calibration *worse* is
the signature #2825 investigated; on the sub-patch band it appears to be live.
Worth a follow-up rather than a conclusion — see the power caveat below.

## Power caveat — read this before quoting wave 2

**The banding comparison is under-powered, and the fault is mine.** I left the
scale-band category selector on for datasets that are *already* box-banded, so it
re-banded within each set and most bands came up empty. The result:

| dataset | categories | cells |
|---|---:|---|
| `vg_box_large` | 5 (`barn, court, dresser, sheet, station`) | 15 |
| `vg_box_small` | 4 (`hands, lips, mask, mustache`) | 12 |
| `vg_box_medium` | **2** (`chest, collar`) | 6 |

`vg_box_medium` rests on **two categories**. Each set has 40 available; a
prevalence-spread selection would have used far more of them. The large-vs-small
direction is big enough (0.32 vs 0.65) to survive this, and the
region-voting crossover is consistent across two bands and echoed by wave 1, but
**the medium row specifically should not be quoted as a point estimate**, and no
between-band difference smaller than ~0.05 should be read as real.

Re-running wave 2 with `CALIB_N_CATEGORIES` on a prevalence spread (the boxless
path) instead of scale bands is cheap — the pile cells are built — and is the
first thing to do before anyone acts on these numbers.
