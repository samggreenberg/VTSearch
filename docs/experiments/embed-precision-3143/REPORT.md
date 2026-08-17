# Half precision for image embedding: what it costs, and what costs more

**Run:** 2026-08-17 · branch `claude/issue-3143-fp16` · GPU jobs `507429`–`507434`,
`507464`, `507543`, `507547`, `507728` · bench arrays `507479`/`507493` (6-seed)
and `507773`/`507786` (64-seed power run)
**Data:** `/expscratch/sgreenberg/precision-3143/{piles,results,bench,bench-power}`
**Code:** `scripts/experiments/precision/`

Issue [#3143](https://github.com/samggreenberg/VTSearch/issues/3143) asks whether
the image encoders can run in half precision. It reports the `siglip2_l` GPU
forward at **4.2×** faster in fp16 and requires three things before the default
flips: rebuild a pile cell in both precisions, report cosine drift and rank
correlation, and re-run a benchmark arm both ways to confirm cost/AP move by less
than the 0.005 the calibration studies resolve.

All three ran. The answer to the question as posed is that **fp16 is a very small
perturbation** — about 3e-6 in cosine, with retrieval order intact. But the study
turned up two things that matter more than the answer:

1. **The speedup mostly is not there any more.** End to end, fp16 buys the
   **shipped default embedder nothing at all** (0.99×). The 4.2× was a
   forward-only figure, and PR #3151 has since overlapped decode with the
   forward, so on `siglip` the forward is no longer the bottleneck. Only
   `siglip2_l` still gains (2.0–2.5×).
2. **Something already shipped perturbs the vectors 50× harder than fp16 does.**
   Two `fp32` runs of the same commit on two different GPU nodes disagree by
   **1.5e-4** on `siglip2_l` — where fp16 on a fixed node costs 2.9e-6. This is
   reproducible and is *not* explained by precision, TF32, or cuDNN algorithm
   selection. #3144 landed a GPU auto-pick (PR #3150) on the stated grounds that
   "cross-GPU fp32 drift is ~1e-7 — far below anything the studies resolve."
   That premise does not hold for `siglip2_l`.

---

# Verdict

| question | answer |
|---|---|
| Does fp16 change the vectors? | Yes, by **2.9e-6** (`siglip2_l`) / **1.3e-6** (`siglip`) median 1−cos, within a fixed node. |
| Does it change retrieval order? | No. Spearman ρ = **1 ± 4e-7**, top-10/50/100 overlap **100%**, same top-1 item on every category. |
| Does it change the benchmark? | No detectable effect at 1013 paired cells (largest 1.3 SE). `regret` is **resolved below** the 0.005 margin; `cost` (+0.0028 ± 0.0023) and AP (−0.00085 ± 0.0021) are centred within 0.003 but their intervals graze it. |
| Is it worth adopting? | **On `siglip2_l` yes** (2.0–2.5× for 2.9e-6). **On `siglip` there is nothing to buy** (0.99×). |
| Should the default flip globally? | **No.** A global flip pays a real (if tiny) numeric cost on the default embedder for zero speed, and would strand the fp32 pile. |
| Which implementation? | **Weight cast** (`fp16`). `autocast_fp16` drifts slightly *less* (2.0e-6) but is slower (1.55× vs 2.03×). **`bf16` is disqualified**: 1.3e-4 drift, as disruptive as changing hardware. |

**Recommendation.** Keep `VTSEARCH_EMBED_PRECISION=fp32` as the default. Adopt
`fp16` selectively for the heavy encoders where the forward dominates, as a
documented opt-in — which is what this PR ships. Do **not** rebuild the shared
pile in fp16: it is all cells or none (per the issue), and the gain on the two
cheap columns is zero.

---

# 1. What was run

Ten arms, one dataset (`visual_genome_m`, 4193 medias), two embedders. The axis
under test is precision; the **card is crossed with it deliberately**, because
"fp16 moves cosines by 1e-3" is not a finding without a denominator.

| arm | precision | card / node | role |
|---|---|---|---|
| `fp32_l40s` | fp32 | L40S | reference |
| `fp32_v100` | fp32 | V100 (rack5n03) | control — same math, different card |
| `fp16_l40s` | fp16 weight cast | L40S | candidate |
| `fp16_v100` | fp16 weight cast | V100 (rack5n03) | candidate on a 2nd card |
| `autocast_l40s` | `torch.autocast` fp16 | L40S | candidate — fp32 weights, per-op cast |
| `bf16_l40s` | bf16 weight cast | L40S | candidate — needs sm_80+ |
| `fp32_notf32_l40s` | fp32, TF32 off | L40S | diagnostic (added mid-run) |
| `fp32_det_l40s` | fp32, cuDNN deterministic | L40S | diagnostic (added mid-run) |
| `fp32_det_v100` | fp32, cuDNN deterministic | V100 (rack5n03) | diagnostic (added mid-run) |
| `fp32_v100_rack7n03` | fp32 | V100 (**rack7n03**) | diagnostic (added mid-run) |

`dinov3_patch` is excluded on purpose, and not for cost: its patch grids are
**already stored `float16`** (`vtscore/datasets/stages/embedding.py`), so its
region path is quantised to half before any of this. It is the arm where fp16
compute is least likely to matter and a null would be least informative — but
nobody has measured what that *storage* cast costs the max-pooled region score,
which is a different question and is filed as **#3159**.

**Nothing touched the shared pile.** Each arm built its own side pile; weights
were read from the shared `models` dir so no arm re-downloaded.

## The premise checks, before any number

Three gates ran, in this order, because a knob that silently did nothing looks
exactly like a treatment with no effect (#2877, #2897, #2905):

- **Build probe** — resolved precision, parameter dtype, card name, compute
  capability, and TF32/determinism flags recorded per arm; the build **refuses to
  write cells** if any contradicts the arm table. All ten arms matched.
- **`check_arms.py`** — provenance present, the card each arm *names* is the card
  it *ran on*, all arms cover the same 4193 medias, no zero-byte cells, and the
  fp32 rebuild reproduces the **published** pile cell.
- **`verify_pairing.py`** — both bench arms select identical categories in
  identical order with identical exemplar candidates. Verified for both the
  6-seed and 64-seed grids.

---

# 2. Throughput: the 4.2× is mostly gone

End-to-end wall time around one `(dataset, embedder)` build — model load, dataset
load, decode, processor, forward, pickling. Same commit, same images, one node per
row. **Speedups are within a card**; across cards is #3144's effect, not this one.

| arm | card | `siglip` | vs fp32 | `siglip2_l` | vs fp32 |
|---|---|---:|---:|---:|---:|
| `fp32_l40s` | L40S | 75 s | (base) | 177 s | (base) |
| `fp16_l40s` | L40S | 76 s | **0.99×** | 87 s | **2.03×** |
| `autocast_l40s` | L40S | 85 s | 0.89× | 114 s | 1.55× |
| `bf16_l40s` | L40S | 72 s | 1.05× | 113 s | 1.56× |
| `fp32_v100` | V100 | 144 s | (base) | 442 s | (base) |
| `fp16_v100` | V100 | 145 s | **0.99×** | 179 s | **2.47×** |

**`siglip` gains nothing, on either card.** That is the single most
decision-relevant number here and it is not in the issue, because the issue
measured the forward in isolation (1.3 s → 0.4 s, 3.3×) before #3151 landed. With
decode overlapped, base SigLIP at 4193 images is no longer GPU-bound, so making
the forward 3× faster moves the wall clock by 1%.

`autocast` is *slower than fp32* on `siglip` (0.89×): it adds per-op cast overhead
to a stage that was not the bottleneck.

**The card is worth more than the precision, for the default embedder.** fp32
L40S vs fp32 V100: **1.92×** on `siglip`, **2.50×** on `siglip2_l`. That
independently reproduces #3144's 1.7×/2.3× on different images and a different
commit, so PR #3150's auto-pick is well founded as a *performance* change.

---

# 3. Vector drift

Per-media 1−cos against the same node's fp32, over all 4193 medias. Reported as a
distribution: the mean of a long-tailed drift is the number that hides the tail.

| embedder | arm | median | p95 | max | mean ± SE |
|---|---|---:|---:|---:|---:|
| `siglip` | `autocast_l40s` | 9.2e-07 | 2.5e-06 | 2.4e-04 | 1.3e-06 ± 6.6e-08 |
| `siglip` | `fp16_l40s` | 1.3e-06 | 3.3e-06 | 8.1e-05 | 1.7e-06 ± 3.5e-08 |
| `siglip` | `fp16_v100` | 1.3e-06 | 3.6e-06 | 2.6e-04 | 1.8e-06 ± 7.1e-08 |
| `siglip` | `bf16_l40s` | 7.3e-05 | 1.6e-04 | 6.1e-03 | 8.7e-05 ± 2.0e-06 |
| `siglip2_l` | `autocast_l40s` | 2.0e-06 | 9.3e-06 | 1.9e-04 | 3.3e-06 ± 8.9e-08 |
| `siglip2_l` | `fp16_l40s` | 2.9e-06 | 1.2e-05 | 3.5e-04 | 4.4e-06 ± 1.2e-07 |
| `siglip2_l` | `fp16_v100` | 2.8e-06 | 1.2e-05 | 3.9e-04 | 4.6e-06 ± 1.6e-07 |
| `siglip2_l` | `bf16_l40s` | 1.3e-04 | 3.2e-04 | 3.2e-03 | 1.6e-04 ± 2.0e-06 |

Figures: `figures/drift_siglip.png`, `figures/drift_siglip2_l.png` (bar = median,
whisker = p95, log scale, floor arm in grey).

Three readings:

- **fp16 costs ~3e-6, and it costs the same on both cards** (2.9e-6 vs 2.8e-6 on
  `siglip2_l`). There is no card-dependent fp16 penalty.
- **`autocast` drifts slightly less than the weight cast** (2.0e-6 vs 2.9e-6),
  which is what keeping softmax and layer norm in fp32 is supposed to buy. It is
  not worth 0.5× of the speedup.
- **`bf16` is a different animal**: 1.3e-4, ~45× the fp16 drift, because it trades
  mantissa bits for exponent range that image embedding does not need.

**Correction.** An earlier reading of this study reported fp16-on-V100 as
markedly worse (ρ = 1 ± 2.5e-5, 98% top-10 overlap). That was an artifact of
differencing the V100 arms against the **L40S** reference, which measures the
node displacement in §5 rather than the precision change. Against its own node's
fp32, `fp16_v100` is 2.8e-6 with ρ = 1 ± 5.2e-7. The tooling now requires the
reference to be named (`--reference`) and labels every table with it.

---

# 4. Retrieval order is preserved

Cosine drift is not what a user sees; the **order** is. Measured per category from
two ranking sources — the **text query** (cross-modal search) and the **exemplar
vector** (what the benchmark's startup sort actually uses). Both towers shift
under half precision, so both are measured.

Against the same node's fp32:

| embedder | arm | source | n | Spearman ρ | top-10 | top-50 | top-100 | same top-1 |
|---|---|---|---:|---|---:|---:|---:|---:|
| `siglip` | `fp16_l40s` | exemplar | 89 | 1 ± 4.6e-07 | 100% | 100% | 100% | 100% |
| `siglip` | `fp16_l40s` | text query | 100 | 1 ± 1.4e-07 | 100% | 100% | 100% | 100% |
| `siglip` | `fp16_v100` | exemplar | 89 | 1 ± 5.1e-07 | 100% | 100% | 100% | 100% |
| `siglip2_l` | `fp16_l40s` | exemplar | 89 | 1 ± 4.1e-07 | 100% | 100% | 100% | 100% |
| `siglip2_l` | `fp16_l40s` | text query | 100 | 1 ± 3.9e-07 | 100% | 100% | 100% | 100% |
| `siglip2_l` | `fp16_v100` | exemplar | 89 | 1 ± 5.2e-07 | 100% | 100% | 100% | 100% |
| `siglip2_l` | `fp16_v100` | text query | 100 | 1 ± 3.9e-07 | 99% | 100% | 100% | 99% |
| `siglip` | `bf16_l40s` | text query | 100 | 1 ± 6.7e-06 | 98% | 99% | 99% | **97%** |
| `siglip2_l` | `bf16_l40s` | text query | 100 | 1 ± 1.3e-05 | 99% | 98% | 99% | **98%** |

Figures: `figures/rank_stability_exemplar.png`,
`figures/rank_stability_text_query.png` (one dot per category, bar = mean — the
average alone would hide a single category that fell apart), and
`figures/topk_overlap.png`.

`bf16` is the only arm that changes the **top-1 result**, on 2–3% of categories.
For a search tool that is the visible failure: the first thing the user sees.

## Literal examples: every large rank move is a tie

Rank moves look alarming in isolation and are not. Each row below is the largest
move for its arm, with the score gap between the reference's items at that rank:

| arm | category | media | file | rank | → | moved | score gap to next |
|---|---|---:|---|---:|---|---:|---:|
| `fp16_l40s` (`siglip2_l`, text) | building | 1592 | `3936.jpg` | 2600 | 2697 | 97 | 2.6e-07 |
| `fp16_l40s` (`siglip`, exemplar) | chair | 2949 | `150386.jpg` | 2102 | 2044 | 58 | 6.4e-07 |
| `fp16_l40s` (`siglip`, text) | bus | 3615 | `497922.jpg` | 1805 | 1864 | 59 | 4.9e-06 |
| `bf16_l40s` (`siglip2_l`, text) | building | 1592 | `3936.jpg` | 1595 | 1154 | **441** | 5.2e-06 |
| `bf16_l40s` (`siglip2_l`, exemplar) | boat | 1875 | `4226.jpg` | 2709 | 3025 | 316 | 1.3e-04 |

A 97-place move across a 2.6e-07 cosine gap is a **tie broken differently**, not a
retrieval failure — thousands of near-identical scores sit in that band, and any
perturbation reshuffles them. Every move is in the **middle** of the ranking
(rank 800–3000 of 4193), never the head. That is why the top-k overlaps are 100%
while the full-list order technically changes.

Note `3936.jpg` under `building` appears for both fp16 and bf16: it is a
genuinely ambiguous image for that label rather than an arm-specific artifact.
Full dumps in `results/examples.json`.

---

# 5. The finding that outgrew the question

Two arms differing **only in which GPU node they ran on**, same commit, same
fp32, same images:

Pairwise median 1−cos, `siglip2_l` (full matrix in `results/pairwise_drift.csv`):

|  | `fp32_l40s` | `fp32_notf32` | `fp32_det_l40s` | `fp16_l40s` | `fp32_v100` | `fp32_det_v100` | `fp16_v100` | published |
|---|---|---|---|---|---|---|---|---|
| **`fp32_l40s`** | – | **0** | **0** | 2.9e-06 | 1.5e-04 | 1.5e-04 | 1.5e-04 | 2.7e-12 |
| **`fp32_v100`** | 1.5e-04 | 1.5e-04 | 1.5e-04 | 1.5e-04 | – | **0** | 2.8e-06 | 1.5e-04 |

The arms cluster **by node, not by precision**. Within a node, fp16 costs 2.9e-6;
across nodes, fp32 costs **1.5e-04** — about **50× more than the precision
change this study was commissioned to evaluate**. On `siglip` the same
cross-node comparison is 7.6e-13, so it is specific to the SO400M/384 geometry.

## What it is not

Two hypotheses were tested and both are **refuted**, which is why this section
does not name a cause:

- **TF32.** `cudnn.allow_tf32` defaults to `True`, and an L40S (sm_89) has TF32
  while a V100 (sm_70) does not — so "fp32" plausibly meant two different formats.
  `fp32_notf32_l40s` came out **bit-identical** to `fp32_l40s` (median exactly
  0.0, mean 4e-17). TF32 was never active. (`matmul.allow_tf32` defaults to
  `False`, which is likely why.)
- **cuDNN algorithm nondeterminism.** Both `fp32_det_*` arms came out
  **bit-identical** to their own node's fp32 (exactly 0). Each node is perfectly
  self-consistent across runs and flag changes; the two simply disagree.

So the difference is **deterministic, reproducible, and hardware-associated**.
Remaining candidates, untested: a different SDPA/attention backend selected by
compute capability, or different GEMM tiling and accumulation order at this
shape. The pile's own docstring says drift comes from "kernel selection" and puts
it at ~1e-7; the mechanism may be right, but the **magnitude is off by ~3 orders
of magnitude** for this cell.

## And the node, not the card, may be the axis

The published `visual_genome_m__siglip2_l.pkl` was built by job `495266` on
**rack7n03** (`sacct`: `gres/gpu:v100=1`). Both V100 arms here landed on
**rack5n03**. And the published cell agrees with the **L40S** cluster (2.7e-12),
not with the V100 arms (1.5e-04) — while the published `siglip` cell agrees with
`fp32_v100` **exactly (0.0)**, as it should.

`gres/gpu:v100` is a *type* label, not a device: this cluster's V100s include
SXM2 and PCIE parts with different SM counts, and a different SM count means
different tiling and a different accumulation order. `fp32_v100_rack7n03` pins the
exact node the published cell was built on to test that.

## Resolved: one specific V100 part is the outlier

It is the node, and it is one device. `gres/gpu:v100` covers at least two parts:

| node | GPU as torch reports it | `siglip` | `siglip2_l` |
|---|---|---:|---:|
| rack7n03 | **Tesla V100S-PCIE-32GB** | 123 s | 380 s |
| rack5n03 | **Tesla V100-SXM2-32GB-LS** | 144 s | 442 s |

Both `sm_70`, both `gres/gpu:v100`, ~15% apart in throughput — and SLURM cannot
distinguish them. Pinning rack7n03:

| `siglip2_l` fp32 | vs `published_pile` | vs `fp32_l40s` | vs `fp32_v100` (rack5n03) |
|---|---|---|---|
| `fp32_v100_rack7n03` | **0.0 — bit-identical** | 2.7e-12 | 1.5e-04 |
| `fp32_v100` (rack5n03) | 1.5e-04 | 1.5e-04 | – |

**The V100S-PCIE part reproduces the published cell bit-for-bit**, and agrees with
the L40S to 2.7e-12. So three of the four devices tested (L40S, V100S-PCIE) agree
to ~1e-12, and **one specific part — the V100-SXM2-32GB-LS on rack5n03 — differs
by 1.5e-04** on `siglip2_l` while agreeing to 7.6e-13 on `siglip`.

That closes the puzzle in the direction that makes the pile look *better* and the
scheduler look *worse*: the published cell is exactly reproducible, on the node
that built it. Nothing about the pile is corrupt. What is not safe is assuming
that "the same GPU type" means the same arithmetic.

The remaining unknown is narrow and no longer blocking: *why* that one part
differs. Both hypotheses that were testable from here are refuted (TF32, cuDNN
algorithm selection), and the candidates left — a capability-selected attention
backend, or GEMM tiling that differs with SM count — need a targeted numerics
probe rather than another pile build.

## Why this matters beyond #3143

- **#3144's stated premise does not hold.** PR #3150 landed a GPU auto-pick on
  the grounds that cross-GPU fp32 drift is "~1e-7 — far below anything the
  studies resolve. This one is safe to land on its own." For `siglip2_l` on one
  of the two V100 parts the measured figure is **1.5e-04**, three orders of
  magnitude larger and 30× the 0.005 the studies resolve when it reaches a
  score. The auto-pick remains a good *performance* change (§2 independently
  confirms 1.9–2.5×) — but it requests a **type**, and a type is not a device, so
  **a pile rebuild under it is not numerically reproducible**.
- **`pick_gpu.py` cannot express what it needs to.** Nothing in
  `scontrol`/`--gres` distinguishes a V100S-PCIE from a V100-SXM2-LS. A study
  that needs numeric reproducibility has to pin `--nodelist`, or at minimum
  *record* `torch.cuda.get_device_name()` so a later reader can tell whether two
  cells are comparable. This study's `provenance.json` does; the pile's build
  does not.
- **It reframes this study's own question.** "Is fp16 safe?" is bounded above by
  something already shipped and unremarked: on that one part, plain fp32
  perturbs `siglip2_l` **50× harder** than fp16 does on a fixed node.

Filed as **#3160** rather than resolved here — the fix is a
scheduling/provenance change, not a numerics change, and it belongs with #3144.

---

# 6. The benchmark: what it can and cannot say

Same production defaults on both arms, only the gallery precision differing.
Paired on `(dataset, embedder, category, seed, style, t)`; **SE taken over cells,
not steps**, because steps within a cell are autocorrelated and a step-level SE
would be anti-conservative (#2825).

## 6-seed grid (192 cells)

95 of 96 cells per arm analysed; `task_0001.csv` was **header-only in both arms**
(the same cell, so pairing is unaffected) — reported rather than silently dropped.

Deep regime (t ≥ 100), `fp16_l40s` − `fp32_l40s`:

| metric | ref | arm | paired diff ± SE | cells | verdict |
|---|---:|---:|---|---:|---|
| cost | 0.38 | 0.38 | −0.0059 ± 0.0076 | 95 | cannot resolve at this n |
| regret | 0.10 | 0.10 | −0.00026 ± 0.0064 | 95 | cannot resolve at this n |
| average_precision | 0.44 | 0.46 | +0.011 ± 0.0077 | 95 | cannot resolve at this n |
| fnr | 0.14 | 0.13 | −0.0037 ± 0.009 | 95 | cannot resolve at this n |
| fpr | 0.25 | 0.24 | −0.0022 ± 0.011 | 95 | cannot resolve at this n |
| rule_inefficiency | −0.019 | −0.012 | +0.0063 ± 0.0063 | 94 | cannot resolve at this n |
| calibration_shift | 0.12 | 0.11 | −0.0065 ± 0.0073 | 94 | cannot resolve at this n |
| n_good (count) | 9.3 | 8.5 | **−0.87 ± 0.38** | 95 | 2.3 SE from zero |

Split by embedder, the signs **oppose** — which is what noise looks like, and
what a pooled average would have hidden:

| embedder | cells | cost | regret | average_precision |
|---|---:|---|---|---|
| `siglip` | 47 | +0.0044 ± 0.010 | +0.006 ± 0.008 | +0.0096 ± 0.0094 |
| `siglip2_l` | 48 | −0.016 ± 0.011 | −0.0064 ± 0.010 | +0.013 ± 0.012 |

Threshold provenance was identical on **100%** of steps, yet `cost` was
bit-identical on only **3.5%**. That is the mechanism: a 3e-6 vector change
reroutes the vote sequence, so trajectories decorrelate even though the decision
*rule* never changes.

**`n_good` is the one resolvable difference**: fp16 finished with 0.87 fewer
positives per cell (9.3 → 8.5), 2.3 SE from zero, unadjusted for testing eight
metrics. #3129 found positives to be the binding constraint on these runs, so
this looked worth a follow-up — not a blocker, but not nothing either. Filed as
**#3158**, which named the check that would dissolve it: whether the effect
shrinks toward zero at the 64-seed n. **It did** — see below. #3158 is closed on
that measurement.

## Power: why the issue's criterion needed a bigger run

"Cannot resolve" is **not** "no effect", and an underpowered null is worthless.
At the measured between-cell spread, resolving the 0.005 margin needs:

| metric | SE at n=95 | cells for 2·SE < 0.005 |
|---|---:|---:|
| rule_inefficiency | 0.0063 | ~592 |
| regret | 0.0064 | ~626 |
| cost | 0.0076 | ~878 |
| calibration_shift | 0.0073 | ~798 |
| average_precision | 0.0077 | ~897 |
| fnr | 0.009 | ~1226 |
| fpr | 0.011 | ~1698 |

Those are affordable, so the criterion was met rather than excused: a **64-seed,
2048-cell** run followed, in its own study dir (a different grid is a different
study).

## 64-seed grid (2048 cells) — the criterion, met and not met

1024 cell files per arm, all present, no zero-byte cells, no failed array tasks.
**11 cells were header-only in both arms at identical indices** (`task_0001`,
`task_0013`, `task_0017`, …) — systematic, not a disk incident, and paired-safe:
**1013 paired cells, 146,730 paired steps**.

| metric | ref | arm | paired diff ± SE | verdict |
|---|---:|---:|---|---|
| cost | 0.39 | 0.39 | +0.0028 ± 0.0023 | cannot resolve |
| **regret** | 0.10 | 0.10 | **+0.00026 ± 0.0019** | **below margin (resolved)** |
| average_precision | 0.45 | 0.45 | −0.00085 ± 0.0021 | cannot resolve |
| fnr | 0.15 | 0.16 | +0.0035 ± 0.0027 | cannot resolve |
| fpr | 0.24 | 0.24 | −0.00070 ± 0.0031 | cannot resolve |
| rule_inefficiency | −0.012 | −0.010 | +0.0015 ± 0.0021 | cannot resolve |
| calibration_shift | 0.11 | 0.11 | −0.0011 ± 0.0020 | cannot resolve |
| n_good (count) | 8.8 | 8.8 | **−0.0034 ± 0.11** | within noise |

Per embedder, `cost`: `siglip` +0.0042 ± 0.0033 (504 cells), `siglip2_l`
+0.0014 ± 0.0033 (509 cells). Both now the same sign — **the 6-seed grid's
opposing signs were noise**, which is what the larger n was for.

Four readings, in order of what they license:

1. **No metric shows a detectable effect.** The largest is `cost` at 1.2 SE and
   `fnr` at 1.3 SE. Nothing here is significant at any conventional threshold.
2. **Strict equivalence at 0.005 is demonstrated for `regret` only.** For `cost`
   the 95% interval reaches +0.0074 and for `average_precision` −0.0051, so both
   graze the margin. Note *why*: `2·SE` is 0.0046 and 0.0042 — **both inside the
   margin**. The intervals cross it because the point estimates are not exactly
   zero, not because the run is underpowered. More cells shrink the SE and would
   not move the centre.
3. **Failing an equivalence test is not detecting an effect**, and the distinction
   is the whole content of this section. The honest summary is: fp16 moves the
   shipped decision metrics by **less than 0.003 in expectation**, with a
   confidence interval that cannot quite exclude 0.005 on two of seven metrics.
   Anyone who wants a clean two-sided equivalence result on `cost` needs a
   different experiment, not a bigger one — see the trajectory note below.
4. **`n_good` dissolved.** −0.87 ± 0.38 at n=95 became **−0.0034 ± 0.11** at
   n=1013 — a factor of 250 smaller and centred on zero. The 6-seed result was a
   multiplicity artifact of testing eight metrics, exactly as #3158 suspected;
   that issue is answered and closed on this measurement.

`cost` was bit-identical on 5.7% of steps (up from 3.5%, as expected with more
cells), `average_precision` on 0.04%, and the threshold was chosen the same way
on **100%** of steps. That combination is the mechanism restated: the decision
*rule* never changes, the *trajectory* always does.

---

# 7. Ops notes and what they cost

- **A launcher that exits 1 with no output submitted nothing.** `find "$dir" |
  wc -l` under `set -euo pipefail`, on a dir that does not exist on the first
  run, aborts the assignment and takes the script with it — silently, from
  inside the preflight whose job was to prevent silent skips. Cost ~15 min and
  two blind re-runs. Both launchers now carry an ERR trap naming the line and
  stating that nothing was submitted. Lesson:
  `lessons/2026-08-17-a-launcher-that-exits-1-with-no-output.md`.
- **A zero-byte cell in a *live* array is a file being written.** The preflight's
  zero-byte check is correct before a launch; a monitor applying it to a running
  array false-alarms. One such alert here was benign, confirmed by a re-check
  finding 0 zero-byte cells and 0 failed tasks.
- **Don't point `suite.sbatch` at the worktree you launch from.** It does `git
  checkout --detach`, and `preflight.sh` then compares `HEAD` against
  `origin/HEAD` (i.e. `main`) and fails. The test suite got its own worktree.
- **`--mem=16G` was 30× oversized.** Measured peak RSS for a bench cell is
  **542 MB**; the power run used 4G. Memory is the binding per-user quota, so an
  oversized array throttles your own later jobs.
- **The reference arm is part of the measurement.** Two separate mistakes traced
  to it: fp16-on-V100 was reported ~3 orders of magnitude worse than it is, and a
  table was printed under the wrong reference's name. Both are fixed in the
  tooling (`--reference`, and headers labelled with the reference in use), not
  just in the prose.

---

# 8. Reproducing

```bash
source scripts/experiments/pile/pile_env.sh
cd scripts/experiments/precision
bash launch_precision.sh arms          # the six pre-registered arms
bash launch_precision.sh check         # MUST pass before analysing
python analyze_drift.py --pairwise --include-published
python report_timings.py
bash launch_bench.sh prepare && bash launch_bench.sh verify-pairing
bash launch_bench.sh cells && bash launch_bench.sh analyze
```

Full method, arm rationale, and the three gates: `scripts/experiments/precision/README.md`.
