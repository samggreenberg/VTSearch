# A type is not a device: what the V100 split costs (#3160)

**Issue:** [#3160](https://github.com/samggreenberg/VTSearch/issues/3160) ·
**Predecessor:** [#3143](../embed-precision-3143/REPORT.md) §5 ·
**Code:** `scripts/experiments/gpu_node/`

#3143 measured, as a control, that two nodes both answering to `gres/gpu:v100`
produce `siglip2_l` vectors **1.5e-04** apart in median 1−cos while three other
devices agree to ~1e-12. #3144 had landed the GPU auto-pick on the stated
premise that cross-GPU fp32 drift is "~1e-7 — far below anything the studies
resolve". The premise is false; #3160 asks what follows from that.

The issue proposes three things. This study runs the measurement (3) and ships
the plumbing (1, 2).

## Pre-registered questions

**Q1 — How many devices, and how likely is a rebuild to hit an outlier?**
Census every node `pick_gpu.py` can hand out (13 v100, 6 a100, 2 l40s; h100/h200
are capped at 0 by the `4gpu_tier` QOS and are not candidates). Each node
embeds the **same fixed 256 VG images** with the shipped forward and reports
`torch.cuda.get_device_name()`, compute capability, SM count and driver.
*Deliverable:* the device table `--gres` cannot express, and the share of the
pool that does not reproduce the published cell.

**Q2 — Why does that one part differ, and can it be turned off?**
On three nodes (the outlier `rack5n03`, the reference `rack7n03`, an L40S), run
the shipped `siglip2_l` forward with each SDPA backend forced, capturing every
vision block's output. Two predictions are distinguishable:
a **step** (bit-identical up to block *k*) means a per-op implementation choice;
a **ramp** (block 0 already differs, growing with depth) means reordered
arithmetic. And decisively: if a forced backend makes the nodes agree, a
determinism knob exists. Bare GEMM/conv fingerprints separate "the model chose
differently" from "the hardware adds differently".

**Q3 — Does 1.5e-04 move a shipped decision?**
Paired benchmark, production defaults, **only** the gallery's build node
differing:

| arm | pile | device |
|---|---|---|
| `fp32_v100_rack7n03` (reference) | #3143 job 507728 | Tesla V100S-PCIE-32GB — bit-identical to the published pile cell |
| `fp32_v100` | #3143 job 507430 | Tesla V100-SXM2-32GB-LS — the outlier |

Both piles already exist; **nothing is rebuilt**, because a rebuild would land
wherever the scheduler chose and that is the defect under study.

`visual_genome_m`, 8 categories, 128 seeds, `max_steps=150`, 2048 cells per arm.
Two embedders with **different roles**:

- `siglip2_l` — **treated**: the two piles differ by 1.5e-04 (median 1−cos), 0%
  of rows bit-identical, max 1.1e-02.
- `siglip` — **placebo**: the same two piles agree to a median of **exactly 0**,
  78% of rows bit-identical, max 1.3e-15. Verified from the pickles before
  launch. Its paired difference has no cause to be non-zero, so it is the
  falsifier for the whole design: if the placebo moves as much as the treated
  arm, this bench is measuring trajectory chaos and licenses no verdict.

**Margin:** 0.005, the figure the calibration studies resolve. **Analysis:**
`analyze_bench_precision.py` — paired on `(dataset, embedder, category, seed,
style, t)`, cells collapsed to their own mean first and **SE taken over cells**
(steps within a cell are autocorrelated, #2825), deep regime t ≥ 100.
128 seeds rather than #3143's 64 because the per-embedder split *is* the design
here and 64 seeds left per-embedder 2·SE at 0.0066, above the margin.

**Verdicts are pre-committed:** "resolvably below the margin" requires
|diff| + 2·SE < 0.005; anything else is reported as "cannot resolve at this
cell count", which is not the same as "no effect".

## What ships regardless of the outcome

1. **Per-cell provenance** (`build_pile.py`): device name, capability, SM count,
   driver, torch/cuDNN, precision, node, SLURM job, commit, and a
   `vectors_sha256` fingerprint — the last of which is what lets a *future*
   rebuild be checked against a cell that no longer exists.
   `--provenance` prints the table and flags a pile that mixes devices;
   `--backfill-provenance` fingerprints the pre-#3160 cells (device recorded as
   `null`, because a guess would be worse than a gap).
2. **Node pinning** (`launch_pile.sh`): `VTS_GPU_NODE=<node>` pins `--nodelist`
   and derives the gres type from the node, so a rebuild can target the machine
   that produced the cell. Cross-referencing is only possible once (1) exists.

## Known limits

- The two piles were built ~45 minutes apart during #3143's run, from the same
  worktree. Their equality on `siglip` (median exactly 0, 78% of rows
  bit-identical) is the evidence that nothing else changed between them.
- The census fingerprint is 256 images, not 4193: enough to separate devices at
  1e-4, not a substitute for a cell.
- Q3 answers "does it move the metric", not "is any single run reproducible".
  A trajectory *always* changes; #3143 measured that a 3e-6 perturbation already
  reroutes the vote sequence on 94% of steps.
