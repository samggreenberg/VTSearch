# Embedding-precision study (issue #3143)

Does running the image encoders in half precision change the results enough to
matter? fp16 makes the `siglip2_l` forward **4.2x** faster, but it changes the
vectors, and the whole pre-embedded pile plus every published result is fp32.

The knob under test is `VTSEARCH_EMBED_PRECISION` (default `fp32`, i.e.
unchanged). See `docs/DEPLOYMENT.md` for the modes.

## Run it

```bash
# 1. build one side pile per precision arm (GPU, ~minutes each)
bash launch_precision.sh size fp32_l40s      # time one arm first
bash launch_precision.sh arms                # then all six
bash launch_precision.sh status
bash launch_precision.sh check               # provenance + reproduction, MUST pass

# 2. vector drift and retrieval-order stability (CPU)
python analyze_drift.py

# 3. the same benchmark arm against each gallery (CPU array)
bash launch_bench.sh prepare
bash launch_bench.sh verify-pairing          # MUST pass before the arrays
bash launch_bench.sh size fp32_l40s 0        # time one real cell
bash launch_bench.sh cells
bash launch_bench.sh status
bash launch_bench.sh analyze
```

Nothing writes to the shared pile at `/expscratch/$USER/vts-cache`. Each arm gets
its own pile root under `/expscratch/$USER/precision-3143/piles/<arm>`, and
weights are read from the shared pile's models dir so no arm re-downloads.

## Why the arms are what they are

| arm | precision | GPU | role |
|---|---|---|---|
| `fp32_l40s` | `fp32` | L40S | reference — everything is differenced against it |
| `fp32_v100` | `fp32` | V100 | **control**: same math, different card → the irreducible drift floor |
| `fp16_l40s` | `fp16` | L40S | candidate (weight cast, the fast one) |
| `fp16_v100` | `fp16` | V100 | the same candidate on a second card — does the drift travel? |
| `autocast_l40s` | `autocast_fp16` | L40S | candidate (fp32 weights, per-op autocast: safer, slower) |
| `bf16_l40s` | `bf16` | L40S | candidate (wider exponent; needs sm_80+, so no V100 twin) |

**The cross-GPU arm is the point.** "fp16 moves cosines by 1e-3" is not a finding
without a denominator: the same fp32 code on two different cards already drifts
through kernel selection alone. Every treatment number is reported as a ratio to
that floor.

**`fp16` runs on both card types** because a single-environment measurement
generalises about as well as #2877's acquisition offset did — which is to say, it
did not.

**The GPU type is pinned per arm, and this launcher deliberately does not call
`scripts/slurm/pick_gpu.py`.** Everywhere else in the tree an auto-pick is right
(#3144 / PR #3150); here the card *is* an experimental factor, and auto-picking
would collapse the treatment and the control into one confounded difference.

**`dinov3_patch` is excluded on purpose.** Its patch grids are already stored
`float16` (`vtscore/datasets/stages/embedding.py`), so its region path is
quantised to half before any of this — it is the arm where fp16 compute is least
likely to matter and where a null would be least informative. It is also
licence-gated. Worth a follow-up; not a blocker for the default flip.

## The checks that gate the numbers

Three scripts exist only to stop a wrong conclusion, in the order they run:

- **`build_arm.py`'s probe** — resolves the precision, loads each model, and
  records the *parameter dtype*, card name and compute capability, then **refuses
  to build** if any of it contradicts the arm table. A mode that silently
  degraded (bf16 on an sm_70 card, a typo'd env var, a CPU fallback) produces
  vectors identical to fp32, which reads as "half precision is harmless" when it
  never ran. Assert the premise, not the parameter.
- **`check_arms.py`** — provenance present and matching, the card the arm *names*
  is the card it *ran on*, all arms cover the same medias, no zero-byte cells,
  and the fp32 rebuild reproduces the **published** pile cell. If the fp32 arm is
  not the pile's fp32, every difference is against nothing in particular.
- **`verify_pairing.py`** — the two bench arms select identical categories in
  identical order with identical exemplar candidates. They should by
  construction; unpaired arms still produce a table, just one whose error bars
  are a fiction.

## Reading the output

- Drift is reported as a **distribution** (median / p95 / max), never a mean
  alone — the mean of a long-tailed drift is what hides the tail.
- Ranking stability is measured from **both** a text query and an exemplar
  vector, because both towers shift and the benchmark seeds from the exemplar.
- Rank moves come with the **score gap to the next item**: a 40-place move across
  a 1e-6 cosine gap is a tie broken differently, not a retrieval failure.
- The bench verdict distinguishes *below margin (resolved)* from *cannot resolve
  at this n*. An underpowered null is not evidence of no effect.

## Follow-up: the float16 patch grid (#3159)

`dinov3_patch` was excluded above because its patch grid is **stored** float16
whatever the forward ran in. #3159 measures that cast. The dtype now has one
name, `vtscore.embedding.matrix.PATCH_ROW_DTYPE`, read at call time by the
ingest cast, the live region matrix and the harness's MaxPatch stack, so the
study flips all three in-process. It is not an env knob.

```bash
bash launch_grid_3159.sh build            # fp32-grid cells, 1 GPU on rack7n03 (the pile's node)
bash launch_grid_3159.sh verify           # MUST pass: fp32 cell -> fp16 cast == the pile, bit for bit
bash launch_grid_3159.sh drift            # pooled-score / rank drift (CPU)
bash launch_grid_3159.sh prepare          # bench arms: fp16 = the pile via symlink, fp32 = the rebuild
bash launch_grid_3159.sh cells            # paired arrays, siglip+dinov3_patch, production defaults
bash launch_grid_3159.sh analyze
```

| file | job |
|---|---|
| `build_grid_3159.py` | build (PATCH_ROW_DTYPE=float32), `adopt-cls`, `verify` |
| `grid_drift_3159.py` | drift of the max-pooled score, per query, split by box band |
| `run_cells_3159.py` | the calibration harness with the dtype pinned and asserted per pickle |
| `analyze_grid_3159.py` | paired bench, per dataset and band, plus the report's measurement files |
| `launch_grid_3159.sh` | all of the above |

Two reproduction traps, both found by `verify`: the grid is **not
batch-invariant** (the pile was embedded at batch 32; today's table says 64,
which moves ~1.3% of cast elements), and a *second* build in the same side
pile reads the demo loader's cached base pickle (`visual_genome_m.pkl`, written
by the first), whose serializer drops `categories` and `regions`: the cell
comes out with every label gone and nothing fails. Delete it before a rebuild. Report:
`docs/experiments/2026-09-22-patch-grid-fp16-3159/REPORT.md`.
