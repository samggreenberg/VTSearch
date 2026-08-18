# #3146 image-processor study — run status

**State as of 2026-08-18 08:45 EDT: the run is in flight on the GRID and needs
no live connection.** Everything below is here so the run can be picked up cold.

## Where things are

| what | where |
|---|---|
| local worktree | `/home/samiam/Code/vts-fastproc-3146`, branch `claude/fast-processor-3146` |
| GRID worktree | `/exp/sgreenberg/projects/vts-fastproc-3146`, same branch |
| study dir | `/expscratch/sgreenberg/fastproc-3146` |
| live progress log | `/expscratch/sgreenberg/fastproc-3146/STATE.md` (the driver appends to it) |
| artifacts | `/expscratch/sgreenberg/fastproc-3146/results/` |

## What is running

`sbatch` job **511756** (`fprocdrv`, cpu partition, 12 h limit) is a GRID-side
driver that runs the whole study in sequence and records each stage into
`STATE.md`. It survives a dropped VPN. Stages:

1. wait for the four side-pile arms (`fproc-*`, jobs 511474 / 511740 / 511741 / 511742)
2. `check_arms.py` → `results/CHECK_ARMS.txt`
3. `analyze_proc_drift.py` → `results/DRIFT_TABLES.txt`, `drift.csv`, `adjudication.csv`, `rank_stability.csv`, `examples.json`, `figures/`
4. GPU probes → `results/PIXEL_TABLES.txt`, `results/ODD_INPUTS.txt`
5. bench `prepare`
6. bench `verify-pairing` → `results/PAIRING.txt` — **stops the run if it fails**, because unpaired arms make the standard error a fiction
7. bench `cells` (3 arms × ~72 cells, cpu partition)
8. bench analysis → `results/BENCH_TABLES.txt`

To check on it: `ssh grid 'tail -40 /expscratch/sgreenberg/fastproc-3146/STATE.md'`

## Sizing, measured rather than guessed

One arm = **288 s** on an L40S (`siglip` 87 s at 48 medias/s, `siglip2_l` 182 s
at 23 medias/s), from job 511474. The four arms run concurrently against a
4-GPU QOS cap, so the pile stage is minutes. The bench array is the long pole.

## Findings already established (before the arms finished)

These came out of the premise-check and are the reason the arm table is not the
one the issue proposed. They are recorded here because they are already
load-bearing.

1. **#3146's premise is false.** With the installed `transformers 5.12.1`, v5
   removed the `Fast` suffix: `SiglipImageProcessor` **is** the torchvision
   implementation and the PIL one was renamed `SiglipImageProcessorPil`.
   Passing nothing already selects torchvision, so every image embedder has
   been on the "fast" path all along. `use_fast` is itself deprecated in favour
   of `backend=`, and `use_fast=True` on an explicitly-named concrete class is a
   no-op.
2. **The default flipped inside our own dependency range.**
   `requirements/image-embedders.txt` pins only `transformers>=4.49`, and the
   default backend differs across that range — so the same code and weights
   produce different pixels depending on which transformers a host resolved,
   with nothing in the pile recording which.
3. **Measured on 128 real VG images, L40S, one node** (probe, not yet the
   study's own arms): `pil/cpu` 986 ms, `torchvision/cpu` 306 ms,
   `torchvision/cuda` 73 ms. So the issue's *other* proposed fix — GPU
   preprocessing — is worth ~4× on the stage, and is the live candidate.
4. **The GPU path is the larger numeric perturbation, not the smaller.**
   `torchvision/cuda` differs from `torchvision/cpu` by 7.1e-2 max abs pixel
   (mean 9.0e-4), while `torchvision/cpu` differs from `pil/cpu` by 7.8e-3.
5. **DINOv3 has no PIL backend at all.** Asked for `backend="pil"` transformers
   warns and hands back torchvision, which is why `dinov3_patch` has no `pil`
   arm — a silent fallback there would be the reference arm under another name.
6. **EXIF orientation is not handled anywhere.** Nothing in
   `vtscore/media/image/decode.py` calls `exif_transpose`, so a rotated JPEG
   reaches the model un-rotated. Real, pre-existing, and *constant across
   backends*, so it is reported and excluded from the arm comparison.

## What still has to be decided when the numbers land

- Does `tv_cpu` reproduce the published pile cell? That adjudicates which
  backend built the pile (`results/CHECK_ARMS.txt` and `adjudication.csv`).
  The peer session working #3160 confirms existing cells record neither device
  nor transformers version, so this comparison is the only way to know.
- Is `tv_cuda` adoptable — i.e. does the benchmark move by less than 0.005
  paired? Its pixel perturbation is large enough that a null is not the
  expected outcome.
- Does `tv_cpu_rep` come out at exactly zero? #3143 found this stack bit-
  identical across repeat runs on one device, so a non-zero floor here would
  itself be the finding.

## Related

Issue #3146. Adjacent: #3143 (precision, merged), #3160 (device provenance, in
flight in a parallel session — its `build_pile.py` provenance sidecar will
record `transformers.__version__` and the resolved processor class as a result
of finding 2 above).
