# Running eval experiments on the GRID (SLURM) — playbook

Practical, hard-won ops notes for running the `scripts/experiments/*` sweeps on
the JHU-HLTCOE GRID (a shared SLURM cluster) — and the older `scripts/sod`
sweeps, which live **only on the `evaluation-framework` branch** and are not
present on `dev` (see `LESSONS.md`, the 2026-08-07 entry, and
`docs/experiments/spike-check-2847/REPORT.md` for why that harness can't be
pointed at `dev`). The patterns are general; HLTCOE-specific values are marked
**[HLTCOE]**. Read this before launching a big sweep — most of it was learned by
getting it wrong.

This file covers **SLURM resources**. Its companions cover the rest of running a
study, and the `grid-experiments` skill ties all three together:

- **`preflight.sh`** — run before submitting arms; blocks the mistakes that are
  mechanically checkable (results-dir collisions, the wrong mount's free space,
  a stale worktree).
- **`LESSONS.md`** — the incident log: what broke on real studies, what it cost,
  and whether it is now prevented or still only advice. Append when something
  breaks.

## 1. Right-size `--mem` (the #1 silent time-sink)

**An over-fat `--mem` wedges your job off *idle* GPUs.** If `squeue` shows your
job pending with reason **`Resources`** while `sinfo` shows **free GPUs on the
node**, the blocker is almost always CPU/**memory** headroom on that GPU node,
not the GPU: the GPUs are free but the node's RAM is already reserved by other
jobs, so your fat request doesn't fit.

- Probe the real peak first: `sacct -j <jobid> -o MaxRSS` on a completed run.
  These sweeps typically peak at **~6–12 GB**, not the tens of GB people request.
- Fix a *pending* job live without resubmitting:
  `scontrol update JobId=<id> MinMemoryNode=<MB>` — **plain MB, no `G` suffix**
  (`16384`, not `16G`; running array elements error "no longer pending", pending
  ones update). Dropping 48G→16G has taken a wedged run from 1 GPU to the full 4.
- When the pending reason flips to **`JobArrayTaskLimit`**, you're saturated at
  your own `%N` array throttle (good — that means you're using your full cap).

## 2. Know your QOS before chasing idle GPUs

`sacctmgr -nP show qos <name> format=Name,MaxTRESPerUser` tells you your real caps.
**[HLTCOE]** `4gpu_tier` = **4 GPUs total**, with `l40s`/`v100`/`a100` ≤ 4 each but
**`h100=0` and `h200=0` — forbidden**. So idle H100/H200 nodes are *unusable* to
you; a job requesting them sits pending and probing throws **`QOSMaxGRESPerUser`**.
Don't burn time trying to grab premium GPUs your tier can't touch — check the QOS,
and if you need a GPU *now*, request an allowed type that's actually idle
(`sinfo -p gpu -O NodeHost,Gres,GresUsed,StateCompact`).

## 3. Prefer fewer, longer allocations over one-per-task

A naive `--array=0-N` with one element per unit of work **re-enters the scheduler
for every unit** — you pay the priority/backfill/contention gauntlet N times, and
node-local caches are wiped between allocations (see §4). Instead, **chunk the work
so one allocation processes a *series* of units on the GPU it grabbed**:

- Size chunks to fill your GPU cap in **one wave** (e.g. 4 chunks for a 4-GPU cap),
  so all GPUs stay busy and nothing re-queues mid-run.
- **Tradeoff — longer reservations backfill slower.** A 14 h job waits longer for a
  slot than a 1.5 h one, and the up-front scheduling gap can cancel the savings.
  Pick the smallest chunk that still amortizes scheduling + enables cache reuse;
  don't request 14 h if 6 h covers the chunk.
- The wins are real but bounded: chunking removes re-scheduling stalls and
  redundant embedding, **not** the per-unit compute (the model-training/scoring
  sims dominate and don't shrink).

## 4. Know which mount you are on

**[HLTCOE]** Three mounts, three jobs. Putting work on the wrong one is the most
common self-inflicted wound here.

| mount | size | use it for |
|---|---|---|
| `/exp/$USER` | **50 G** | the checkout and its venv. Nothing else. |
| `/expscratch/$USER` | **500 G**, flash | the embedding pile, study outputs, archives |
| `/scratch/jobs/$USER/$SLURM_JOB_ID` | ~286 G, node-local | per-job temp; wiped when the job ends |
| `/exp/scale26` | 25 T, shared | staged source datasets (read-mostly, ~94% full) |

- **`/exp` is a small quota** and the venv alone is ~13 G of it. Write no study
  output there at all — an ENOSPC kills the whole array mid-run.
- **`/expscratch` is where data lives**, and it is fast (~85 MB/s rsync, flash).
  Treat it as **purgeable**: keep the rebuild path in the repo so anything there
  can be regenerated from staged sources.
- **`HF_HOME` leak:** the grid shell points `HF_HOME` at `/exp`; one model
  download then fills the quota. Point it at the pile's models dir
  (`pile_env.sh` does this) or at node scratch in run wrappers.
- **Reuse one `--cache-dir` across a chunk's units.** Across e.g. object classes
  the negative pools are ~the same images ("images without class X" overlap
  heavily), so a shared cache embeds each image ~once per chunk instead of once
  per unit. This is the main payoff of §3's chunking.

## 4a. Use the shared pile; do not embed your own copy

`/expscratch/$USER/vts-cache` holds a `(dataset, embedder)` grid that is already
embedded. `source scripts/experiments/pile/pile_env.sh` points
`VTSEARCH_DATA_DIR` / `VTSEARCH_MODELS_DIR` / `HF_HOME` at it, and a study then
reads cells in place instead of re-embedding. Full docs:
`scripts/experiments/pile/README.md`.

**Datasets.** `visual_genome_m` (4193, boxed), `caltech101_m` (838, boxless),
`coco_val` (4952, boxed), and the box-size-banded VG sets `vg_box_small` /
`vg_box_medium` / `vg_box_large` (12000 each, boxed), drawn from the whole VG
source rather than the demo pipeline's 4% slice.

**Embedders.** `siglip` (768-d, the shipped default), `siglip2_l` (1152-d,
premium) and `dinov3_patch` (768-d). Differing dims mean `siglip` and
`siglip2_l` galleries are **not** interchangeable. The middle rungs
(`siglip_l`, `siglip2`) were dropped deliberately — see the pile README for the
tradeoff that buys.

**Which arms can region-vote.** Region voting drags a ground-truth box and pools
it over a patch grid, so it needs **both** halves:

| | boxed dataset | patch embedder | region-votes? |
|---|:--:|:--:|:--:|
| `visual_genome_m` / `coco_val` / `vg_box_*` x `dinov3_patch` | yes | yes | **yes** |
| any boxed dataset x `siglip` or `siglip2_l` | yes | no | no — **binary** |
| `caltech101_m` x anything | no | — | no — **binary** |

`dinov3_patch` is the only patch-capable embedder, so it is the only way to get
a region arm. **A boxed dataset on a single-vector embedder does not error — it
silently runs as binary voting**, which has now cost three studies (#2877,
#2897, #2905). Assert the geometry rather than trusting the arm table:
`build_pile.py --verify` checks that every region-capable cell actually carries
`patch_grid`, and `launch_*.sh` has a `--require-region-voting` preflight.

## 5. Monitor from the GRID, not from your laptop

- **Local background pollers get culled** (editor/session caps kill long-running
  local loops). For anything that must survive, submit a **GRID-side dependency
  job** — `sbatch --dependency=afterany:<runjob>` — to run the analysis /
  consolidation / final data-pull. Those are kill-immune and fire when the run
  ends regardless of whether anything local is still watching.
- **VPN flaps kill local SSH watchers; the GRID jobs don't care.** Use short
  retry-loop probes (`timeout … ssh … 'squeue …'`) rather than one long-lived ssh
  session, and lean on the dependency jobs above for the actual work.

## 6. Environment gotchas

- **libpython:** the venv's `python` fails to load `libpython3.12.so` on the login
  node until you `module load python/3.12.3` — source the project's grid-env
  wrapper before anything.
- **Editable-finder shadow trap:** with multiple worktrees, a worktree's `app.py`
  can silently import the *other* checkout's package via the editable install.
  Pin the intended worktree with the shadow-module `PYTHONPATH` trick.
- **GPU nodes are `Exclusive_Process`** (**[HLTCOE]**): one CUDA process at a time —
  serialize GPU stages within a job.

## 7. If you're orchestrating this with Claude

- `Agent(isolation:"remote")` can **silently downgrade to a local agent** when
  remote is gated/unavailable — then it has no Python env and flails. Check the
  spawn/stop result's `task_type`; if it says `local_agent`, it never had a real
  test env.
- App-tier work that needs `./run-tests.sh` (deps, frontend build) belongs in the
  **Claude Code webapp** (`CLAUDE_CODE_REMOTE=true`, deps auto-install), not a local
  session or local agent. GRID SSH from a sandboxed remote won't work (no VPN/keys).
