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

## 4. Cache aggressively; keep `/exp` tiny

- Node-local **`/scratch/jobs/$USER/$SLURM_JOB_ID`** is per-job, ephemeral, and
  large (**[HLTCOE]** ~286 G). Embed there.
- **Reuse one `--cache-dir` across a chunk's units.** Across e.g. object classes
  the negative pools are ~the same images ("images without class X" overlap
  heavily), so a shared cache embeds each image ~once per chunk instead of once
  per unit. This is the main payoff of §3's chunking.
- **`/exp` is a small quota** (**[HLTCOE]** ~50 G, chronically ~98% full). Write
  only **small, trace-free outputs** there (per-run `results.jsonl`), never
  per-step traces/PNGs — those blow ENOSPC and kill the whole array mid-run. Pull
  results off `/exp` to durable storage (your laptop, a project dir) promptly;
  don't treat `/exp` as durable.
- **`HF_HOME` leak:** the grid shell often points `HF_HOME` at `/exp`; one model
  download then fills the quota. Hard-set `HF_HOME` to node scratch in run wrappers.

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

## Memory is a per-user quota, not just a per-job request

`cpu_limit` caps **cpu=240 and mem=1100000M (~1.07 TB) per user**. CPU is rarely
the binding constraint; memory usually is, and it binds against *your own* jobs.

An array of `16 x --mem=64G` claims 1024G — 95 % of the allowance — so every
later job you submit sits in `QOSMaxMemoryPerUser` behind it. That is
indistinguishable from a busy cluster and it is entirely self-inflicted. In
#3129 it delayed a prepare job, throttled a second array to 2 slots instead of
12, and parked five small diagnostic jobs for 25 minutes.

**Size `--mem` from a real cell, the same way you size wall-clock:**

```bash
sacct -j <jobid> --format=JobID,JobName%20,MaxRSS,Elapsed
```

Measured peaks for the calibration harness (#3129):

| cell | medias | peak RSS | sensible `--mem` |
|---|---:|---:|---|
| whole-image (siglip / siglip2_l) | 4–5k | ~1.1 GB | 8G |
| whole-image, 12k set | 12k | ~3 GB | 8G |
| `max_patch` (dinov3, 4–5k) | 4–5k | ~13–14 GB | 24G |
| `max_patch` (dinov3, 12k) | 12k | ~14 GB | 24G |
| prepare (loads every pickle) | — | ~3.4 GB | 24G |

Two QOS can bind and they disagree: `squeue %q` reports the association QOS
while the partition carries its own. Read both and use the tightest —
`preflight.sh --mem --conc` does this.

Levers once an array is already running:

```bash
scontrol update JobId=<id> ArrayTaskThrottle=<n>   # frees quota as tasks finish
scontrol update JobId=<id> MinMemoryNode=<MB>      # PENDING tasks only
```

`MinMemoryNode` cannot retarget tasks the scheduler has already dispatched, so
throttling is usually the faster lever.
