# Running eval experiments on the GRID (SLURM) — playbook

SLURM *resource* practice for the `scripts/experiments/*` sweeps on the
JHU-HLTCOE GRID (a shared SLURM cluster): memory, QOS, GPU type, chunking,
mounts. The patterns are general; HLTCOE-specific values are marked
**[HLTCOE]**. Read this before sizing a big sweep.

The rest of running a study — worktrees, preflight, launch, monitoring,
reporting — is in the `grid-experiments` skill
(`.claude/skills/grid-experiments/SKILL.md`). `preflight.sh` enforces the
checkable subset of both; [`LESSONS.md`](LESSONS.md) indexes the incidents
behind them.

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

## 1a. Memory is a per-user quota, not just a per-job request

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
| `max_patch` (dinov3, `vg_scale`) | 7.7k | ~9.1 GB | 16G |
| prepare (loads every pickle) | — | ~3.4 GB | 24G |

`vg_scale` peaks *lower* than the 4–5k sets above despite holding more medias,
because its cells are designated: three-valued labels cut each cell's evaluable
pool to 4,000, and the working set is the cell, not the pickle. Do not read that
as "patch cells got cheaper" — read it as "measure the cell you are about to
run", which is the whole point of this table.

**Size from a cell of the same KIND, and check the kind is the one you meant.**
A `dinov3` cell that has silently fallen back to `whole_image` peaks near 4 GB
and finishes in the same two minutes as a `siglip` cell — it looks like a
perfectly good patch measurement and is not one (#3156: 74 of 108 cells then
died at a `--mem` sized from it). `styles=` in the harness's own resolution
output is the thing to read, not the runtime.

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

## 2. Know your QOS before chasing idle GPUs

`sacctmgr -nP show qos <name> format=Name,MaxTRESPerUser` tells you your real caps.
**[HLTCOE]** `4gpu_tier` = **4 GPUs total**, with `l40s`/`v100`/`a100` ≤ 4 each but
**`h100=0` and `h200=0` — forbidden**. So idle H100/H200 nodes are *unusable* to
you; a job requesting them sits pending and probing throws **`QOSMaxGRESPerUser`**.
Don't burn time trying to grab premium GPUs your tier can't touch — check the QOS,
and if you need a GPU *now*, request an allowed type that's actually idle
(`sinfo -p gpu -O NodeHost,Gres,GresUsed,StateCompact`).

**Don't hardcode the type in a launcher, either.** This cluster rejects an
untyped `--gres=gpu:1`, so a type must be named — but a named type is a pin that
outlives its reason in both directions: `v100` silently cost 2.3x on every
`siglip2_l` embed while L40S nodes idled, and `l40s` once meant ~5-day waits back
when only two L40S nodes existed. `python3 scripts/slurm/pick_gpu.py --explain`
answers the question above from `scontrol` and prints its reasoning; new
launchers should call it rather than picking a favourite
(`GRES="gpu:$(python3 "$REPO/scripts/slurm/pick_gpu.py"):1"`).

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
- **Count job records, not running tasks.** Slurm's `MaxJobCount` (**10,000**
  for the whole cluster) counts every array task from the moment it is queued,
  so a `%6` throttle does not make a 720-task array cheap. #4668's sixteen such
  arrays filled 85% of it and the last four were refused (#4701).
  `python3 scripts/slurm/job_records.py --tasks N` says whether N more fit.

## 4. Know which mount you are on

**[HLTCOE]** Four mounts, four jobs. Putting work on the wrong one is the most
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
  It has no snapshots: record every delete in [`scratch-deletions.md`](scratch-deletions.md).
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

**What is in it.** `pile_config.DATASETS` x `pile_config.EMBEDDERS` is the
grid; `build_pile.py --list` shows what is on disk, and the pile README's
[grid section](pile/README.md#the-grid) describes each row and column. Differing
embedding dims mean galleries are **not** interchangeable across embedders.

**Which arms can region-vote.** Region voting drags a ground-truth box and pools
it over a patch grid, so it needs **both** halves:

| | boxed dataset | patch embedder | region-votes? |
|---|:--:|:--:|:--:|
| `coco_better` / `coco_val` / `visual_genome_m` x `dinov3_patch` | yes | yes | **yes** |
| any boxed dataset x a single-vector embedder (`siglip`, `clip`, …) | yes | no | no — **binary** |
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
  node until you `module load python/3.12.3`. `source gridenv.sh` (repo root)
  does that and activates the venv; source it before anything.
- **Editable-finder shadow trap:** with multiple worktrees, a worktree's `app.py`
  can silently import the *other* checkout's package via the editable install.
  Pin the intended worktree with the shadow-module `PYTHONPATH` trick. A study's
  stage scripts get this for free from `scripts/experiments/_expcommon.py`, which
  the per-study `common.py` files delegate to (`umap_params` predates it); a
  shell wrapper gets it from the repo-root `gridenv.sh`.
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
