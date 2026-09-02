#!/bin/bash
# Record one real timing profile and read the r2 off it (#3345).
#
# Drives five task families with the GENERIC recorder armed
# (VTSEARCH_TIMING_RECORD), plus the older per-load profiler
# (VTSEARCH_PROFILE_LOAD) on the same loads, so recorder-vs-recorder is a
# controlled contrast rather than a machine one. Ends by fitting three profiles:
#
#   profile_generic.json   this run, generic recorder, five task families
#   profile_loadprof.json  this run, legacy load profiler, dataset_load only
#   profile_resweep.json   the #3062 resweep rows refit through vtscore.timing
#
# Four size tiers per media type is the whole design: r2 only exists where the
# fitter had spread in n. One dataset load leaves OLS with no x-variance, every
# step collapses to a median, and the profile carries no r2 at all.
#
# Completion is the file $EXP/DONE.run, never a live process (watchers die with
# the VPN).
#SBATCH --job-name=timing3345
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/expscratch/sgreenberg/timing-3345/slurm-%j.out

set -euo pipefail

EXP=/expscratch/sgreenberg/timing-3345
WT=/exp/sgreenberg/projects/vts-timing-3345
DEMOS=caltech101_s,caltech101_m,caltech101_l,caltech101_a,esc50_s,esc50_m,esc50_l,esc50_a
TUNE="$WT/scripts/profiling/tune_timing_profile.py"
ROWS="$EXP/rows.jsonl"

module load python/3.12.3
source /exp/sgreenberg/projects/VTSearch/.venv/bin/activate
cd "$WT"

export VTSEARCH_DATA_DIR="$EXP/data"
export VTSEARCH_MODELS_DIR=/exp/scale26/datasets/external/vtsearch-calib-2623/models
export VTSEARCH_TORCH_THREADS="${SLURM_CPUS_ON_NODE:-8}"
export VTSEARCH_PROFILE_LOAD="$EXP/loadprof.jsonl"
unset VTSEARCH_TIMING_PROFILE || true

# Provenance. A load-cost number without its core count is not comparable to any
# other, and the V100 nodes are two different machines (#3160, #3062).
{
  echo "host=$(hostname)"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "slurm_job=${SLURM_JOB_ID:-none}"
  echo "cpus=${SLURM_CPUS_ON_NODE:-?} torch_threads=$VTSEARCH_TORCH_THREADS"
  echo "cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
  nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"
} > "$EXP/env.txt"
python "$WT/scripts/experiments/timing_r2/env_probe.py" >> "$EXP/env.txt" 2>&1 || true
cat "$EXP/env.txt"

step () { echo "=== [$(date +%H:%M:%S)] $* ==="; }

# 1. Imports: four sizes per media type, one rep. The spread comes from the
#    sizes, not from repetition.
step "dataset_load"
python "$TUNE" --drive --allow-mutating --tasks dataset_load \
  --demo "$DEMOS" --reps 1 --data-dir "$EXP/data" \
  --record "$ROWS" --out "$EXP/profile_step1.json" 2>&1 | tee "$EXP/step1-load.log"

# 2. Staging: the same acquire+embed work under a different step structure and
#    no registry write.
step "dataset_stage"
python "$TUNE" --drive --allow-mutating --tasks dataset_stage \
  --demo "$DEMOS" --reps 1 --data-dir "$EXP/data" \
  --record "$ROWS" --out "$EXP/profile_step2.json" 2>&1 | tee "$EXP/step2-stage.log"

# 3. The read-only families over everything now in the registry. find is
#    expected to skip: a scratch data dir has no detectors.
step "dataset_open,text_sort,find"
python "$TUNE" --drive --tasks dataset_open,text_sort,find \
  --reps 2 --data-dir "$EXP/data" \
  --record "$ROWS" --out "$EXP/profile_step3.json" 2>&1 | tee "$EXP/step3-readonly.log"

# 4. Promote last: it adds _tuning_* datasets, which would change what the
#    read-only drivers above enumerate.
step "dataset_promote"
python "$TUNE" --drive --allow-mutating --tasks dataset_promote \
  --reps 1 --data-dir "$EXP/data" \
  --record "$ROWS" --out "$EXP/profile_step4.json" 2>&1 | tee "$EXP/step4-promote.log"

# 5. The three fits.
step "fit generic recorder"
python "$TUNE" --fit-only "$ROWS" --out "$EXP/profile_generic.json" \
  --notes "#3345 generic recorder, five families" 2>&1 | tee "$EXP/fit-generic.log"

step "fit legacy load profiler (same loads)"
python "$TUNE" --fit-only "$EXP/loadprof.jsonl" --out "$EXP/profile_loadprof.json" \
  --notes "#3345 legacy _load_profiler rows from the same loads" 2>&1 | tee "$EXP/fit-loadprof.log"

step "fit #3062 resweep rows"
python "$TUNE" --fit-only /exp/sgreenberg/resweep-3062/*.filt.jsonl \
  --out "$EXP/profile_resweep.json" \
  --notes "#3062 resweep rows refit through vtscore.timing.fit" 2>&1 | tee "$EXP/fit-resweep.log"

step "done"
wc -l "$ROWS" "$EXP/loadprof.jsonl"
touch "$EXP/DONE.run"
