#!/bin/bash
# The same eight imports again, after the #3345 embedder fix (leg 2 of the run).
#
# Leg 1 measured the defect: `drive_dataset_load` names no embedder, both
# recorders stamped the caller's blank, and the profile came back with three
# rollup cells and no exact `(device, media, embedder)` cell at all. This leg
# re-measures the identical workload on the fixed tree so the before/after is a
# property of the code and not of the machine, the cache, or the day.
#
# A FRESH data dir on purpose: the registry and the cold/warm model sequence
# both have to start where leg 1 started, or the comparison is between two
# different experiments.
#SBATCH --job-name=timing3345fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/expscratch/sgreenberg/timing-3345/fixed/slurm-%j.out

set -euo pipefail

EXP=/expscratch/sgreenberg/timing-3345/fixed
WT=/exp/sgreenberg/projects/vts-timing-3345
DEMOS=caltech101_s,caltech101_m,caltech101_l,caltech101_a,esc50_s,esc50_m,esc50_l,esc50_a
TUNE="$WT/scripts/profiling/tune_timing_profile.py"
ROWS="$EXP/rows.jsonl"

mkdir -p "$EXP/data"
ln -sfn /exp/scale26/datasets/external/vtsearch-resweep/j3-gpu-cuml-on/caltech-101 "$EXP/data/caltech-101"
ln -sfn /exp/scale26/datasets/external/vtsearch-demos/ESC-50-master "$EXP/data/ESC-50-master"

module load python/3.12.3
source /exp/sgreenberg/projects/VTSearch/.venv/bin/activate
cd "$WT"

export VTSEARCH_DATA_DIR="$EXP/data"
export VTSEARCH_MODELS_DIR=/exp/scale26/datasets/external/vtsearch-calib-2623/models
export VTSEARCH_TORCH_THREADS="${SLURM_CPUS_ON_NODE:-8}"
export VTSEARCH_PROFILE_LOAD="$EXP/loadprof.jsonl"
unset VTSEARCH_TIMING_PROFILE || true

{
  echo "host=$(hostname)"
  echo "commit=$(git rev-parse HEAD)"
  echo "slurm_job=${SLURM_JOB_ID:-none}"
  echo "cpus=${SLURM_CPUS_ON_NODE:-?} torch_threads=$VTSEARCH_TORCH_THREADS"
  echo "cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
  nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"
} > "$EXP/env.txt"
python "$WT/scripts/experiments/timing_r2/env_probe.py" >> "$EXP/env.txt" 2>&1 || true
cat "$EXP/env.txt"

echo "=== [$(date +%H:%M:%S)] dataset_load (post-fix) ==="
python "$TUNE" --drive --allow-mutating --tasks dataset_load \
  --demo "$DEMOS" --reps 1 --data-dir "$EXP/data" \
  --record "$ROWS" --out "$EXP/profile_fixed.json" 2>&1 | tee "$EXP/load.log"

# dataset_promote never ran in leg 1: its driver called `GET /api/medias`,
# which was replaced by `/api/medias/ids`, so every dataset was skipped and the
# family reported NOT MEASURED. Re-driven here to prove the corrected endpoint
# actually measures it.
echo "=== [$(date +%H:%M:%S)] dataset_promote (post-fix) ==="
python "$TUNE" --drive --allow-mutating --tasks dataset_promote \
  --reps 1 --data-dir "$EXP/data" \
  --record "$ROWS" --out "$EXP/profile_fixed.json" 2>&1 | tee "$EXP/promote.log"

echo "=== [$(date +%H:%M:%S)] fit legacy profiler rows ==="
python "$TUNE" --fit-only "$EXP/loadprof.jsonl" --out "$EXP/profile_fixed_loadprof.json" \
  --notes "#3345 post-fix legacy _load_profiler rows" 2>&1 | tee "$EXP/fit-loadprof.log"

echo "=== [$(date +%H:%M:%S)] done ==="
wc -l "$ROWS" "$EXP/loadprof.jsonl"
touch "$EXP/DONE.run"
