#!/bin/bash
# How does the coverage-atlas rebuild scale past n = 2954? (#3595)
#
# #3521 drove the rebuild on caltech101_{s,m,l,a} (412..2954) and fitted
# 0.0026 s/item, r^2 0.95. Everything larger -- including the "~131 s at the
# 50 000-item auto-build threshold" figure the shipped comments quote -- was an
# extrapolation across a 17x gap. This job measures that gap.
#
# Workload, one V100 node, one data dir of its own:
#   image  caltech101_{s,m,l,a}   412 / 838 / 1704 / 2954   (#3521's points, re-measured
#                                                            here so the fit is one node's)
#          places365_{s,m,a}      5110 / 10220 / 36500
#   audio  esc50_a                2000
#          urbansound8k_{l,a}     4995 / 8732
# Each dataset is loaded once (embedding is not what is measured), then opened
# with --cold-atlas TWICE, so every size carries two real rebuilds.
#
# Source media are symlinks into the shared read-only demo caches, so nothing
# downloads. Completion is the file $EXP/DONE.run, never a live process.
#SBATCH --job-name=atlas3595
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/expscratch/sgreenberg/slurm-atlas3595-%j.out

set -euo pipefail

EXP=${EXP:-/expscratch/sgreenberg/atlas-3595}
WT=${VTS_REPO:?set VTS_REPO to the worktree to run}
TUNE="$WT/scripts/profiling/tune_timing_profile.py"
DEMOS_CACHE=/exp/scale26/datasets/external/vtsearch-demos
CALTECH=/exp/scale26/datasets/external/vtsearch-resweep/j3-gpu-cuml-on/caltech-101
DEMOS=caltech101_s,caltech101_m,caltech101_l,caltech101_a,places365_s,places365_m,places365_a,esc50_a,urbansound8k_l,urbansound8k_a

module load python/3.12.3
source /exp/sgreenberg/projects/VTSearch/.venv/bin/activate
source "$WT/gridenv.sh"
cd "$WT"

mkdir -p "$EXP/data"
ln -sfn "$CALTECH"                     "$EXP/data/caltech-101"
ln -sfn "$DEMOS_CACHE/places365"       "$EXP/data/places365"
ln -sfn "$DEMOS_CACHE/ESC-50-master"   "$EXP/data/ESC-50-master"
ln -sfn "$DEMOS_CACHE/UrbanSound8K"    "$EXP/data/UrbanSound8K"

export HF_HOME=/expscratch/sgreenberg/cache/huggingface
export VTSEARCH_MODELS_DIR=/exp/scale26/datasets/external/vtsearch-calib-2623/models
export VTSEARCH_TORCH_THREADS="${SLURM_CPUS_ON_NODE:-8}"
export VTSEARCH_DATA_DIR="$EXP/data"
export VTSEARCH_PROFILE_LOAD="$EXP/loadprof.jsonl"
unset VTSEARCH_TIMING_PROFILE || true
ROWS="$EXP/rows.jsonl"

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

step () { echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "dataset_load (once per demo)"
python "$TUNE" --drive --allow-mutating --tasks dataset_load \
  --demo "$DEMOS" --reps 1 --data-dir "$EXP/data" \
  --record "$ROWS" --out "$EXP/profile_load.json" 2>&1 | tee "$EXP/load.log"

for pass in 1 2; do
  step "dataset_open --cold-atlas, pass $pass"
  python "$TUNE" --drive --tasks dataset_open --cold-atlas \
    --reps 1 --data-dir "$EXP/data" \
    --record "$ROWS" --out "$EXP/profile_open_$pass.json" 2>&1 | tee "$EXP/open_$pass.log"
done

step "done"
wc -l "$ROWS"
touch "$EXP/DONE.run"
