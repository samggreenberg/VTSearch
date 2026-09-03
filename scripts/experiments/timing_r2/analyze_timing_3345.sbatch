#!/bin/bash
# Analysis + figures for #3345, on a GPU node on purpose.
#
# `load_step_weights` reads the device from the *process*, not from an argument,
# so the shipped-`LOAD_COST_MODEL` baseline row of the report's pacing table is
# only comparable with the fitted profiles when the analysis resolves the same
# device the sweep ran on. Run this on a login node and that row silently
# becomes a CPU row compared against `cuda+cuml` profiles.
#SBATCH --job-name=timing3345an
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/expscratch/sgreenberg/timing-3345/slurm-analyze-%j.out

set -euo pipefail

EXP=/expscratch/sgreenberg/timing-3345
WT=/exp/sgreenberg/projects/vts-timing-3345
FIGS="$WT/docs/experiments/2026-09-02-timing-r2-3345/figures"

module load python/3.12.3
source /exp/sgreenberg/projects/VTSearch/.venv/bin/activate
cd "$WT"
mkdir -p "$FIGS"

python scripts/experiments/timing_r2/env_probe.py

# Refit every profile from the recorded rows before reading them, so the tables
# always describe the fitter in the tree rather than whichever version happened
# to run during the sweep. The rows are the durable artefact; the profiles are
# derived and cheap to rebuild.
TUNE=scripts/profiling/tune_timing_profile.py
python "$TUNE" --fit-only "$EXP/rows.jsonl" --out "$EXP/profile_generic.json" \
  --notes "#3345 generic recorder, five families"
python "$TUNE" --fit-only "$EXP/loadprof.jsonl" --out "$EXP/profile_loadprof.json" \
  --notes "#3345 legacy _load_profiler rows from the same loads"
python "$TUNE" --fit-only /exp/sgreenberg/resweep-3062/*.filt.jsonl --out "$EXP/profile_resweep.json" \
  --notes "#3062 resweep rows refit through vtscore.timing.fit"
python "$TUNE" --fit-only "$EXP/fixed/rows.jsonl" --out "$EXP/fixed/profile_fixed.json" \
  --notes "#3345 post-fix generic recorder"
python "$TUNE" --fit-only "$EXP/fixed/loadprof.jsonl" --out "$EXP/fixed/profile_fixed_loadprof.json" \
  --notes "#3345 post-fix legacy _load_profiler rows"

python scripts/experiments/timing_r2/analyze_timing_3345.py --exp "$EXP" 2>&1 | tee "$EXP/analysis.log"

python scripts/experiments/timing_r2/figures_timing_3345.py \
  --summary "$EXP/summary_timing3345.json" \
  --rows "$EXP/rows.jsonl" \
  --fixed-rows "$EXP/fixed/rows.jsonl" \
  --outdir "$FIGS"

ls -la "$FIGS"
rm -f "$EXP/DONE.analyze"
touch "$EXP/DONE.analyze"
