#!/bin/bash
# Analysis + figures for #3521, chained off the sweep with --dependency=afterok.
#
# On a GPU node on purpose: `load_step_weights` reads the device from the
# PROCESS, not from an argument, so the `shipped` arm silently becomes a CPU row
# on a login node and every number in the table would describe hardware the
# sweep did not run on (#3345).
#SBATCH --job-name=drive3521a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
# Deliberately NOT inside the study dir. `--output=` is opened by slurmd before
# the first line of this script runs, so a path under a deleted study dir kills
# the job before any in-body guard can speak -- and the failure has nowhere to
# be written, because the log IS the missing file (#4006). The scratch root
# outlives any one study, so the job name carries the study instead.
#SBATCH --output=/expscratch/sgreenberg/slurm-drive3521-analyze-%j.out

set -euo pipefail
EXP=/expscratch/sgreenberg/drive-3521

# The study output dir is read, not created: fail naming it (#4001).
if [ ! -d "$EXP" ]; then
  echo "$EXP does not exist: this study's output was deleted 2026-09-18 (see #4001 and /expscratch/sgreenberg/keep/deleted-20260918.md)." >&2
  echo "Re-run the study, or set EXP to a directory that has its results." >&2
  exit 2
fi

WT=/exp/sgreenberg/projects/vts-drive-3521

module load python/3.12.3
source /exp/sgreenberg/projects/VTSearch/.venv/bin/activate
cd "$WT"

python scripts/experiments/drive_cold/analyze_drive_3521.py --exp "$EXP" 2>&1 | tee "$EXP/analyze.log"
python scripts/experiments/drive_cold/figures_drive_3521.py --exp "$EXP" 2>&1 | tee -a "$EXP/analyze.log"
touch "$EXP/DONE.analyze"
