#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:50:00
#SBATCH --output=/expscratch/sgreenberg/rapids-4390/logs/%x-%j.log
# #4390: one GPU allocation, every candidate venv through gpu_probe.sh in turn (A100/L40S slots are scarce).
for arm in cu129 v100 v100mix; do bash /expscratch/sgreenberg/rapids-4390/gpu_probe.sh $arm; done
