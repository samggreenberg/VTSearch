#!/bin/bash
#SBATCH --job-name=negpool3986
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=360
#SBATCH --output=/expscratch/sgreenberg/results/negpool-3986/robust-%j.log
set -euo pipefail
cd /expscratch/sgreenberg/vts-3986
source gridenv.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
D=/expscratch/sgreenberg/results/negpool-3986
for e in siglip2_l clip clip_l; do
  echo "########## $e shipped"
  python scripts/experiments/pile/coco_negative_pool_shortcut.py \
      --embedder "$e" --train-pool shipped --json "$D/shortcut-$e-shipped.json" | tail -14
done
echo "ALL DONE"
