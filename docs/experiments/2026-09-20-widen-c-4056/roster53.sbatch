#!/bin/bash
#SBATCH --job-name=roster53
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=360
#SBATCH --output=/expscratch/sgreenberg/results/roster54-4056/roster53-%j.log
set -euo pipefail
export LD_LIBRARY_PATH=/cluster/apps/python/3.12.3/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
source /expscratch/sgreenberg/vtsearch-venv/bin/activate
cd /expscratch/sgreenberg/vts-3986/scripts/experiments/pile
D=/expscratch/sgreenberg/results/roster54-4056
for e in siglip siglip2_l; do
  echo "########## $e (wide = the shipped 53)"
  python coco_roster_width_pool.py --embedder "$e" \
    --roster-json /expscratch/sgreenberg/results/roster53.json \
    --json "$D/roster53-$e.json" | tail -18
done
echo "ALL DONE"
