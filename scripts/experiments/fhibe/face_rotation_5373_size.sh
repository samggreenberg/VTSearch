#!/bin/bash
# #5373 sizing, 16 shards: SIZE=1024|640 QMAX=<faces under this quality> sbatch face_rotation_5373_size.sh
#SBATCH -c 4 --mem=8G --time=2:00:00 --array=0-15 -o /expscratch/sgreenberg/fhibe/rotation-5373/logs/size-%A_%a.out
export LD_LIBRARY_PATH=/cluster/apps/python/3.12.3/lib OMP_NUM_THREADS=4
cd /expscratch/sgreenberg/worktrees/vts-5373
PYTHONPATH=/expscratch/sgreenberg/worktrees/vts-5373 exec /expscratch/sgreenberg/vtsearch-venv/bin/python scripts/experiments/fhibe/face_rotation_5373_size.py "${SIZE:-1024}" "${QMAX:-18}" "$SLURM_ARRAY_TASK_ID" 16 "/expscratch/sgreenberg/fhibe/rotation-5373/size_${SIZE:-1024}_q${QMAX:-18}_${SLURM_ARRAY_TASK_ID}.csv"
