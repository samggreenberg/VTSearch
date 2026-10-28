#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodelist=rack7n11
#SBATCH --cpus-per-task=60
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
# #5330: build fhibe_faces_ik_<size> x face under buffalo_<v> into a private pile + FHIBE root.
#   sbatch buffalo_m_build_5330.sh <v: l|m> <size>
set -euo pipefail
V=$1 SIZE=$2 W=/expscratch/sgreenberg/fhibe/ik-buffalo_m
WT=/expscratch/sgreenberg/worktrees/vts-5330-run
cd "$WT"
source gridenv.sh >/dev/null 2>&1
export VTS_PILE=$W/$V/pile VTS_FHIBE_ROOT=$W/$V/root
source scripts/experiments/pile/pile_env.sh
export VTS_REPO="$WT" VTS_LAUNCH_COMMIT=$(git -C "$WT" rev-parse HEAD)
export IMAGEKIT_FACE_PROVIDER=oss IMAGEKIT_FACE_CONFIG=/expscratch/sgreenberg/imagekit-models/buffalo_$V.json
export VTSEARCH_DEVICE=cpu VTSEARCH_TORCH_THREADS=60 OMP_NUM_THREADS=60
echo "venv=$VIRTUAL_ENV host=$(hostname) commit=$VTS_LAUNCH_COMMIT pile=$VTS_PILE root=$VTS_FHIBE_ROOT cfg=$IMAGEKIT_FACE_CONFIG"
cd scripts/experiments/pile
python build_pile.py --datasets "fhibe_faces_ik_$SIZE" --embedders face
python build_pile.py --datasets "fhibe_faces_ik_$SIZE" --embedders face --verify
