#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/expscratch/sgreenberg/rapids-4390/logs/%x-%j.log
# #4390 V100 path, stage 2: lift every nvidia-*-cu12 runtime lib to the version the cu129 venv carries
# (cudart, nvrtc, nvjitlink, cublas, cusolver, cusparse, cufft, curand, cccl, ...), so RAPIDS sees no mixed headers.
set -uo pipefail
D=/expscratch/sgreenberg/rapids-4390; V=$D/venv-v100
source /etc/profile.d/modules.sh; module load python/3.12.3
export PIP_CACHE_DIR=/expscratch/sgreenberg/cache/pip
source "$V/bin/activate"
mapfile -t pins < <(grep -E '^nvidia-.*-cu12==' $D/freeze-cu129.txt)
echo "cu129 nvidia set:"; printf '  %s\n' "${pins[@]}"
pip install --progress-bar off --no-deps --extra-index-url https://pypi.nvidia.com "${pins[@]}"; rc=$?
pip freeze --all > $D/freeze-v100.txt; pip check > $D/check-v100.txt 2>&1
diff <(sort $D/freeze-v100-stage1.txt) <(sort $D/freeze-v100.txt)
echo "== nvidia libs in v100 venv NOT on cu129's version:"
comm -23 <(grep -E '^nvidia-' $D/freeze-v100.txt | sort) <(grep -E '^nvidia-' $D/freeze-cu129.txt | sort)
echo "== pip check"; cat $D/check-v100.txt
exit $rc
