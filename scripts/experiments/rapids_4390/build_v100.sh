#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=/expscratch/sgreenberg/rapids-4390/logs/%x-%j.log
# #4390 V100 path, stage 1: `install.sh cu126` (torch 2.14.1+cu126, cuML step skipped), then cuml-cu12>=26.8 + pandas>=3.
set -uo pipefail
D=/expscratch/sgreenberg/rapids-4390; V=$D/venv-v100; W=/expscratch/sgreenberg/worktrees/vts-4390
source /etc/profile.d/modules.sh; module load python/3.12.3
export PIP_CACHE_DIR=/expscratch/sgreenberg/cache/pip VTSEARCH_VERBOSE=1
[ -e "$V" ] && { echo "exists: $V"; exit 2; }
python3 -m venv "$V" && source "$V/bin/activate"
echo "[$(date +%T)] $(hostname)"
cd "$W" && bash scripts/install.sh cu126; rc=$?
echo "[$(date +%T)] install.sh cu126 rc=$rc"
pip freeze --all > $D/freeze-v100-stage0.txt
pip install --progress-bar off --extra-index-url https://pypi.nvidia.com --prefer-binary "cuml-cu12>=26.8" "pandas>=3"; rc2=$?
echo "[$(date +%T)] cuml rc=$rc2"
pip freeze --all > $D/freeze-v100-stage1.txt; pip check > $D/check-v100-stage1.txt 2>&1
diff <(sort $D/freeze-v100-stage0.txt) <(sort $D/freeze-v100-stage1.txt)
echo "== pip check"; cat $D/check-v100-stage1.txt
exit $(( rc || rc2 ))
