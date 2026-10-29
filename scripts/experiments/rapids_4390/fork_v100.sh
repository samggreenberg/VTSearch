#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --output=/expscratch/sgreenberg/rapids-4390/logs/%x-%j.log
# #4390: split the V100 venv into the two candidate shapes.
#  venv-v100mix: install.sh cu126 + cuml-cu12>=26.8 as pip left it (12.6 libs, nvjitlink 12.9.86) = freeze-v100-stage1.
#  venv-v100:    the issue's lift (12.9 cudart/nvrtc/nvjitlink/cublas/cusolver/...) but cuDNN back at torch's 9.10.2.21
#                (cuDNN 9.20 refuses SM < 7.5: torch + onnxruntime fail on the V100, probe 1145779).
set -uo pipefail
D=/expscratch/sgreenberg/rapids-4390
source /etc/profile.d/modules.sh; module load python/3.12.3
export PIP_CACHE_DIR=/expscratch/sgreenberg/cache/pip
[ -e $D/venv-v100mix ] && { echo exists; exit 2; }
echo "[$(date +%T)] copy"; cp -a $D/venv-v100 $D/venv-v100mix || exit 3
/expscratch/sgreenberg/venv-4838/relocate.sh $D/venv-v100mix $D/venv-v100 $D/venv-v100mix
mapfile -t pins < <(grep -E '^nvidia-.*-cu12==' $D/freeze-v100-stage1.txt)
$D/venv-v100mix/bin/python -m pip install --progress-bar off --no-deps "${pins[@]}" || exit 4
$D/venv-v100/bin/python -m pip install --progress-bar off --no-deps nvidia-cudnn-cu12==9.10.2.21 || exit 5
for v in v100mix v100; do
  $D/venv-$v/bin/python -m pip freeze --all > $D/freeze-$v.txt; $D/venv-$v/bin/python -m pip check > $D/check-$v.txt 2>&1
  echo "== $v vs stage1:"; diff <(sort $D/freeze-v100-stage1.txt) <(sort $D/freeze-$v.txt); echo "== pip check $v"; cat $D/check-$v.txt
done
echo "[$(date +%T)] done"
