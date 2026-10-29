#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/expscratch/sgreenberg/rapids-4390/logs/%x-%j.log
# #4390: on one GPU node, run the scratch venv ARM through (1) install.sh's vts_smoke_test_gpu probe,
# (2) #4838's validate.py (torch matmul/SDPA/conv, EmbeddingGemma 2 + pile re-embed, cuML UMAP/k-means fit
# with gpu_backends._cuml_runtime_failed checked), (3) the onnxruntime-gpu face path on GPU vs CPU.
# Usage: sbatch --gres=gpu:<type>:1 -J probe-<arm>-4390 gpu_probe.sh <arm>   (arm = cu129 | v100 | live)
set -uo pipefail
ARM=$1; D=/expscratch/sgreenberg/rapids-4390; W=/expscratch/sgreenberg/worktrees/vts-4390
case $ARM in live) V=/expscratch/sgreenberg/vtsearch-venv ;; *) V=$D/venv-$ARM ;; esac
source /etc/profile.d/modules.sh; module load python/3.12.3
export VIRTUAL_ENV=$V PATH=$V/bin:$PATH VTS_REPO=$W PYTHONPATH=$W \
  VTSEARCH_DATA_DIR=/expscratch/sgreenberg/vts-cache/datadir VTSEARCH_MODELS_DIR=/expscratch/sgreenberg/vts-cache/models \
  HF_HOME=/expscratch/sgreenberg/vts-cache/models OPENBLAS_CORETYPE=Haswell TOKENIZERS_PARALLELISM=false
TAG=$ARM-$(hostname)
echo "== $ARM on $(hostname) python=$(command -v python) cvd=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader; nvidia-smi | grep -m1 "CUDA Version"
python -c 'import torch, pandas, numpy; print("torch", torch.__version__, "cuda", torch.version.cuda, "pandas", pandas.__version__, "numpy", numpy.__version__)'
python -c 'import cuml, cudf; print("cuml", cuml.__version__, "cudf", cudf.__version__)' || echo "cuml/cudf import FAILED"

echo "== (1) install.sh smoke test probe"
awk '/^    cat >"\$pyfile" <<.PY.$/{f=1;next} /^PY$/{f=0} f' $W/scripts/install.sh > /tmp/smoke-$$.py
python /tmp/smoke-$$.py; echo "smoke rc=$?"; rm -f /tmp/smoke-$$.py

echo "== (2) validate.py"
cd $D && python validate.py --arm "$TAG" --out $D/results --cuml 2>&1 | grep -v -E 'pthread|VerifyOutput'

echo "== (3) onnxruntime-gpu face path (FHIBE buffalo_s, 50 photos @1024)"
cd $W && source scripts/experiments/pile/pile_env.sh
export IMAGEKIT_FACE_PROVIDER=oss IMAGEKIT_FACE_CONFIG=/expscratch/sgreenberg/imagekit-models/grid-default/imagekit/face.json \
  VTSEARCH_TORCH_THREADS=8 OMP_NUM_THREADS=8 VIRTUAL_ENV=$V PATH=$V/bin:$PATH
S=scripts/experiments/fhibe/ort_gpu_5332.py
for dev in cuda cpu; do
  VTSEARCH_DEVICE=$dev python $S run 50 1024 $TAG-$dev $D/results/face-$TAG-$dev.npz 2>&1 | grep -v -E 'pthread|VerifyOutput' | grep -E "^ort|^sessions|ms/face|Error|error|Traceback" | sort -u
done
python $S compare $D/results/face-$TAG-cuda.npz $D/results/face-$TAG-cpu.npz | tail -5
echo "== done $(date +%T)"
