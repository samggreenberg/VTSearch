#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
# #5332: which onnxruntime-gpu releases run on a V100 (sm_70): CUDA session on w600k_mbf + det_500m, output vs CPU.
cd /expscratch/sgreenberg/worktrees/vts-5332
source scripts/grid/gridenv.sh >/dev/null 2>&1
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
M=/expscratch/sgreenberg/imagekit-models/grid-default/imagekit
for v in 1.23.2 1.24.4 1.25.1 1.26.0 1.27.0 1.28.0 1.29.0 1.30.0 1.31.0; do
  d=$TMPDIR/ort-$v; pip install -q --no-deps --target $d onnxruntime-gpu==$v >/dev/null 2>&1 || { echo "$v install failed"; continue; }
  PYTHONPATH=$d python - $v $M <<'PY' 2>&1 | grep -v -E "pthread|VerifyOutput" | tail -1
import sys, numpy as np, torch, onnxruntime as ort
v, m = sys.argv[1], sys.argv[2]
try:
    out = []
    for name, shape in (("w600k_mbf.onnx", (1, 3, 112, 112)), ("det_500m.onnx", (1, 3, 640, 640))):
        x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        r = {}
        for p in ("CUDAExecutionProvider", "CPUExecutionProvider"):
            s = ort.InferenceSession(f"{m}/{name}", providers=[p])
            assert s.get_providers()[0] == p, s.get_providers()
            r[p] = s.run(None, {s.get_inputs()[0].name: x})[0].ravel()
        a, b = r.values()
        out.append(f"{name} maxdiff {np.abs(a - b).max():.2e}")
    print(v, "OK", ort.__version__, "; ".join(out))
except Exception as e:
    print(v, "FAIL", type(e).__name__, str(e)[:160])
PY
done
