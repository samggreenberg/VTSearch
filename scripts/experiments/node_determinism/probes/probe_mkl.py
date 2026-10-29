"""#4110: which MKL / ATen pin makes torch's CPU float32 scoring identical across node types; prices a 50k-row scoring pass."""

import hashlib
import json
import os
import sys
import time
import numpy as np


def h(a):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:12]


variant = sys.argv[1]
import torch

torch.set_num_threads(1)
if "nomkldnn" in variant:
    torch.backends.mkldnn.enabled = False
out = {
    "node": os.uname().nodename.split(".")[0],
    "variant": variant,
    "cap": torch.backends.cpu.get_cpu_capability(),
    "env": {
        k: os.environ.get(k)
        for k in ("MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS", "ATEN_CPU_CAPABILITY", "NPY_DISABLE_CPU_FEATURES")
    },
}
cfg = torch.__config__.show()
out["blas"] = [ln.strip() for ln in cfg.splitlines() if "BLAS" in ln or "MKL" in ln or "LAPACK" in ln][:6]
rng = np.random.default_rng(0)
X32 = rng.standard_normal((4000, 768)).astype(np.float32)
w32 = rng.standard_normal(768).astype(np.float32)
v32 = rng.uniform(-30, 5, 200_000).astype(np.float32)
r = {}
lin = torch.nn.Linear(768, 1)
with torch.no_grad():
    lin.weight.copy_(torch.from_numpy(w32).reshape(1, -1))
    lin.bias.fill_(0.25)
    tX = torch.from_numpy(X32)
    r["linear"] = h(lin(tX).numpy())
    r["matmul"] = h((tX[:500] @ tX[:500].T).numpy())
    r["mv"] = h(torch.mv(tX, torch.from_numpy(w32)).numpy())
    r["exp"] = h(torch.exp(torch.from_numpy(v32)).numpy())
    r["sigmoid"] = h(torch.sigmoid(torch.from_numpy(v32)).numpy())
sys.path.insert(0, os.environ["VTS_REPO"])
from vtscore.training.svm import fit_linear_svm_head
from vtscore.detectors.training_scoring import _forward_sigmoid_chunked

y = (X32[:300, 0] + 0.5 * rng.standard_normal(300) > 0.8).astype(np.float32)
m = fit_linear_svm_head(X32[:300], y, 768)
m = m[0] if isinstance(m, tuple) else m
r["svm_head_coef"] = h(next(m.parameters()).detach().numpy())
r["vts_scores"] = h(np.asarray(_forward_sigmoid_chunked(m, X32), dtype=np.float32))
# price: one production-size scoring pass (50k rows x 1152, siglip2-ish) through the head
Xb = rng.standard_normal((50_000, 1152)).astype(np.float32)
mb = torch.nn.Sequential(torch.nn.Linear(1152, 1))
t0 = time.perf_counter()
for _ in range(5):
    _forward_sigmoid_chunked(mb, Xb)
out["score_50k_ms"] = round((time.perf_counter() - t0) / 5 * 1000, 1)
out["r"] = r
json.dump(out, open(f"{os.environ['PROBE_OUT']}/{out['node']}__{variant}.json", "w"), indent=1)
print(out["node"], variant, out["cap"], out["score_50k_ms"], "ms", r)
