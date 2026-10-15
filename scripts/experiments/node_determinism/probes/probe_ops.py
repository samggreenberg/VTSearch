"""#4110: hash the outputs of the numeric ops on a calibration cell's path (numpy, sklearn, vtscore, torch), per node and env pin."""

import hashlib
import json
import os
import sys

import numpy as np


def h(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]


out = {"node": os.uname().nodename, "variant": sys.argv[1]}
cpuinfo = open("/proc/cpuinfo").read().splitlines()
out["cpu"] = next(ln.split(":", 1)[1].strip() for ln in cpuinfo if ln.startswith("model name"))
flags = set(next(ln.split(":", 1)[1].split() for ln in cpuinfo if ln.startswith("flags")))
out["avx512f"] = "avx512f" in flags
out["env"] = {
    k: os.environ.get(k)
    for k in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "ATEN_CPU_CAPABILITY", "MKL_CBWR", "OMP_NUM_THREADS")
}
try:
    from numpy._core._multiarray_umath import __cpu_features__ as F
except ImportError:
    from numpy.core._multiarray_umath import __cpu_features__ as F
out["np"] = np.__version__
out["np_avx512_on"] = sorted(k for k, v in F.items() if v and k.startswith("AVX512"))

rng = np.random.default_rng(0)
X32 = rng.standard_normal((4000, 768)).astype(np.float32)
w32 = rng.standard_normal(768).astype(np.float32)
X64 = X32.astype(np.float64)
r = {}
r["np_matvec_f32"] = h(X32 @ w32)
r["np_matmul_f32"] = h(X32[:500] @ X32[:500].T)
r["np_matvec_f64"] = h(X64 @ w32.astype(np.float64))
v64 = rng.uniform(-30, 5, 200_000)
r["np_exp_f64"] = h(np.exp(v64))
r["np_log_f64"] = h(np.log(np.abs(v64) + 1e-3))
v32 = v64.astype(np.float32)
r["np_exp_f32"] = h(np.exp(v32))
r["np_log_f32"] = h(np.log(np.abs(v32) + np.float32(1e-3)))
r["np_tanh_f64"] = h(np.tanh(v64 / 10))
r["np_sum_f64"] = float(np.sum(v64)).hex()
r["np_sum_f32"] = h(np.sum(X32, axis=0))
r["np_var_f64"] = float(np.var(v64)).hex()
r["np_mean_f32_axis0"] = h(X32.mean(axis=0))
ties = np.round(rng.random(100_000), 3)
r["np_argsort_quick"] = h(np.argsort(ties))
r["np_argsort_stable"] = h(np.argsort(ties, kind="stable"))
r["np_sort_f64"] = h(np.sort(v64))

import torch

torch.set_num_threads(1)
out["torch"] = torch.__version__
out["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
lin = torch.nn.Linear(768, 1)
with torch.no_grad():
    lin.weight.copy_(torch.from_numpy(w32).reshape(1, -1))
    lin.bias.fill_(0.25)
    tX = torch.from_numpy(X32)
    r["torch_linear_f32"] = h(lin(tX).numpy())
    r["torch_sigmoid_linear"] = h(torch.sigmoid(lin(tX)).numpy())
    r["torch_matmul_f32"] = h((tX[:500] @ tX[:500].T).numpy())
    r["torch_sum_dim0"] = h(tX.sum(dim=0).numpy())
    r["torch_exp_f32"] = h(torch.exp(torch.from_numpy(v32)).numpy())

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

y = (X32[:300, 0] + 0.5 * rng.standard_normal(300) > 0.8).astype(int)
r["sk_scaler"] = h(StandardScaler().fit_transform(X32[:300]))
svc = LinearSVC(C=1.0, random_state=0, max_iter=5000).fit(X32[:300], y)
r["sk_linearsvc_coef"] = h(svc.coef_)

sys.path.insert(0, os.environ["VTS_REPO"])
from vtscore.training.thresholds import fit_gmm_threshold
from vtscore.training.thresholds.gmm import fit_score_gmm

scores = np.concatenate([rng.normal(0.2, 0.08, 3000), rng.normal(0.7, 0.1, 60)]).clip(0, 1)
fit = fit_score_gmm(scores)
r["vts_fit_score_gmm"] = (
    repr(fit)
    if fit is None
    else h(np.array([float(getattr(fit, f)) for f in fit.__dataclass_fields__ if isinstance(getattr(fit, f), float)]))
)
thr, _ = fit_gmm_threshold(list(map(float, scores)))
r["vts_gmm_threshold"] = float(thr).hex()
from vtscore.training.svm import fit_linear_svm_head

try:
    head = fit_linear_svm_head(X32[:300], y.astype(np.float32), 768)
    m = head[0] if isinstance(head, tuple) else head
    with torch.no_grad():
        r["vts_svm_head_scores"] = h(m(tX).detach().numpy())
except Exception as e:  # signature drift: record, keep going
    r["vts_svm_head_scores"] = f"ERR {type(e).__name__}: {e}"[:120]

try:
    from threadpoolctl import threadpool_info

    out["threadpools"] = [
        {k: d.get(k) for k in ("internal_api", "architecture", "version", "num_threads")} for d in threadpool_info()
    ]
except Exception as e:
    out["threadpools"] = str(e)
out["r"] = r
d = os.environ["PROBE_OUT"]
with open(f"{d}/{out['node']}__{out['variant']}.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("node", "variant", "cpu", "np_avx512_on", "torch_cpu_capability")}))
