"""#4110: what the pins cost - torch GEMM and an encoder layer on the CPU, numpy exp/log, the anchored EM."""

import json
import os
import sys
import time
import numpy as np

variant = sys.argv[1]


def best(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return round(min(ts) * 1000, 2)


out = {"node": os.uname().nodename.split(".")[0], "variant": variant}
rng = np.random.default_rng(0)
v = rng.uniform(-30, 5, 1_000_000)
buf = np.empty_like(v)
out["np_exp_1M_ms"] = best(lambda: np.exp(v, out=buf))
out["np_log_1M_ms"] = best(lambda: np.log(np.abs(v) + 1e-3, out=buf))
sys.path.insert(0, os.environ["VTS_REPO"])
from vtscore.training.thresholds.gmm import fit_anchored_score_gmm, gmm_fit_array

scores = np.concatenate([rng.normal(0.2, 0.08, 40_000), rng.normal(0.7, 0.1, 400)]).clip(0, 1)
arr = gmm_fit_array(scores)
anc_s = np.concatenate([rng.normal(0.2, 0.08, 20), rng.normal(0.7, 0.1, 10)])
anc_l = np.array([0.0] * 20 + [1.0] * 10)
out["anchored_em_ms"] = best(lambda: fit_anchored_score_gmm(arr, anc_s, anc_l))
out["anchored_em_fit"] = repr(fit_anchored_score_gmm(arr, anc_s, anc_l)[0])[:200]

import torch

for nt in (1, 4):
    torch.set_num_threads(nt)
    A = torch.randn(2048, 1152)
    B = torch.randn(1152, 1152)
    out[f"gemm_2048x1152x1152_t{nt}_ms"] = best(lambda: A @ B)
    layer = torch.nn.TransformerEncoderLayer(768, 12, 3072, batch_first=True).eval()
    x = torch.randn(8, 197, 768)
    with torch.no_grad():
        out[f"vit_layer_b8_t{nt}_ms"] = best(lambda: layer(x), reps=3)
json.dump(out, open(f"{os.environ['PROBE_OUT']}/{out['node']}__{variant}.json", "w"), indent=1)
print(json.dumps(out))
