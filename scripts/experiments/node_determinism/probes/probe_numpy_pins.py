"""#4930: which NPY_DISABLE_CPU_FEATURES value takes a numpy build off its AVX-512 kernels, per numpy version and CPU.

Run once per (numpy, pin) in a fresh process: numpy reads the variable when it loads.
Writes ``$PROBE_OUT/<node>__<tag>.json``: the build's dispatch targets, which of them
are on, any warning numpy raised about the variable, the target each optimized ufunc
runs, and a hash of each op's output (the ops #4110 found differing by CPU, plus the
sorts and partitions x86-simd-sort dispatches).
"""

import hashlib
import json
import os
import sys
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import numpy as np

try:
    from numpy._core import _multiarray_umath as mu
except ImportError:  # numpy < 2
    from numpy.core import _multiarray_umath as mu


def h(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]


tag = sys.argv[1]
out = {"node": os.uname().nodename.split(".")[0], "tag": tag, "np": np.__version__}
cpuinfo = open("/proc/cpuinfo").read().splitlines()
out["cpu"] = next(ln.split(":", 1)[1].strip() for ln in cpuinfo if ln.startswith("model name"))
flags = set(next(ln.split(":", 1)[1].split() for ln in cpuinfo if ln.startswith("flags")))
out["cpu_avx512"] = sorted(f for f in flags if f.startswith("avx512"))
out["pin"] = os.environ.get("NPY_DISABLE_CPU_FEATURES")
out["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
out["baseline"] = list(mu.__cpu_baseline__)
out["dispatch"] = list(mu.__cpu_dispatch__)
out["dispatch_on"] = [t for t in mu.__cpu_dispatch__ if mu.__cpu_features__.get(t)]
out["features_on"] = sorted(k for k, v in mu.__cpu_features__.items() if v)

current: dict[str, int] = {}
per_func: dict[str, str] = {}
try:
    from numpy.lib import introspect

    for func, sigs in introspect.opt_func_info().items():
        for sig, d in sigs.items():
            current[d["current"]] = current.get(d["current"], 0) + 1
            if func in ("exp", "log", "tanh", "sin", "argmax", "maximum", "add") and sig.rstrip("i") in (
                "d",
                "f",
                "dd",
                "ff",
                "ddd",
                "fff",
            ):
                per_func[f"{func}:{sig}"] = d["current"]
except Exception as e:  # noqa: BLE001  # a probe records what it could not read
    current = {"error": repr(e)}
out["ufunc_current"] = current
out["ufunc_current_some"] = per_func

rng = np.random.default_rng(0)
X32 = rng.standard_normal((2000, 768)).astype(np.float32)
v64 = rng.uniform(-30, 5, 200_000)
v32 = v64.astype(np.float32)
ties = np.round(rng.random(100_000), 3)
ints = rng.integers(-1000, 1000, 100_000)
r = {}
r["exp_f64"] = h(np.exp(v64))
r["log_f64"] = h(np.log(np.abs(v64) + 1e-3))
r["exp_f32"] = h(np.exp(v32))
r["log_f32"] = h(np.log(np.abs(v32) + np.float32(1e-3)))
r["tanh_f64"] = h(np.tanh(v64 / 10))
r["sin_f32"] = h(np.sin(v32))
r["power_f64"] = h(np.power(np.abs(v64), 0.37))
r["sum_f64"] = float(np.sum(v64)).hex()
r["sum_f32_axis0"] = h(np.sum(X32, axis=0))
r["matmul_f32"] = h(X32[:500] @ X32[:500].T)
r["argsort_ties_f64"] = h(np.argsort(ties))
r["argsort_ties_f32"] = h(np.argsort(ties.astype(np.float32)))
r["argsort_i64"] = h(np.argsort(ints))
r["argsort_stable_f64"] = h(np.argsort(ties, kind="stable"))
r["sort_f64"] = h(np.sort(v64))
r["sort_f16"] = h(np.sort(v32.astype(np.float16)))
r["argsort_i16"] = h(np.argsort(ints.astype(np.int16)))
r["partition_f64"] = h(np.partition(ties, 5000))
r["argpartition_f64"] = h(np.argpartition(ties, 5000))
r["argmax_f32_axis1"] = h(np.argmax(X32, axis=1))
r["unique_ties"] = h(np.unique(ties, return_index=True)[1])
out["hash"] = r

dest = os.environ.get("PROBE_OUT", ".")
os.makedirs(dest, exist_ok=True)
with open(os.path.join(dest, f"{out['node']}__{tag}.json"), "w") as f:
    json.dump(out, f, indent=1)
print(tag, out["np"], "warn" if out["warnings"] else "ok", "dispatch_on:", out["dispatch_on"])
