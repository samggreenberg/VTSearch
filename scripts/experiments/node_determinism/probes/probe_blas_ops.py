"""#4930 / #4938: run one BLAS op through numpy and print a hash of its result, or crash trying.

One op a process, so a segfault names its op:

    python probe_blas_ops.py OP M K      # OP: {s,d}{gemm,syrk,gemv,svd}, or {s,d}gemm_scipy

``A`` is M x K from a fixed seed; ``gemm`` is ``A @ B`` with ``B`` K x M, ``syrk`` is
``A @ A.T`` (numpy hands it to syrk rather than gemm), ``gemv`` is ``A @ v``, ``svd`` is
numpy's economy SVD of ``A`` (LAPACK, which calls GEMM itself; the singular values); ``_scipy``
runs the GEMM through ``scipy.linalg.blas``, i.e. scipy's own bundled OpenBLAS.  K is the
reduction length, which OpenBLAS cuts into chunks of its ``GEMM_Q``: on OpenBLAS 0.3.34 a
Zen 4 host overrides Q under a forced ``OPENBLAS_CORETYPE`` (OpenMathLib/OpenBLAS#6021),
so a float64 GEMM with K above ~299 overruns the Haswell kernel's stack and segfaults,
and a float32 one sums in a different order than any other host.  ``openblas_l2.sh``
drives it.
"""

import hashlib
import os
import sys

import numpy as np

op, m, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dtype = {"s": np.float32, "d": np.float64}[op[0]]
rng = np.random.default_rng(0)
A = rng.standard_normal((m, k)).astype(dtype)
B = rng.standard_normal((k, m)).astype(dtype)
if op.endswith("_scipy"):
    from scipy.linalg import blas

    out = getattr(blas, op.removesuffix("_scipy"))(1.0, A, B)
else:
    out = {
        "gemm": lambda: A @ B,
        "syrk": lambda: A @ A.T,
        "gemv": lambda: A @ B[:, 0].copy(),
        "svd": lambda: np.linalg.svd(A, full_matrices=False)[1],
    }[op[1:]]()
digest = hashlib.sha1(np.ascontiguousarray(out).tobytes()).hexdigest()[:12]
env = " ".join(f"{v}={os.environ.get(v, '-')}" for v in ("OPENBLAS_CORETYPE", "OPENBLAS_L2_SIZE"))
print(f"{os.uname().nodename.split('.')[0]}\tnumpy={np.__version__}\t{env}\t{op}\t{m}\t{k}\t{digest}")
