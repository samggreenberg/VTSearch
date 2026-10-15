"""#4930 / #4938: run one BLAS op through numpy and print a hash of its result, or crash trying.

One op a process, so a segfault names its op:

    for op in sgemm ssyrk sgemv dgemm sdot; do for n in 16 64 500; do
      OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 python probe_blas_ops.py $op $n || echo "$op $n: exit $?"
    done; done

``A`` is n x 768 and ``B`` 768 x n, float32 from a fixed seed; ``ssyrk`` is ``A @ A.T``,
which numpy hands to syrk rather than gemm.  On numpy 2.5.4 (OpenBLAS 0.3.34) under
``OPENBLAS_CORETYPE=Haswell`` an EPYC 9534 segfaults on ``dgemm`` from n = 64.
"""

import hashlib
import os
import sys

import numpy as np

op, n = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(0)
A = rng.standard_normal((n, 768)).astype(np.float32)
B = rng.standard_normal((768, n)).astype(np.float32)
ops = {
    "sgemm": lambda: A @ B,
    "ssyrk": lambda: A @ A.T,
    "sgemv": lambda: A @ B[:, 0].copy(),
    "dgemm": lambda: A.astype(np.float64) @ B.astype(np.float64),
    "sdot": lambda: np.asarray(np.dot(A[0], B[:, 0])),
}
out = np.ascontiguousarray(ops[op]())
digest = hashlib.sha1(out.tobytes()).hexdigest()[:12]
node = os.uname().nodename.split(".")[0]
print(f"{node} numpy={np.__version__} OPENBLAS_CORETYPE={os.environ.get('OPENBLAS_CORETYPE', '-')} {op} n={n} {digest}")
