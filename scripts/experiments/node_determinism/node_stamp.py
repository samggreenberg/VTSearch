"""Print one TSV line naming the node a task ran on and the numeric kernels it resolved (#4110).

Columns: job, array task, host, CPU model, whether the CPU has AVX-512, numpy's
enabled AVX-512 dispatch targets, torch's ATen CPU capability, and the pins
(OPENBLAS_CORETYPE, NPY_DISABLE_CPU_FEATURES, MKL_CBWR) as the process saw them.
A cross-node pair is read by these columns, not by node name: one partition
holds more than one CPU model.
"""

from __future__ import annotations

import os


def _cpu() -> tuple[str, bool]:
    model, avx512 = "?", False
    with open("/proc/cpuinfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("model name") and model == "?":
                model = line.split(":", 1)[1].strip()
            elif line.startswith("flags"):
                avx512 = "avx512f" in line.split()
                break
    return model, avx512


def main() -> None:
    import numpy as np  # noqa: PLC0415  # after the env is read, as in a cell
    import torch  # noqa: PLC0415

    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats  # noqa: PLC0415
    except ImportError:  # numpy < 2
        from numpy.core._multiarray_umath import __cpu_features__ as feats  # noqa: PLC0415
    model, avx512 = _cpu()
    row = [
        os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "")),
        os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        os.uname().nodename.split(".")[0],
        model,
        str(avx512),
        ",".join(sorted(k for k, v in feats.items() if v and k.startswith("AVX512"))) or "-",
        torch.backends.cpu.get_cpu_capability(),
        os.environ.get("OPENBLAS_CORETYPE", "-"),
        os.environ.get("NPY_DISABLE_CPU_FEATURES", "-"),
        os.environ.get("MKL_CBWR", "-"),
        np.__version__,
    ]
    print("\t".join(row))


if __name__ == "__main__":
    main()
