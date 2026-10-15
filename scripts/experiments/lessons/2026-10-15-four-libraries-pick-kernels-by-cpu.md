# 2026-10-15 — Four libraries pick kernels by CPU, so a cell differed by node (#4110)

**What happened.** #3840 ran the same arm twice on one commit, and 3 of 114 cells differed in cost; 36 of
114 differed somewhere, by one unit in a threshold's last printed digit. Every one of the 36 had run on an AMD
node in one run and an Intel node in the other. None of the 60 pairs that ran on the same vendor twice differed
at all.

#4481 had already pinned OpenBLAS (`OPENBLAS_CORETYPE=Haswell`), but three more libraries on a cell's path
pick their kernels by CPU, and the GRID mixes AVX-512 CPUs (Xeon Silver 4114 on the cpu partition, Gold 5218R
and EPYC 9534 on GPU nodes) with AVX2-only ones (EPYC 7713, Xeon E5-2698 v4):

- **numpy** runs float64 `exp`/`log` (the anchored EM) and `argsort` (tie order) on AVX-512 code where it can;
- **torch's bundled MKL** picks sgemv/sgemm/VML kernels per CPU (the head's scores, the text tower);
- **torch's ATen** runs the text tower's LayerNorm, softmax and GELU on AVX-512 or AVX2 kernels. A cell embeds
  its typed query live, so this reaches the opening's sort.

Pinning one or two of the three left every cell still different (the text query's vector alone moved). With
all three, 72 of 72 cells were bit-identical on an Intel and an AMD node
(`docs/experiments/2026-10-15-node-determinism-4110/`).

**Cost.** An exactness check that only held when the arms happened to share a vendor: #3839 read a paired Δ of
exactly zero as "the harness is the same". #4496 spent half an hour chasing a "bug" that was the node, and
#4910's opening picks moved between two V100 node types, which the numpy half of this likely explains. The
drift was mean-zero each time, so no verdict was wrong, but no exactness check on a mixed-node run could pass.

**The rule.** A cell run by `run_cells.py` is now bit-identical on both cpu-partition CPU models (tested end to
end), and every op probed agrees on the GPU nodes' three CPU models too. The runner defaults four pins
before numpy and torch load (`_expcommon.CPU_KERNEL_PINS`: `OPENBLAS_CORETYPE=Haswell`,
`NPY_DISABLE_CPU_FEATURES="AVX512F AVX512_SKX"`, `MKL_CBWR=COMPATIBLE`, `ATEN_CPU_CAPABILITY=avx2`), and logs
the CPU and pins of every cell (`kernels: ...`). `gridenv.sh` sets the numpy pin for every GRID job; the MKL and
ATen pins make CPU GEMMs 2–3x slower, so they stay in the cell runner. `VTS_KERNEL_PINS=0` runs a cell as it
ran before.

- There is no vendor to pin any more. The 2026-10-04 advice to pin `Features=intel` is obsolete for cells.
- **A grid from before the pins is a different baseline.** The pins move a cell's last bits on every node,
  AMD included, so re-run a control rather than pair a pinned arm with an unpinned one.
- The pins do not reach a GPU: cuBLAS differs by device (#3160).

**Status.** *Prevented* for `run_cells.py` cells: the runner sets the pins itself, and
`tests_lib/meta/test_cpu_kernel_pins.py` checks that it does so before numpy loads, that `gridenv.sh` agrees,
and that the installed numpy still accepts the pin without a warning. *Advice only* for any other runner.
