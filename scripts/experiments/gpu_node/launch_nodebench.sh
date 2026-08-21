#!/usr/bin/env bash
# #3160, part 3: does the 1.5e-04 *node* drift move the shipped decision metrics?
#
#   bash launch_nodebench.sh prepare        # per-arm category selection + exemplars
#   bash launch_nodebench.sh verify-pairing # MUST pass before cells
#   bash launch_nodebench.sh size <arm> 0   # time ONE real cell
#   bash launch_nodebench.sh cells
#   bash launch_nodebench.sh status | analyze
#
# This is #3143's bench harness with one substitution: the axis under test is no
# longer precision but **which physical V100 built the gallery**.  Both arms are
# fp32, same commit, same images, same 4193 medias -- they differ only in the
# device that ran the forward:
#
#   fp32_v100_rack7n03  Tesla V100S-PCIE-32GB  -- bit-identical to the published pile
#   fp32_v100           Tesla V100-SXM2-32GB-LS -- 1.5e-04 median 1-cos on siglip2_l
#
# Both piles already exist, built by #3143 (job 507728 and 507430).  Nothing is
# rebuilt here: a rebuild would land on whichever V100 the scheduler picked,
# which is precisely the defect under study.
#
# `siglip` rides along as a **placebo arm**, and that is the point of including
# it: the same two nodes agree to 7.6e-13 on it, so its paired difference has no
# cause to be non-zero.  If siglip moves as much as siglip2_l, this bench is
# measuring trajectory chaos rather than the drift, and no verdict is available
# from it.
set -euo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

# The piles live in #3143's study dir (read-only for us); the results do NOT --
# a different grid is a different study (grid-experiments skill).
export VTS_PRECISION_STUDY="${VTS_PRECISION_STUDY:-/expscratch/$USER/precision-3143}"
export VTS_BENCH_ROOT="${VTS_BENCH_ROOT:-/expscratch/$USER/gpu-node-3160/bench}"
export VTS_BENCH_ARMS="${VTS_BENCH_ARMS:-fp32_v100_rack7n03,fp32_v100}"

# 128 seeds, not #3143's 64.  That run resolved the 0.005 margin *pooled* but not
# per embedder (per-embedder SE 0.0033, so 2*SE = 0.0066 > margin), and here the
# per-embedder split IS the design -- one treated embedder, one placebo.  128
# seeds doubles the cells per embedder and puts 2*SE just inside the margin.
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-128}"

# Measured on #3143's power array: 63 s and 0.7 GB peak per cell.  16G (that
# study's first pass) was 30x oversized, and memory is the binding per-user quota.
export CALIB_MEM="${CALIB_MEM:-4G}"
export CALIB_CPUS="${CALIB_CPUS:-4}"
export CALIB_CONC="${CALIB_CONC:-24}"

exec bash "$WT/scripts/experiments/precision/launch_bench.sh" "$@"
