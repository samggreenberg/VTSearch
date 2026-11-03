# Source to run experiment code from THIS worktree (shadows main venv editable finder).
#
#   source scripts/grid/gridenv.sh        # from the worktree root
#   source "$WT/scripts/grid/gridenv.sh"  # from anywhere, as the launchers do
#
# Self-locating: the *worktree* path is derived from this file (two levels up),
# not hardcoded to whichever worktree happened to write it last.  A stale
# absolute path there silently puts ANOTHER worktree's checkout on PYTHONPATH,
# which is how a run measures code you did not commit.
#
# The venv and the module are a different thing: they are shared machine state,
# not per-worktree, so they are named rather than derived.  Both are overridable
# for a different user, machine, or venv; the defaults are the GRID's current
# ones.  Set VTS_PYTHON_MODULE= (empty) to skip `module load` on a host without
# environment modules, and VTS_VENV= (empty) to skip venv activation entirely
# (e.g. when the caller has already activated one).
if [ -n "${VTS_PYTHON_MODULE-python/3.12.3}" ]; then
  module load "${VTS_PYTHON_MODULE:-python/3.12.3}"
fi
if [ -n "${VTS_VENV-/exp/sgreenberg/projects/VTSearch/.venv}" ]; then
  source "${VTS_VENV:-/exp/sgreenberg/projects/VTSearch/.venv}/bin/activate"
fi
_VTS_WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The no-op module that neutralises the venv's editable-install finder.  It is
# untracked, so a *fresh* worktree does not have one -- and without it the finder
# still resolves `import vtscore` to the MAIN checkout while PYTHONPATH looks
# correct.  Create it here rather than writing it down: the failure is silent and
# the remedy is two lines (#2846).  `.shadow/` is gitignored on purpose: it was
# once committed, which made this block unreachable and left the prevention
# resting on git tracking that nobody had written down (#3437).
if [ ! -f "${_VTS_WT}/.shadow/__editable___vtsearch_0_1_0_finder.py" ]; then
  mkdir -p "${_VTS_WT}/.shadow"
  printf 'def install():\n    pass\n' > "${_VTS_WT}/.shadow/__editable___vtsearch_0_1_0_finder.py"
fi
export PYTHONPATH="${_VTS_WT}/.shadow:${_VTS_WT}:${PYTHONPATH:-}"

# `common.setup_env()` inserts VTS_REPO at sys.path[0] and *defaults it to the
# shared vts-calib worktree*, which lands ahead of PYTHONPATH -- the shadow shim
# cannot save you from that.  Pin it to this worktree unless the caller means
# otherwise; `preflight.sh` checks the value.
export VTS_REPO="${VTS_REPO:-$_VTS_WT}"
export TOKENIZERS_PARALLELISM=false
# One set of OpenBLAS kernels on every node (#4481): the AMD EPYC nodes otherwise get
# AVX-512 kernels and the Xeons AVX2, whose float32 rounding differs enough to move
# document VLAD/tile vectors, and with them closed-loop sessions, between node types.
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"
# OpenBLAS 0.3.34 (numpy 2.5.2 on) re-tunes GEMM blocking on a Zen 4 CPU (the L40S
# nodes' EPYC 9534) even under a forced core (#4938): Haswell float64 GEMM and any
# SVD then segfault, and float32 GEMM sums in another order. It fires only on a 1 MB
# L2, which the Haswell table reads nowhere else, so 2 MB keeps its blocking. An
# exported value, an empty one included, wins.
if [ "${OPENBLAS_CORETYPE,,}" = haswell ]; then
  export OPENBLAS_L2_SIZE="${OPENBLAS_L2_SIZE-2048}"
fi
# And one set of numpy kernels (#4110): numpy dispatches float64 exp/log and argsort
# to AVX-512 code where the CPU has it (Xeon Silver 4114 and Gold 5218R, EPYC 9534)
# and not where it does not (EPYC 7713, Xeon E5-2698 v4). The two round exp/log
# differently and order ties differently, which moved the anchored EM's threshold by
# 1e-6 and, votes later, a pick. Turning those targets off costs that EM ~40% on an
# AVX-512 node and nothing on the others. x86-64 only: numpy warns about a feature
# its build cannot dispatch. The cell runner pins MKL and ATen as well
# (scripts/experiments/_expcommon.py `cpu_kernel_pins`); they cost a CPU GEMM 2-3x,
# so they are not set here for every job.
# The value names numpy's dispatch targets, which numpy 2.4 renamed (#4930), and
# numpy skips a name its build lacks with a warning Python hides, so the old value
# disables nothing on numpy 2.4. It comes from the numpy this python has, chosen by
# the cell runner's own function. An exported value wins, an empty one included.
if [ "$(uname -m)" = x86_64 ] && [ -z "${NPY_DISABLE_CPU_FEATURES+set}" ]; then
  if _vts_npy_pin="$(python -c 'import sys; sys.path.insert(0, sys.argv[1]); import _expcommon; print(_expcommon.numpy_kernel_pin())' \
    "${_VTS_WT}/scripts/experiments" 2>/dev/null)"; then
    export NPY_DISABLE_CPU_FEATURES="$_vts_npy_pin"
  else
    echo "gridenv.sh: could not read numpy's version, so numpy is not pinned and runs AVX-512 kernels where the CPU has them (#4930)" >&2
  fi
  unset _vts_npy_pin
fi
