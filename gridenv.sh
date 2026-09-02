# Source to run experiment code from THIS worktree (shadows main venv editable finder).
# Self-locating: the *worktree* path is derived from this file, not hardcoded to
# whichever worktree happened to write it last.  A stale absolute path there
# silently puts ANOTHER worktree's checkout on PYTHONPATH, which is how a run
# measures code you did not commit.
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
_VTS_WT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
