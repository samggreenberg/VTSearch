# Source to run experiment code from THIS worktree (shadows main venv editable finder).
# Self-locating: the path is derived from this file, not hardcoded to whichever
# worktree happened to write it last.  A stale absolute path here silently puts
# ANOTHER worktree's checkout on PYTHONPATH, which is how a run measures code you
# did not commit.
module load python/3.12.3
source /exp/sgreenberg/projects/VTSearch/.venv/bin/activate
_VTS_WT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The no-op module that neutralises the venv's editable-install finder.  It is
# untracked, so a *fresh* worktree does not have one -- and without it the finder
# still resolves `import vtscore` to the MAIN checkout while PYTHONPATH looks
# correct.  Create it here rather than writing it down: the failure is silent and
# the remedy is two lines (#2846).
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
