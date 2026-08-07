# Source to run experiment code from THIS worktree (shadows main venv editable finder).
# Self-locating: the path is derived from this file, not hardcoded to whichever
# worktree happened to write it last.  A stale absolute path here silently puts
# ANOTHER worktree's checkout on PYTHONPATH, which is how a run measures code you
# did not commit.
module load python/3.12.3
source /exp/sgreenberg/projects/VTSearch/.venv/bin/activate
_VTS_WT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_VTS_WT}/.shadow:${_VTS_WT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
