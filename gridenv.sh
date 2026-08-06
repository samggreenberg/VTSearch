# Source to run experiment code from THIS worktree (shadows main venv editable finder)
module load python/3.12.3
source /exp/sgreenberg/projects/VTSearch/.venv/bin/activate
export PYTHONPATH="/exp/sgreenberg/projects/vts-rate-2861/.shadow:/exp/sgreenberg/projects/vts-rate-2861:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
