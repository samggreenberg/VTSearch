# Sourced helper: structured step/progress output for install scripts.
#
# Usage:
#   source "$SCRIPT_DIR/_progress.sh"
#   vts_progress_init <total_steps> "Banner title"
#   vts_progress_step "Doing the thing"
#   ...work...
#   vts_progress_step "Doing the next thing"
#   ...work...
#   vts_progress_done "All done"
#
# Each step prints a header like:
#   [2/4] (0:00:12 elapsed) Doing the next thing
# followed by an underline.  Colors are emitted only when stdout is a TTY.

if [ -t 1 ]; then
    _VTS_BOLD="$(printf '\033[1m')"
    _VTS_CYAN="$(printf '\033[36m')"
    _VTS_GREEN="$(printf '\033[32m')"
    _VTS_DIM="$(printf '\033[2m')"
    _VTS_RESET="$(printf '\033[0m')"
else
    _VTS_BOLD=""
    _VTS_CYAN=""
    _VTS_GREEN=""
    _VTS_DIM=""
    _VTS_RESET=""
fi

_VTS_TOTAL=0
_VTS_CURRENT=0
_VTS_START=0

_vts_elapsed() {
    local now elapsed h m s
    now=$(date +%s)
    elapsed=$((now - _VTS_START))
    h=$((elapsed / 3600))
    m=$(((elapsed % 3600) / 60))
    s=$((elapsed % 60))
    printf '%d:%02d:%02d' "$h" "$m" "$s"
}

vts_progress_init() {
    _VTS_TOTAL="$1"
    _VTS_CURRENT=0
    _VTS_START=$(date +%s)
    local title="$2"
    local bar
    bar=$(printf '%.0s=' $(seq 1 ${#title}))
    printf '\n%s%s%s%s\n' "$_VTS_BOLD" "$_VTS_CYAN" "$title" "$_VTS_RESET"
    printf '%s%s%s\n\n' "$_VTS_CYAN" "$bar" "$_VTS_RESET"
}

vts_progress_step() {
    _VTS_CURRENT=$((_VTS_CURRENT + 1))
    local msg="$1"
    local header
    header="[${_VTS_CURRENT}/${_VTS_TOTAL}]"
    printf '\n%s%s%s %s(%s elapsed)%s %s%s%s\n' \
        "$_VTS_BOLD" "$header" "$_VTS_RESET" \
        "$_VTS_DIM" "$(_vts_elapsed)" "$_VTS_RESET" \
        "$_VTS_BOLD" "$msg" "$_VTS_RESET"
}

vts_progress_done() {
    local msg="$1"
    printf '\n%s%s✓ %s (total: %s)%s\n\n' \
        "$_VTS_BOLD" "$_VTS_GREEN" "$msg" "$(_vts_elapsed)" "$_VTS_RESET"
}
