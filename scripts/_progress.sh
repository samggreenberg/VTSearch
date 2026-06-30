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
    _VTS_RED="$(printf '\033[31m')"
    _VTS_DIM="$(printf '\033[2m')"
    _VTS_RESET="$(printf '\033[0m')"
else
    _VTS_BOLD=""
    _VTS_CYAN=""
    _VTS_GREEN=""
    _VTS_RED=""
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

# --- live "heartbeat" command runner -----------------------------------------
# Long install steps (dnf metadata refresh, kernel-module compile, multi-GB pip
# downloads, the GPU smoke-test import) can sit silent for 10s-several minutes,
# which looks frozen. And the driver install is a try-until-one-works cascade
# whose doomed early attempts dump scary-but-recoverable dnf/pip error walls
# (the "Problem 1..6 / nothing provides dkms" tree, "filtered out by modular
# filtering", the red "pip's dependency resolver" report, the unregistered-RHEL
# subscription-manager noise). Users read those as the installer crashing.
#
# vts_run / vts_try fix both at once: they run a command with its combined
# output captured to a temp log while an animated heartbeat (elapsed seconds)
# shows the step is alive, then print a single ✓/✗ verdict line. The noisy
# output stays in the log instead of scrolling past as a wall.
#
#   vts_run "Label" cmd args...   # on failure: print ✗ and DUMP the log
#   vts_try "Label" cmd args...   # on failure: print a dim "trying another
#                                 #   approach" and stay quiet -- for a
#                                 #   best-effort step in a cascade whose
#                                 #   failure is expected and handled by the
#                                 #   caller's next fallback.
#
# Both return the command's exit status and leave the log path in $VTS_LAST_LOG.
# Set VTSEARCH_VERBOSE=1 to stream every command's raw output live instead (no
# capture, no spinner) -- the escape hatch for debugging a genuinely stuck run.
#
# Pass the command as separate args (NO shell operators / pipes / redirects).
# For an optional sudo prefix, use an UNQUOTED variable so an empty value (the
# already-root case) vanishes instead of becoming a literal empty argument:
#   vts_try "Installing X" ${sudo_cmd} "$dnf" install -y x
#
# Used as a bare statement under `set -e`, a failing vts_try aborts the script;
# append `|| true` when the failure is meant to be tolerated outside an `if`.

VTS_LAST_LOG=""

_vts_run_ok()   { printf '%s✓%s %s\n' "$_VTS_GREEN" "$_VTS_RESET" "$1"; }
_vts_run_fail() { printf '%s✗%s %s\n' "$_VTS_RED" "$_VTS_RESET" "$1"; }
_vts_run_soft() {
    printf '  %s↻ %s — not available here; trying another approach…%s\n' \
        "$_VTS_DIM" "$1" "$_VTS_RESET"
}

# Echo a captured log, indented and fenced, only if it has content.
vts_dump_log() {
    local log="${1:-$VTS_LAST_LOG}"
    [ -n "$log" ] && [ -s "$log" ] || return 0
    printf '%s    ---- captured output (set VTSEARCH_VERBOSE=1 to see this live) ----%s\n' \
        "$_VTS_DIM" "$_VTS_RESET"
    sed 's/^/    /' "$log"
    printf '%s    ----------------------------------------------------------------%s\n' \
        "$_VTS_DIM" "$_VTS_RESET"
}

# Animate a heartbeat next to $2 until background pid $1 exits. TTY: an in-place
# spinner with elapsed seconds. Non-TTY (CI, `curl ... | bash`): a dot every few
# seconds so piped logs still show the step is alive.
_vts_spin() {
    local pid="$1" label="$2"
    local frames='|/-\' i=0 start now el
    start=$(date +%s)
    if [ -t 1 ]; then
        while kill -0 "$pid" 2>/dev/null; do
            i=$(((i + 1) % 4))
            now=$(date +%s)
            el=$((now - start))
            printf '\r%s%s%s %s %s(%ds)%s\033[K' \
                "$_VTS_CYAN" "${frames:$i:1}" "$_VTS_RESET" \
                "$label" "$_VTS_DIM" "$el" "$_VTS_RESET"
            sleep 0.2
        done
        printf '\r\033[K'
    else
        printf '   %s … ' "$label"
        while kill -0 "$pid" 2>/dev/null; do
            printf '.'
            sleep 5
        done
        printf '\n'
    fi
}

_vts_mktemp() { mktemp 2>/dev/null || printf '/tmp/vts_run.%s.%s' "$$" "${RANDOM:-0}"; }

# mode: hard (print ✗ and dump the log on failure) | soft (stay quiet on failure,
# for a best-effort cascade step the caller will fall back from).
_vts_run_impl() {
    local mode="$1"; shift
    local label="$1"; shift
    local rc=0

    if [ "${VTSEARCH_VERBOSE:-0}" = "1" ]; then
        VTS_LAST_LOG="/dev/null"
        printf '%s ▸ %s%s\n' "$_VTS_DIM" "$label" "$_VTS_RESET"
        if "$@"; then rc=0; else rc=$?; fi
        if [ "$rc" -eq 0 ]; then _vts_run_ok "$label"
        elif [ "$mode" = soft ]; then _vts_run_soft "$label"
        else _vts_run_fail "$label"; fi
        return "$rc"
    fi

    local log
    log="$(_vts_mktemp)"
    VTS_LAST_LOG="$log"
    # stdin from /dev/null so a backgrounded command never blocks invisibly on a
    # prompt (e.g. an un-warmed sudo); it fails fast and we dump/handle the log.
    "$@" >"$log" 2>&1 </dev/null &
    local pid=$!
    _vts_spin "$pid" "$label"
    wait "$pid" || rc=$?

    if [ "$rc" -eq 0 ]; then
        _vts_run_ok "$label"
    elif [ "$mode" = soft ]; then
        _vts_run_soft "$label"
    else
        _vts_run_fail "$label"
        vts_dump_log "$log"
    fi
    # Keep the log only for a hard failure (already dumped, but handy for VERBOSE
    # re-runs); drop it otherwise. Guarded so `set -e` can't fire on the test.
    if [ "$rc" -eq 0 ] || [ "$mode" = soft ]; then
        rm -f "$log" 2>/dev/null || true
    fi
    return "$rc"
}

vts_run() { local l="$1"; shift; _vts_run_impl hard "$l" "$@"; }
vts_try() { local l="$1"; shift; _vts_run_impl soft "$l" "$@"; }

# --- sudo credential keep-alive ----------------------------------------------
# vts_run backgrounds commands with stdin from /dev/null, so a sudo step that
# needs a password would fail fast rather than prompt. Warm the sudo timestamp
# up front (one foreground prompt) and refresh it in the background so the whole
# driver install runs without a mid-step password stall. $1 = sudo prefix (""
# => already root, nothing to do). Returns non-zero if creds can't be cached
# (no TTY and no NOPASSWD); the caller proceeds and backgrounded sudo fails fast.
VTS_SUDO_KEEPALIVE_PID=""
vts_sudo_keepalive() {
    local sudo_cmd="$1"
    [ -n "$sudo_cmd" ] || return 0
    $sudo_cmd -v || return 1
    ( while true; do
          sleep 50
          $sudo_cmd -n -v >/dev/null 2>&1 || exit 0
      done ) &
    VTS_SUDO_KEEPALIVE_PID=$!
    return 0
}
vts_sudo_keepalive_stop() {
    [ -n "$VTS_SUDO_KEEPALIVE_PID" ] && kill "$VTS_SUDO_KEEPALIVE_PID" 2>/dev/null
    VTS_SUDO_KEEPALIVE_PID=""
    return 0
}
