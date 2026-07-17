#!/bin/bash
set -euo pipefail

# Unified dependency installer for VTSearch.
#
# ONE script for both CPU and GPU installs. With no argument it AUTO-DETECTS
# whether this host has a usable NVIDIA GPU (via nvidia-smi) and installs the
# matching dependency set, so you don't have to know -- or tell it -- what
# hardware you have:
#
#   - GPU present -> CUDA torch, with the right cuXYZ wheel tag auto-selected
#     from the GPU's compute capability (scripts/detect_cuda_tag.py).
#   - No GPU      -> the smaller CPU-only torch wheel (~200 MB vs ~2 GB).
#
# "GPU present" means nvidia-smi can actually see the card. If the GPU is
# physically present (an NVIDIA PCI device) but nvidia-smi can't -- the usual
# state on a fresh cloud GPU box like an AWS g4dn whose NVIDIA driver isn't
# installed yet -- auto mode OFFERS TO INSTALL THE DRIVER FOR YOU (a distro-aware,
# best-effort sudo install; default choice at the prompt; falls back to NVIDIA's
# self-contained .run installer when the distro's package path dead-ends, e.g. on
# a bare unregistered RHEL g4dn), or to fall back to a CPU-only install, or to
# stop. On a non-interactive shell it stops unless told
# otherwise: VTSEARCH_AUTO_DRIVER=1 auto-installs the driver, VTSEARCH_ASSUME_CPU=1
# proceeds with CPU. An explicit 'cpu'/'gpu'/'cuXYZ' argument skips the check.
#
# PERSISTENCE (set-and-forget): a driver installed here is registered with DKMS
# whenever dkms is available (the module then AUTO-REBUILDS on future kernel
# upgrades instead of silently breaking on the next reboot), NVIDIA's persistence
# daemon is enabled so the driver initializes at every boot, and every GPU install
# reports whether the result is kernel-update-proof or still kernel-pinned. If the
# driver is already up but NOT DKMS-managed, the installer OFFERS to convert it in
# place (reinstall via .run --dkms); VTSEARCH_AUTO_DKMS=1 converts unattended and
# VTSEARCH_SKIP_DKMS=1 skips the offer. See docs/DEPLOYMENT.md, "Making the GPU
# driver survive reboots and kernel updates".
#
# Override the auto-detection by passing an argument:
#   bash scripts/install.sh            # auto-detect CPU vs GPU (recommended)
#   bash scripts/install.sh cpu        # force the CPU install
#   bash scripts/install.sh gpu        # force the GPU install (auto-detect CUDA tag)
#   bash scripts/install.sh cu118      # force the GPU install with an explicit CUDA tag
#   bash scripts/install.sh cu121      # ...
#   bash scripts/install.sh cu124      # for CUDA 12.4 (V100/Volta, A100, H100)
#   bash scripts/install.sh cu128      # for CUDA 12.8 (Blackwell; drops Volta)
#
# About the CUDA tag (only relevant to GPU installs): it selects which prebuilt
# torch wheel you get, and each wheel only ships kernel images for a fixed set
# of GPU architectures. There is a floor AND a ceiling. Newer GPUs need newer
# tags: Ampere/Ada work on cu118+, Hopper (H100) on cu121+, Blackwell (B100/B200,
# RTX 50xx) on cu128+. But the newest wheels also DROP the oldest architectures,
# so "just use the latest tag" is wrong for old hardware: cu128 dropped Volta
# (sm_70), so a Tesla V100 needs cu124 (or cu121/cu118), NOT cu128. Rule of
# thumb: use the oldest tag your driver supports that still covers your GPU.
# cu124 is a safe default that spans Volta through Hopper, and what the
# auto-detect picks for those cards.
#
# (VTSearch also smoke-tests CUDA at runtime and falls back to CPU if the
# installed wheel can't run on the GPU, so a mismatch degrades instead of
# crashing - but you only get GPU acceleration with a matching wheel.)
#
# OUTPUT / QUIET MODE: the driver install is a try-until-one-works cascade, and
# its doomed early attempts (on a bare cloud GPU box) print scary-but-recoverable
# dnf/pip error walls. By default those attempts run under a live heartbeat with
# their output captured to a log, so you see "trying X… not available here, trying
# the next approach" instead of a wall of red, and the raw output only surfaces if
# a step actually fails. Set VTSEARCH_VERBOSE=1 to stream every command's raw
# output live (no capture, no spinner) when debugging a genuinely stuck install.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Shared progress-printing helpers.
# shellcheck source=_progress.sh
source "$SCRIPT_DIR/_progress.sh"

# --- GPU presence ------------------------------------------------------------
# True (exit 0) when nvidia-smi exists and lists at least one GPU. This is the
# CPU-vs-GPU decision; the finer cuXYZ tag choice is detect_cuda_tag.py's job.
vts_have_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi -L >/dev/null 2>&1 || return 1
    [ -n "$(nvidia-smi -L 2>/dev/null)" ]
}

# True (exit 0) when an NVIDIA GPU is PHYSICALLY present, regardless of whether
# the driver is installed. This is the key distinction nvidia-smi can't make:
# on a fresh cloud GPU instance (e.g. an AWS g4dn with a Tesla T4) the card is
# in the box but the kernel driver isn't installed yet, so nvidia-smi is absent
# and vts_have_gpu() above returns false -- which would silently install the
# CPU wheels even though the user wants GPU. We detect the hardware itself via
# the PCI vendor ID 0x10de (NVIDIA) on a display/3D controller (class 0x03xxxx).
# sysfs is read directly because it's always present on Linux, needs no driver,
# and needs no lspci; lspci is only a last-resort fallback when sysfs is empty.
vts_nvidia_hardware_present() {
    local dev vendor class
    for dev in /sys/bus/pci/devices/*; do
        [ -r "$dev/vendor" ] && [ -r "$dev/class" ] || continue
        read -r vendor < "$dev/vendor" || continue
        [ "$vendor" = "0x10de" ] || continue
        read -r class < "$dev/class" || continue
        case "$class" in 0x03*) return 0 ;; esac  # 0x03xxxx = display controller
    done
    # Fallback for non-Linux / unusual sysfs layouts: ask lspci if it exists.
    if command -v lspci >/dev/null 2>&1; then
        lspci 2>/dev/null | grep -iE 'NVIDIA.*(VGA|3D|Display)' >/dev/null 2>&1 && return 0
    fi
    return 1
}

# Print the situation when the GPU is physically present but unusable because
# the driver is missing (nvidia-smi absent / not listing a device).
vts_report_driver_missing() {
    cat >&2 <<'EOF'
NOTICE: An NVIDIA GPU is physically present, but no usable driver was found
        (nvidia-smi is absent or lists no device), so the GPU cannot be used yet.

This is the common state on a fresh cloud GPU instance (e.g. an AWS g4dn with a
Tesla T4) booted from a base AMI: the card is attached but the NVIDIA kernel
driver isn't installed. pip cannot fix this -- the driver is a system package.
EOF
}

# Drop NVIDIA's CUDA .repo into /etc/yum.repos.d so `cuda-drivers` becomes
# installable. A base RHEL/Fedora/Amazon-Linux AMI does NOT ship this repo, so
# a plain `dnf install cuda-drivers` fails with "No match for argument:
# cuda-drivers" until it's enabled (the common state on a fresh RHEL GPU box).
# The repo slug is keyed to the distro + major version (rhel9, fedora41,
# amzn2023, ...) and the CPU arch (x86_64, or sbsa on arm64). Idempotent:
# re-fetching the .repo just overwrites an identical file.
#   $1 = sudo prefix ("" or "sudo");  $2 = the dnf/yum command name.
vts_enable_nvidia_cuda_repo() {
    local sudo_cmd="$1" dnf="$2"
    local id="" version_id="" id_like=""
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        id="$(. /etc/os-release && echo "${ID:-}")"
        # shellcheck disable=SC1091
        version_id="$(. /etc/os-release && echo "${VERSION_ID:-}")"
        # shellcheck disable=SC1091
        id_like="$(. /etc/os-release && echo "${ID_LIKE:-}")"
    fi
    local major="${version_id%%.*}"

    # Map distro -> NVIDIA repo slug. Amazon Linux uses the full VERSION_ID
    # (amzn2023, amzn2); the RHEL family uses just the major (rhel8, rhel9).
    local slug=""
    case "$id" in
        rhel | centos | rocky | almalinux) slug="rhel${major}" ;;
        fedora) slug="fedora${major}" ;;
        amzn) slug="amzn${version_id}" ;;
        *)
            # Unrecognized ID: fall back to the RHEL slug for RHEL-likes.
            case " $id_like " in
                *rhel* | *centos* | *fedora*) slug="rhel${major}" ;;
            esac
            ;;
    esac
    if [ -z "$slug" ] || [ -z "$major" ]; then
        echo "  Could not determine the NVIDIA CUDA repo for '${id:-unknown} ${version_id:-?}'." >&2
        return 1
    fi

    local arch
    arch="$(uname -m)"
    case "$arch" in aarch64) arch="sbsa" ;; esac

    local url="https://developer.download.nvidia.com/compute/cuda/repos/${slug}/${arch}/cuda-${slug}.repo"
    echo "  Enabling NVIDIA's CUDA repo for ${slug} (${arch}):" >&2
    echo "    $url" >&2

    # `dnf config-manager` is the canonical tool, but its syntax differs between
    # dnf4 (--add-repo) and dnf5 (addrepo --from-repofile), so just place the
    # .repo file directly -- tool-agnostic and works under yum too.
    local repo_file="/etc/yum.repos.d/cuda-${slug}.repo"
    if command -v curl >/dev/null 2>&1; then
        vts_run "Adding NVIDIA's CUDA repo (${slug})" \
            ${sudo_cmd} curl -fsSL -o "$repo_file" "$url" \
            || { echo "  Failed to download the CUDA .repo file (network/proxy/404?)." >&2; return 1; }
    elif command -v wget >/dev/null 2>&1; then
        vts_run "Adding NVIDIA's CUDA repo (${slug})" \
            ${sudo_cmd} wget -qO "$repo_file" "$url" \
            || { echo "  Failed to download the CUDA .repo file (network/proxy/404?)." >&2; return 1; }
    else
        echo "  Neither curl nor wget is available to fetch the CUDA .repo file." >&2
        return 1
    fi

    # Refresh metadata so the newly added repo is visible to the installer.
    vts_try "Refreshing package metadata" ${sudo_cmd} "$dnf" makecache || true
    return 0
}

# Install the NVIDIA driver packages on the RHEL/Fedora/Amazon-Linux family,
# coping with the two ways `cuda-drivers` can be unavailable:
#
#   1. The plain meta-package. On Fedora, Amazon Linux, and RHEL once the CUDA
#      repo is enabled and NOT modular, `dnf install cuda-drivers` just works.
#   2. The DNF *module*. On RHEL 8/9 (and Rocky/Alma/CentOS Stream) NVIDIA's CUDA
#      repo ships the driver as a module stream, so a bare `dnf install
#      cuda-drivers` is REJECTED with "All matches were filtered out by modular
#      filtering for argument: cuda-drivers" -- the package exists but is hidden
#      behind a module that must be enabled first. This is the failure on a fresh
#      RHEL 9 GPU box (e.g. an AWS g4dn) even after the repo is enabled. The fix
#      is `dnf module install nvidia-driver:<stream>`. We try two flavors of
#      stream in order:
#        a. DKMS streams (latest-dkms, open-dkms): build the module from source,
#           so they're kernel-agnostic -- but they require the `dkms` package,
#           which on the RHEL family lives in EPEL, not the base repos. If dkms is
#           missing the stream is REJECTED with "nothing provides dkms >= 3.1.8
#           needed by kmod-nvidia-latest-dkms" -- the exact failure on a fresh,
#           UNregistered RHEL 9 box where `dnf install epel-release` finds nothing
#           (epel-release isn't in any enabled repo there). So we make sure dkms
#           is actually installed first (vts_rhel_ensure_dkms, which bootstraps
#           EPEL from its canonical URL when the packaged release isn't found) and
#           skip the DKMS streams entirely if it still can't be had.
#        b. Precompiled kABI-tracking streams (latest, open): ship a prebuilt
#           module (kmod-nvidia-latest) that needs NO dkms, only a kernel whose
#           kABI matches. They're the fallback for boxes where EPEL/dkms can't be
#           reached at all. They need a matching kernel, so they're tried last.
#      latest*/open* order within each flavor: proprietary first (covers
#      Turing/Ampere/Ada like the T4), open as the fallback for newer datacenter
#      GPUs (Hopper, Blackwell) that ONLY have an open kernel module.
#   $1 = sudo prefix ("" or "sudo");  $2 = the dnf/yum command name.
vts_rhel_install_driver_pkgs() {
    local sudo_cmd="$1" dnf="$2"
    if vts_try "Installing NVIDIA driver (cuda-drivers package)" \
        ${sudo_cmd} "$dnf" install -y cuda-drivers; then
        return 0
    fi
    # `dnf module` doesn't exist under plain yum (RHEL 7), but there cuda-drivers
    # is a plain package and the line above already succeeded, so we only reach
    # here on dnf-based RHEL 8/9 where the modular path is the right one.
    echo "  (cuda-drivers is a dnf module here; using a driver module stream instead.)" >&2

    # The -dkms streams need `dkms` (from EPEL). Make it present before trying
    # them; if it can't be installed, skip straight to the precompiled streams
    # rather than letting each -dkms stream fail with "nothing provides dkms".
    local streams
    if vts_rhel_ensure_dkms "$sudo_cmd" "$dnf"; then
        streams="latest-dkms open-dkms latest open"
    else
        echo "  (dkms unavailable; trying the precompiled kABI-tracking streams.)" >&2
        streams="latest open"
    fi

    local stream
    for stream in $streams; do
        $sudo_cmd "$dnf" module reset -y nvidia-driver >/dev/null 2>&1 || true
        if vts_try "Installing NVIDIA driver module stream: ${stream}" \
            ${sudo_cmd} "$dnf" module install -y "nvidia-driver:${stream}"; then
            return 0
        fi
    done
    return 1
}

# Make the `dkms` package available so the -dkms driver streams can build their
# kernel module. On the RHEL family dkms ships from EPEL, NOT the base repos, so
# a plain `dnf install dkms` finds nothing until EPEL is enabled. We try, in
# order: (1) the packaged `epel-release` (present in the `extras` repo on
# Rocky/Alma/CentOS Stream, and on Fedora/Amazon Linux dkms is in the base repos
# so this is a harmless no-op); then if dkms still won't install, (2) bootstrap
# EPEL straight from the project's canonical URL keyed to the RHEL major
# (`rpm -E %rhel`) -- the case that matters on a fresh, UNregistered RHEL box
# where epel-release is in no enabled repo. Returns 0 iff dkms ends up installed.
# All steps are best-effort (|| true); the return value is the single source of
# truth so the caller can choose the precompiled streams when this fails.
#   $1 = sudo prefix ("" or "sudo");  $2 = the dnf/yum command name.
vts_rhel_ensure_dkms() {
    local sudo_cmd="$1" dnf="$2"
    if command -v dkms >/dev/null 2>&1 || rpm -q dkms >/dev/null 2>&1; then
        return 0
    fi
    vts_try "Enabling EPEL (for dkms)" ${sudo_cmd} "$dnf" install -y epel-release || true
    vts_try "Installing dkms" ${sudo_cmd} "$dnf" install -y dkms || true
    if command -v dkms >/dev/null 2>&1 || rpm -q dkms >/dev/null 2>&1; then
        return 0
    fi
    # epel-release wasn't reachable as a package -> bootstrap EPEL from its URL.
    # `rpm -E %rhel` yields the RHEL major (9 on RHEL/Rocky/Alma/CentOS 9); it's
    # empty on Fedora (where dkms is in the base repos and we'd have it already)
    # and unreliable on Amazon Linux (whose cuda-drivers is a plain package, so
    # we never reach the dkms streams there).
    local rhelver
    rhelver="$(rpm -E %rhel 2>/dev/null || true)"
    case "$rhelver" in
        [0-9]*)
            # Import EPEL's signing key FIRST. A base box may leave
            # localpkg_gpgcheck ON (common on hardened / unregistered RHEL), and
            # then `dnf install <epel url.rpm>` dies with "Public key ... is not
            # installed / GPG check FAILED" instead of the no-check the default
            # assumes. With the key imported the signed epel-release verifies; if
            # the install still fails we retry once with --nogpgcheck as a last
            # resort (the .rpm is fetched over HTTPS from Fedora's own host).
            local epel_key="https://dl.fedoraproject.org/pub/epel/RPM-GPG-KEY-EPEL-${rhelver}"
            local epel_url="https://dl.fedoraproject.org/pub/epel/epel-release-latest-${rhelver}.noarch.rpm"
            vts_try "Importing EPEL signing key" ${sudo_cmd} rpm --import "$epel_key" || true
            vts_try "Bootstrapping EPEL from ${epel_url}" \
                ${sudo_cmd} "$dnf" install -y "$epel_url" \
                || vts_try "Bootstrapping EPEL (retry, skipping GPG check)" \
                    ${sudo_cmd} "$dnf" install -y --nogpgcheck "$epel_url" || true
            vts_try "Installing dkms" ${sudo_cmd} "$dnf" install -y dkms || true
            ;;
    esac
    command -v dkms >/dev/null 2>&1 || rpm -q dkms >/dev/null 2>&1
}

# Universal last-resort fallback: install the driver from NVIDIA's self-contained
# `.run` installer instead of the distro's packages. This is the path AWS itself
# documents for EC2 GPU instances, and it sidesteps EVERY failure mode of the
# RPM/module/dkms route: it needs no CUDA repo, no `dnf` module stream, no EPEL,
# and no `dkms` package. The installer compiles the kernel module in place against
# the running kernel's source, so all it needs is a C toolchain + kernel headers.
#
# This is exactly what rescues a bare, UNregistered RHEL 9 GPU box (e.g. an AWS
# g4dn) where the dnf path dead-ends completely: `cuda-drivers` is hidden behind
# a module, the `*-dkms` streams can't get `dkms` (EPEL/base repos dark without a
# subscription), AND the precompiled kABI streams have no satisfiable `nvidia-kmod`
# provider either ("filtered out by modular filtering" / "missing module
# nvidia-driver:open"). The `.run` installer doesn't care about any of that.
#
# The `.run` URL is resolved in this order:
#   1. $VTSEARCH_NVIDIA_RUNFILE_URL, if set -- pin a version or point at the
#      public Tesla compute driver, e.g.
#      https://us.download.nvidia.com/tesla/<ver>/NVIDIA-Linux-x86_64-<ver>.run
#   2. The latest driver in AWS's public, credential-free S3 bucket
#      (ec2-linux-nvidia-drivers), which AWS keeps current for EC2 GPU instances
#      and serves over plain HTTPS -- no AWS CLI, no AWS credentials needed.
#   $1 = sudo prefix ("" or "sudo").
vts_install_driver_runfile() {
    local sudo_cmd="$1"

    # A self-contained `.run` still compiles a kernel module, so it needs the
    # kernel headers/devel for the RUNNING kernel plus a C toolchain. Install
    # them best-effort with whatever package manager is present. We also try to
    # get `dkms`: when it's present we hand the module to DKMS (--dkms below) so
    # it AUTO-REBUILDS on future kernel upgrades. That is the difference between a
    # set-and-forget install and one that silently dies on the box's next kernel
    # bump -- the exact recurring breakage this fallback path otherwise causes.
    if command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
        local dnf="dnf"
        command -v dnf >/dev/null 2>&1 || dnf="yum"
        vts_try "Installing kernel headers + build tools" \
            ${sudo_cmd} "$dnf" install -y "kernel-devel-$(uname -r)" kernel-headers gcc make tar || true
        # dkms lives in EPEL on RHEL, not the base repos, so a plain `dnf install
        # dkms` finds nothing on a bare box. vts_rhel_ensure_dkms bootstraps EPEL
        # first (even on an unregistered box) so the --dkms registration below can
        # actually take -- the difference between a set-and-forget install and a
        # kernel-pinned one on a bare RHEL g4dn.
        vts_rhel_ensure_dkms "$sudo_cmd" "$dnf" || true
    elif command -v apt-get >/dev/null 2>&1; then
        vts_try "Installing kernel headers + build tools" \
            ${sudo_cmd} apt-get install -y "linux-headers-$(uname -r)" gcc make || true
        vts_try "Installing dkms (so the module auto-rebuilds on kernel updates)" \
            ${sudo_cmd} apt-get install -y dkms || true
    fi

    # Register the built module with DKMS (--dkms) when dkms is available so a
    # later kernel upgrade triggers an automatic rebuild; without it the .run
    # module is pinned to the current kernel and the next kernel bump breaks
    # nvidia-smi. Unquoted so an empty value vanishes instead of passing "".
    local dkms_flag=""
    if command -v dkms >/dev/null 2>&1; then
        dkms_flag="--dkms"
    fi

    local url="${VTSEARCH_NVIDIA_RUNFILE_URL:-}"
    if [ -z "$url" ]; then
        # Discover the newest driver in AWS's public bucket. The S3 REST list
        # endpoint is anonymous-readable, so a plain GET + a grep for the .run
        # key works without the AWS CLI. Keys look like
        # "latest/NVIDIA-Linux-x86_64-<ver>-grid-aws.run".
        local arch_dir="x86_64"
        case "$(uname -m)" in aarch64 | arm64) arch_dir="aarch64" ;; esac
        local bucket="https://ec2-linux-nvidia-drivers.s3.amazonaws.com"
        local listing="" key=""
        if command -v curl >/dev/null 2>&1; then
            listing="$(curl -fsSL "${bucket}/?list-type=2&prefix=latest/" 2>/dev/null || true)"
        elif command -v wget >/dev/null 2>&1; then
            listing="$(wget -qO- "${bucket}/?list-type=2&prefix=latest/" 2>/dev/null || true)"
        fi
        key="$(printf '%s' "$listing" \
            | grep -oE "latest/NVIDIA-Linux-${arch_dir}-[0-9.]+(-grid-aws)?\.run" \
            | sort -V | tail -n1 || true)"
        [ -n "$key" ] && url="${bucket}/${key}"
    fi
    if [ -z "$url" ]; then
        echo "  Could not determine an NVIDIA .run installer URL automatically." >&2
        echo "  Set VTSEARCH_NVIDIA_RUNFILE_URL to a driver .run and re-run, e.g.:" >&2
        echo "    VTSEARCH_NVIDIA_RUNFILE_URL=https://us.download.nvidia.com/tesla/<ver>/NVIDIA-Linux-x86_64-<ver>.run \\" >&2
        echo "      bash scripts/install.sh" >&2
        return 1
    fi

    echo "  Downloading NVIDIA .run installer:" >&2
    echo "    $url" >&2
    local runfile
    runfile="$(mktemp --suffix=.run 2>/dev/null || mktemp)"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL -o "$runfile" "$url" \
            || { echo "  Download failed (network/proxy/404?)." >&2; rm -f "$runfile"; return 1; }
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$runfile" "$url" \
            || { echo "  Download failed (network/proxy/404?)." >&2; rm -f "$runfile"; return 1; }
    else
        echo "  Neither curl nor wget is available to download the .run installer." >&2
        rm -f "$runfile"
        return 1
    fi

    # --silent implies --no-questions/--ui=none. --disable-nouveau blacklists the
    # conflicting OSS driver (a reboot then loads nvidia). --no-cc-version-check
    # tolerates a gcc that differs from the one the kernel was built with.
    local rc=0
    vts_run "Compiling the NVIDIA kernel module (a few minutes; kernel $(uname -r))" \
        ${sudo_cmd} sh "$runfile" --silent --disable-nouveau --no-cc-version-check ${dkms_flag} || rc=$?
    rm -f "$runfile"
    if [ "$rc" -ne 0 ]; then
        echo "  The NVIDIA .run installer exited non-zero (see /var/log/nvidia-installer.log)." >&2
        echo "  Most common cause: kernel headers/devel for the running kernel" >&2
        echo "  ($(uname -r)) aren't installed and couldn't be fetched." >&2
        return 1
    fi
    return 0
}

# Tell the user whether the currently-installed NVIDIA driver is SET-AND-FORGET
# (survives future kernel upgrades on its own) or kernel-pinned (the next kernel
# bump breaks nvidia-smi until the module is rebuilt). A DKMS-managed module is
# rebuilt automatically for each new kernel; a precompiled/kABI-stream or plain
# `.run` module is tied to the kernel it was built against -- the usual cause of
# "GPU worked yesterday, gone today" after an unattended kernel update on a cloud
# box. Detected via `dkms status` (readable without root). Purely informational.
vts_report_driver_persistence() {
    if command -v dkms >/dev/null 2>&1 && dkms status 2>/dev/null | grep -qi nvidia; then
        echo "  GPU driver is DKMS-managed: it will auto-rebuild for new kernels, so a" >&2
        echo "  routine kernel update won't break the GPU. Set and forget." >&2
        return 0
    fi
    cat >&2 <<'EOF'
  NOTE: the GPU driver is NOT DKMS-managed, so it is tied to the CURRENT kernel.
  A later kernel upgrade (unattended-upgrades / dnf automatic) can stop the module
  loading, and nvidia-smi will go dark until you re-run this installer. To make it
  set-and-forget: install `dkms` and re-run this script (so the module auto-rebuilds
  for new kernels), pin the kernel so it stops changing, or bake a custom AMI / use
  the AWS Deep Learning AMI. See docs/DEPLOYMENT.md, "Making the GPU driver survive
  reboots and kernel updates".
EOF
    return 0
}

# Decide whether to convert a working-but-kernel-pinned driver to a DKMS build.
# Echo a single token on stdout: "convert" or "skip". Mirrors
# vts_decide_driver_missing: an env override wins (VTSEARCH_AUTO_DKMS=1 ->
# convert, VTSEARCH_SKIP_DKMS=1 -> skip), otherwise prompt on a TTY (default
# yes, since the whole point of running the installer is a durable GPU), and on
# a non-interactive shell SKIP -- a driver reinstall is privileged and implies a
# reboot, so it must never happen silently. Human text goes to stderr.
vts_decide_dkms_conversion() {
    if [ "${VTSEARCH_AUTO_DKMS:-0}" = "1" ]; then
        echo "  VTSEARCH_AUTO_DKMS=1 set -> converting the driver to a DKMS build." >&2
        echo convert
        return 0
    fi
    if [ "${VTSEARCH_SKIP_DKMS:-0}" = "1" ]; then
        echo "  VTSEARCH_SKIP_DKMS=1 set -> leaving the current driver as-is." >&2
        echo skip
        return 0
    fi
    if [ ! -t 0 ]; then
        echo "  Non-interactive shell; not converting automatically (a driver reinstall" >&2
        echo "  is privileged and needs a reboot). Re-run with VTSEARCH_AUTO_DKMS=1 to" >&2
        echo "  convert unattended, or leave it to keep the current kernel-pinned driver." >&2
        echo skip
        return 0
    fi
    local reply
    cat >&2 <<'EOF'

Convert it to a DKMS build now? This reinstalls the driver via NVIDIA's .run
installer with --dkms (needs sudo; a reboot is recommended afterward so the
freshly built module loads). Your GPU keeps working either way -- declining just
leaves it kernel-pinned.
EOF
    printf 'Convert now? [Y/n]: ' >&2
    read -r reply
    case "$reply" in
        "" | [yY] | [yY][eE][sS]) echo convert ;;
        *) echo skip ;;
    esac
}

# Make `dkms` available across package managers (best-effort). On RHEL this goes
# through vts_rhel_ensure_dkms (which bootstraps EPEL, importing its GPG key);
# on Debian/Ubuntu dkms is in the base repos. Returns 0 iff dkms ends up present.
#   $1 = sudo prefix.
vts_ensure_dkms_any() {
    local sudo_cmd="$1"
    command -v dkms >/dev/null 2>&1 && return 0
    if command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
        local dnf="dnf"
        command -v dnf >/dev/null 2>&1 || dnf="yum"
        vts_rhel_ensure_dkms "$sudo_cmd" "$dnf" || true
    elif command -v apt-get >/dev/null 2>&1; then
        vts_try "Installing dkms" ${sudo_cmd} apt-get install -y dkms || true
    fi
    command -v dkms >/dev/null 2>&1
}

# Reinstall the current driver as a DKMS build so it auto-rebuilds on future
# kernel updates. Reuses vts_install_driver_runfile, which downloads the matching
# .run and runs it with --dkms. Privileged; returns 0 on success. Sudo handling
# mirrors vts_install_nvidia_driver.
vts_convert_driver_to_dkms() {
    local sudo_cmd=""
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            sudo_cmd="sudo"
        else
            echo "  Cannot convert: need root and 'sudo' is not available." >&2
            return 1
        fi
    fi
    vts_sudo_keepalive "$sudo_cmd" || true
    trap 'vts_sudo_keepalive_stop' RETURN

    # dkms MUST be present or the .run --dkms registration is a no-op and we'd do
    # a pointless (still kernel-pinned) reinstall. If it can't be installed -- the
    # unregistered-RHEL / EPEL-unreachable case -- bail without touching the
    # working driver and let the caller point at kernel-pinning instead.
    if ! vts_ensure_dkms_any "$sudo_cmd"; then
        echo "  Could not install dkms on this host (e.g. an unregistered RHEL box" >&2
        echo "  where EPEL is unreachable), so a DKMS conversion isn't possible. The" >&2
        echo "  current driver is untouched; pin the kernel instead so it keeps" >&2
        echo "  matching. See docs/DEPLOYMENT.md, \"Making the GPU driver survive...\"." >&2
        return 1
    fi

    # The .run installer refuses to replace a driver whose modules are LOADED
    # (it aborts with "nvidia-modeset appears to be already loaded"), which is the
    # normal state here since the driver is up. Stop the persistence daemon and
    # unload the modules first (dependents before the base module). Best-effort:
    # if something still holds a module, the .run install fails and we warn.
    if command -v systemctl >/dev/null 2>&1; then
        $sudo_cmd systemctl stop nvidia-persistenced >/dev/null 2>&1 || true
    fi
    $sudo_cmd rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia >/dev/null 2>&1 || true

    vts_install_driver_runfile "$sudo_cmd" || return 1

    # Best-effort: load the fresh module and (re-)enable the persistence daemon,
    # same as a from-scratch driver install.
    $sudo_cmd modprobe nvidia >/dev/null 2>&1 || true
    if command -v systemctl >/dev/null 2>&1; then
        $sudo_cmd systemctl enable --now nvidia-persistenced >/dev/null 2>&1 || true
    fi
    return 0
}

# When the driver is up (nvidia-smi works) but NOT DKMS-managed, it's pinned to
# the running kernel and the next kernel update will break it. Offer to convert
# it in place so it becomes set-and-forget. No-op when the driver is absent (the
# driver-install path handles that) or already DKMS-managed (nothing to do).
vts_maybe_convert_driver_to_dkms() {
    vts_have_gpu || return 0
    if command -v dkms >/dev/null 2>&1 && dkms status 2>/dev/null | grep -qi nvidia; then
        return 0
    fi
    echo >&2
    echo "The GPU driver works, but it is NOT DKMS-managed -- it's pinned to the" >&2
    echo "running kernel ($(uname -r)), so the next kernel update will break it." >&2
    case "$(vts_decide_dkms_conversion)" in
        convert)
            if vts_convert_driver_to_dkms; then
                echo "  Driver reinstalled and registered with DKMS: it will now auto-rebuild" >&2
                echo "  on kernel updates. Reboot soon so the fresh module loads (until then" >&2
                echo "  the GPU smoke test below may report CPU)." >&2
            else
                echo "warning: DKMS conversion did not complete; the driver still works but" >&2
                echo "         stays kernel-pinned. See the output above, retry later, or use" >&2
                echo "         the manual steps in docs/DEPLOYMENT.md." >&2
            fi
            ;;
        *)
            echo "  Leaving the driver kernel-pinned. Set VTSEARCH_AUTO_DKMS=1 on a later" >&2
            echo "  run to convert it, or see docs/DEPLOYMENT.md for the manual steps." >&2
            ;;
    esac
}

# Best-effort, distro-aware NVIDIA driver install so we can fix the missing
# driver FOR the user instead of just telling them how. It's a privileged,
# system-mutating step (kernel driver), so it only runs on an explicit choice
# (the interactive prompt's default, or VTSEARCH_AUTO_DRIVER=1) -- never silently.
#
# Returns 0 only if nvidia-smi can see the GPU afterward (driver live this
# session). Returns 1 if the install couldn't run (no root, unsupported distro,
# package failure) OR if it installed but the GPU isn't visible yet -- almost
# always meaning a REBOOT is needed before the kernel module loads. The caller
# treats a non-zero return as "can't do GPU in this session" and explains next
# steps. Guarded throughout so `set -e` can't abort the whole install on a
# package-manager hiccup (the function is invoked in an `if`, which disables
# `set -e` for its body, but we also check each step explicitly).
vts_install_nvidia_driver() {
    local sudo_cmd=""
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            sudo_cmd="sudo"
        else
            echo "  Cannot auto-install: need root and 'sudo' is not available." >&2
            return 1
        fi
    fi

    local distro=""
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        distro="$(. /etc/os-release && echo "${ID:-} ${ID_LIKE:-}")"
    fi
    echo "  Installing the NVIDIA driver (distro: ${distro:-unknown}). This needs" >&2
    echo "  network access and a few minutes; you may be prompted for your password." >&2

    # Cache sudo credentials now (one prompt) and keep them warm for the duration,
    # so the backgrounded heartbeat-wrapped steps below never stall invisibly on a
    # password prompt. The RETURN trap stops the refresher on every exit path.
    vts_sudo_keepalive "$sudo_cmd" || true
    trap 'vts_sudo_keepalive_stop' RETURN

    # Set to 1 by any branch that gets the driver packages installed. When it
    # stays 0, we fall through to the universal `.run` installer fallback below
    # rather than giving up -- this is what saves a bare RHEL 9 box where every
    # dnf path dead-ends.
    local pkg_ok=0
    case "$distro" in
        *ubuntu* | *debian*)
            if vts_try "Refreshing apt package lists" ${sudo_cmd} apt-get update; then
                # ubuntu-drivers picks the recommended driver for THIS GPU; fall
                # back to a recent fixed branch if the helper isn't installed.
                if ! command -v ubuntu-drivers >/dev/null 2>&1; then
                    vts_try "Installing ubuntu-drivers-common" \
                        ${sudo_cmd} apt-get install -y ubuntu-drivers-common || true
                fi
                if command -v ubuntu-drivers >/dev/null 2>&1; then
                    if vts_try "Installing the recommended NVIDIA driver (ubuntu-drivers)" ${sudo_cmd} ubuntu-drivers install \
                        || vts_try "Installing NVIDIA driver (nvidia-driver-535)" ${sudo_cmd} apt-get install -y nvidia-driver-535; then
                        pkg_ok=1
                    fi
                elif vts_try "Installing NVIDIA driver (nvidia-driver-535)" ${sudo_cmd} apt-get install -y nvidia-driver-535; then
                    pkg_ok=1
                fi
                [ "$pkg_ok" -eq 1 ] || echo "  driver package install failed." >&2
            else
                echo "  apt-get update failed." >&2
            fi
            ;;
        *amzn* | *rhel* | *centos* | *rocky* | *almalinux* | *fedora*)
            local dnf=""
            command -v dnf >/dev/null 2>&1 && dnf="dnf"
            [ -z "$dnf" ] && command -v yum >/dev/null 2>&1 && dnf="yum"
            if [ -z "$dnf" ]; then
                echo "  No dnf/yum found; cannot install driver packages on this RPM distro." >&2
            else
                # A DKMS-built driver needs the kernel headers/devel matching the
                # running kernel plus a C toolchain; install them best-effort here.
                # (The `dkms` package itself comes from EPEL and is handled inside
                # vts_rhel_install_driver_pkgs -> vts_rhel_ensure_dkms, which knows
                # how to bootstrap EPEL when the packaged epel-release isn't found.)
                vts_try "Installing kernel headers + build tools" \
                    ${sudo_cmd} "$dnf" install -y "kernel-devel-$(uname -r)" kernel-headers gcc make || true
                # A base AMI doesn't have NVIDIA's CUDA repo (which provides
                # cuda-drivers), so the first install attempt typically fails with
                # "No match for argument: cuda-drivers". Enable the repo and retry;
                # this is the fix for the common fresh-RHEL-GPU-box failure. The
                # retry goes through vts_rhel_install_driver_pkgs, which ALSO handles
                # the second RHEL failure mode -- "All matches were filtered out by
                # modular filtering" -- by installing the nvidia-driver module stream.
                if vts_rhel_install_driver_pkgs "$sudo_cmd" "$dnf"; then
                    pkg_ok=1
                else
                    echo "  cuda-drivers not available yet -- enabling NVIDIA's CUDA repo and retrying." >&2
                    if vts_enable_nvidia_cuda_repo "$sudo_cmd" "$dnf" \
                        && vts_rhel_install_driver_pkgs "$sudo_cmd" "$dnf"; then
                        pkg_ok=1
                    else
                        echo "  cuda-drivers install still failed via dnf (repo/module/dkms)." >&2
                    fi
                fi
            fi
            ;;
        *)
            echo "  Unrecognized distro -- no package-manager path; will try the .run installer." >&2
            ;;
    esac

    # Every package-manager path failed -> fall back to NVIDIA's self-contained
    # `.run` installer, which needs no repo/module/dkms and is the route AWS
    # documents for EC2. This is the rescue for a bare, unregistered RHEL 9 g4dn.
    if [ "$pkg_ok" -ne 1 ]; then
        echo "  Package-based driver install didn't complete; falling back to" >&2
        echo "  NVIDIA's self-contained .run installer (the path AWS documents for EC2)." >&2
        if ! vts_install_driver_runfile "$sudo_cmd"; then
            echo "  The .run installer fallback did not succeed either. See AWS's guide:" >&2
            echo "    https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html" >&2
            return 1
        fi
    fi

    # Try to load the freshly built module without a reboot (best-effort).
    $sudo_cmd modprobe nvidia >/dev/null 2>&1 || true

    # Enable NVIDIA's persistence daemon so the driver initializes at every boot
    # without waiting for something to open the device first -- AWS's recommended
    # setup for EC2 GPU instances. Best-effort: a missing unit / no systemd no-ops.
    if command -v systemctl >/dev/null 2>&1; then
        $sudo_cmd systemctl enable --now nvidia-persistenced >/dev/null 2>&1 || true
    fi

    if vts_have_gpu; then
        echo "  Driver installed and the GPU is now visible to nvidia-smi." >&2
        return 0
    fi
    cat >&2 <<'EOF'
  Driver installed, but nvidia-smi still can't see the GPU -- the kernel module
  usually loads only after a REBOOT. Reboot, then re-run:

      sudo reboot
      bash scripts/install.sh
EOF
    return 1
}

# The GPU is in the box but the driver is missing. Decide what to do and echo a
# single token on stdout: "driver" (install the driver, then GPU), "cpu"
# (CPU-only install), or "stop". The human-readable notice/prompt goes to
# stderr so the caller can capture just the token via $(...).
#
# Interactive: prompt with three choices, defaulting to installing the driver
# (the user has GPU hardware and almost certainly wants it). Non-interactive
# (no TTY: CI, `curl ... | bash`, Docker build): can't prompt, so honor an env
# override -- VTSEARCH_AUTO_DRIVER=1 -> driver, VTSEARCH_ASSUME_CPU=1 -> cpu --
# and otherwise stop so a headless run can't silently land on CPU or run sudo.
vts_decide_driver_missing() {
    vts_report_driver_missing

    if [ "${VTSEARCH_AUTO_DRIVER:-0}" = "1" ]; then
        echo "  VTSEARCH_AUTO_DRIVER=1 set -> installing the NVIDIA driver." >&2
        echo driver
        return 0
    fi
    if [ "${VTSEARCH_ASSUME_CPU:-0}" = "1" ]; then
        echo "  VTSEARCH_ASSUME_CPU=1 set -> proceeding with a CPU-only install." >&2
        echo cpu
        return 0
    fi
    if [ ! -t 0 ]; then
        cat >&2 <<'EOF'

Non-interactive shell; not prompting. Stopping so this doesn't silently install
a driver (sudo) or land on CPU. Re-run with one of:
  bash scripts/install.sh                       # interactive prompt
  VTSEARCH_AUTO_DRIVER=1 bash scripts/install.sh # auto-install the driver
  bash scripts/install.sh cpu                    # CPU-only, no GPU
EOF
        echo stop
        return 0
    fi

    local reply
    cat >&2 <<'EOF'

What would you like to do?
  [i] Install the NVIDIA driver now (needs sudo; may require a reboot) -- recommended
  [c] Install CPU-only torch instead (no GPU acceleration)
  [s] Stop and fix it yourself
EOF
    printf 'Choice [I/c/s]: ' >&2
    read -r reply
    case "$reply" in
        "" | [iI] | [iI][nN][sS][tT][aA][lL][lL]) echo driver ;;
        [cC] | [cC][pP][uU]) echo cpu ;;
        *) echo stop ;;
    esac
}

# --- CUDA tag resolution -----------------------------------------------------
# Echo a cuXYZ tag for the GPU. An explicit arg wins; otherwise auto-detect via
# detect_cuda_tag.py (which prints its reasoning to stderr -> the terminal), and
# fall back to the safe Volta..Hopper default if it can't tell.
vts_resolve_cuda_tag() {
    local explicit="${1:-}"
    if [ -n "$explicit" ]; then
        echo "$explicit"
        return 0
    fi
    local pybin detected
    pybin="$(command -v python || command -v python3 || true)"
    if [ -n "$pybin" ] && detected="$("$pybin" "$SCRIPT_DIR/detect_cuda_tag.py")"; then
        echo "$detected"
    else
        echo "  (auto-detect failed; falling back to default tag cu124)" >&2
        echo "cu124"
    fi
}

# --- pre-commit wiring (shared) ----------------------------------------------
vts_install_precommit() {
    if [ -d "$REPO_ROOT/.git" ] && [ -f "$REPO_ROOT/.pre-commit-config.yaml" ]; then
        (cd "$REPO_ROOT" && pre-commit install --install-hooks) || \
            echo "warning: pre-commit install failed; run it manually to enable git hooks"
    else
        echo "  (skipped: no git checkout or .pre-commit-config.yaml)"
    fi
}

# --- CPU install -------------------------------------------------------------
# Installs runtime + dev dependencies and the vtsearch package itself (editable)
# by forwarding to pyproject.toml via requirements/base.txt, which is just
# `--extra-index-url <cpu wheel index>` + `-e .[dev]`.
# --- toponymy (VTSBrowse signpost naming) ------------------------------------
# Two quirks force a dedicated step (see docs/plans/vtsbrowse-toponymy.md):
#   - apricot-select (a real toponymy dep, declared in pyproject.toml) ships a
#     legacy setup.py sdist. It must be installed BEFORE the main requirements
#     pass so its build quirks stay contained to this step. Do NOT wrap it in
#     SETUPTOOLS_USE_DISTUTILS=stdlib: pip's isolated build env installs the
#     latest setuptools, and setuptools >= 74 refuses to even import with that
#     value set, so the build dies with "BackendUnavailable: Cannot import
#     'setuptools.build_meta'" (and Python >= 3.12 has no stdlib distutils for
#     the shim to point at anyway). The historical "AttributeError:
#     install_layout" crash the shim once worked around no longer reproduces
#     with current setuptools (verified on Python 3.12 and 3.14).
#   - toponymy 0.5.2 pins transformers<5.0.0, which a plain install would
#     honor by downgrading the app's transformers stack; the pin is
#     empirically unnecessary for our usage (validated end-to-end on
#     transformers 5.x), so toponymy is installed --no-deps with its real
#     dependencies declared in pyproject.toml instead. The tests_lib
#     projection smoke test guards this bypass against future breakage.
vts_install_toponymy() {
    pip install apricot-select --progress-bar on
    pip install --no-deps "toponymy==0.5.2" --progress-bar on
}

# --- facenet-pytorch (FaceNet face-identity embedder) ------------------------
# The `face` embedder (vtscore.media.face.embedder_facenet) loads FaceNet's
# InceptionResnetV1 via facenet-pytorch. facenet-pytorch 2.6.0 hard-pins
# torch<2.3, torchvision<0.18, numpy<2.0 and Pillow<10.3 -- a plain install
# would honor those by DOWNGRADING the app's entire torch/numpy/Pillow stack.
# Those pins are empirically unnecessary: the model builds and runs a correct
# forward pass on the app's modern stack (verified on torch 2.13 / numpy 2.x /
# Pillow 12). So install it --no-deps -- exactly the toponymy pattern -- and let
# the app's own torch, torchvision, numpy, Pillow, requests and tqdm (all
# already present) satisfy it at runtime. Pretrained weights are lazy-downloaded
# from GitHub on first model use, not here.
vts_install_face_deps() {
    pip install --no-deps facenet-pytorch --progress-bar on
}

vts_install_cpu() {
    vts_progress_init 6 "Installing VTSearch CPU dependencies"
    echo "Heads-up: the pip steps below show a download bar, but pip also goes quiet"
    echo "for 10-30s at a time while it resolves dependency versions. That silence is"
    echo "normal -- it is working, not frozen."

    vts_progress_step "Checking Python version (>= 3.10)"
    # shellcheck source=_check-python.sh
    source "$SCRIPT_DIR/_check-python.sh"

    vts_progress_step "Upgrading pip / setuptools / wheel"
    pip install --upgrade pip "setuptools<82" wheel --progress-bar on

    vts_progress_step "Installing signpost naming deps (apricot-select + toponymy)"
    vts_install_toponymy

    vts_progress_step "Installing runtime + dev dependencies (this may take several minutes)"
    pip install -r "$REPO_ROOT/requirements/base.txt" --progress-bar on

    vts_progress_step "Installing FaceNet face embedder (facenet-pytorch, --no-deps)"
    vts_install_face_deps

    vts_progress_step "Wiring up pre-commit git hook"
    vts_install_precommit

    vts_progress_done "CPU dependencies installed successfully"
}

# --- optional cuML / RAPIDS accelerator (best-effort) ------------------------
# cuML powers GPU UMAP (VTSBrowse projection) and GPU k-means (diversity tree);
# VTSearch auto-detects it at runtime and falls back to the CPU umap-learn /
# scikit-learn paths when it's absent (see vtscore/gpu_backends.py). We install
# it by default on GPU hosts, but as an ISOLATED, NON-FATAL step:
#
#   - It's a multi-gigabyte RAPIDS stack (cudf, rmm, cupy, ...) and lives on a
#     separate index (pypi.nvidia.com), so a slow/unreachable index or a
#     resolver hiccup must NOT abort an otherwise-good GPU install.
#   - The wheel is CUDA-major-pinned: cu11* tags -> cuml-cu11, cu12* -> cuml-cu12.
#   - The cu12 wheel is ALSO capped below RAPIDS 26 (see VTS_CUML_CU12_SPEC).
#     RAPIDS 26.x raised its CUDA floor to require nvidia-nvjitlink-cu12>=12.9,
#     but the torch CUDA wheels we pin (cu124 = CUDA 12.4 .. cu128 = CUDA 12.8)
#     top out at 12.8 and no torch wheel ships 12.9 yet, so an UNpinned
#     cuml-cu12 floats to 26.x and pip dies with an nvjitlink conflict (and
#     cupy then JIT-compiles cuVS/raft kernels against mismatched CUDA-13 fp8
#     headers -> "cuda_fp8.hpp: this declaration has no storage class" at fit
#     time). 25.x declares cuda-toolkit==12.* and resolves cleanly against the
#     pinned torch. Bump the cap once a torch wheel ships the CUDA minor 26.x
#     needs. See docs/DEPLOYMENT.md "cuML crashes compiling a kernel".
#   - RAPIDS ships linux-only wheels for a fixed Python range; on an unsupported
#     platform/Python this step just warns and the app uses the CPU fallback.
#
# Override/skip with VTSEARCH_SKIP_CUML=1 (e.g. air-gapped installs that can't
# reach the NVIDIA index, or when you simply don't want the download).
vts_install_cuml() {
    local cuda_tag="$1"

    if [ "${VTSEARCH_SKIP_CUML:-0}" = "1" ]; then
        echo "  (skipped: VTSEARCH_SKIP_CUML=1; GPU UMAP/k-means will use the CPU fallback)"
        return 0
    fi

    # The cu12 spec is capped below RAPIDS 26 (whose CUDA-12.9 floor outruns the
    # torch CUDA wheels we pin); see the header comment above. Override the cap
    # via VTS_CUML_CU12_SPEC, e.g. once a matching torch wheel exists:
    #   VTS_CUML_CU12_SPEC='cuml-cu12' bash scripts/install.sh cu128
    local cuml_pkg
    case "$cuda_tag" in
        cu11*) cuml_pkg="cuml-cu11" ;;
        *) cuml_pkg="${VTS_CUML_CU12_SPEC:-cuml-cu12<26}" ;;  # cu12x + anything else
    esac

    # Heads-up: cuML depends on NEWER nvidia-*-cu12 runtime wheels than the torch
    # build pins exactly (e.g. cu124 torch pins ==12.4.x), so installing it upgrades
    # those libs and pip emits a red "dependency conflicts" report naming torch's
    # now-unsatisfied pins. This is distinct from the FATAL RAPIDS-26 nvjitlink/fp8
    # break the cuml-cu12<26 cap above prevents -- this one is cosmetic. That is
    # EXPECTED and non-fatal -- pip still completes the install and does not roll
    # anything back, and CUDA 12.x runtimes are compatible across minor versions.
    # vts_run captures that report into a log instead of letting it scroll past as
    # a scary red wall (it surfaces only if the install actually fails); the GPU
    # smoke test below then confirms torch + cuML coexist. This is a multi-GB
    # RAPIDS download, so the heartbeat's elapsed-time counter is the liveness cue.
    echo "  (cuML is a large multi-GB download; the heartbeat below shows progress."
    echo "   A cosmetic pip 'dependency conflicts' report is captured, not shown.)"

    # Isolated pass against the NVIDIA index. vts_run is guarded so `set -e` can't
    # abort the install on failure; the runtime cuML detection degrades to CPU.
    if vts_run "Installing cuML/RAPIDS (${cuml_pkg})" \
        pip install --extra-index-url https://pypi.nvidia.com \
            --prefer-binary "$cuml_pkg"; then
        echo "  cuML installed: GPU UMAP + k-means enabled."
    else
        echo "warning: cuML (${cuml_pkg}) install failed; GPU UMAP/k-means will fall" >&2
        echo "         back to the CPU umap-learn / scikit-learn paths. This is" >&2
        echo "         non-fatal. Re-run later with a reachable pypi.nvidia.com, or" >&2
        echo "         set VTSEARCH_SKIP_CUML=1 to silence this step." >&2
    fi
}

# --- post-install GPU smoke test (best-effort) -------------------------------
# After the GPU install, actually exercise the stack so the red pip "dependency
# conflicts" report from the cuML step turns into a definitive pass/fail instead
# of an ambiguous wall of text: import torch, run a tiny CUDA matmul, and import
# cuML. This catches the one case where the conflict WOULD have mattered -- a
# runtime-lib bump that breaks torch's GPU path -- at install time rather than
# leaving a silent landmine. Diagnostic only and non-fatal: VTSearch smoke-tests
# CUDA at runtime and falls back to CPU, so a failure here warns but never aborts.
vts_smoke_test_gpu() {
    local pybin
    pybin="$(command -v python || command -v python3 || true)"
    if [ -z "$pybin" ]; then
        echo "  (skipped: no python on PATH to run the smoke test)"
        return 0
    fi

    # Write the probe to a temp file so it can be run as a backgrounded command
    # under a heartbeat (importing torch alone is a ~15s silent stretch that looks
    # frozen); a heredoc on stdin can't be backgrounded with stdin from /dev/null.
    local rc=0 pyfile log pid
    pyfile="$(mktemp --suffix=.py 2>/dev/null || mktemp)"
    cat >"$pyfile" <<'PY'
import sys

try:
    import torch
except Exception as e:  # noqa: BLE001 - report any import failure verbatim
    print(f"  torch: IMPORT FAILED: {type(e).__name__}: {e}")
    sys.exit(2)

print(f"  torch {torch.__version__}")
gpu_ok = False
if not torch.cuda.is_available():
    print("  torch.cuda.is_available() = False -> VTSearch will run on CPU.")
    print("  (a driver/CUDA-tag mismatch, NOT the cuML dependency warning.)")
else:
    try:
        x = torch.randn(256, 256, device="cuda")
        checksum = (x @ x).sum().item()
        print(f"  GPU op OK on {torch.cuda.get_device_name(0)} (matmul checksum {checksum:.1f})")
        gpu_ok = True
    except Exception as e:  # noqa: BLE001 - report any runtime failure verbatim
        print(f"  GPU op FAILED: {type(e).__name__}: {e}")
        print("  The installed CUDA runtime may be incompatible with this torch build.")
        sys.exit(3)

try:
    import cuml

    print(f"  cuML {getattr(cuml, '__version__', '?')} import OK -> GPU UMAP/k-means enabled.")
except Exception:  # noqa: BLE001 - absence is fine; app uses the CPU fallback
    print("  cuML not importable -> GPU UMAP/k-means will use the CPU fallback.")

sys.exit(0 if gpu_ok else 1)
PY

    if [ "${VTSEARCH_VERBOSE:-0}" = "1" ]; then
        "$pybin" "$pyfile" || rc=$?
    else
        log="$(_vts_mktemp)"
        "$pybin" "$pyfile" >"$log" 2>&1 </dev/null &
        pid=$!
        _vts_spin "$pid" "Importing torch + exercising the GPU"
        wait "$pid" || rc=$?
        cat "$log"
        rm -f "$log" 2>/dev/null || true
    fi
    rm -f "$pyfile" 2>/dev/null || true

    case "$rc" in
        0) ;;  # GPU verified end-to-end; the python output already said so.
        1) ;;  # torch fine but no usable CUDA -> CPU mode; python explained why.
        *)
            echo "warning: GPU smoke test did not pass (see above). The install may" >&2
            echo "         still be usable -- VTSearch falls back to CPU automatically --" >&2
            echo "         but you will NOT get GPU acceleration. If you expected it," >&2
            echo "         re-check the driver and the CUDA tag (bash scripts/install.sh cuXYZ)." >&2
            ;;
    esac
    return 0
}

# --- GPU install -------------------------------------------------------------
# The PyTorch extra-index (download.pytorch.org/whl/cu*) sometimes serves source
# tarballs for packages like numpy and scipy, so we pre-install them as
# binary-only wheels before the full requirements pass, avoiding the need for a
# C++ compiler.
vts_install_gpu() {
    local cuda_tag="$1"
    local extra_index="https://download.pytorch.org/whl/${cuda_tag}"

    vts_progress_init 10 "Installing VTSearch GPU dependencies (CUDA tag: ${cuda_tag})"
    echo "Heads-up: the pip steps below show a download bar, but pip also goes quiet"
    echo "for 10-30s at a time while it resolves dependency versions. That silence is"
    echo "normal -- it is working, not frozen."

    # If the driver works but is kernel-pinned (non-DKMS), offer to convert it to
    # a DKMS build now so tomorrow's kernel update doesn't take the GPU down; then
    # report the (possibly updated) persistence status.
    vts_maybe_convert_driver_to_dkms
    vts_report_driver_persistence

    vts_progress_step "Checking Python version (>= 3.10)"
    # shellcheck source=_check-python.sh
    source "$SCRIPT_DIR/_check-python.sh"

    vts_progress_step "Upgrading pip / setuptools / wheel"
    pip install --upgrade pip "setuptools<82" wheel --progress-bar on

    vts_progress_step "Pre-installing binary-only wheels (numpy, scipy) from PyPI"
    pip install --only-binary :all: \
      --index-url https://pypi.org/simple \
      "numpy" \
      "scipy" \
      --progress-bar on

    # Pin torch/torchvision/torchaudio to the chosen CUDA index with --index-url
    # (NOT --extra-index-url). With --extra-index-url, PyPI stays in the candidate
    # set, and when PyPI ships a *newer* torch than this CUDA index tops out at
    # (e.g. cu124 caps at 2.6.0 while PyPI has 2.7.x, a cu126 build), pip prefers
    # the higher version and silently installs the wrong-arch wheel -- which then
    # fails with cudaErrorNoKernelImageForDevice on older GPUs. Installing from the
    # CUDA index alone forces the matching +${cuda_tag} build; the PyTorch index
    # mirrors torch's dependency closure, so a sole --index-url resolves cleanly.
    vts_progress_step "Installing CUDA torch from ${cuda_tag} index (pinned so PyPI can't substitute a mismatched build)"
    pip install --index-url "$extra_index" \
      --prefer-binary \
      torch torchvision torchaudio \
      --progress-bar on

    vts_progress_step "Installing signpost naming deps (apricot-select + toponymy)"
    vts_install_toponymy

    # Install everything else. torch is already satisfied by the pinned build above,
    # so this --extra-index-url pass won't replace it (no --upgrade).
    vts_progress_step "Installing remaining dependencies via ${extra_index} (this may take several minutes)"
    pip install --extra-index-url "$extra_index" \
      --prefer-binary \
      -r "$REPO_ROOT/requirements/gpu.txt" \
      --progress-bar on

    vts_progress_step "Installing FaceNet face embedder (facenet-pytorch, --no-deps)"
    vts_install_face_deps

    vts_progress_step "Installing optional cuML/RAPIDS accelerator (best-effort; CUDA tag: ${cuda_tag})"
    vts_install_cuml "$cuda_tag"

    vts_progress_step "Verifying the GPU stack (torch CUDA op + cuML import)"
    vts_smoke_test_gpu

    vts_progress_step "Wiring up pre-commit git hook"
    vts_install_precommit

    vts_progress_done "GPU dependencies installed successfully"
}

# --- dispatch ----------------------------------------------------------------
MODE="${1:-auto}"
case "$MODE" in
    cpu)
        vts_install_cpu
        ;;
    gpu)
        vts_install_gpu "$(vts_resolve_cuda_tag)"
        ;;
    cu*)
        vts_install_gpu "$(vts_resolve_cuda_tag "$MODE")"
        ;;
    auto|"")
        if vts_have_gpu; then
            echo "Detected an NVIDIA GPU -> installing the GPU dependency set." >&2
            vts_install_gpu "$(vts_resolve_cuda_tag)"
        elif vts_nvidia_hardware_present; then
            # Card is in the box but nvidia-smi can't see it -> driver missing.
            # Offer to install the driver for them, fall back to CPU, or stop.
            case "$(vts_decide_driver_missing)" in
                driver)
                    if vts_install_nvidia_driver; then
                        echo "GPU is online -> installing the GPU dependency set." >&2
                        vts_install_gpu "$(vts_resolve_cuda_tag)"
                    else
                        echo "Could not bring the GPU online in this session. See the" >&2
                        echo "specific cause and next step printed above (a reboot if the" >&2
                        echo "driver installed but its module isn't loaded yet; otherwise the" >&2
                        echo "named repo/package error), resolve it, then re-run:" >&2
                        echo "  bash scripts/install.sh" >&2
                        echo "For the full, unfiltered output of every step, re-run with:" >&2
                        echo "  VTSEARCH_VERBOSE=1 bash scripts/install.sh" >&2
                        exit 1
                    fi
                    ;;
                cpu)
                    echo "Proceeding with a CPU-only install." >&2
                    vts_install_cpu
                    ;;
                *)
                    echo "Stopping. Install the NVIDIA driver and re-run, or run" >&2
                    echo "'bash scripts/install.sh cpu' to install CPU-only torch." >&2
                    exit 1
                    ;;
            esac
        else
            echo "No NVIDIA GPU detected (no NVIDIA PCI device found) -> installing the CPU dependency set." >&2
            vts_install_cpu
        fi
        ;;
    *)
        cat >&2 <<EOF
ERROR: unknown argument '$MODE'.

Usage:
  bash scripts/install.sh            # auto-detect CPU vs GPU
  bash scripts/install.sh cpu        # force CPU install
  bash scripts/install.sh gpu        # force GPU install (auto-detect CUDA tag)
  bash scripts/install.sh cu124      # force GPU install with an explicit CUDA tag
EOF
        exit 2
        ;;
esac
