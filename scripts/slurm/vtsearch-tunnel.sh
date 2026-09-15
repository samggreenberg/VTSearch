#!/bin/bash
# Find your running VTSearch GPU job on the cluster and forward its port to your
# local machine. Run this LOCALLY, AFTER starting VTSearch on the cluster (the
# `vtsearch` command from scripts/slurm/vtsearch-slurm.sh).
#
# It auto-discovers the node and the per-user port. Browse the printed
# http://localhost:<port> once it connects. Ctrl+C closes the tunnel.
#
# Requires an SSH host entry for the cluster login node. By default this script
# connects to the host alias `cluster`; define it in ~/.ssh/config, e.g.
#
#     Host cluster
#         HostName login.your-cluster.edu
#         User your-cluster-username
#
# or point this script at a different alias with:  CLUSTER_HOST=mycluster vtsearch-tunnel
#
# TWO KNOBS FOR AN ALWAYS-ON RELAY BOX (a machine that holds the VPN at home so
# other devices need not):
#
#   VTS_BIND    Address the forwarded port is bound to locally. Unset (the
#               default) binds loopback only, so the port exists for this
#               machine alone. Set it to reach the app from another device --
#               typically a private-mesh address, e.g.
#                   VTS_BIND=$(tailscale ip -4) vtsearch-tunnel
#               Bind to that specific address rather than 0.0.0.0: the latter
#               also serves the app to everything else on the local network.
#
#   --no-shell  Hold the tunnel open and nothing else (ssh -N, no TTY, no login
#               shell). This is the mode a service unit wants; the interactive
#               default is the mode a human wants. See vtsearch-tunnel.service.
set -u

NO_SHELL=0
for arg in "$@"; do
    case "$arg" in
        --no-shell) NO_SHELL=1 ;;
        -h|--help)
            sed -n '2,/^set -u/p' "$0" | sed 's/^# \{0,1\}//; $d'
            exit 0 ;;
        *)
            echo "!!! Unknown argument: $arg" >&2
            echo "    Usage: vtsearch-tunnel [--no-shell]" >&2
            exit 2 ;;
    esac
done

CLUSTER_HOST=${CLUSTER_HOST:-cluster}
# Where the VTSearch checkout lives on the cluster (used only to drop you into
# the project dir for git/edits). Mirrors VTS_DIR in vtsearch-slurm.sh.
VTS_DIR=${VTS_DIR:-/exp/\$USER/projects/VTSearch}

# The login node may be reachable directly (on-campus / same network) or only
# through a VPN. Rather than guess the path, just probe whether we can actually
# reach it over SSH.
echo ">>> Checking connectivity to '$CLUSTER_HOST'..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$CLUSTER_HOST" true 2>/dev/null; then
    echo "!!! Can't reach '$CLUSTER_HOST' over SSH."
    echo "    - Make sure '$CLUSTER_HOST' is defined in ~/.ssh/config (see top of this script)."
    echo "    - If your cluster requires it, connect to its network / VPN first."
    exit 1
fi

echo ">>> Locating your VTSearch job on the cluster..."
# One round-trip returns the per-user PORT (computed on the cluster with the
# same UID formula vtsearch-slurm.sh uses, so it matches what the app bound to)
# followed by the running job's NODE. Port first so an empty node still parses.
read -r REMOTE_PORT NODE < <(ssh -o BatchMode=yes "$CLUSTER_HOST" \
    'node=$(squeue --me -h -o "%j %N %T" | awk "\$1==\"vtsearch\" && \$3==\"RUNNING\" {print \$2; exit}"); echo "$((10000 + $(id -u) % 20000)) $node"')

if [ -z "$NODE" ]; then
    echo "!!! No RUNNING VTSearch job found."
    echo "    Start it on the cluster first:  ssh $CLUSTER_HOST   then   vtsearch"
    echo "    (If it's still queuing, wait for it to start, then re-run this.)"
    exit 1
fi

# VTS_PORT overrides locally (set it to match if you overrode it on the
# cluster); otherwise use the cluster-computed per-user port.
PORT=${VTS_PORT:-$REMOTE_PORT}

# The local end of the forward. VTS_BIND unset keeps ssh's default (loopback
# only); set, it prefixes the spec as `bind:port:node:port` so another device on
# that network can reach the app through this machine.
FORWARD="${VTS_BIND:+$VTS_BIND:}$PORT:$NODE:$PORT"
BROWSE_HOST=${VTS_BIND:-localhost}

echo ">>> VTSearch is on $NODE:$PORT. Forwarding $BROWSE_HOST:$PORT -> $NODE:$PORT."
if [ "$NO_SHELL" -eq 1 ]; then
    echo ">>> Browse  http://$BROWSE_HOST:$PORT   (tunnel only; no shell)"
    # -N: no remote command. Nothing to keep alive but the forward itself, which
    # is what a service wants -- and the keepalives let a dead link be noticed in
    # a minute rather than hanging until TCP gives up.
    exec ssh -N -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
        -L "$FORWARD" "$CLUSTER_HOST"
fi

echo ">>> Browse  http://$BROWSE_HOST:$PORT   (Ctrl+C here closes the tunnel)"
echo ">>> Dropping you into the VTSearch dir on the login node for git/edits."
# -t: interactive TTY. The remote command cd's into the project (same shared
# filesystem the compute node sees) and starts a login shell, so this terminal
# is both the tunnel and a ready-to-use git workspace.
exec ssh -t -L "$FORWARD" "$CLUSTER_HOST" \
    "cd \"$VTS_DIR\" 2>/dev/null; exec bash -l"
