#!/bin/bash
# Find your running VTSearch GPU job on the cluster and forward its port to your
# local machine. Run this LOCALLY, AFTER starting VTSearch on the cluster (the
# `vtsearch` command from scripts/slurm/vtsearch-slurm.sh).
#
# Browse http://localhost:5000 once it connects. Ctrl+C closes the tunnel.
#
# Requires an SSH host entry for the cluster login node. By default this script
# connects to the host alias `cluster`; define it in ~/.ssh/config, e.g.
#
#     Host cluster
#         HostName login.your-cluster.edu
#         User your-cluster-username
#
# or point this script at a different alias with:  CLUSTER_HOST=mycluster vtsearch-tunnel
set -u

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
NODE=$(ssh -o BatchMode=yes "$CLUSTER_HOST" \
    'squeue --me -h -o "%j %N %T" | awk "\$1==\"vtsearch\" && \$3==\"RUNNING\" {print \$2; exit}"')

if [ -z "$NODE" ]; then
    echo "!!! No RUNNING VTSearch job found."
    echo "    Start it on the cluster first:  ssh $CLUSTER_HOST   then   vtsearch"
    echo "    (If it's still queuing, wait for it to start, then re-run this.)"
    exit 1
fi

echo ">>> VTSearch is on $NODE. Forwarding localhost:5000 -> $NODE:5000."
echo ">>> Browse  http://localhost:5000   (Ctrl+C here closes the tunnel)"
echo ">>> Dropping you into the VTSearch dir on the login node for git/edits."
# -t: interactive TTY. The remote command cd's into the project (same shared
# filesystem the compute node sees) and starts a login shell, so this terminal
# is both the tunnel and a ready-to-use git workspace.
exec ssh -t -L 5000:"$NODE":5000 "$CLUSTER_HOST" \
    "cd \"$VTS_DIR\" 2>/dev/null; exec bash -l"
