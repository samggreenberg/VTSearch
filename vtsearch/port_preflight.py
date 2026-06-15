"""Startup port-collision preflight for ``python app.py``.

A common dev annoyance is leaving an old ``python app.py`` running and only
discovering it when a fresh launch can't bind :5000. These helpers detect a
prior listener, identify its PID via ``/proc`` (Linux, best-effort, no extra
deps), and let the user kill it before we try to bind.

Extracted from ``app.py`` so the WSGI ``app`` object gunicorn imports stays
free of this CLI-only concern; only the ``__main__`` launch path (via
``vtsearch.cli_main``) calls :func:`_acquire_single_instance_lock` and
:func:`_preflight_port`.
"""

import os
import sys


def _port_is_free(port: int) -> bool:
    """Return True if nothing is bound to ``0.0.0.0:port``.

    Probes with ``SO_REUSEADDR``, matching how werkzeug actually binds.  On
    Linux an active LISTEN socket still surfaces as ``EADDRINUSE`` (only
    ``SO_REUSEPORT`` would mask it), but orphaned ``TIME_WAIT`` remnants - the
    parent-less sockets a ``kill -9``'d server leaves behind for ~60s - do
    not.  Without the flag those remnants made the preflight refuse a restart
    that werkzeug's own bind would have happily performed.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _listen_inodes_for_port(port: int) -> "set[str]":
    """Socket inodes of LISTEN sockets bound to ``port`` (from ``/proc/net``)."""
    port_hex = f"{port:04X}"
    inodes: set[str] = set()
    for proc_net in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_net) as fh:
                next(fh, None)  # header row
                for line in fh:
                    fields = line.split()
                    # fields[3] == "0A" is TCP_LISTEN; fields[1] is HEXADDR:HEXPORT.
                    if len(fields) >= 10 and fields[3] == "0A" and fields[1].rsplit(":", 1)[-1] == port_hex:
                        inodes.add(fields[9])
        except OSError:
            continue
    return inodes


def _fd_points_to(fd_dir: str, fd: str, targets: "set[str]") -> bool:
    """True if ``fd_dir/fd`` is a symlink to one of the ``socket:[inode]`` targets."""
    try:
        return os.readlink(f"{fd_dir}/{fd}") in targets
    except OSError:
        return False


def _pids_owning_inodes(inodes: "set[str]") -> "list[int]":
    """Scan ``/proc/<pid>/fd`` for any process holding one of ``inodes``."""
    pids: list[int] = []
    targets = {f"socket:[{inode}]" for inode in inodes}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue  # process gone or not ours to inspect
        if any(_fd_points_to(fd_dir, fd, targets) for fd in fds):
            pids.append(int(entry))
    return pids


def _find_listener_pids(port: int) -> "list[int]":
    """Best-effort PIDs holding a LISTEN socket on ``port`` (Linux ``/proc``).

    Returns an empty list if the port is free or if PID resolution fails (e.g.
    ``/proc`` unavailable). Detection of *whether* the port is taken is done
    separately via :func:`_port_is_free`; this only enriches the warning.
    """
    inodes = _listen_inodes_for_port(port)
    return _pids_owning_inodes(inodes) if inodes else []


def _describe_pid(pid: int) -> str:
    """A short ``pid (cmdline)`` label, falling back to just the PID."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        cmdline = ""
    return f"{pid} ({cmdline})" if cmdline else str(pid)


def _wait_for_free(port: int, deadline: float = 5.0) -> bool:
    """Poll until ``port`` is free or ``deadline`` seconds elapse; return final state."""
    import time

    waited = 0.0
    while waited < deadline and not _port_is_free(port):
        time.sleep(0.2)
        waited += 0.2
    return _port_is_free(port)


def _terminate_listeners(pids: "list[int]", port: int) -> bool:
    """SIGTERM the listeners, escalate to SIGKILL if needed; return True once free."""
    import signal

    def _signal_all(sig: int) -> None:
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    _signal_all(signal.SIGTERM)
    if _wait_for_free(port):
        return True
    _signal_all(signal.SIGKILL)
    return _wait_for_free(port)


def _acquire_single_instance_lock(port: int):
    """Fail fast if another ``python app.py`` is already running on ``port``.

    Closes the gap :func:`_preflight_port` cannot: during the first instance's
    minute-long model load the port is not bound yet, so a second launch in
    that window slips past the port check and loads the model stack (~17 GB) a
    second time, OOM-killing the job. The lock here is taken before any model
    load, so the duplicate dies in milliseconds. Returns a handle the caller
    must hold for the process lifetime.
    """
    from vtscore.single_instance import AlreadyRunningError, acquire

    try:
        return acquire(port)
    except AlreadyRunningError as exc:
        print(
            f"⛔ {exc}. Refusing to start a second copy, which would reload "
            f"the model stack and risk OOM-killing the job. Stop the first "
            f"instance (e.g. `fuser -k {port}/tcp`), or set VTSEARCH_RUNDIR to "
            f"isolate this one.",
            flush=True,
        )
        sys.exit(1)


def _preflight_port(port: int) -> None:
    """Warn-and-prompt if ``port`` is already bound before we try to bind it.

    If the user agrees, SIGTERM (then SIGKILL) the prior listener and wait for
    the port to free. Non-interactive stdin or a declined prompt exits rather
    than letting ``app.run`` crash with an opaque ``Address already in use``.
    """
    if _port_is_free(port):
        return

    pids = _find_listener_pids(port)
    if pids:
        listed = ", ".join(_describe_pid(p) for p in pids)
        print(f"⚠️  Port {port} is already in use by PID {listed}.", flush=True)
    else:
        print(f"⚠️  Port {port} is already in use (could not identify the owning process).", flush=True)

    if not pids or not sys.stdin.isatty():
        # Nothing safe to kill, or no TTY to ask on: don't guess, just bail.
        print(f"   Free port {port} and try again (e.g. `fuser -k {port}/tcp`).", flush=True)
        sys.exit(1)

    reply = input(f"   Kill PID {', '.join(str(p) for p in pids)} and continue? [y/N] ").strip().lower()
    if reply not in ("y", "yes"):
        print("   Leaving the existing instance alone; not starting.", flush=True)
        sys.exit(1)

    if not _terminate_listeners(pids, port):
        print(f"   Could not free port {port}; aborting.", flush=True)
        sys.exit(1)
    print(f"   Freed port {port}; starting up.", flush=True)
