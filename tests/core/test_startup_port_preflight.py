"""Tests for the startup port-collision preflight helpers in ``app.py``.

These cover the detection + safe-exit paths (``_port_is_free``,
``_find_listener_pids``, the non-TTY bail) and the real SIGTERM/SIGKILL
``_terminate_listeners`` path against a throwaway child process. They use
ephemeral ports so they never collide with a real instance on :5000.
"""

import io
import os
import socket
import subprocess
import sys
import time

import pytest

import app as app_module


def _listening_socket():
    """Bind+listen on an OS-assigned ephemeral port; return ``(sock, port)``."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


def _free_ephemeral_port():
    """Reserve then release an ephemeral port, returning its number."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("0.0.0.0", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_port_is_free_reflects_binding():
    sock, port = _listening_socket()
    try:
        assert app_module._port_is_free(port) is False
    finally:
        sock.close()
    assert app_module._port_is_free(port) is True


def test_port_is_free_ignores_orphaned_time_wait():
    """Sockets a dead server leaves in TIME_WAIT must not read as "in use".

    A ``kill -9``'d server's accepted connections linger in TIME_WAIT (local
    port == the listen port, owned by no process) for ~60s. werkzeug binds
    with ``SO_REUSEADDR`` and would start fine, so the probe - which matches
    that bind - must report the port free rather than block the restart.
    Closing the server side of the connection first parks its socket in
    TIME_WAIT, surviving all three ``close()`` calls below.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    conn, _ = server.accept()
    conn.close()  # server closes first -> this socket heads to TIME_WAIT
    client.close()
    server.close()
    assert app_module._port_is_free(port) is True


def test_find_listener_pids_identifies_self():
    sock, port = _listening_socket()
    try:
        pids = app_module._find_listener_pids(port)
    finally:
        sock.close()
    assert os.getpid() in pids


def test_find_listener_pids_empty_for_free_port():
    port = _free_ephemeral_port()
    assert app_module._find_listener_pids(port) == []


def test_preflight_returns_when_port_free():
    port = _free_ephemeral_port()
    # No listener: must return without prompting or exiting.
    app_module._preflight_port(port)


def test_preflight_bails_when_occupied_and_no_tty(monkeypatch):
    sock, port = _listening_socket()
    # A non-interactive stdin (StringIO.isatty() is False) must never prompt;
    # the preflight should print guidance and exit(1) instead of hanging.
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    try:
        with pytest.raises(SystemExit) as excinfo:
            app_module._preflight_port(port)
        assert excinfo.value.code == 1
    finally:
        sock.close()


def test_terminate_listeners_frees_port():
    port = _free_ephemeral_port()
    code = f"import socket, time; s = socket.socket(); s.bind(('0.0.0.0', {port})); s.listen(1); time.sleep(30)"
    proc = subprocess.Popen([sys.executable, "-c", code])  # noqa: S603  # interpreter + constant literal code
    try:
        # Wait until the child is actually holding the port (poll, don't sleep-guess).
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and app_module._port_is_free(port):
            time.sleep(0.05)
        assert not app_module._port_is_free(port), "child never bound the port"

        assert app_module._terminate_listeners([proc.pid], port) is True
        assert app_module._port_is_free(port)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
