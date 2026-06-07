"""Single-instance lock: a second acquire for the same port is refused,
and the lock is released once the holder exits (no stale lock)."""

import os
import subprocess
import sys
import textwrap

from vtscore import single_instance


def test_second_acquire_same_port_raises(tmp_path, monkeypatch):
    # Set the rundir in our OWN env so both this process and the child below
    # resolve the same lockfile (the child inherits this env).
    monkeypatch.setenv("VTSEARCH_RUNDIR", str(tmp_path))
    handle = single_instance.acquire(5099)
    try:
        # flock is per-open-file-description, so a true second *process* is
        # needed to observe the conflict.
        code = textwrap.dedent(
            """
            import sys
            from vtscore import single_instance
            try:
                single_instance.acquire(5099)
            except single_instance.AlreadyRunningError as exc:
                print("HOLDER", exc.holder_pid)
                sys.exit(7)
            sys.exit(0)
            """
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 7, result.stderr
        assert str(os.getpid()) in result.stdout
    finally:
        handle.close()


def test_lock_released_after_holder_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("VTSEARCH_RUNDIR", str(tmp_path))
    # A short-lived process takes the lock and exits without explicit release.
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", "from vtscore import single_instance; single_instance.acquire(5098)"],
        check=True,
    )
    # No stale lock left behind: we can acquire the same port now.
    handle = single_instance.acquire(5098)
    handle.close()
