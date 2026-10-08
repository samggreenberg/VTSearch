"""Tests for .claude/hooks/ensure-test-deps.sh, the cloud container's installer.

Issue #4662: the cloud image shipped with `python` on 3.11 and `pip` on 3.13.
The hook installed with a bare `pip install`, so the whole stack landed in 3.13
while every documented `python ...` command ran on 3.11 and died on
`No module named 'flask'`. The hook's marker then made it worse: it was written
after the (wrong-interpreter) install, so every later run exited early and
re-running the hook changed nothing.

The installer is exercised for real, against stubs: a `python` on `PATH` that
records its `-m pip` calls and answers the import probe from a state file, and
a `pip` / `pip3` that records any call at all. No package is installed, and the
stack only "imports" once the stubbed `python` has installed the requirements,
which is the property the hook must now check rather than assume.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
INSTALLER = HOOKS / "ensure-test-deps.sh"

pytestmark = pytest.mark.skipif(shutil.which("sha256sum") is None, reason="the hook hashes itself with sha256sum")

# Answers `python -m pip ...` with success, and marks the stack importable once
# the requirements file goes in -- unless the test says the install is broken.
# The import probe (`python -c 'import flask, ...'`) succeeds only after that.
STUB_PYTHON = """#!/bin/bash
echo "python $*" >> "$STUB_LOG"
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  if [[ " $* " == *"requirements/base.txt"* ]] && [ -z "${STUB_INSTALL_IS_BROKEN:-}" ]; then
    touch "$STUB_STATE/importable"
  fi
  exit 0
fi
if [ "$1" = "-c" ] && [[ "$2" == *"import flask"* ]]; then
  [ -f "$STUB_STATE/importable" ]
  exit $?
fi
exit 0
"""

STUB_BARE_PIP = """#!/bin/bash
echo "bare-pip $*" >> "$STUB_LOG"
exit 0
"""


class Sandbox:
    def __init__(self, root: Path):
        self.root = root
        self.bin = root / "bin"
        self.state = root / "state"
        self.markers = root / "markers"
        self.log = root / "calls.log"
        for d in (self.bin, self.state, self.markers):
            d.mkdir()
        for name, body in (("python", STUB_PYTHON), ("pip", STUB_BARE_PIP), ("pip3", STUB_BARE_PIP)):
            exe = self.bin / name
            exe.write_text(body)
            exe.chmod(0o755)
        self.log.touch()
        script_hash = hashlib.sha256(INSTALLER.read_bytes()).hexdigest()[:12]
        self.marker = self.markers / f".vtsearch-deps-installed-{script_hash}"

    def run(self, **extra_env: str) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "CLAUDE_CODE_REMOTE": "true",
            "VTSEARCH_DEPS_MARKER_DIR": str(self.markers),
            "STUB_LOG": str(self.log),
            "STUB_STATE": str(self.state),
            **extra_env,
        }
        # cwd has no frontend/, so the hook's npm step is skipped.
        return subprocess.run(  # noqa: S603  # bash + repo-local hook path, no shell
            ["bash", str(INSTALLER)],  # noqa: S607
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines()

    def pip_installs(self) -> list[str]:
        return [c for c in self.calls() if c.startswith("python -m pip install")]

    def bare_pip_calls(self) -> list[str]:
        return [c for c in self.calls() if c.startswith("bare-pip")]


@pytest.fixture
def sandbox(tmp_path) -> Sandbox:
    return Sandbox(tmp_path)


class TestInstallsIntoThePythonInUse:
    def test_every_install_goes_through_python_m_pip(self, sandbox):
        result = sandbox.run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert sandbox.bare_pip_calls() == [], "a bare `pip` installs into whatever interpreter it belongs to"
        assert any("requirements/base.txt" in c for c in sandbox.pip_installs())
        assert sandbox.marker.exists()

    def test_the_user_site_upgrade_goes_through_python_m_pip_too(self, sandbox):
        """The stub enables the user site, so the one conditional install runs as well."""
        sandbox.run()
        assert any(" --user " in c for c in sandbox.pip_installs())


class TestMarker:
    def test_a_marker_whose_stack_does_not_import_is_not_trusted(self, sandbox):
        """The issue's trap: a marker from another interpreter short-circuited every re-run."""
        sandbox.marker.touch()
        result = sandbox.run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert sandbox.pip_installs(), "a marker over an unimportable stack skipped the install"
        assert "reinstalling" in result.stdout

    def test_a_marker_whose_stack_imports_skips_the_install(self, sandbox):
        sandbox.marker.touch()
        (sandbox.state / "importable").touch()
        result = sandbox.run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert sandbox.pip_installs() == []

    def test_an_install_that_leaves_the_stack_unimportable_writes_no_marker(self, sandbox):
        result = sandbox.run(STUB_INSTALL_IS_BROKEN="1")
        assert result.returncode != 0
        assert "do not import" in result.stderr
        assert not sandbox.marker.exists(), "the next run would trust a marker over a broken install"


# `pip install` / `pip3 install` / `pip3.13 install` not preceded by `-m `.
_BARE_PIP_INSTALL = re.compile(r"(?<!-m )\bpip(?:3(?:\.\d+)?)? install\b")


@pytest.mark.parametrize("hook", sorted(HOOKS.glob("*.sh")), ids=lambda p: p.name)
def test_no_hook_installs_with_a_bare_pip(hook: Path):
    """Covers the hooks the behavioural tests above do not run (session-start.sh)."""
    offenders = [
        f"{hook.name}:{n}: {line.strip()}"
        for n, line in enumerate(hook.read_text().splitlines(), 1)
        if not line.lstrip().startswith("#") and _BARE_PIP_INSTALL.search(line)
    ]
    assert offenders == [], "install with `python -m pip`, not a bare `pip` (issue #4662):\n" + "\n".join(offenders)
