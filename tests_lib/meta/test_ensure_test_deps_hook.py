"""Tests for the .claude/hooks/ensure-test-deps-gate.py PreToolUse gate.

The gate installs the project's stack before a command that needs it, and its
predecessor -- an inline `echo "$TOOL_INPUT" | grep ...` in settings.json --
never fired once: the harness delivers the tool call on stdin and sets no
`TOOL_INPUT` variable, so the grep matched against an empty string forever
(issue #3440). `run-tests.sh` calls the installer directly, which masked the
failure and left a dead safety net looking alive.

These are a regression guard, not the verification: a synthesized payload can
only re-encode whatever contract the test author assumed, which is the exact
mistake that produced the bug. The contract was pinned by watching a live
session -- see `.claude/hooks/hook_payload.py`. What is checkable here is that
the gate reads the channel the harness actually uses, that it stays wired up in
settings.json, and that it never blocks a tool call.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
GATE = HOOKS / "ensure-test-deps-gate.py"
INSTALLER = HOOKS / "ensure-test-deps.sh"

ALLOW = 0
# A PreToolUse hook blocks on exit 2. The gate must never return it: a
# dependency problem should surface as the command's own error.
BLOCK = 2


@pytest.fixture
def stub_installer(tmp_path):
    """Stand in for ensure-test-deps.sh, recording whether the gate ran it."""
    receipt = tmp_path / "installed"
    script = tmp_path / "stub-installer.sh"
    script.write_text(f"#!/bin/bash\ntouch {receipt}\n")
    return script, receipt


def run_gate(payload, installer: Path, *, on_stdin: bool = True) -> subprocess.CompletedProcess:
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    env = {**os.environ, "VTSEARCH_DEPS_INSTALLER": str(installer)}
    if not on_stdin:
        env["TOOL_INPUT"] = raw
        raw = ""
    return subprocess.run(  # noqa: S603  # interpreter + repo-local hook path, no shell
        [sys.executable, str(GATE)], input=raw, capture_output=True, text=True, timeout=60, env=env
    )


def bash(command: str) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": command, "description": "a description"}}


class TestPayloadDelivery:
    """The bug was a wrong guess about *where* the payload arrives."""

    def test_a_stdin_payload_triggers_the_install(self, stub_installer):
        script, receipt = stub_installer
        assert run_gate(bash("python -m pytest tests/ -q"), script).returncode == ALLOW
        assert receipt.exists(), "the gate did not read the payload the harness actually sends"

    def test_an_env_payload_still_triggers_the_install(self, stub_installer):
        """The env fallback is vestigial, but it must not rot silently."""
        script, receipt = stub_installer
        assert run_gate(bash("python -m pytest tests/ -q"), script, on_stdin=False).returncode == ALLOW
        assert receipt.exists()

    def test_bare_argument_payload_is_still_matched(self, stub_installer):
        """Some harness versions pass the arguments dict without an envelope."""
        script, receipt = stub_installer
        assert run_gate({"command": "npm run build:prod"}, script).returncode == ALLOW
        assert receipt.exists()


class TestMatching:
    @pytest.mark.parametrize(
        "command",
        [
            "python -m pytest tests/ tests_lib/ -q",
            "cd frontend && npm run build:prod",
            "cd frontend && npm test",
            "ng serve --port 4200",
            "ng test",
        ],
    )
    def test_commands_that_need_the_stack_install_it(self, stub_installer, command):
        script, receipt = stub_installer
        assert run_gate(bash(command), script).returncode == ALLOW
        assert receipt.exists()

    @pytest.mark.parametrize("command", ["git status", "ls -la docs/", "ruff check .", "echo hello"])
    def test_unrelated_commands_do_not_install(self, stub_installer, command):
        script, receipt = stub_installer
        assert run_gate(bash(command), script).returncode == ALLOW
        assert not receipt.exists()


class TestPassthrough:
    """A hook that fails closed would wedge ordinary work."""

    def test_other_tools_are_ignored(self, stub_installer):
        script, receipt = stub_installer
        payload = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/pytest.log"}}
        assert run_gate(payload, script).returncode == ALLOW
        assert not receipt.exists()

    @pytest.mark.parametrize(
        "raw",
        ["", "   ", "not json at all", "[]", "null", '"a string"'],
        ids=["empty", "blank", "garbage", "list", "null", "string"],
    )
    def test_unparseable_payloads_allow(self, stub_installer, raw):
        script, receipt = stub_installer
        assert run_gate(raw, script).returncode == ALLOW
        assert not receipt.exists()

    def test_a_failing_installer_does_not_block_the_command(self, tmp_path):
        """Exit 1 is a visible hook error; exit 2 would refuse to run the command."""
        script = tmp_path / "broken-installer.sh"
        script.write_text("#!/bin/bash\necho 'pip exploded' >&2\nexit 1\n")
        result = run_gate(bash("python -m pytest -q"), script)
        assert result.returncode != BLOCK
        assert "failed" in result.stderr


class TestWiring:
    """The gate is inert unless settings.json actually points at it."""

    def test_gate_and_installer_both_exist(self):
        assert GATE.is_file()
        assert INSTALLER.is_file(), "the gate's default installer path must resolve"

    def test_registered_as_a_pretooluse_hook_for_bash(self):
        settings = json.loads((HOOKS.parent / "settings.json").read_text())
        entry = next(h for h in settings["hooks"]["PreToolUse"] if h["matcher"] == "Bash")
        assert any(GATE.name in hook["command"] for hook in entry["hooks"])

    def test_no_hook_command_reads_the_env_var_the_harness_never_sets(self):
        """The original defect, in the shape it took: an inline `$TOOL_INPUT` matcher."""
        settings = (HOOKS.parent / "settings.json").read_text()
        assert "TOOL_INPUT" not in settings
