"""Tests for the .claude/hooks/require-issue-labels.py PreToolUse gate.

The hook enforces CLAUDE.md's "Label every issue you file" rule at tool-call
time. It is the only mechanical check on that rule -- there is no CI, and
`run-tests.sh` never sees a GitHub issue -- so its two failure directions both
matter and are tested here:

* a miss (allowing an unlabeled issue) silently contaminates the human-issue
  view, which is the regression that produced issue #3127;
* a false block (rejecting an unrelated GitHub call) would wedge ordinary work,
  so every non-create, non-issue payload must pass straight through.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "require-issue-labels.py"

ALLOW = 0
BLOCK = 2

# An experiment-shaped body: two weak signals ("calibration", "measure") and
# no strong one, so it also covers the >=2-weak-signals branch.
EXPERIMENT_BODY = "Re-run the calibration arm and measure mAP against the fold-anchored threshold."
PLAIN_BODY = "The Back button in the importer modal is left-aligned but should use .back-btn."


def run_hook(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603  # interpreter + repo-local hook path, no shell
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )


def create(body: str = PLAIN_BODY, labels: list[str] | None = None, **overrides) -> dict:
    args = {"method": "create", "owner": "samggreenberg", "repo": "vtsearch", "title": "A title", "body": body}
    if labels is not None:
        args["labels"] = labels
    args.update(overrides)
    return {"tool_name": "mcp__github__issue_write", "tool_input": args}


class TestClaudeLabel:
    """`claude` is mechanically decidable: Claude is making the call."""

    def test_create_without_any_labels_is_blocked(self):
        result = run_hook(create())
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_create_with_unrelated_labels_is_blocked(self):
        result = run_hook(create(labels=["bug", "enhancement"]))
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_create_with_claude_label_is_allowed(self):
        assert run_hook(create(labels=["claude"])).returncode == ALLOW

    def test_label_matching_is_case_and_whitespace_insensitive(self):
        assert run_hook(create(labels=[" Claude "])).returncode == ALLOW


class TestExperimentLabel:
    """`experiment` is a judgment call, so the hook blocks heuristically."""

    def test_experiment_shaped_body_without_the_label_is_blocked(self):
        result = run_hook(create(body=EXPERIMENT_BODY, labels=["claude"]))
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr
        assert "MISSING `claude`" not in result.stderr

    def test_a_single_strong_signal_is_enough(self):
        body = "Add a new arm to `python -m vtscore.eval` for the region-voting path."
        result = run_hook(create(body=body, labels=["claude"]))
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr

    def test_experiment_shaped_body_with_the_label_is_allowed(self):
        assert run_hook(create(body=EXPERIMENT_BODY, labels=["claude", "experiment"])).returncode == ALLOW

    def test_the_heuristic_reads_the_title_too(self):
        result = run_hook(create(title="Re-run the GRID sweep", body=PLAIN_BODY, labels=["claude"]))
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr

    def test_a_single_weak_signal_does_not_block(self):
        body = "The progress bar should measure elapsed time, not step count."
        assert run_hook(create(body=body, labels=["claude"])).returncode == ALLOW

    def test_opt_out_marker_releases_the_heuristic(self):
        body = f"{EXPERIMENT_BODY}\n\n<!-- not-an-experiment: the numbers already exist in #3077 -->"
        assert run_hook(create(body=body, labels=["claude"])).returncode == ALLOW

    def test_opt_out_marker_does_not_release_the_claude_label(self):
        body = f"{EXPERIMENT_BODY}\n\n<!-- not-an-experiment: already measured -->"
        result = run_hook(create(body=body))
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_both_problems_are_reported_together(self):
        """One round-trip must be enough to fix both labels."""
        result = run_hook(create(body=EXPERIMENT_BODY))
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr
        assert "MISSING `experiment`" in result.stderr


class TestPassthrough:
    """A hook that fails closed would wedge unrelated GitHub work."""

    def test_updates_are_never_blocked(self):
        """Relabeling an existing issue -- including a human's -- must pass."""
        payload = create(body=EXPERIMENT_BODY, method="update", issue_number=3127)
        payload["tool_input"].pop("labels", None)
        assert run_hook(payload).returncode == ALLOW

    def test_other_tools_are_ignored(self):
        payload = {"tool_name": "mcp__github__create_pull_request", "tool_input": {"method": "create", "repo": "x"}}
        assert run_hook(payload).returncode == ALLOW

    @pytest.mark.parametrize(
        "raw",
        ["", "   ", "not json at all", "[]", "null", '"a string"'],
        ids=["empty", "blank", "garbage", "list", "null", "string"],
    )
    def test_unparseable_payloads_allow(self, raw):
        result = subprocess.run(  # noqa: S603  # interpreter + repo-local hook path, no shell
            [sys.executable, str(HOOK)], input=raw, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == ALLOW

    def test_bare_argument_payload_is_still_policed(self):
        """Some harness versions pass the arguments dict without an envelope."""
        result = run_hook({"method": "create", "owner": "samggreenberg", "repo": "vtsearch", "body": PLAIN_BODY})
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr


class TestWiring:
    """The hook is inert unless settings.json actually points at it."""

    def test_hook_file_exists(self):
        assert HOOK.is_file()

    def test_registered_as_a_pretooluse_hook_for_issue_write(self):
        settings = json.loads((HOOK.parents[1] / "settings.json").read_text())
        matchers = settings["hooks"]["PreToolUse"]
        entry = next(h for h in matchers if h["matcher"] == "mcp__github__issue_write")
        assert any(HOOK.name in hook["command"] for hook in entry["hooks"])
