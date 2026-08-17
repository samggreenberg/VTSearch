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
    args: dict = {"method": "create", "owner": "samggreenberg", "repo": "vtsearch", "title": "A title", "body": body}
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


class TestDevLabelOnClose:
    """`dev` means "on `dev`, NOT on `main`", so a close must strip it.

    The hook only ever sees the call's arguments, never the issue's current
    state, so the enforceable form is "a completing close must state its label
    set explicitly" -- that being the only shape of the call that provably
    strips the label.
    """

    @staticmethod
    def close(**overrides) -> dict:
        args = {"method": "update", "owner": "samggreenberg", "repo": "vtsearch", "issue_number": 3077}
        args.update(overrides)
        return {"tool_name": "mcp__github__issue_write", "tool_input": args}

    def test_completed_close_carrying_dev_is_blocked(self):
        result = run_hook(self.close(state="closed", state_reason="completed", labels=["claude", "dev"]))
        assert result.returncode == BLOCK
        assert "KEEPS `dev` ON A CLOSED ISSUE" in result.stderr

    def test_completed_close_without_explicit_labels_is_blocked(self):
        result = run_hook(self.close(state="closed", state_reason="completed"))
        assert result.returncode == BLOCK
        assert "CLOSE DOES NOT STRIP `dev`" in result.stderr

    def test_the_denial_warns_that_labels_replaces_the_whole_set(self):
        """Passing `[]` to satisfy the hook would silently wipe `claude`."""
        result = run_hook(self.close(state="closed", state_reason="completed"))
        assert "REPLACES the whole set" in result.stderr

    def test_completed_close_with_labels_minus_dev_is_allowed(self):
        assert run_hook(self.close(state="closed", state_reason="completed", labels=["claude"])).returncode == ALLOW

    def test_an_issue_with_no_labels_left_can_still_be_closed(self):
        assert run_hook(self.close(state="closed", state_reason="completed", labels=[])).returncode == ALLOW

    def test_dev_is_blocked_on_any_close_not_just_completed(self):
        """A not_planned close carrying `dev` is just as false a statement."""
        result = run_hook(self.close(state="closed", state_reason="not_planned", labels=["claude", "dev"]))
        assert result.returncode == BLOCK
        assert "KEEPS `dev` ON A CLOSED ISSUE" in result.stderr

    def test_non_completed_close_does_not_require_explicit_labels(self):
        """Only the release sweep must strip; a not_planned close need not restate labels."""
        assert run_hook(self.close(state="closed", state_reason="not_planned")).returncode == ALLOW

    def test_dev_label_matching_is_case_insensitive(self):
        result = run_hook(self.close(state="closed", state_reason="completed", labels=["claude", "DEV"]))
        assert result.returncode == BLOCK

    def test_create_path_rules_do_not_leak_onto_closes(self):
        """A close needs no `claude` label -- the issue may well be a human's."""
        assert run_hook(self.close(state="closed", state_reason="completed", labels=["enhancement"])).returncode == ALLOW


class TestPassthrough:
    """A hook that fails closed would wedge unrelated GitHub work."""

    def test_non_close_updates_are_never_blocked(self):
        """Relabeling an existing issue -- including a human's -- must pass.

        Only a *close* is policed on the update path; an edit that does not
        touch `state` is none of the hook's business, and the create-path
        label rules must not leak onto it.
        """
        payload = create(body=EXPERIMENT_BODY, method="update", issue_number=3127)
        payload["tool_input"].pop("labels", None)
        assert run_hook(payload).returncode == ALLOW

    def test_reopening_is_not_policed(self):
        payload = create(method="update", issue_number=3127, state="open")
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
