"""Tests for the .claude/hooks/require-issue-labels.py PreToolUse gate.

The hook enforces CLAUDE.md's "Label every issue you file" rule at tool-call
time. It is the only mechanical check on that rule -- there is no CI, and
`run-tests.sh` never sees a GitHub issue -- so its two failure directions both
matter and are tested here:

* a miss (allowing an unlabeled issue) silently contaminates the human-issue
  view, which is the regression that produced issue #3127;
* a false block (rejecting an unrelated GitHub call) would wedge ordinary work,
  so every non-create, non-issue payload must pass straight through.

Both directions are tested twice over, because the rule has two call paths:
`mcp__github__issue_write` and `gh issue create` run through `Bash`. The `gh`
path went unwatched for weeks while being the only one this repo's sessions
actually used, so `TestGhCli` carries the shell shapes those sessions really
emit -- compound commands, heredoc bodies, `--body-file`, `ssh grid '...'` --
rather than a tidy flag list that would pass without proving anything.
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


class TestSolvedLabelOnClose:
    """`solved` means "development done, only merges remain", so a close must strip it.

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

    def test_completed_close_carrying_solved_is_blocked(self):
        result = run_hook(self.close(state="closed", state_reason="completed", labels=["claude", "solved"]))
        assert result.returncode == BLOCK
        assert "KEEPS `solved` ON A CLOSED ISSUE" in result.stderr

    def test_completed_close_without_explicit_labels_is_blocked(self):
        result = run_hook(self.close(state="closed", state_reason="completed"))
        assert result.returncode == BLOCK
        assert "CLOSE DOES NOT STRIP `solved`" in result.stderr

    def test_the_denial_warns_that_labels_replaces_the_whole_set(self):
        """Passing `[]` to satisfy the hook would silently wipe `claude`."""
        result = run_hook(self.close(state="closed", state_reason="completed"))
        assert "REPLACES the whole set" in result.stderr

    def test_completed_close_with_labels_minus_solved_is_allowed(self):
        assert run_hook(self.close(state="closed", state_reason="completed", labels=["claude"])).returncode == ALLOW

    def test_an_issue_with_no_labels_left_can_still_be_closed(self):
        assert run_hook(self.close(state="closed", state_reason="completed", labels=[])).returncode == ALLOW

    def test_solved_is_blocked_on_any_close_not_just_completed(self):
        """A not_planned close carrying `solved` is just as false a statement."""
        result = run_hook(self.close(state="closed", state_reason="not_planned", labels=["claude", "solved"]))
        assert result.returncode == BLOCK
        assert "KEEPS `solved` ON A CLOSED ISSUE" in result.stderr

    def test_non_completed_close_does_not_require_explicit_labels(self):
        """Only the release sweep must strip; a not_planned close need not restate labels."""
        assert run_hook(self.close(state="closed", state_reason="not_planned")).returncode == ALLOW

    def test_solved_label_matching_is_case_insensitive(self):
        result = run_hook(self.close(state="closed", state_reason="completed", labels=["claude", "SOLVED"]))
        assert result.returncode == BLOCK

    def test_create_path_rules_do_not_leak_onto_closes(self):
        """A close needs no `claude` label -- the issue may well be a human's."""
        assert (
            run_hook(self.close(state="closed", state_reason="completed", labels=["enhancement"])).returncode == ALLOW
        )


class TestGhCli:
    """`gh issue create` through Bash -- the path the sessions here actually take.

    The shapes below are taken from real transcripts, not invented: bodies
    arrive as `$(cat <<'EOF' ... EOF)` heredocs, commands are chained with
    `&&`, some run inside `ssh grid '...'`, and some pass `--body-file` for a
    body that never appears in the command at all.
    """

    @staticmethod
    def bash(command: str) -> dict:
        return {"tool_name": "Bash", "tool_input": {"command": command}}

    def run(self, command: str) -> subprocess.CompletedProcess:
        return run_hook(self.bash(command))

    def test_unlabeled_create_is_blocked(self):
        result = self.run(f'gh issue create --title "A title" --body "{PLAIN_BODY}"')
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_the_denial_names_the_flag_not_the_api_field(self):
        """The caller is holding a shell command; `labels: [...]` is no help to them."""
        result = self.run('gh issue create --title "A title" --body "x"')
        assert "--label claude" in result.stderr

    def test_claude_label_is_enough_for_a_plain_issue(self):
        assert self.run(f'gh issue create --title "T" --body "{PLAIN_BODY}" --label claude').returncode == ALLOW

    @pytest.mark.parametrize(
        "flag",
        ["--label claude", "--label=claude", "-l claude", '--label "claude"', "--label 'claude'"],
        ids=["space", "equals", "short", "double-quoted", "single-quoted"],
    )
    def test_every_flag_spelling_is_recognised(self, flag):
        assert self.run(f'gh issue create --title "T" --body "{PLAIN_BODY}" {flag}').returncode == ALLOW

    @pytest.mark.parametrize(
        "flag",
        ["--label claude,experiment", '--label "claude, experiment"', "--label claude --label experiment"],
        ids=["comma", "comma-spaced-quoted", "repeated"],
    )
    def test_both_labels_in_every_combining_form(self, flag):
        assert self.run(f'gh issue create --title "T" --body "{EXPERIMENT_BODY}" {flag}').returncode == ALLOW

    def test_experiment_shaped_body_without_the_label_is_blocked(self):
        result = self.run(f'gh issue create --title "T" --body "{EXPERIMENT_BODY}" --label claude')
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr
        assert "MISSING `claude`" not in result.stderr

    def test_a_heredoc_body_is_read_by_the_heuristic(self):
        """The real shape: the body is a heredoc inside the command string."""
        command = (
            'gh issue create --repo samggreenberg/VTSearch --title "Widen the sweep" '
            f"--body \"$(cat <<'EOF'\n{EXPERIMENT_BODY}\nEOF\n)\" --label claude"
        )
        result = self.run(command)
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr

    def test_a_body_file_on_disk_is_read_back(self, tmp_path):
        body = tmp_path / "issue_body.md"
        body.write_text(EXPERIMENT_BODY)
        result = self.run(f'gh issue create --title "T" --body-file {body} --label claude')
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr

    def test_a_body_file_that_does_not_resolve_still_enforces_claude(self):
        """`$SP/issue.md` and GRID-side paths cannot be read; the label rule still binds."""
        result = self.run('gh issue create --title "T" --body-file $SP/issue.md')
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_a_body_file_that_does_not_resolve_does_not_crash_the_heuristic(self):
        assert self.run('gh issue create --title "T" --body-file $SP/issue.md --label claude').returncode == ALLOW

    def test_a_compound_command_is_still_policed(self):
        command = f'gh issue create --title "T" --body "{PLAIN_BODY}"'
        assert self.run(command).returncode == BLOCK

    @pytest.mark.parametrize(
        "prefix",
        ["", "timeout 120 ", "cd /tmp && ", "nohup ", "cat body.md | "],
        ids=["bare", "timeout", "chained", "nohup", "piped"],
    )
    def test_every_invocation_shape_is_policed(self, prefix):
        assert self.run(f'{prefix}gh issue create --title "T" --body "{PLAIN_BODY}"').returncode == BLOCK

    def test_a_create_after_a_heredoc_on_its_own_line_is_policed(self):
        command = f"cat > /tmp/body.md <<'EOF'\n{PLAIN_BODY}\nEOF\ngh issue create --title \"T\" -F /tmp/body.md"
        assert self.run(command).returncode == BLOCK

    def test_a_create_inside_ssh_quotes_is_still_policed(self):
        command = 'timeout 300 ssh grid \'cd /exp/sgreenberg/projects/vts && gh issue create --title "T" --body "x"\''
        assert self.run(command).returncode == BLOCK

    def test_add_label_does_not_satisfy_a_create(self):
        """A follow-up edit is not a label at creation time -- and often never happens."""
        command = f'gh issue create --title "T" --body "{PLAIN_BODY}" && gh issue edit 1 --add-label claude'
        result = self.run(command)
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_opt_out_marker_works_from_the_command_line_too(self):
        body = f"{EXPERIMENT_BODY} <!-- not-an-experiment: measured already in #3077 -->"
        assert self.run(f'gh issue create --title "T" --body "{body}" --label claude').returncode == ALLOW

    @pytest.mark.parametrize(
        "command",
        [
            "gh issue create --help",
            "gh issue create -h",
            "gh issue create --web",
            'gh issue create --web --title "T"',
        ],
        ids=["help", "short-help", "web", "web-with-title"],
    )
    def test_non_filing_invocations_are_not_policed(self, command):
        """`--help` files nothing, and is what you type *after* the denial message.

        Blocking it turns the denial into a dead end -- the hook tells you to
        add `--label` and then refuses to let you look up the flag. `--web`
        hands the form to a browser, where the CLI's flags do not apply.
        """
        assert self.run(command).returncode == ALLOW

    def test_the_smoke_probe_shape_is_blocked(self):
        """`false && gh issue create ...` is the live-session probe for this hook.

        The lesson this change exists to fix is that a gate is worth nothing
        until it has been observed to fire. This shape is the honest way to
        watch it: the hook sees a command-position `gh issue create` and blocks,
        and if the hook were ever broken the shell would still run nothing,
        because `false &&` short-circuits. No issue can be filed either way.
        """
        assert self.run('false && gh issue create --title "probe" --body "probe"').returncode == BLOCK

    def test_bare_bash_payload_is_still_policed(self):
        result = run_hook({"command": f'gh issue create --title "T" --body "{PLAIN_BODY}"'})
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr


class TestGhPassthrough:
    """A Bash hook sees every command in the session, so it must be near-silent."""

    @staticmethod
    def run(command: str) -> subprocess.CompletedProcess:
        return run_hook({"tool_name": "Bash", "tool_input": {"command": command}})

    @pytest.mark.parametrize(
        "command",
        [
            "./run-tests.sh",
            "python -m pytest tests_lib/meta -q",
            "git commit -m 'gh issue create is mentioned only in this message'",
            "gh issue view 3127 --json labels",
            "gh issue list --label claude",
            "gh issue comment 3127 --body 'Addressed in #3130'",
            "gh issue edit 3127 --add-label solved",
            "gh pr create --base dev --title 'T' --body 'B'",
            "echo 'gh issue created earlier today'",
        ],
        ids=[
            "run-tests",
            "pytest",
            "commit-message",
            "issue-view",
            "issue-list",
            "issue-comment",
            "issue-edit",
            "pr-create",
            "prose-mention",
        ],
    )
    def test_unrelated_commands_pass_straight_through(self, command):
        assert self.run(command).returncode == ALLOW

    @pytest.mark.parametrize(
        "command",
        [
            "git commit -m 'Teach the hook about gh issue create'",
            'gh pr create --title "T" --body "This makes gh issue create enforce labels"',
            "grep -rn 'gh issue create' docs/",
            'echo "run gh issue create with --label claude" >> docs/RELEASE.md',
            "# gh issue create --title 'T' --body 'B'",
            "echo hi\n# gh issue create --title 'T' --body 'B'",
        ],
        ids=["commit-message", "pr-body", "grep", "doc-line", "commented-out", "commented-out-line"],
    )
    def test_merely_mentioning_the_command_is_not_running_it(self, command):
        """The hook must not block work *about* the rule -- including this PR.

        Every one of these is a real command from the change that added this
        test. A hook that blocked them would make its own repo unworkable.
        """
        assert self.run(command).returncode == ALLOW

    def test_issue_close_is_not_policed_here(self):
        """`gh issue close` has the same `solved`-strip hole, tracked separately.

        The MCP path blocks a `completed` close that does not restate its
        labels; `gh issue close` cannot restate them at all, so the equivalent
        rule needs its own design rather than a guess bolted on here.
        """
        assert self.run("gh issue close 3319 --reason completed").returncode == ALLOW


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

    @staticmethod
    def _matcher(name: str) -> dict:
        settings = json.loads((HOOK.parents[1] / "settings.json").read_text())
        return next(h for h in settings["hooks"]["PreToolUse"] if h["matcher"] == name)

    def test_registered_as_a_pretooluse_hook_for_issue_write(self):
        entry = self._matcher("mcp__github__issue_write")
        assert any(HOOK.name in hook["command"] for hook in entry["hooks"])

    def test_registered_as_a_pretooluse_hook_for_bash(self):
        """The `gh` path is the one this repo's sessions actually use.

        Registering the hook only for the MCP tool is what let 29 unlabeled
        issues through: the file existed, the tests passed, and the matcher
        named a tool nobody was calling.
        """
        entry = self._matcher("Bash")
        assert any(HOOK.name in hook["command"] for hook in entry["hooks"])

    def test_bash_registration_keeps_the_dep_gate(self):
        """Two hooks share the Bash matcher; adding one must not evict the other."""
        entry = self._matcher("Bash")
        assert any("ensure-test-deps-gate" in hook["command"] for hook in entry["hooks"])
