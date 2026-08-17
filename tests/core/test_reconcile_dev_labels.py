"""Tests for scripts/reconcile-dev-labels.py.

The script decides which open issues are "fixed on `dev`, not yet on `main`"
and therefore carry the `dev` label. It reuses docs/RELEASE.md step 6's
resolution logic, whose sharp edges are the point of testing it:

* a closing keyword means resolved; `Refs` / `Part of` / a bare `#N` do not;
* `Partially addressed in #M` is the documented marker for work still owed,
  and must not read as `Addressed in #M`;
* a comment posted *after* a fix pointer is ambiguous -- it may be chatter or
  a dispute -- and the script must surface it rather than guess either way.

Getting any of these wrong corrupts the awaiting-release view silently, which
is exactly the failure mode the script exists to prevent.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reconcile-dev-labels.py"


def _load():
    spec = importlib.util.spec_from_file_location("reconcile_dev_labels", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load()


def issue(number: int, *, state: str = "open", labels: list[str] | None = None, comments: list[str] | None = None):
    return {
        "number": number,
        "state": state,
        "labels": labels or [],
        "comments": [{"body": body} for body in (comments or [])],
    }


def plan_for(prs: list[dict], issues: list[dict]) -> dict[str, dict[int, str]]:
    """Reconcile, then flatten each bucket to {issue number: reason}."""
    raw = mod.reconcile({"release_prs": prs, "issues": issues})
    return {action: {number: reason for number, reason in entries} for action, entries in raw.items()}


class TestClosingKeywords:
    """Only a closing keyword means the issue is finished on `dev`."""

    @pytest.mark.parametrize("keyword", ["Closes", "closes", "Fixes", "fixed", "Resolves", "RESOLVED"])
    def test_closing_keywords_resolve(self, keyword):
        prs = [{"number": 3128, "body": f"Some work.\n\n{keyword} #3077"}]
        assert 3077 in plan_for(prs, [issue(3077)])["add"]

    @pytest.mark.parametrize("keyword", ["Refs", "Part of", "See", "Related to"])
    def test_non_closing_keywords_do_not_resolve(self, keyword):
        """A genuine `Refs` is doing its job: work is still owed on that issue."""
        prs = [{"number": 3128, "body": f"{keyword} #3077"}]
        assert 3077 in plan_for(prs, [issue(3077)])["none"]

    def test_bare_mention_does_not_resolve(self):
        prs = [{"number": 3128, "body": "Follow-up to #3077 (its 'watch out for' note)."}]
        assert 3077 in plan_for(prs, [issue(3077)])["none"]

    def test_multiple_issues_in_one_body_each_resolve(self):
        """CLAUDE.md mandates one keyword per issue, not a comma-list."""
        prs = [{"number": 3128, "body": "Closes #12, closes #15"}]
        result = plan_for(prs, [issue(12), issue(15)])
        assert set(result["add"]) == {12, 15}

    def test_already_labelled_issue_needs_no_change(self):
        prs = [{"number": 3128, "body": "Closes #3077"}]
        assert 3077 in plan_for(prs, [issue(3077, labels=["claude", "dev"])])["none"]

    def test_the_reason_names_the_pr(self):
        prs = [{"number": 3128, "body": "Closes #3077"}]
        assert "#3128" in plan_for(prs, [issue(3077)])["add"][3077]


class TestCommentPointers:
    """The orphan backstop: a PR that under-claimed, caught via issue comments."""

    PRS = [{"number": 3128, "body": "Refs #3077"}]

    def test_addressed_in_pointer_resolves(self):
        result = plan_for(self.PRS, [issue(3077, comments=["Addressed in #3128"])])
        assert 3077 in result["add"]

    def test_partially_addressed_does_not_resolve(self):
        """CLAUDE.md's marker for work still owed must not read as resolution."""
        comment = "Partially addressed in #3128 — still open: the video arm."
        assert 3077 in plan_for(self.PRS, [issue(3077, comments=[comment])])["none"]

    def test_pointer_at_a_pr_outside_the_release_does_not_resolve(self):
        """A fix shipped in an earlier release is already on `main`."""
        assert 3077 in plan_for(self.PRS, [issue(3077, comments=["Addressed in #2001"])])["none"]

    def test_issue_with_no_comments_is_untouched(self):
        assert 3077 in plan_for(self.PRS, [issue(3077)])["none"]


class TestMostRecentCommentRule:
    """A comment after the pointer is ambiguous, so it is surfaced, not guessed."""

    PRS = [{"number": 3128, "body": "Refs #3077"}]

    def test_pointer_as_newest_comment_resolves(self):
        comments = ["I'll take a look.", "Addressed in #3128"]
        assert 3077 in plan_for(self.PRS, [issue(3077, comments=comments)])["add"]

    def test_a_comment_after_the_pointer_flags_for_review(self):
        """The dispute case: the reporter says it is not actually fixed."""
        comments = ["Addressed in #3128", "This didn't actually fix it — still repros on audio."]
        result = plan_for(self.PRS, [issue(3077, comments=comments)])
        assert 3077 in result["review"]
        assert 3077 not in result["add"]

    def test_harmless_chatter_also_flags_rather_than_vanishing_the_issue(self):
        """The false-negative case a strict newest-comment rule would hide."""
        result = plan_for(self.PRS, [issue(3077, comments=["Addressed in #3128", "Thanks!"])])
        assert 3077 in result["review"]
        assert 3077 not in result["none"]

    def test_the_review_reason_says_how_many_comments_followed(self):
        comments = ["Addressed in #3128", "Thanks!", "Confirmed."]
        assert "2 later comment" in plan_for(self.PRS, [issue(3077, comments=comments)])["review"][3077]

    def test_a_later_re_pointer_wins(self):
        """Fixed, disputed, then fixed again: the newest pointer is the truth."""
        comments = ["Addressed in #3128", "Reopening, still broken.", "Addressed in #3128 (take two)"]
        assert 3077 in plan_for(self.PRS, [issue(3077, comments=comments)])["add"]

    def test_a_closing_pr_body_beats_a_disputed_comment_thread(self):
        """An explicit `Closes` in the PR is a stronger claim than comment order."""
        prs = [{"number": 3128, "body": "Closes #3077"}]
        comments = ["Addressed in #3128", "Thanks!"]
        assert 3077 in plan_for(prs, [issue(3077, comments=comments)])["add"]


class TestRemovalAndStaleness:
    """The label is transient, so the script reconciles in both directions."""

    def test_closed_issue_still_carrying_dev_is_flagged_for_removal(self):
        prs = [{"number": 3128, "body": "Closes #3077"}]
        result = plan_for(prs, [issue(3077, state="closed", labels=["claude", "dev"])])
        assert 3077 in result["remove"]

    def test_closed_issue_without_the_label_needs_no_change(self):
        assert 3077 in plan_for([], [issue(3077, state="closed", labels=["claude"])])["none"]

    def test_open_issue_labelled_but_unresolved_is_flagged_for_review(self):
        """Stale from a prior release, or the label was applied by hand in error."""
        assert 3077 in plan_for([], [issue(3077, labels=["dev"])])["review"]

    def test_dev_label_matching_is_case_insensitive(self):
        prs = [{"number": 3128, "body": "Closes #3077"}]
        assert 3077 in plan_for(prs, [issue(3077, labels=["DEV"])])["none"]


class TestCommandLine:
    def run(self, data: dict, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603  # interpreter + repo-local script path, no shell
            [sys.executable, str(SCRIPT), *args],
            input=json.dumps(data),
            capture_output=True,
            text=True,
            timeout=30,
        )

    RESOLVED = {"release_prs": [{"number": 3128, "body": "Closes #3077"}], "issues": [issue(3077)]}

    def test_reads_stdin_and_reports(self):
        result = self.run(self.RESOLVED)
        assert result.returncode == 0
        assert "#3077" in result.stdout

    def test_check_exits_nonzero_on_drift(self):
        assert self.run(self.RESOLVED, "--check").returncode == 1

    def test_check_exits_zero_when_settled(self):
        settled = {"release_prs": [], "issues": [issue(3077, labels=["claude"])]}
        assert self.run(settled, "--check").returncode == 0

    def test_json_output_is_machine_readable(self):
        payload = json.loads(self.run(self.RESOLVED, "--json").stdout)
        assert payload["add"] == [{"number": 3077, "reason": payload["add"][0]["reason"]}]
        assert "#3128" in payload["add"][0]["reason"]

    def test_invalid_json_is_rejected_with_a_clear_error(self):
        result = subprocess.run(  # noqa: S603  # interpreter + repo-local script path, no shell
            [sys.executable, str(SCRIPT)], input="not json", capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 2
        assert "not valid JSON" in result.stderr

    def test_non_object_input_is_rejected(self):
        result = subprocess.run(  # noqa: S603  # interpreter + repo-local script path, no shell
            [sys.executable, str(SCRIPT)], input="[]", capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 2

    def test_empty_input_is_not_an_error(self):
        assert self.run({}).returncode == 0
