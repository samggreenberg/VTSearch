"""Tests for scripts/reconcile-solved-labels.py.

The script decides which open issues have no development left in them -- the
problem is solved and only git merges remain -- and therefore carry the
`solved` label. It reuses docs/RELEASE.md step 6's resolution logic, whose sharp edges
are the point of testing it:

* a closing keyword means resolved; `Refs` / `Part of` / a bare `#N` do not;
* `Partially addressed in #M` is the documented marker for work still owed,
  and must not read as `Addressed in #M`;
* a comment posted *after* a fix pointer is ambiguous -- it may be chatter or
  a dispute -- and the script must surface it rather than guess either way;
* a pointer naming a *commit* rather than a PR cannot be resolved from this
  input, and must be surfaced rather than reported as settled;
* an *open* fix PR resolves an issue just as a merged one does (the label
  tracks "solved", not "merged"), while a PR closed *without* merging un-solves
  it and must take the label back off.

It also reconciles the *assignee*, which carries the companion status "a
session is working this right now". That half removes only: a solved or closed
issue must end up unassigned, but nothing in this input can distinguish "nobody
is on it" from "a session just started", so an assignment is never invented and
an ambiguous issue is never touched.

Getting any of these wrong silently corrupts both views the label powers:
`-label:solved` (what a human should pick up next) and `label:solved`
(solved, waiting only on merges).
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reconcile-solved-labels.py"


def _load():
    spec = importlib.util.spec_from_file_location("reconcile_solved_labels", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load()


def issue(
    number: int,
    *,
    state: str = "open",
    labels: list[str] | None = None,
    comments: list[str] | None = None,
    assignees: list | None = None,
):
    return {
        "number": number,
        "state": state,
        "labels": labels or [],
        "assignees": assignees or [],
        "comments": [{"body": body} for body in (comments or [])],
    }


def plan_for(
    prs: list[dict],
    issues: list[dict],
    *,
    open_prs: list[dict] | None = None,
    abandoned_prs: list[dict] | None = None,
) -> dict[str, dict[int, str]]:
    """Reconcile, then flatten each bucket to {issue number: reason}."""
    raw = mod.reconcile(
        {
            "release_prs": prs,
            "open_prs": open_prs or [],
            "abandoned_prs": abandoned_prs or [],
            "issues": issues,
        }
    )
    return {action: {number: reason for number, reason in entries} for action, entries in raw.items()}


class TestClosingKeywords:
    """Only a closing keyword means the issue's development is finished."""

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
        assert 3077 in plan_for(prs, [issue(3077, labels=["claude", "solved"])])["none"]

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


class TestCommitShaPointers:
    """A fix claimed by commit SHA is surfaced, never reported as settled.

    Regression cover for #2911, which shipped to `main` with its only pointer
    reading "Fixed on `dev` by `de9ae81ac`". That matched no pattern, so the
    script reported it as unresolved -- indistinguishable from an issue
    nobody had started -- and the release sweep had nothing to go on.
    """

    PRS = [{"number": 3128, "body": "Refs #3077"}]

    @pytest.mark.parametrize(
        "comment",
        [
            "Fixed on `dev` by `de9ae81ac`",
            "Fixed in de9ae81ac",
            "Addressed by commit `de9ae81acaafb9e267996e6096ff9b59c140b8e8`",
            "Resolved on dev by de9ae81a",
        ],
    )
    def test_sha_pointer_is_flagged_for_review(self, comment):
        result = plan_for(self.PRS, [issue(3077, comments=[comment])])
        assert 3077 in result["review"]
        assert 3077 not in result["none"]

    def test_the_review_reason_names_the_commit(self):
        result = plan_for(self.PRS, [issue(3077, comments=["Fixed on `dev` by `de9ae81ac`"])])
        assert "de9ae81ac" in result["review"][3077]

    def test_a_proper_pr_pointer_still_resolves_and_is_not_downgraded(self):
        """The `#M` form is the documented one; it must not be dragged into review."""
        assert 3077 in plan_for(self.PRS, [issue(3077, comments=["Addressed in #3128"])])["add"]

    def test_partially_addressed_by_commit_does_not_resolve_either(self):
        """The `(?<!ly )` guard applies to the SHA form too."""
        comment = "Partially addressed by `de9ae81ac` — still open: the video arm."
        assert 3077 in plan_for(self.PRS, [issue(3077, comments=[comment])])["none"]

    def test_prose_mentioning_a_commit_without_a_fix_claim_is_ignored(self):
        """Not every SHA in a comment is a resolution."""
        comment = "This regressed somewhere around de9ae81ac, still bisecting."
        assert 3077 in plan_for(self.PRS, [issue(3077, comments=[comment])])["none"]

    def test_a_newer_pr_pointer_beats_an_older_sha_pointer(self):
        comments = ["Fixed on `dev` by `de9ae81ac`", "Addressed in #3128"]
        assert 3077 in plan_for(self.PRS, [issue(3077, comments=comments)])["add"]

    def test_a_closed_issue_with_a_sha_pointer_is_not_flagged(self):
        """Closed is closed; the backstop only guards the open pile."""
        result = plan_for(self.PRS, [issue(3077, state="closed", comments=["Fixed by `de9ae81ac`"])])
        assert 3077 in result["none"]

    def test_check_exits_nonzero_so_the_claim_cannot_pass_silently(self):
        data = {"release_prs": self.PRS, "issues": [issue(3077, comments=["Fixed on `dev` by `de9ae81ac`"])]}
        result = subprocess.run(  # noqa: S603  # interpreter + repo-local script path, no shell
            [sys.executable, str(SCRIPT), "--check"],
            input=json.dumps(data),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 1


class TestOpenFixPrs:
    """The label tracks "solved", not "merged", so an open fix PR resolves an issue.

    This is the widened front edge: the label used to require a merge into `dev`,
    which no session ever observes, so nothing applied the label in practice.
    From a planning perspective an issue solved in an open PR has exactly as
    little left to think about as one already merged.
    """

    def test_an_open_pr_closing_an_issue_resolves_it(self):
        result = plan_for([], [issue(3081)], open_prs=[{"number": 3160, "body": "Closes #3081"}])
        assert 3081 in result["add"]

    def test_the_reason_distinguishes_an_open_pr_from_a_merged_one(self):
        result = plan_for([], [issue(3081)], open_prs=[{"number": 3160, "body": "Closes #3081"}])
        assert "open PR #3160" in result["add"][3081]

    def test_a_comment_pointing_at_an_open_pr_resolves_it(self):
        """The `Addressed in #M` comment lands before the merge, and counts."""
        result = plan_for(
            [],
            [issue(3081, comments=["Addressed in #3160"])],
            open_prs=[{"number": 3160, "body": "Refs #3081"}],
        )
        assert 3081 in result["add"]

    def test_a_non_closing_open_pr_still_does_not_resolve(self):
        """Widening which PRs count does not weaken which keywords count."""
        result = plan_for([], [issue(3081)], open_prs=[{"number": 3160, "body": "Refs #3081"}])
        assert 3081 in result["none"]


class TestAbandonedFixPrs:
    """A fix PR closed without merging un-solves its issue, so the label comes off.

    This removal case only exists because the label goes on before the merge:
    under the old "merged into `dev`" meaning there was no window in which a
    labelled issue could lose its fix.
    """

    ABANDONED = [{"number": 3155, "body": "Closes #3090"}]

    def test_labelled_issue_whose_fix_pr_was_abandoned_loses_the_label(self):
        result = plan_for([], [issue(3090, labels=["claude", "solved"])], abandoned_prs=self.ABANDONED)
        assert 3090 in result["remove"]

    def test_the_removal_reason_names_the_abandoned_pr(self):
        result = plan_for([], [issue(3090, labels=["claude", "solved"])], abandoned_prs=self.ABANDONED)
        assert "#3155" in result["remove"][3090]

    def test_an_unlabelled_issue_with_an_abandoned_fix_needs_no_change(self):
        """It is already in the human queue, which is where it belongs."""
        result = plan_for([], [issue(3090, labels=["claude"])], abandoned_prs=self.ABANDONED)
        assert 3090 in result["none"]

    def test_an_abandoned_comment_pointer_also_removes(self):
        result = plan_for(
            [],
            [issue(3090, labels=["solved"], comments=["Addressed in #3155"])],
            abandoned_prs=[{"number": 3155, "body": "Refs #3090"}],
        )
        assert 3090 in result["remove"]

    def test_a_superseding_open_pr_beats_the_abandoned_one(self):
        """Take two of a fix keeps the issue solved; the dead claim must not win."""
        result = plan_for(
            [],
            [issue(3090, labels=["claude", "solved"])],
            open_prs=[{"number": 3170, "body": "Closes #3090"}],
            abandoned_prs=self.ABANDONED,
        )
        assert 3090 in result["none"]
        assert 3090 not in result["remove"]

    def test_a_superseding_merged_pr_beats_the_abandoned_one(self):
        result = plan_for(
            [{"number": 3180, "body": "Closes #3090"}],
            [issue(3090)],
            abandoned_prs=self.ABANDONED,
        )
        assert 3090 in result["add"]

    def test_an_abandoned_pr_does_not_flag_an_unrelated_labelled_issue_for_review(self):
        """Only the issue the dead PR claimed is affected."""
        result = plan_for(
            [{"number": 3180, "body": "Closes #3077"}],
            [issue(3077, labels=["solved"])],
            abandoned_prs=self.ABANDONED,
        )
        assert 3077 in result["none"]


class TestRemovalAndStaleness:
    """The label is transient, so the script reconciles in both directions."""

    def test_closed_issue_still_carrying_solved_is_flagged_for_removal(self):
        prs = [{"number": 3128, "body": "Closes #3077"}]
        result = plan_for(prs, [issue(3077, state="closed", labels=["claude", "solved"])])
        assert 3077 in result["remove"]

    def test_closed_issue_without_the_label_needs_no_change(self):
        assert 3077 in plan_for([], [issue(3077, state="closed", labels=["claude"])])["none"]

    def test_open_issue_labelled_but_unresolved_is_flagged_for_review(self):
        """Stale from a prior release, or the label was applied by hand in error."""
        assert 3077 in plan_for([], [issue(3077, labels=["solved"])])["review"]

    def test_solved_label_matching_is_case_insensitive(self):
        prs = [{"number": 3128, "body": "Closes #3077"}]
        assert 3077 in plan_for(prs, [issue(3077, labels=["SOLVED"])])["none"]


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


class TestAssigneeRemoval:
    """The assignee comes off once an issue is solved or closed -- and only then."""

    def test_issue_solved_by_this_run_is_unassigned(self):
        prs = [{"number": 3128, "body": "Closes #3077"}]
        plan = plan_for(prs, [issue(3077, assignees=["samggreenberg"])])
        assert 3077 in plan["add"]
        assert "@samggreenberg" in plan["unassign"][3077]

    def test_already_solved_issue_still_assigned_is_unassigned(self):
        """The label went on but the assignee was forgotten -- the common miss."""
        prs = [{"number": 3128, "body": "Closes #3077"}]
        plan = plan_for(prs, [issue(3077, labels=["claude", "solved"], assignees=["samggreenberg"])])
        assert 3077 in plan["none"]  # label is already correct...
        assert 3077 in plan["unassign"]  # ...but the assignee is not

    def test_closed_issue_is_unassigned(self):
        plan = plan_for([], [issue(3077, state="closed", assignees=["samggreenberg"])])
        assert 3077 in plan["unassign"]

    def test_unsolved_issue_keeps_its_assignee(self):
        """An assignee on an unsolved issue means a session is working it now."""
        plan = plan_for([], [issue(3077, assignees=["samggreenberg"])])
        assert 3077 not in plan["unassign"]

    def test_refs_only_issue_keeps_its_assignee(self):
        prs = [{"number": 3128, "body": "Refs #3077"}]
        assert 3077 not in plan_for(prs, [issue(3077, assignees=["samggreenberg"])])["unassign"]

    def test_unassigned_issue_is_never_planned_for_assignment(self):
        """The script only ever removes -- it cannot know who, if anyone, is working."""
        prs = [{"number": 3128, "body": "Closes #3077"}]
        plan = plan_for(prs, [issue(3077)])
        assert plan["unassign"] == {}

    def test_ambiguous_issue_is_left_alone(self):
        """A `NEEDS REVIEW` label verdict must not drag the assignee with it."""
        plan = plan_for([], [issue(3077, labels=["solved"], assignees=["samggreenberg"])])
        assert 3077 in plan["review"]
        assert 3077 not in plan["unassign"]

    def test_fallen_through_fix_leaves_the_assignee_alone(self):
        """The label comes off, but whoever is picking the work back up may be assigned."""
        plan = plan_for(
            [],
            [issue(3077, labels=["solved"], assignees=["samggreenberg"])],
            abandoned_prs=[{"number": 3155, "body": "Closes #3077"}],
        )
        assert 3077 in plan["remove"]
        assert 3077 not in plan["unassign"]

    @pytest.mark.parametrize(
        "assignees", [["samggreenberg"], [{"login": "samggreenberg"}]], ids=["mcp-strings", "rest-objects"]
    )
    def test_both_assignee_payload_shapes_are_understood(self, assignees):
        """MCP tools hand back logins; the REST API hands back objects."""
        plan = plan_for([{"number": 3128, "body": "Closes #3077"}], [issue(3077, assignees=assignees)])
        assert "@samggreenberg" in plan["unassign"][3077]

    def test_multiple_assignees_are_all_named(self):
        plan = plan_for(
            [{"number": 3128, "body": "Closes #3077"}], [issue(3077, assignees=["samggreenberg", "Khamersk"])]
        )
        assert "@samggreenberg" in plan["unassign"][3077]
        assert "@Khamersk" in plan["unassign"][3077]

    def test_empty_and_missing_assignees_are_both_quiet(self):
        prs = [{"number": 3128, "body": "Closes #3077"}]
        assert plan_for(prs, [{"number": 3077, "state": "open", "labels": [], "comments": []}])["unassign"] == {}

    def test_check_flags_an_owed_unassignment_alone(self):
        """A tree whose labels are all correct still fails the gate on a stale assignee."""
        data = {
            "release_prs": [{"number": 3128, "body": "Closes #3077"}],
            "issues": [issue(3077, labels=["solved"], assignees=["samggreenberg"])],
        }
        assert mod.reconcile(data)["add"] == []
        assert mod.reconcile(data)["unassign"] != []

    def test_render_names_the_bucket(self):
        plan = mod.reconcile(
            {
                "release_prs": [{"number": 3128, "body": "Closes #3077"}],
                "issues": [issue(3077, assignees=["samggreenberg"])],
            }
        )
        assert "CLEAR ASSIGNEE" in mod.render(plan)
