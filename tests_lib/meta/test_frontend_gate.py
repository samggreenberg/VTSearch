"""The frontend-gate decision (#4019): when may run-tests.sh skip, and when must it block?

The bug this guards is not a gate that broke -- it is a gate that never ran
while the banner still read `RUN PASSED (all gates green)`. PR #4018 was run
through the suite in a fresh worktree, skipped the frontend build and 2,446
Vitest specs on a frontend change, and reported every gate green.

So the cases that matter are the two directions: a frontend diff with no
`node_modules` must block, and a backend-only diff in the same worktree must
still be allowed to skip. Both are exercised against real git repositories
rather than a stubbed diff, because the premise being tested is precisely
"what does the diff say", and a synthesized answer would re-encode the
assumption instead of checking it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[2] / "scripts" / "check-frontend-gate.py"


def git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603  # fixed argv, no shell
        ["git", *args],  # noqa: S607  # git resolves on PATH
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo with an origin/dev to diff against, and no frontend/node_modules."""
    r = tmp_path / "wt"
    r.mkdir()
    git(r, "init", "-q", "-b", "dev")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / "vtscore").mkdir()
    (r / "vtscore" / "core.py").write_text("x = 1\n")
    (r / "frontend").mkdir()
    (r / "frontend" / "angular.json").write_text("{}\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "base")
    # A local ref standing in for origin/dev; merge-base reads it the same way.
    git(r, "update-ref", "refs/remotes/origin/dev", "HEAD")
    git(r, "checkout", "-q", "-b", "feature")
    return r


def decide(repo: Path) -> tuple[str, str]:
    out = subprocess.run(  # noqa: S603  # this interpreter + a repo-local gate path
        [sys.executable, str(GATE), "--repo", str(repo)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip(), out.stderr


class TestTheTwoDirections:
    def test_a_frontend_change_without_node_modules_blocks(self, repo: Path) -> None:
        (repo / "frontend" / "src").mkdir()
        (repo / "frontend" / "src" / "app.ts").write_text("export const a = 1;\n")
        decision, why = decide(repo)
        assert decision == "block"
        assert "frontend/src/app.ts" in why
        assert "npm install" in why

    def test_a_backend_change_without_node_modules_may_skip(self, repo: Path) -> None:
        (repo / "vtscore" / "core.py").write_text("x = 2\n")
        decision, _ = decide(repo)
        assert decision == "skip"

    def test_node_modules_present_always_runs(self, repo: Path) -> None:
        (repo / "frontend" / "node_modules").mkdir()
        (repo / "frontend" / "src").mkdir()
        (repo / "frontend" / "src" / "app.ts").write_text("export const a = 1;\n")
        assert decide(repo)[0] == "run"


class TestWhatCountsAsAChange:
    def test_committed_staged_unstaged_and_untracked_all_count(self, repo: Path) -> None:
        # Untracked counts for the reason run-tests.sh gives: a stray new file
        # would be gated by a full run, so it must not ride along unchecked.
        (repo / "frontend" / "new.ts").write_text("export const b = 2;\n")
        assert decide(repo)[0] == "block"

        git(repo, "add", "-A")
        assert decide(repo)[0] == "block"

        git(repo, "commit", "-qm", "frontend work")
        assert decide(repo)[0] == "block"

    def test_a_frontend_change_is_found_among_backend_ones(self, repo: Path) -> None:
        (repo / "vtscore" / "core.py").write_text("x = 3\n")
        (repo / "frontend" / "src").mkdir()
        (repo / "frontend" / "src" / "app.ts").write_text("export const a = 1;\n")
        decision, why = decide(repo)
        assert decision == "block"
        # Only the frontend paths are named; the backend edit is not the reason.
        assert "frontend/src/app.ts" in why
        assert "vtscore/core.py" not in why

    def test_a_path_merely_containing_frontend_does_not_count(self, repo: Path) -> None:
        (repo / "docs").mkdir()
        (repo / "docs" / "frontend-notes.md").write_text("# notes\n")
        assert decide(repo)[0] == "skip"


class TestFailClosed:
    def test_no_origin_dev_blocks_rather_than_assuming_backend_only(self, repo: Path) -> None:
        git(repo, "update-ref", "-d", "refs/remotes/origin/dev")
        decision, why = decide(repo)
        assert decision == "block"
        assert "origin/dev" in why

    def test_an_empty_diff_still_skips(self, repo: Path) -> None:
        # Nothing changed at all: the gates have nothing to judge either way.
        assert decide(repo)[0] == "skip"
