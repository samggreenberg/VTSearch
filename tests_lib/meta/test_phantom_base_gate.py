"""The stale-tree gate itself: `scripts/check-phantom-base.py`.

The gate exists because five PRs silently reverted the work that merged just
before them, and nothing noticed for as long as two days (#2821's revert
reached `main`). Every one of them shared a shape that the obvious checks miss:
the merge-base was `origin/dev`'s tip -- zero commits behind -- while the
*worktree* held pre-merge content, so the commit recorded a deletion for every
file `dev` had gained meanwhile.

So the tests below pin the properties that make the gate able to see that, each
of which a plausible simpler implementation would get wrong:

* it fires on a fresh-parent/stale-tree branch, and names the ancestor the
  content actually came from -- that ancestor is what a restore needs, since
  the clobbering commit also carries work worth keeping;
* it sees a clobber that has not been committed yet, because that is how the
  damage exists first and `run-tests.sh` may run before the commit;
* it stays quiet on a deliberate deletion of a long-standing file, which no
  ancestor can explain away;
* it stays quiet on a rename scoring between 40% and 50%, which git's default
  threshold reports as a deletion (the one such false positive in the
  historical sweep scored 47%);
* it never leans on the *size* of the deletion, because two of the four real
  clobbers deleted only 2 and 3 files;
* the deliberate-deletion override releases it, since `CLAUDE.md` requires
  deleting a plan file when it ships.

The gate carries a second signal for the half of the damage deletions cannot
show (#3210): a clobber whose reverted hunks sit inside files that survive.
That one fires on a run of two or more consecutive base commits the branch
keeps *nothing* of, so the tests below also pin why it is two and not one --
reverting a single merge is something people do deliberately -- and that it
needs an actual revert, not merely a branch that rewrote the same files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-phantom-base.py"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", *args],  # noqa: S607 - git resolved from PATH
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _run_gate(repo: Path, base_ref: str = "base", **env_extra: str):
    """Run the gate inside `repo`, returning (exit_code, output)."""
    import os

    env = {**os.environ, "VTSEARCH_BASE_REF": base_ref, **env_extra}
    env.pop("VTSEARCH_ALLOW_DELETIONS", None)
    env.pop("VTSEARCH_ALLOW_REVERTS", None)
    env.update(env_extra)
    proc = subprocess.run(  # noqa: S603  # interpreter + repo-local script path, no shell
        [sys.executable, str(SCRIPT)],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo whose `base` branch has advanced three commits past `first`."""
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q", "-b", "base", ".")
    (r / "kept.txt").write_text("kept\n")
    (r / "ancient.txt").write_text("ancient\n")
    _commit(r, "first")
    (r / "landed_a.txt").write_text("feature a\n")
    _commit(r, "land A")
    (r / "landed_b.txt").write_text("feature b\n")
    _commit(r, "land B")
    return r


def _stale_checkout(repo: Path, drop: list[str]) -> None:
    """Reproduce the mechanism: fresh HEAD, worktree missing what base gained."""
    _git(repo, "checkout", "-q", "-b", "work")
    for name in drop:
        (repo / name).unlink()
    (repo / "my_feature.txt").write_text("the legitimate change\n")


class TestFiresOnAClobber:
    def test_flags_a_stale_tree_committed_onto_a_fresh_parent(self, repo: Path):
        _stale_checkout(repo, ["landed_a.txt", "landed_b.txt"])
        _commit(repo, "my feature")

        code, out = _run_gate(repo)

        assert code == 1
        assert "deletes 2 file(s) that it never created" in out
        assert "landed_a.txt" in out
        assert "landed_b.txt" in out
        # The branch's own work must not be reported as a deletion.
        assert "my_feature.txt" not in out

    def test_names_the_ancestor_the_content_actually_came_from(self, repo: Path):
        root = _git(repo, "rev-list", "--max-parents=0", "HEAD")
        # Ask git for the abbreviation it would print; %h is 7 chars in a
        # repo this small and longer in a real one.
        first = _git(repo, "rev-parse", "--short", root)
        _stale_checkout(repo, ["landed_a.txt", "landed_b.txt"])
        _commit(repo, "my feature")

        code, out = _run_gate(repo)

        assert code == 1
        # A restore has to start from the real base, so the gate must say which.
        assert first in out
        assert "2 commits earlier" in out

    def test_sees_a_clobber_that_is_not_committed_yet(self, repo: Path):
        """The damage exists in the worktree before it is ever recorded."""
        _stale_checkout(repo, ["landed_a.txt", "landed_b.txt"])
        # Deliberately no commit.

        code, out = _run_gate(repo)

        assert code == 1
        assert "landed_a.txt" in out

    def test_flags_a_two_file_clobber_the_same_as_a_large_one(self, repo: Path):
        """Size is not evidence: real clobbers ran as small as 2 files."""
        _stale_checkout(repo, ["landed_a.txt"])
        _commit(repo, "my feature")

        code, out = _run_gate(repo)

        assert code == 1
        assert "deletes 1 file(s)" in out


class TestStaysQuiet:
    def test_clean_branch_that_deletes_nothing(self, repo: Path):
        _git(repo, "checkout", "-q", "-b", "work")
        (repo / "my_feature.txt").write_text("added only\n")
        _commit(repo, "my feature")

        code, out = _run_gate(repo)

        assert code == 0
        assert "OK" in out

    def test_deliberate_deletion_of_a_long_standing_file(self, repo: Path):
        """No ancestor explains it away, so it is a real deletion."""
        _git(repo, "checkout", "-q", "-b", "work")
        (repo / "ancient.txt").unlink()
        _commit(repo, "drop the shipped plan")

        code, out = _run_gate(repo)

        assert code == 0

    def test_rename_scoring_between_40_and_50_percent(self, repo: Path):
        """git's default 50% threshold would read this as a deletion."""
        _git(repo, "checkout", "-q", "-b", "work")
        (repo / "renamed_from.py").write_text("".join(f"original line {i}\n" for i in range(20)))
        _commit(repo, "add the file that will be renamed")

        lines = [f"original line {i}\n" for i in range(20)]
        for i in range(10):
            lines[i] = f"wholly other text {i}\n"
        (repo / "renamed_to.py").write_text("".join(lines))
        (repo / "renamed_from.py").unlink()
        _commit(repo, "rename it")

        # Pin the premise: this really is in the band git's default misses.
        status = _git(repo, "diff", "--find-renames=10%", "--name-status", "HEAD~1", "HEAD")
        assert status.startswith("R04"), f"premise broken, got {status!r}"

        code, out = _run_gate(repo)

        assert code == 0, out


class TestEscapeHatches:
    def test_override_releases_a_deliberate_deletion(self, repo: Path):
        """Deleting one recently-added file needs only the deletion hatch.

        One commit's worth of the base is missing, which is a deliberate
        deletion (a shipped plan file) far more often than a clobber -- so the
        reverted-window signal, which needs two, has nothing to say here.
        """
        _git(repo, "checkout", "-q", "-b", "work")
        (repo / "landed_b.txt").unlink()
        _commit(repo, "drop the shipped plan")

        code, out = _run_gate(repo, VTSEARCH_ALLOW_DELETIONS="1")

        assert code == 0
        assert "VTSEARCH_ALLOW_DELETIONS" in out

    def test_a_full_clobber_needs_both_hatches(self, repo: Path):
        """Each hatch releases its own signal, and a clobber trips both.

        Waving through a tree that is missing whole commits of `dev` should
        take more than one reflex, so the deletion hatch deliberately does not
        speak for the reverted-window signal.
        """
        _stale_checkout(repo, ["landed_a.txt", "landed_b.txt"])
        _commit(repo, "my feature")

        code, out = _run_gate(repo, VTSEARCH_ALLOW_DELETIONS="1")
        assert code == 1
        assert "carries none of what" in out

        code, out = _run_gate(repo, VTSEARCH_ALLOW_DELETIONS="1", VTSEARCH_ALLOW_REVERTS="1")
        assert code == 0, out

    def test_missing_base_ref_skips_rather_than_fails(self, repo: Path):
        _stale_checkout(repo, ["landed_a.txt"])
        _commit(repo, "my feature")

        code, out = _run_gate(repo, base_ref="origin/nonexistent")

        assert code == 0
        assert "skipping" in out


@pytest.fixture
def edited_repo(tmp_path: Path) -> Path:
    """A repo whose `base` branch advances by *editing* files, adding none.

    This is the window the deletion signal cannot see into: whatever a stale
    tree reverts here, no path ever disappears.
    """
    r = tmp_path / "e"
    r.mkdir()
    _git(r, "init", "-q", "-b", "base", ".")
    for name in ("a.txt", "b.txt", "c.txt"):
        (r / name).write_text("v1\n")
    _commit(r, "first")
    (r / "a.txt").write_text("v2\n")
    _commit(r, "edit A")
    (r / "b.txt").write_text("v2\n")
    _commit(r, "edit B")
    return r


class TestFiresOnAHunkOnlyClobber:
    """#3210: the damage need not delete anything to be a clobber."""

    def test_flags_a_stale_tree_that_deletes_nothing(self, edited_repo: Path):
        _git(edited_repo, "checkout", "-q", "-b", "work")
        _git(edited_repo, "checkout", "-q", "HEAD~2", "--", ".")  # the stale content
        (edited_repo / "mine.txt").write_text("the legitimate change\n")
        _commit(edited_repo, "my feature")

        # Premise: the deletion signal has nothing to work with.
        assert _git(edited_repo, "diff", "--diff-filter=D", "--name-only", "base") == ""

        code, out = _run_gate(edited_repo)

        assert code == 1
        assert "carries none of what base gained across 2 consecutive" in out
        assert "a.txt" in out and "b.txt" in out
        # The branch's own work is not part of the damage.
        assert "mine.txt" not in out
        # And the older signal stayed silent, as it must: nothing was deleted.
        assert "never created" not in out

    def test_sees_a_hunk_clobber_that_is_not_committed_yet(self, edited_repo: Path):
        _git(edited_repo, "checkout", "-q", "-b", "work")
        _git(edited_repo, "checkout", "-q", "HEAD~2", "--", ".")
        # Deliberately no commit: this is how the damage exists first.

        code, out = _run_gate(edited_repo)

        assert code == 1
        assert "a.txt" in out

    def test_names_the_commit_the_content_came_from(self, edited_repo: Path):
        root = _git(edited_repo, "rev-parse", "--short", _git(edited_repo, "rev-list", "--max-parents=0", "HEAD"))
        _git(edited_repo, "checkout", "-q", "-b", "work")
        _git(edited_repo, "checkout", "-q", "HEAD~2", "--", ".")
        _commit(edited_repo, "my feature")

        code, out = _run_gate(edited_repo)

        assert code == 1
        # A restore starts from the real content point, so the gate says which.
        assert root in out

    def test_a_run_that_does_not_touch_the_merge_base(self, edited_repo: Path):
        """#2821's shape: the branch has the newest merge but not the two before it."""
        (edited_repo / "c.txt").write_text("v2\n")
        _commit(edited_repo, "edit C")
        _git(edited_repo, "checkout", "-q", "-b", "work")
        _git(edited_repo, "checkout", "-q", "HEAD~3", "--", "a.txt", "b.txt")  # keep edit C
        _commit(edited_repo, "my feature")

        code, out = _run_gate(edited_repo)

        assert code == 1
        assert "across 2 consecutive" in out
        assert "c.txt" not in out


class TestHunkSignalStaysQuiet:
    def test_deliberate_revert_of_a_single_commit(self, edited_repo: Path):
        """One merge undone is a thing people do; two in a row is not."""
        _git(edited_repo, "checkout", "-q", "-b", "work")
        _git(edited_repo, "checkout", "-q", "HEAD~1", "--", "b.txt")
        _commit(edited_repo, "revert that change")

        code, out = _run_gate(edited_repo)

        assert code == 0, out

    def test_branch_that_rewrites_the_same_files_without_reverting(self, edited_repo: Path):
        """Touching every recently-changed path is not evidence by itself.

        Only holding the *pre-merge blob* is; a third value means the branch
        did its own work on top of the base, which is the normal case.
        """
        _git(edited_repo, "checkout", "-q", "-b", "work")
        (edited_repo / "a.txt").write_text("v3\n")
        (edited_repo / "b.txt").write_text("v3\n")
        _commit(edited_repo, "my feature")

        code, out = _run_gate(edited_repo)

        assert code == 0, out

    def test_ordinary_branch_on_a_fresh_base(self, edited_repo: Path):
        _git(edited_repo, "checkout", "-q", "-b", "work")
        (edited_repo / "mine.txt").write_text("added only\n")
        _commit(edited_repo, "my feature")

        code, out = _run_gate(edited_repo)

        assert code == 0
        assert "OK" in out

    def test_override_releases_a_deliberate_revert(self, edited_repo: Path):
        _git(edited_repo, "checkout", "-q", "-b", "work")
        _git(edited_repo, "checkout", "-q", "HEAD~2", "--", ".")
        _commit(edited_repo, "revert the last two merges on purpose")

        code, out = _run_gate(edited_repo, VTSEARCH_ALLOW_REVERTS="1")

        assert code == 0
        assert "VTSEARCH_ALLOW_REVERTS" in out
