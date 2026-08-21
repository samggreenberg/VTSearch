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
        first = _git(repo, "rev-list", "--max-parents=0", "HEAD")
        _stale_checkout(repo, ["landed_a.txt", "landed_b.txt"])
        _commit(repo, "my feature")

        code, out = _run_gate(repo)

        assert code == 1
        # A restore has to start from the real base, so the gate must say which.
        assert first[:8] in out
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
        _stale_checkout(repo, ["landed_a.txt", "landed_b.txt"])
        _commit(repo, "my feature")

        code, out = _run_gate(repo, VTSEARCH_ALLOW_DELETIONS="1")

        assert code == 0
        assert "VTSEARCH_ALLOW_DELETIONS" in out

    def test_missing_base_ref_skips_rather_than_fails(self, repo: Path):
        _stale_checkout(repo, ["landed_a.txt"])
        _commit(repo, "my feature")

        code, out = _run_gate(repo, base_ref="origin/nonexistent")

        assert code == 0
        assert "skipping" in out
