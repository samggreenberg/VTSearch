"""The GRID suite launcher, ``scripts/slurm/suite.sbatch``, behaves as its guard says.

``suite.sbatch`` is the only gate a VTSearch branch passes before it is pushed
from a 3 GB laptop, and until #3694 it lived outside the repository, where
nothing reviewed or tested it. These tests pin the behaviour its comments
promise, by running the real script (not a copy of its logic) against a
throwaway origin + clone + worktree:

* **in sync** -- the local branch matches ``origin/<ref>``: the suite runs and
  the ``=== ref`` line says so;
* **remote-only** -- no local branch at all: resolved through ``origin/<ref>``
  instead of dying on ``--detach``'s unrelated DWIM error;
* **behind** -- a force-push the non-forced fetch could not fast-forward (#3677):
  refused, exit 2, and the suite does NOT run;
* **ahead** -- a local commit not yet pushed (the #3292 direction): refused,
  exit 2, never "corrected" to the older ``origin/<ref>``;
* **diverged** -- refused, exit 2;
* in every case the worktree is put back on the branch it was on (the EXIT
  trap), and a non-worktree is refused before anything is touched.

``run-tests.sh`` and ``gridenv.sh`` are stubs here: what is under test is the
launcher's ref handling, not the suite it launches.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "slurm" / "suite.sbatch"

pytestmark = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("bash", "git", "flock")),
    reason="needs bash, git and flock (util-linux)",
)

BRANCH = "feature"
GIT = shutil.which("git") or "git"
BASH = shutil.which("bash") or "bash"


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(k, None)
    return subprocess.run([GIT, *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603  # fixed argv, no shell


def _commit(cwd: Path, name: str) -> str:
    (cwd / name).write_text(name)
    _git(cwd, "add", name)
    _git(cwd, "commit", "-q", "-m", name)
    return _git(cwd, "rev-parse", "HEAD")


@pytest.fixture
def grid(tmp_path: Path) -> dict:
    """An origin, a clone of it, and a second worktree the suite is pointed at."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "dev", str(origin))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", "-q", str(origin), str(seed))
    _git(seed, "checkout", "-q", "-b", "dev")
    # The stubs are committed, so every checkout the script makes still has them.
    (seed / "gridenv.sh").write_text(
        'mkdir -p "$PWD/.bin"; printf "#!/bin/sh\\n" > "$PWD/.bin/ruff"\n'
        'chmod +x "$PWD/.bin/ruff"; export PATH="$PWD/.bin:$PATH"\n'
    )
    (seed / "run-tests.sh").write_text('#!/usr/bin/env bash\necho "SUITE RAN at $(git rev-parse HEAD)"\n')
    (seed / "run-tests.sh").chmod(0o755)
    (seed / ".gitignore").write_text(".bin/\n")
    _git(seed, "add", ".")
    _git(seed, "commit", "-q", "-m", "stubs")
    _git(seed, "push", "-q", "origin", "dev")
    base = _commit(seed, "b1")
    _git(seed, "push", "-q", "origin", f"HEAD:refs/heads/{BRANCH}")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    wt = tmp_path / "wt"
    _git(clone, "worktree", "add", "-q", "--detach", str(wt), "origin/dev")
    _git(wt, "checkout", "-q", "-b", "scratch")
    return {"seed": seed, "clone": clone, "wt": wt, "base": base}


def _run(wt: Path, ref: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "SUITE_HF_HOME": str(wt.parent / "hf")}
    return subprocess.run([BASH, str(SCRIPT), str(wt), ref], env=env, capture_output=True, text=True, timeout=120)  # noqa: S603  # fixed argv, no shell


def _restored(grid: dict) -> bool:
    return _git(grid["wt"], "symbolic-ref", "--short", "HEAD") == "scratch"


def test_script_is_tracked_and_parses() -> None:
    assert SCRIPT.is_file()
    subprocess.run([BASH, "-n", str(SCRIPT)], check=True)  # noqa: S603  # fixed argv, no shell
    text = SCRIPT.read_text()
    # The headers were never recorded anywhere else before #3694; a port that
    # drops them silently changes the resources every suite job gets.
    for header in ("--partition=cpu", "--cpus-per-task=8", "--mem=32G", "--time=02:00:00"):
        assert f"#SBATCH {header}" in text


def test_in_sync_runs_the_suite(grid: dict) -> None:
    _git(grid["clone"], "branch", "-q", BRANCH, f"origin/{BRANCH}")
    r = _run(grid["wt"], BRANCH)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"SUITE RAN at {grid['base']}" in r.stdout
    assert f"=== ref  '{BRANCH}' matches origin/{BRANCH}" in r.stdout
    assert _restored(grid)


def test_remote_only_resolves_through_origin(grid: dict) -> None:
    r = _run(grid["wt"], BRANCH)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"SUITE RAN at {grid['base']}" in r.stdout
    assert "resolved through origin" in r.stdout
    assert _restored(grid)


def test_behind_origin_is_refused(grid: dict) -> None:
    _git(grid["clone"], "branch", "-q", BRANCH, f"origin/{BRANCH}")
    new = _commit(grid["seed"], "b2")
    _git(grid["seed"], "push", "-q", "origin", f"HEAD:refs/heads/{BRANCH}")
    r = _run(grid["wt"], BRANCH)
    assert r.returncode == 2, r.stdout + r.stderr
    assert f"but origin/{BRANCH} is {new}" in r.stdout
    assert "BEHIND origin" in r.stdout
    assert "SUITE RAN" not in r.stdout
    assert _restored(grid)


def test_force_push_is_refused_not_tested_stale(grid: dict) -> None:
    """#3677: the branch was amended and force-pushed; the fetch is not forced."""
    _git(grid["clone"], "branch", "-q", BRANCH, f"origin/{BRANCH}")
    _git(grid["seed"], "commit", "-q", "--amend", "-m", "b1 amended")
    _git(grid["seed"], "push", "-q", "-f", "origin", f"HEAD:refs/heads/{BRANCH}")
    r = _run(grid["wt"], BRANCH)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "DIVERGED" in r.stdout
    assert "SUITE RAN" not in r.stdout


def test_ahead_of_origin_is_refused_not_corrected(grid: dict) -> None:
    """#3292's direction: never silently test the OLDER origin commit."""
    _git(grid["clone"], "branch", "-q", BRANCH, f"origin/{BRANCH}")
    other = grid["clone"].parent / "wt2"
    _git(grid["clone"], "worktree", "add", "-q", str(other), BRANCH)
    _commit(other, "local-only")
    r = _run(grid["wt"], BRANCH)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "AHEAD of origin" in r.stdout
    assert "SUITE RAN" not in r.stdout
    assert _restored(grid)


def test_diverged_is_refused(grid: dict) -> None:
    _git(grid["clone"], "branch", "-q", BRANCH, f"origin/{BRANCH}")
    other = grid["clone"].parent / "wt2"
    _git(grid["clone"], "worktree", "add", "-q", str(other), BRANCH)
    _commit(other, "local-only")
    _commit(grid["seed"], "remote-only")
    _git(grid["seed"], "push", "-q", "origin", f"HEAD:refs/heads/{BRANCH}")
    r = _run(grid["wt"], BRANCH)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "DIVERGED" in r.stdout
    assert "SUITE RAN" not in r.stdout


def test_unknown_ref_is_refused(grid: dict) -> None:
    r = _run(grid["wt"], "no-such-branch")
    assert r.returncode == 2
    assert "names no commit here and none on origin either" in r.stdout
    assert _restored(grid)


def test_non_worktree_is_refused(tmp_path: Path) -> None:
    r = subprocess.run([BASH, str(SCRIPT), str(tmp_path), BRANCH], capture_output=True, text=True)  # noqa: S603  # fixed argv, no shell
    assert r.returncode == 2
    assert "is not a VTSearch worktree" in r.stdout
