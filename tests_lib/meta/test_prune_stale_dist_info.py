"""Tests for .claude/hooks/prune-stale-dist-info.py, run by ensure-test-deps.sh last.

Issue #5153: a fresh cloud container's first `./run-tests.sh` failed pip-audit on
certifi 2026.7.22 although 2026.10.19 was installed and importable. The upgrade
had uninstalled the user site's copy (first on `sys.path`, so the one pip
resolves) and left the image's uv-installed dist-info in /usr/local beside the
new one, and `pip list` read the old one first. `--ignore-installed` left the
same pairs for cryptography, pycparser, pyjwt and urllib3.

The script runs for real against fake site directories: dist-info directories
with a METADATA and a RECORD, aged with `os.utime` so which one was written last
is set by the test rather than by how fast it runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PRUNE = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "prune-stale-dist-info.py"

IMAGE_BUILD = 1_792_500_000  # the image's uv install
HOOK_RUN = IMAGE_BUILD + 40_000  # ensure-test-deps.sh, the next day


def dist_info(site_dir: Path, dirname: str, name: str, version: str, *, written: int) -> Path:
    module = site_dir / name.lower()
    module.mkdir(exist_ok=True)
    (module / "__init__.py").write_text(f'__version__ = "{version}"\n')
    path = site_dir / f"{dirname}-{version}.dist-info"
    path.mkdir()
    (path / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (path / "RECORD").write_text(f"{name.lower()}/__init__.py,,\n{path.name}/METADATA,,\n{path.name}/RECORD,,\n")
    os.utime(path, (written, written))
    return path


def prune(*site_dirs: Path) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603  # interpreter + repo-local hook path, no shell
        [sys.executable, str(PRUNE), *map(str, site_dirs)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


def test_the_issues_certifi_pair_keeps_the_upgrade(tmp_path):
    old = dist_info(tmp_path, "certifi", "certifi", "2026.7.22", written=IMAGE_BUILD)
    new = dist_info(tmp_path, "certifi", "certifi", "2026.10.19", written=HOOK_RUN)
    result = prune(tmp_path)
    assert not old.exists()
    assert new.exists()
    assert "certifi-2026.7.22.dist-info" in result.stdout


def test_only_the_metadata_goes_not_the_files_its_record_lists(tmp_path):
    """The upgrade wrote its files over the old copy's: they are the survivor's now."""
    dist_info(tmp_path, "certifi", "certifi", "2026.7.22", written=IMAGE_BUILD)
    dist_info(tmp_path, "certifi", "certifi", "2026.10.19", written=HOOK_RUN)
    prune(tmp_path)
    assert (tmp_path / "certifi" / "__init__.py").read_text() == '__version__ = "2026.10.19"\n'


def test_the_survivor_is_the_last_written_not_the_highest_version(tmp_path):
    """After a downgrade the files on disk are the lower version's, so its metadata is the true one."""
    higher = dist_info(tmp_path, "urllib3", "urllib3", "2.8.0", written=IMAGE_BUILD)
    lower = dist_info(tmp_path, "urllib3", "urllib3", "2.7.0", written=HOOK_RUN)
    prune(tmp_path)
    assert lower.exists()
    assert not higher.exists()


def test_one_project_spelled_two_ways_is_one_project(tmp_path):
    old = dist_info(tmp_path, "PyJWT", "PyJWT", "2.7.0", written=IMAGE_BUILD)
    new = dist_info(tmp_path, "pyjwt", "pyjwt", "2.15.1", written=HOOK_RUN)
    prune(tmp_path)
    assert not old.exists()
    assert new.exists()


def test_every_older_copy_goes_when_there_are_three(tmp_path):
    copies = [
        dist_info(tmp_path, "pycparser", "pycparser", version, written=HOOK_RUN + n)
        for n, version in enumerate(("2.22", "3.0", "3.11"))
    ]
    prune(tmp_path)
    assert [c.exists() for c in copies] == [False, False, True]


def test_projects_with_one_dist_info_are_untouched(tmp_path):
    lone = [
        dist_info(tmp_path, "certifi", "certifi", "2026.10.19", written=HOOK_RUN),
        dist_info(tmp_path, "cryptography", "cryptography", "50.0.2", written=IMAGE_BUILD),
    ]
    result = prune(tmp_path)
    assert all(p.exists() for p in lone)
    assert result.stdout == ""


def test_a_copy_in_another_directory_is_not_a_duplicate(tmp_path):
    """The user site shadowing /usr/local is by design: the first on sys.path wins both import and pip list."""
    user_site, local = tmp_path / "user", tmp_path / "local"
    user_site.mkdir()
    local.mkdir()
    shadowing = dist_info(user_site, "urllib3", "urllib3", "2.8.0", written=HOOK_RUN)
    shadowed = dist_info(local, "urllib3", "urllib3", "2.7.0", written=IMAGE_BUILD)
    prune(user_site, local)
    assert shadowing.exists()
    assert shadowed.exists()
