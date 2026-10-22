#!/usr/bin/env python3
"""Delete the dist-info a later install of the same project left behind (#5153).

`ensure-test-deps.sh` can leave two `*.dist-info` directories for one project in
one site directory. `pip list` -- and so pip-audit -- then reports whichever one
the directory listing yields first, so a package that is actually patched can be
flagged, or flip red on a reshuffle. Two paths lead there on the cloud image:

- `--ignore-installed` never uninstalls, so its upgrades (cryptography, pyjwt,
  urllib3, ...) land beside the image's own uv-installed copies.
- A plain upgrade uninstalls the copy pip *resolves*, which is the first one on
  `sys.path`. The per-user site sits ahead of /usr/local, so when both hold
  certifi, the requirements pass's upgrade into /usr/local uninstalls the
  user-site copy and leaves the /usr/local one's metadata where it was.

Either way the newer install wrote its files over the older one's, so what is
on disk is the last install's and only the older metadata is stale. This keeps
the most recently written dist-info of each project in each directory and
deletes the others -- the directory alone, never the files its RECORD lists,
which now belong to the survivor.

Run with no arguments, it prunes the directories pip installs into (the default
scheme and the user site); directories given as arguments replace those.
"""

from __future__ import annotations

import importlib.metadata
import re
import shutil
import site
import sys
import sysconfig
from collections import defaultdict
from pathlib import Path


def project_name(dist_info: Path) -> str:
    """The PEP 503 normalised project name, from METADATA or else the directory name."""
    name = importlib.metadata.Distribution.at(dist_info).metadata.get("Name") or dist_info.name.split("-", 1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def stale_dist_infos(site_dir: Path) -> list[tuple[Path, Path]]:
    """`(stale, survivor)` for every dist-info that a later one of its project shadows."""
    by_project: dict[str, list[Path]] = defaultdict(list)
    for dist_info in site_dir.glob("*.dist-info"):
        by_project[project_name(dist_info)].append(dist_info)
    stale = []
    for copies in by_project.values():
        copies.sort(key=lambda p: p.stat().st_mtime_ns)
        stale.extend((old, copies[-1]) for old in copies[:-1])
    return stale


def install_dirs() -> list[Path]:
    dirs = {sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}
    if site.ENABLE_USER_SITE:
        dirs.add(site.getusersitepackages())
    return sorted(Path(d) for d in dirs if Path(d).is_dir())


def main(argv: list[str]) -> int:
    for site_dir in [Path(a) for a in argv] or install_dirs():
        for stale, survivor in stale_dist_infos(site_dir):
            shutil.rmtree(stale)
            print(f"Removed {stale.name} from {site_dir}: {survivor.name} was installed over it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
