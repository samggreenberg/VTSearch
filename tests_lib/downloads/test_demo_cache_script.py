"""Keep scripts/link-demo-cache.sh's DEMO_DIRS list in sync with the downloaders.

The shared demo-cache script hardcodes the set of extraction directories the
demo downloaders create directly under ``DATA_DIR``. That set lives as string
literals scattered across ``vtscore/datasets/downloader/*.py`` (there is no
runtime registry of extraction dir names), so this test re-derives it from the
source and fails when a new demo's extraction dir is missing from the script
(or a stale one lingers).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "link-demo-cache.sh"
DOWNLOADER_SOURCES = [
    *sorted((REPO_ROOT / "vtscore" / "datasets" / "downloader").glob("*.py")),
    REPO_ROOT / "vtscore" / "datasets" / "importers" / "demo" / "__init__.py",
]

# DATA_DIR children that are not shareable demo extraction dirs: the derived
# per-media-type dirs (loaders copy/organize into these; they can contain
# absolute paths specific to one data dir) and single downloaded files (their
# names carry an extension), which ride along inside their dataset's dir or
# stay per-user.
NON_CACHE_NAMES = {"images", "video"}


def _expected_demo_dirs() -> set[str]:
    literals: set[str] = set()
    for source in DOWNLOADER_SOURCES:
        literals.update(re.findall(r'DATA_DIR / "([^"]+)"', source.read_text()))
    return {name for name in literals if name not in NON_CACHE_NAMES and "." not in name}


def _script_demo_dirs() -> set[str]:
    text = SCRIPT.read_text()
    match = re.search(r"DEMO_DIRS=\(\n(.*?)\n\)", text, re.DOTALL)
    assert match, "DEMO_DIRS array not found in scripts/link-demo-cache.sh"
    return set(match.group(1).split())


def test_script_list_matches_downloader_sources():
    expected = _expected_demo_dirs()
    in_script = _script_demo_dirs()
    assert expected, "no DATA_DIR extraction literals found - did the downloaders move?"
    missing = expected - in_script
    stale = in_script - expected
    assert not missing, f"add to scripts/link-demo-cache.sh DEMO_DIRS: {sorted(missing)}"
    assert not stale, f"remove from scripts/link-demo-cache.sh DEMO_DIRS: {sorted(stale)}"


def test_script_is_executable():
    assert SCRIPT.stat().st_mode & 0o111, "scripts/link-demo-cache.sh must be executable"
