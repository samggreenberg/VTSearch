"""Validate that slim requirements files include the core framework deps.

The slim files (image-embedders.txt, labbench.txt, etc.) are hand-edited flat
lists used by specialised Docker images.  They do not forward to pyproject.toml
via `-e .`, so missing entries silently produce broken images.  This test
catches that class of bug at CI time.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent

# Packages that every VTSearch deployment requires regardless of variant.
# These are imported at app startup before any plugin-specific code runs.
_ALWAYS_REQUIRED: frozenset[str] = frozenset(
    {
        "flask",
        "flask-smorest",
        "marshmallow",
        "pydantic",
        "werkzeug",
    }
)


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_packages(path: Path) -> set[str]:
    pkgs: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name = re.split(r"[>=<!;\[", line)[0].strip()
        if name:
            pkgs.add(_normalise(name))
    return pkgs


def _is_forwarding(path: Path) -> bool:
    return any(line.strip().startswith("-e") for line in path.read_text().splitlines())


_SLIM_FILES = [
    p
    for p in sorted((_REPO_ROOT / "requirements").iterdir())
    if p.suffix == ".txt" and not _is_forwarding(p)
]


@pytest.mark.parametrize("req_file", _SLIM_FILES, ids=[p.name for p in _SLIM_FILES])
def test_slim_requirements_include_core_deps(req_file: Path) -> None:
    declared = _parse_packages(req_file)
    missing = {_normalise(p) for p in _ALWAYS_REQUIRED} - declared
    assert not missing, (
        f"{req_file.name} is missing core framework deps: {sorted(missing)}\n"
        "Add them to the '── Core framework ──' section of that file."
    )
