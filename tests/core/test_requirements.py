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

# Packages that every VTSearch deployment requires regardless of variant:
# the web/app framework, the numeric + model stack every embedder and the
# ranker sit on, and umap-learn for the Browse projection. A slim file may
# drop media-type-specific deps (librosa, PyMuPDF, ultralytics, ...) but
# never these.
#
# Only list packages nothing else pulls in transitively. threadpoolctl and
# huggingface_hub are imported directly by vtscore but arrive with
# scikit-learn / transformers, so a slim file that omits them still works;
# umap-learn has no such carrier, which is how issue #2843 (LabBench image
# built without umap-learn) happened.
_ALWAYS_REQUIRED: frozenset[str] = frozenset(
    {
        "flask",
        "flask-smorest",
        "marshmallow",
        "pydantic",
        "werkzeug",
        "gunicorn",
        "numpy",
        "requests",
        "tqdm",
        "scikit-learn",
        "torch",
        # Browse canvas: vtscore.gpu_backends.umap_fit_transform falls back to
        # the CPU umap-learn reducer whenever cuML is absent (always, in the
        # slim images).
        "umap-learn",
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
        name = re.split(r"[>=<!;\[\]]", line)[0].strip()
        if name:
            pkgs.add(_normalise(name))
    return pkgs


def _is_forwarding(path: Path) -> bool:
    return any(line.strip().startswith("-e") for line in path.read_text().splitlines())


_SLIM_FILES = [
    p for p in sorted((_REPO_ROOT / "requirements").iterdir()) if p.suffix == ".txt" and not _is_forwarding(p)
]


@pytest.mark.parametrize("req_file", _SLIM_FILES, ids=[p.name for p in _SLIM_FILES])
def test_slim_requirements_include_core_deps(req_file: Path) -> None:
    declared = _parse_packages(req_file)
    missing = {_normalise(p) for p in _ALWAYS_REQUIRED} - declared
    assert not missing, (
        f"{req_file.name} is missing deps every deployment needs: {sorted(missing)}\n"
        "Add them to that file (core framework / numeric stack entries near the top; "
        "umap-learn under the '── VTSBrowse projection ──' heading)."
    )
