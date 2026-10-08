"""Validate that slim requirements files include the core framework deps.

The slim files (image-embedders.txt, labbench.txt, etc.) are hand-edited flat
lists used by specialised Docker images.  They do not forward to pyproject.toml
via `-e .`, so missing entries silently produce broken images.  This test
catches that class of bug at CI time.
"""

from __future__ import annotations

import re
import tomllib
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
        # Browse signposts: toponymy's real dependencies, declared on its
        # behalf because toponymy itself is installed `--no-deps` by a
        # dedicated Dockerfile step (its transformers<5 pin would downgrade
        # the image's transformers stack -- see
        # docs/plans/vtsbrowse-toponymy.md) AND these images install the
        # package with `--no-deps -e .`, so pyproject.toml's declarations
        # never reach them either. Only the ones with no carrier are listed:
        # jinja2 arrives with flask, numba with fast_hdbscan, httpx and
        # tokenizers with transformers' huggingface_hub, and apricot-select
        # is installed alongside toponymy in that same dedicated step
        # because it ships a legacy setup.py sdist. Omitting these is how
        # issue #3852 stayed half-fixed: `toponymy` imports at build time,
        # so a missing one is an ImportError in the signpost path.
        "fast_hdbscan",
        "vectorizers",
        "tenacity",
    }
)


# Packages that any slim file shipping an *image* embedder requires. The
# image-processor backend is now requested by name (`torchvision`) rather than
# inherited from a transformers default (#3173), so torchvision is part of the
# preprocessing path itself and not an optional accelerator. Nothing carries it
# transitively — `torch` does not pull it in and neither does `transformers` —
# which is how issue #3264 (LabBench image built without torchvision) happened.
# Keyed off Pillow, the dep every image-capable variant declares.
_IMAGE_MARKER = "pillow"
_IMAGE_REQUIRED: frozenset[str] = frozenset({"torchvision"})


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
        "umap-learn under the '── VTSBrowse projection ──' heading; fast_hdbscan / "
        "vectorizers / tenacity under '── VTSBrowse signpost naming (toponymy) ──')."
    )


@pytest.mark.parametrize("req_file", _SLIM_FILES, ids=[p.name for p in _SLIM_FILES])
def test_slim_requirements_include_image_deps(req_file: Path) -> None:
    declared = _parse_packages(req_file)
    if _IMAGE_MARKER not in declared:
        pytest.skip(f"{req_file.name} ships no image media type")
    missing = {_normalise(p) for p in _IMAGE_REQUIRED} - declared
    assert not missing, (
        f"{req_file.name} ships image embedders but is missing: {sorted(missing)}\n"
        "Add them to that file under the PyTorch heading so they resolve against "
        "the same wheel index as torch."
    )


# Packages held to the same version bound in every environment. pandas is held
# on one major (#4381; 3 since #4390) because its majors differ in both inline
# annotations and runtime semantics. The slim files install the app with `--no-deps -e .`, so
# pyproject.toml's bound never reaches their images, and each file has to repeat
# it. Otherwise lifting the bound in pyproject.toml alone would silently leave
# those images on the old major.
_SAME_BOUND_EVERYWHERE = ("pandas",)


def _split_requirement(line: str) -> tuple[str, str]:
    """``"pandas<3  # why"`` -> ``("pandas", "<3")``."""
    line = line.split("#", 1)[0].strip()
    name = re.split(r"[>=<!~;\[\]\s]", line)[0]
    return _normalise(name), line[len(name) :].replace(" ", "")


def _file_bounds(path: Path) -> dict[str, str]:
    bounds: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            name, spec = _split_requirement(line)
            bounds[name] = spec
    return bounds


def _pyproject_bounds() -> dict[str, str]:
    deps = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    return dict(_split_requirement(dep) for dep in deps)


_BOUND_CASES = [
    (req_file, package)
    for req_file in _SLIM_FILES
    for package in _SAME_BOUND_EVERYWHERE
    if package in _file_bounds(req_file)
]


def test_same_bound_packages_are_declared() -> None:
    # Guard the parametrisation below: a package missing from pyproject.toml,
    # or declared in no slim file, would leave nothing for it to check.
    pyproject = _pyproject_bounds()
    for package in _SAME_BOUND_EVERYWHERE:
        assert package in pyproject, f"pyproject.toml no longer declares {package}"
        assert any(p == package for _, p in _BOUND_CASES), f"no slim requirements file declares {package}"


@pytest.mark.parametrize(
    ("req_file", "package"),
    _BOUND_CASES,
    ids=[f"{f.name}-{p}" for f, p in _BOUND_CASES],
)
def test_slim_requirements_repeat_pyproject_bound(req_file: Path, package: str) -> None:
    expected = _pyproject_bounds()[package]
    declared = _file_bounds(req_file)[package]
    assert declared == expected, (
        f"{req_file.name} declares {package}{declared or ' (unbounded)'}, but pyproject.toml "
        f"declares {package}{expected or ' (unbounded)'}. The slim images install the app "
        "`--no-deps`, so they only get the bound this file gives them; change both together."
    )
