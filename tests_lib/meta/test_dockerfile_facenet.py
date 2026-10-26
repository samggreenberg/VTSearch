"""Every Docker image must install ``facenet-pytorch``, and always ``--no-deps``.

``facenet-pytorch`` is the whole face path: the ``image2face`` converter's
MTCNN detector, and the ``face`` media type's FaceNet embedder. It cannot be declared in ``pyproject.toml``: its ``torch<2.3`` /
``numpy<2`` / ``Pillow<10.3`` pins would downgrade the app's stack. So every
install path spells the bypass out by hand: ``scripts/install.sh``,
``.claude/hooks/ensure-test-deps.sh``, and one ``RUN`` per Dockerfile.

Until issue #4886 none of the Dockerfiles had it. Every container offered
``image2face``, which logged a warning and produced no faces, and the face
embedder raised ``ImportError``. The failure is silent at
build time, the same shape as toponymy's #3852, so a static check is the only
thing that catches it. Pure text over the repo: no Docker daemon, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The non-Dockerfile install paths: the one a user runs, and the test container's.
_SCRIPT_PATHS = (
    Path("scripts/install.sh"),
    Path(".claude/hooks/ensure-test-deps.sh"),
)


def _join_continued_lines(text: str) -> str:
    """Collapse backslash-newline continuations (Docker's and the shell's)."""
    return re.sub(r"\\\n\s*", " ", text)


def _facenet_installs(path: Path) -> list[str]:
    """Every pip command segment in *path* that installs facenet-pytorch."""
    text = _join_continued_lines((REPO_ROOT / path).read_text())
    return [
        segment.strip()
        for segment in re.split(r"&&|\|\||;|\n", text)
        if "pip install" in segment and "facenet-pytorch" in segment
    ]


def _dockerfiles() -> list[Path]:
    return sorted(p.relative_to(REPO_ROOT) for p in (REPO_ROOT / "docker").glob("Dockerfile*"))


@pytest.mark.parametrize("dockerfile", _dockerfiles(), ids=lambda p: p.name)
def test_dockerfile_installs_facenet(dockerfile: Path) -> None:
    """A Docker build never runs install.sh, so each image installs it itself."""
    assert _facenet_installs(dockerfile), (
        f"{dockerfile} never installs facenet-pytorch, so the image it builds offers image2face "
        "but produces no faces, and cannot load the face embedder "
        "(issue #4886). Add the step that mirrors scripts/install.sh's vts_install_face_deps:\n"
        "    RUN pip install --no-cache-dir --no-deps facenet-pytorch"
    )


@pytest.mark.parametrize("path", [*_SCRIPT_PATHS, *_dockerfiles()], ids=lambda p: p.name)
def test_facenet_is_installed_without_deps(path: Path) -> None:
    """``--no-deps`` is the point: a plain install downgrades torch, numpy and Pillow."""
    for command in _facenet_installs(path):
        assert "--no-deps" in command, (
            f"{path} installs facenet-pytorch WITHOUT --no-deps:\n  {command}\n"
            "Its torch<2.3 / numpy<2 / Pillow<10.3 pins would then be honoured, downgrading "
            "the stack every embedder sits on."
        )


def test_the_script_install_paths_install_it() -> None:
    """The two non-Docker paths are the reference the images mirror."""
    for path in _SCRIPT_PATHS:
        assert _facenet_installs(path), f"{path} no longer installs facenet-pytorch"
