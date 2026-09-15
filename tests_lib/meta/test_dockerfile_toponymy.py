"""Every Docker image must install ``toponymy``, and always ``--no-deps``.

``toponymy`` letters the VTSBrowse map (``vtscore.projection.signpost_build``).
It cannot be declared in ``pyproject.toml`` or in a requirements file: its
``transformers<5.0.0`` pin would drag the app's transformers stack backwards,
and pip rejects ``--no-deps`` inside a requirements file.  So every install
path has to spell the bypass out by hand -- ``scripts/install.sh`` for a host
install, ``.claude/hooks/ensure-test-deps.sh`` for the test container, and one
dedicated ``RUN`` per Dockerfile.

Nothing connected those copies, which is how issue #3852 happened: all five
images shipped without ``toponymy``, so a LabBench deployment logged
"VTSBrowse signposts are disabled" and rendered every map unlettered.  The
failure is invisible at build time and silent-ish at runtime (the app degrades
rather than crashing), so a static check is the only thing that catches it.

Two invariants, both pure text over the repo -- no Docker daemon, no network:

* every ``docker/Dockerfile*`` installs ``toponymy``, and
* every install path in the repo pins the *same* version and passes
  ``--no-deps``, so bumping the pin in one place fails until the rest follow.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The pinned version inside a ``toponymy==X.Y.Z`` requirement.
_TOPONYMY_PIN = re.compile(r"toponymy==(\d+\.\d+\.\d+)")

#: The non-Dockerfile install paths.  ``scripts/install.sh`` is the one a user
#: runs; the hook is what the test container runs.
_SCRIPT_PATHS = (
    Path("scripts/install.sh"),
    Path(".claude/hooks/ensure-test-deps.sh"),
)


def _join_continued_lines(text: str) -> str:
    """Collapse backslash-newline continuations (Docker's and the shell's)."""
    return re.sub(r"\\\n\s*", " ", text)


def _toponymy_installs(path: Path) -> list[tuple[str, str]]:
    """Return ``(command, version)`` for every toponymy pip install in ``path``.

    A ``RUN``/shell line can chain several installs with ``&&``, so the text is
    split into command segments first: the flags that matter are the ones on
    *toponymy's own* segment, not on a neighbour's.
    """
    text = _join_continued_lines((REPO_ROOT / path).read_text())
    found: list[tuple[str, str]] = []
    for segment in re.split(r"&&|\|\||;|\n", text):
        if "pip install" not in segment:
            continue
        match = _TOPONYMY_PIN.search(segment)
        if match:
            found.append((segment.strip(), match.group(1)))
    return found


def _dockerfiles() -> list[Path]:
    return sorted(p.relative_to(REPO_ROOT) for p in (REPO_ROOT / "docker").glob("Dockerfile*"))


@pytest.mark.parametrize("dockerfile", _dockerfiles(), ids=lambda p: p.name)
def test_dockerfile_installs_toponymy(dockerfile: Path) -> None:
    """A Docker build never runs install.sh, so each image installs it itself."""
    assert _toponymy_installs(dockerfile), (
        f"{dockerfile} never installs toponymy, so the image it builds renders every "
        "VTSBrowse map unlettered (issue #3852). Add the dedicated step that mirrors "
        "scripts/install.sh's vts_install_toponymy:\n"
        '    RUN pip install --no-cache-dir --no-deps "toponymy==<pin>"\n'
        "plus apricot-select in the same step for an image that installs itself with "
        "`--no-deps -e .` (labbench, image-embedders)."
    )


@pytest.mark.parametrize(
    "path",
    [*_SCRIPT_PATHS, *_dockerfiles()],
    ids=lambda p: p.name,
)
def test_toponymy_is_installed_without_deps(path: Path) -> None:
    """``--no-deps`` is the point: a plain install downgrades transformers."""
    for command, version in _toponymy_installs(path):
        assert "--no-deps" in command, (
            f"{path} installs toponymy=={version} WITHOUT --no-deps. Its "
            "transformers<5.0.0 pin would then be honoured, downgrading the "
            "transformers stack every embedder sits on. See "
            "docs/plans/vtsbrowse-toponymy.md."
        )


def test_every_install_path_pins_the_same_version() -> None:
    """One pin, spelled in N places -- so a bump has to land in all of them."""
    pins: dict[str, set[str]] = {}
    for path in (*_SCRIPT_PATHS, *_dockerfiles()):
        for _command, version in _toponymy_installs(path):
            pins.setdefault(version, set()).add(str(path))

    assert pins, "no toponymy install found anywhere in the repo"
    assert len(pins) == 1, (
        "toponymy is pinned to different versions across install paths:\n"
        + "\n".join(f"  {version}: {sorted(paths)}" for version, paths in sorted(pins.items()))
        + "\nBump every path together, or the Docker images and the host install "
        "will letter maps with different labeler_signature values."
    )
