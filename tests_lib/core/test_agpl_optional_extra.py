"""The AGPL-3.0 packages are an opt-*out* extra, not an opt-in one.

``ultralytics`` and ``PyMuPDF`` are AGPL-3.0-or-later, so they live in the
``agpl`` extra rather than in ``[project.dependencies]`` — but every default
install path requests that extra, so a normal install is unchanged. Two things
can silently break that arrangement, and both are cheap to pin down:

* a default install path quietly *losing* the extra (users stop getting the
  YOLO/PDF features, and the test suite stops covering those code paths), or
* the ``*-no-agpl.txt`` mirrors drifting from the files they mirror, so the
  opt-out install differs from the default in ways nobody intended.

The rest of the file checks that opting out degrades with an actionable
message instead of a bare ``ImportError``.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from vtscore.utils.optional_deps import agpl_import_error, agpl_unavailable_message

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "requirements"

AGPL_PACKAGES = ("PyMuPDF", "ultralytics")


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _requirement_lines(path: Path) -> list[str]:
    """Requirement/option lines of a pip requirements file (comments dropped)."""
    lines: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _normalise(name: str) -> str:
    return name.replace("_", "-").lower()


# ---------------------------------------------------------------------------
# Packaging: where the two packages are declared
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", AGPL_PACKAGES)
def test_agpl_package_lives_in_the_agpl_extra(package: str) -> None:
    project = _pyproject()["project"]
    hard = {_normalise(d) for d in project["dependencies"]}
    extra = {_normalise(d) for d in project["optional-dependencies"]["agpl"]}

    assert _normalise(package) not in hard, (
        f"{package} is AGPL-3.0-or-later and must stay in the `agpl` extra, not in "
        "[project.dependencies] — otherwise there is no way to install without it."
    )
    assert _normalise(package) in extra, f"{package} is missing from [project.optional-dependencies].agpl"


# ---------------------------------------------------------------------------
# The default install paths keep requesting the extra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["base", "gpu"])
def test_default_requirements_request_the_agpl_extra(name: str) -> None:
    """The default install is unchanged by the split: it still gets both packages."""
    editable = [line for line in _requirement_lines(REQUIREMENTS / f"{name}.txt") if line.startswith("-e")]
    assert editable == ["-e .[dev,agpl]"], (
        f"requirements/{name}.txt must forward to `-e .[dev,agpl]` so a default install still installs the AGPL deps"
    )


@pytest.mark.parametrize("name", ["base", "gpu"])
def test_no_agpl_mirror_matches_its_default_except_for_the_extra(name: str) -> None:
    """The opt-out file differs from the default only in the requested extras."""
    default = _requirement_lines(REQUIREMENTS / f"{name}.txt")
    opt_out = _requirement_lines(REQUIREMENTS / f"{name}-no-agpl.txt")

    assert "-e .[dev]" in opt_out, f"requirements/{name}-no-agpl.txt must forward to `-e .[dev]` (no `agpl` extra)"
    assert [line for line in default if not line.startswith("-e")] == [
        line for line in opt_out if not line.startswith("-e")
    ], (
        f"requirements/{name}.txt and requirements/{name}-no-agpl.txt have drifted apart. They must stay "
        "identical apart from the requested extras, so opting out of AGPL changes nothing else."
    )


@pytest.mark.parametrize(
    ("dockerfile", "expected"),
    [("Dockerfile", "base.txt"), ("Dockerfile.gpu", "gpu.txt")],
)
def test_dockerfiles_default_to_the_agpl_requirements(dockerfile: str, expected: str) -> None:
    """Images are turnkey deployments: they include the AGPL deps unless told otherwise."""
    text = (REPO_ROOT / "docker" / dockerfile).read_text()
    assert f"ARG REQUIREMENTS={expected}" in text, (
        f"docker/{dockerfile} must default its REQUIREMENTS build arg to {expected}; the "
        f"{expected.replace('.txt', '')}-no-agpl.txt file is the opt-in-to-opting-out path."
    )
    assert "requirements/$REQUIREMENTS" in text, f"docker/{dockerfile} must install the file named by $REQUIREMENTS"


# ---------------------------------------------------------------------------
# Degrading without the extra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", AGPL_PACKAGES)
def test_message_names_the_package_the_feature_and_the_fix(package: str) -> None:
    message = agpl_unavailable_message(package, "Doing the thing")
    assert "Doing the thing" in message
    assert package in message
    assert "VTSEARCH_NO_AGPL=1" in message
    assert f"pip install {package}" in message


def test_import_error_carries_the_message() -> None:
    exc = agpl_import_error("PyMuPDF", "Rendering PDF pages")
    assert isinstance(exc, ImportError)
    assert "PyMuPDF" in str(exc)


@pytest.fixture
def without_module(monkeypatch: pytest.MonkeyPatch):
    """Make ``import <name>`` fail, as it does on a VTSEARCH_NO_AGPL install."""

    def _hide(name: str) -> None:
        # A None entry in sys.modules makes the import machinery raise
        # ImportError, without needing the package to be genuinely absent.
        monkeypatch.setitem(sys.modules, name, None)  # pyright: ignore[reportArgumentType]

    return _hide


def test_pdf_render_reports_the_missing_package(without_module, tmp_path: Path) -> None:
    from vtscore.datasets import pdf

    without_module("fitz")
    with pytest.raises(ImportError, match="PyMuPDF"):
        pdf.render_pdf_pages(tmp_path / "nonexistent.pdf")


def test_pdf_preview_reports_the_missing_package(without_module) -> None:
    from vtscore.datasets import pdf

    without_module("fitz")
    with pytest.raises(ImportError, match="PyMuPDF"):
        pdf.render_pdf_page_png(b"%PDF-1.4 not really a pdf")


@pytest.mark.parametrize(
    ("module_path", "class_name"),
    [
        ("vtscore.converters.document2image", "Document2ImageMediaConverter"),
        ("vtscore.converters.document2text", "Document2TextMediaConverter"),
    ],
)
def test_document_converters_report_the_missing_package(
    without_module,
    capsys: pytest.CaptureFixture[str],
    module_path: str,
    class_name: str,
) -> None:
    """Converters return no clips (their existing contract) but say why."""
    import importlib

    converter = getattr(importlib.import_module(module_path), class_name)()
    without_module("fitz")

    assert converter.convert({"filename": "doc.pdf", "media_bytes": b"%PDF-1.4"}) == []
    assert "PyMuPDF" in capsys.readouterr().out


def test_yolo_extractor_reports_the_missing_package(without_module) -> None:
    from vtscore.media.image.extractor import ImageClassExtractor

    without_module("ultralytics")
    with pytest.raises(ImportError, match="ultralytics"):
        ImageClassExtractor("people", "person").load_model()


def test_yolo_clipper_reports_the_missing_package(without_module) -> None:
    from vtscore.media.image.clipper import ImageObjectClipper

    without_module("ultralytics")
    with pytest.raises(ImportError, match="ultralytics"):
        ImageObjectClipper()._load_model()
