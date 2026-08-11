"""Actionable errors for the opt-out (AGPL-3.0) dependencies.

Two runtime dependencies are AGPL-3.0-or-later: ``ultralytics`` (YOLO) and
``PyMuPDF``.  They live in the ``agpl`` extra in ``pyproject.toml`` rather than
in ``[project.dependencies]``, but **every documented install path requests
that extra** (``requirements/base.txt`` and ``requirements/gpu.txt`` are
``-e .[dev,agpl]``, and both Dockerfiles install those files), so a normal
install has them exactly as it always did.

The split exists so a deployment that cannot take copyleft code can say no:
``VTSEARCH_NO_AGPL=1 bash scripts/install.sh``, or installing the
``requirements/*-no-agpl.txt`` mirrors directly.  On such an install the
features backed by these packages are simply unavailable, and the bare
``ImportError`` a lazy import would raise says nothing about why.  These
helpers turn it into a message that names the package, the feature, and the
way back.

Every call site keeps its plain ``import`` inside a ``try`` (rather than going
through :func:`importlib.import_module` here) so static analysis still sees the
real module and its stubs::

    try:
        import fitz
    except ImportError as exc:
        raise agpl_import_error("PyMuPDF", "Rendering PDF pages") from exc
"""

from __future__ import annotations

# Package name -> the pip install target and a one-line note on what it is.
# Keyed by the distribution name as it appears in the ``agpl`` extra.
_AGPL_PACKAGES: dict[str, str] = {
    "PyMuPDF": "PDF rendering and document text extraction",
    "ultralytics": "YOLO object detection",
}


def agpl_unavailable_message(package: str, feature: str) -> str:
    """Explain that *feature* needs the opt-out AGPL package *package*.

    Args:
        package: Distribution name, e.g. ``"PyMuPDF"`` or ``"ultralytics"``.
        feature: What the caller was trying to do, phrased as a sentence
            subject, e.g. ``"Rendering PDF pages"``.

    Returns:
        A single-line message naming the package, why it may be missing, and
        the two ways to get it back.
    """
    what = _AGPL_PACKAGES.get(package, "")
    what = f" ({what})" if what else ""
    return (
        f"{feature} needs the '{package}' package{what}, which is not installed. "
        f"It is AGPL-3.0-or-later, so it lives in VTSearch's optional 'agpl' extra: a default "
        f"install includes it, but an install made with VTSEARCH_NO_AGPL=1 (or from a "
        f"requirements/*-no-agpl.txt file) leaves it out on purpose. Install it with "
        f"'pip install {package}', or reinstall with the extra: 'pip install -e .[agpl]'. "
        f"See NOTICE for the licensing implications."
    )


def agpl_import_error(package: str, feature: str) -> ImportError:
    """Build the :class:`ImportError` to raise from a failed AGPL-package import.

    Raise it ``from`` the original ``ImportError`` so the underlying failure
    (a genuinely broken install, rather than a missing one) stays visible.
    """
    return ImportError(agpl_unavailable_message(package, feature))
