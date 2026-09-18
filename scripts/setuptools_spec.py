#!/usr/bin/env python3
"""Pick the setuptools requirement ``scripts/install.sh`` should upgrade to.

setuptools below **83.0.0** carries PYSEC-2026-3447, and ``run-tests.sh``'s
pip-audit gate fails on it. The installer used to pin ``setuptools<82``
unconditionally (76813f0ea) because torch 2.12.0 declares ``setuptools<82`` and
upgrading past that printed a red resolver ERROR on every install. That pin
outlived its reason (#3905): torch 2.13 and 2.14 declare ``setuptools>=77.0.3``,
torch <= 2.10 declares plain ``setuptools``, and only **2.11.x-2.12.x** cap it --
which today means the ``cu128`` index, whose newest torch is 2.11.0. A venv
rebuilt with the stale pin (the GRID's, 2026-09-15) landed on setuptools 81 and
turned every suite run red.

So the choice is made against the torch actually in the venv: the fixed floor
(:data:`FLOOR`) unless that torch's own metadata excludes it, in which case
torch's requirement wins -- installing past it would only reproduce the
resolver error -- and a note on stderr says pip-audit will flag the result.

Prints the requirement to **stdout** (e.g. ``setuptools>=83``); the note goes to
**stderr** so the caller can capture just the spec. Always exits ``0``: this
must never be the step that stops an install.

Runs under the venv's own interpreter, *before* project dependencies are
installed, so it uses only the stdlib plus the ``packaging`` that pip vendors.
The selection (:func:`select_setuptools_spec`) is pure and unit-tested in
``tests_lib/meta/test_setuptools_spec.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, requires

#: The first setuptools release without PYSEC-2026-3447.
FIXED_VERSION = "83.0.0"
FLOOR = f"setuptools>={FIXED_VERSION}"


def _packaging():
    """``packaging.requirements.Requirement``, from the package or pip's vendored copy."""
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:  # a bare venv: pip always vendors it
        from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
    return Requirement, InvalidRequirement


def select_setuptools_spec(torch_requires: Iterable[str] | None) -> tuple[str, str | None]:
    """Return ``(spec, note)`` for the torch whose ``Requires-Dist`` is *torch_requires*.

    *torch_requires* is ``None`` when torch is not installed. *note* is ``None``
    unless torch forces a setuptools that still carries the advisory.
    """
    Requirement, InvalidRequirement = _packaging()
    for line in torch_requires or ():
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        if req.name.lower() != "setuptools" or not req.specifier:
            continue
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        if req.specifier.contains(FIXED_VERSION, prereleases=True):
            continue
        return (
            f"setuptools{req.specifier}",
            f"the installed torch requires setuptools{req.specifier}, which excludes the "
            f"PYSEC-2026-3447 fix ({FIXED_VERSION}); keeping torch's pin, so pip-audit "
            "will flag setuptools until torch is upgraded (#3905)",
        )
    return FLOOR, None


def main() -> int:
    try:
        torch_requires = requires("torch")
    except PackageNotFoundError:
        torch_requires = None
    try:
        spec, note = select_setuptools_spec(torch_requires)
    except Exception as exc:  # noqa: BLE001 - never the step that stops an install
        print(f"setuptools_spec: {exc}; defaulting to {FLOOR}", file=sys.stderr)
        spec, note = FLOOR, None
    if note:
        print(f"warning: {note}", file=sys.stderr)
    print(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
