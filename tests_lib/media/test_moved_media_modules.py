"""The public ``vtscore.media`` module paths #5401 moved still import.

Each ``_MOVED_MODULES`` entry is an extension-facing promise: code written
against the old path keeps working, warns, and gets the very module the app
runs (so a sentinel is never executed twice and a ``mock.patch`` through the
old path lands).  A real file re-added at an old path would shadow the alias
for good, so the deprecation warning is asserted too.
"""

import importlib
import sys

import pytest

from vtscore.media import _MOVED_MODULES


@pytest.mark.parametrize(("old", "new"), sorted(_MOVED_MODULES.items()))
def test_old_path_resolves_to_the_moved_module(old, new):
    sys.modules.pop(old, None)
    with pytest.deprecated_call(match=new):
        aliased = importlib.import_module(old)
    assert aliased is importlib.import_module(new)


def test_from_import_through_an_old_path():
    sys.modules.pop("vtscore.media.clipper", None)
    with pytest.deprecated_call():
        # The alias is runtime-only: no file sits at the old path for a type checker to follow.
        from vtscore.media.clipper import MediaClipper  # pyright: ignore[reportMissingImports]

    from vtscore.media import MediaClipper as root_export

    assert MediaClipper is root_export


def test_unmoved_names_are_left_to_the_real_finders():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("vtscore.media.no_such_module")
