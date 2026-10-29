"""The ``vtscore.cli`` package keeps its public surface and its old module names (#5397).

``vtscore/cli.py`` became a package, one module per stage, and the three
``cli_*`` siblings moved into it.  What an out-of-tree caller imports must
still resolve: the entry points from ``vtscore.cli``, and the old module names
as deprecated aliases that hand back the moved module itself - so a format set,
or a stub patched, through either name is the one the CLI reads.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

import vtscore.cli
from vtscore.cli import pipeline, progress, resume

ALIASES = [
    ("vtscore.cli_progress", progress),
    ("vtscore.cli_resume", resume),
    ("vtscore.cli_pipeline", pipeline),
]


@pytest.mark.parametrize(("old_name", "module"), ALIASES)
def test_an_old_module_name_is_the_moved_module_and_warns(old_name, module, monkeypatch):
    monkeypatch.delitem(sys.modules, old_name, raising=False)
    with pytest.warns(DeprecationWarning, match=f"{old_name} has moved to {module.__name__}"):
        assert importlib.import_module(old_name) is module
    assert sys.modules[old_name] is module


def test_state_set_through_the_old_name_is_the_new_modules(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("vtscore.cli_progress")

    monkeypatch.setattr(old, "_format", "json")
    assert progress.get_format() == "json"


@pytest.mark.parametrize("name", vtscore.cli.__all__)
def test_the_public_names_import_from_the_package(name):
    assert getattr(vtscore.cli, name).__module__.startswith("vtscore.cli.")
