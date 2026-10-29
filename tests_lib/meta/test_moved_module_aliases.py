"""The loose top-level ``vtscore`` modules keep their old names (#5398).

``io``, ``gpu_backends``, ``single_instance``, ``host_seams`` and
``achievements_hooks`` moved into packages.  Out-of-tree code may import them
by the old names, so each old name is a deprecated alias that hands back the
moved module itself: a recorder registered, or a stub patched, through either
name is the one the library reads.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import pytest

import vtscore.config as config
from vtscore.concurrency import single_instance
from vtscore.config import gpu_backends
from vtscore.host import achievements, seams
from vtscore.utils import io

ALIASES = [
    ("vtscore.io", io),
    ("vtscore.gpu_backends", gpu_backends),
    ("vtscore.single_instance", single_instance),
    ("vtscore.host_seams", seams),
    ("vtscore.achievements_hooks", achievements),
]


@pytest.mark.parametrize(("old_name", "module"), ALIASES)
def test_an_old_module_name_is_the_moved_module_and_warns(old_name, module, monkeypatch):
    monkeypatch.delitem(sys.modules, old_name, raising=False)
    with pytest.warns(DeprecationWarning, match=f"{old_name} has moved to {module.__name__}"):
        assert importlib.import_module(old_name) is module
    assert sys.modules[old_name] is module


def test_a_recorder_registered_through_the_old_name_is_the_one_the_library_calls(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("vtscore.achievements_hooks")

    seen = []
    monkeypatch.setitem(achievements._recorders, "vote", seen.append)
    old.record_achievement("vote", "via the alias")
    assert seen == ["via the alias"]


def test_the_config_package_re_exports_the_gpu_backend_selectors():
    for name in ("cuml_enabled", "kmeans_fit_predict", "umap_fit_transform"):
        assert getattr(config, name) is getattr(gpu_backends, name)


def test_importing_the_config_package_does_not_load_numpy():
    """``app.py`` sets the native-math thread counts before numpy loads; the
    config package, which re-exports the GPU backends, must not load it first."""
    code = "import sys, vtscore.config; assert 'numpy' not in sys.modules, 'vtscore.config loaded numpy'"
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
