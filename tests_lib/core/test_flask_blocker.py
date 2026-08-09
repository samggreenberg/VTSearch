"""Guards for the ``./run-tests.sh vtscore-clean`` gate itself.

The gate's whole value is the claim "the library tier is import-clean of
Flask".  That claim was silently false for a while: the meta-path hook
was installed only in the controller process, while ``-n auto`` handed
every real import to xdist workers that never saw it.  A ``tests_lib``
module could have grown ``import flask`` and the gate would have stayed
green.

So these tests guard the *gate*, in three links that together can't rot:

* Under the gate (``VTSEARCH_BLOCK_FLASK`` set) the canary below proves
  Flask is genuinely unimportable **inside the worker**, both at
  test-module import time and at test-call time - the exact windows the
  controller-only hook missed.
* Always, :func:`test_conftest_installs_blocker_first` pins the
  install call to the top of ``tests_lib/conftest.py`` (the only place
  that runs early enough in a worker), and
  :func:`test_gate_script_exports_env_var` pins the controller side of
  the handshake.  Without those, the canary would skip silently and the
  gate would go vacuous again.
* Always, the mechanism tests check the hook actually raises.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests_lib import flask_blocker
from tests_lib.flask_blocker import BLOCK_ENV_VAR, BLOCKED_TOP_LEVEL


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GATE_ACTIVE = bool(os.environ.get(BLOCK_ENV_VAR))

# Canary: this runs at test-module import time, inside whichever process
# collects the module (an xdist worker under the gate).  That is precisely
# the window a controller-only blocker leaves open.
_MODULE_IMPORT_BLOCKED: bool | None = None
if _GATE_ACTIVE:
    try:
        import flask  # noqa: F401
    except ImportError:
        _MODULE_IMPORT_BLOCKED = True
    else:
        _MODULE_IMPORT_BLOCKED = False


_gate_only = pytest.mark.skipif(not _GATE_ACTIVE, reason=f"only meaningful when {BLOCK_ENV_VAR} is set")


@_gate_only
def test_flask_blocked_at_module_import_time():
    assert _MODULE_IMPORT_BLOCKED is True, (
        "vtscore-clean gate is not blocking Flask when test modules are imported - "
        "the blocker is missing from this process (an xdist worker?)."
    )


@_gate_only
@pytest.mark.parametrize("name", sorted(BLOCKED_TOP_LEVEL))
def test_blocked_modules_unimportable_at_test_time(name):
    with pytest.raises(ImportError):
        importlib.import_module(name)


@_gate_only
def test_blocker_is_on_this_processes_meta_path():
    assert flask_blocker.is_installed()


def test_conftest_installs_blocker_first():
    """``tests_lib/conftest.py`` must import+install the blocker before anything else.

    A worker imports this conftest before any test module, so it is the
    only hook point early enough to cover library imports.  Anything
    imported ahead of it is outside the gate.
    """
    source = (_REPO_ROOT / "tests_lib" / "conftest.py").read_text()
    tree = ast.parse(source)

    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)) and getattr(node, "module", None) != "__future__"
    ]
    assert imports, "conftest.py has no imports?"
    first = imports[0]
    assert isinstance(first, ast.ImportFrom) and first.module == "tests_lib.flask_blocker", (
        f"tests_lib/conftest.py must import tests_lib.flask_blocker first; found {ast.dump(first)[:120]} instead."
    )

    calls = [node.value.func for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    assert any(isinstance(func, ast.Name) and "flask_blocker" in func.id for func in calls), (
        "tests_lib/conftest.py imports the blocker but never installs it."
    )


def test_gate_script_exports_env_var():
    """The gate must hand the blocker to its xdist workers via the environment.

    ``sys.meta_path`` is per-process and execnet workers start clean, so
    the env var is the only channel; if this assignment disappears the
    gate-only tests above would skip and the gate would pass vacuously.
    """
    source = (_REPO_ROOT / "scripts" / "check-vtscore-clean.py").read_text()
    assert "os.environ[BLOCK_ENV_VAR]" in source or f'os.environ["{BLOCK_ENV_VAR}"]' in source, (
        f"scripts/check-vtscore-clean.py no longer sets {BLOCK_ENV_VAR} for xdist workers."
    )


def test_install_blocks_flask_then_restores():
    """The hook raises a directive-bearing ImportError while installed."""
    saved_meta_path = list(sys.meta_path)
    saved_modules = {k: v for k, v in sys.modules.items() if k.partition(".")[0] in BLOCKED_TOP_LEVEL}
    try:
        flask_blocker.install()
        assert flask_blocker.is_installed()
        with pytest.raises(ImportError, match="blocked in vtscore-clean test mode"):
            importlib.import_module("flask")
    finally:
        sys.meta_path[:] = saved_meta_path
        sys.modules.update(saved_modules)


def test_install_evicts_already_imported_modules():
    """A cached module never consults ``sys.meta_path``, so install() must evict it."""
    saved_meta_path = list(sys.meta_path)
    saved_modules = {k: v for k, v in sys.modules.items() if k.partition(".")[0] in BLOCKED_TOP_LEVEL}
    sentinel = object()
    sys.modules["flask"] = sentinel  # type: ignore[assignment]
    try:
        flask_blocker.install()
        assert sys.modules.get("flask") is not sentinel
    finally:
        sys.meta_path[:] = saved_meta_path
        sys.modules.pop("flask", None)
        sys.modules.update(saved_modules)


@pytest.mark.parametrize("env_value, expect_blocked", [("1", True), ("", False)])
def test_env_var_handshake_in_a_fresh_process(env_value, expect_blocked):
    """End-to-end check of the controller→worker handshake.

    Mirrors what an xdist worker does on start-up: fresh interpreter,
    inherited environment, ``install_if_requested()`` at the top of the
    conftest.  With the var set Flask must be unimportable; without it
    the blocker must stay out of the way.
    """
    env = dict(os.environ)
    if env_value:
        env[BLOCK_ENV_VAR] = env_value
    else:
        env.pop(BLOCK_ENV_VAR, None)

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "from tests_lib.flask_blocker import install_if_requested\n"
            "install_if_requested()\n"
            "try:\n"
            "    import flask\n"
            "except ImportError:\n"
            "    print('BLOCKED')\n"
            "else:\n"
            "    print('IMPORTED')\n",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ("BLOCKED" if expect_blocked else "IMPORTED")
