"""A calibration cell runs on one set of numeric kernels on every node (#4110).

The GRID mixes AVX-512 CPUs with AVX2-only ones, and OpenBLAS, numpy, torch's
MKL and torch's ATen each pick kernels by CPU.  Their last-bit differences moved
a cell's threshold by 1e-6 and, votes later, a pick, so the same arm on the same
commit was not the same cell on an Intel and an AMD node (#3840).  The fix is
four env pins (``_expcommon.CPU_KERNEL_PINS``) that only work if they are set
before numpy and torch load, and that fail silently if they are not -- so these
tests pin the order as well as the values.
"""

from __future__ import annotations

import ast
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS = _ROOT / "scripts" / "experiments"
_RUN_CELLS = _EXPERIMENTS / "calibration" / "run_cells.py"


@pytest.fixture(scope="module")
def expcommon():
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    import _expcommon

    return _expcommon


@pytest.fixture
def x86(monkeypatch, expcommon):
    monkeypatch.setattr(expcommon.platform, "machine", lambda: "x86_64")


class TestPinCpuKernels:
    def test_defaults_all_four(self, expcommon, x86):
        env: dict[str, str] = {}
        assert expcommon.pin_cpu_kernels(env) == expcommon.CPU_KERNEL_PINS
        assert env == expcommon.CPU_KERNEL_PINS

    def test_an_exported_value_wins(self, expcommon, x86):
        env = {"MKL_CBWR": "AVX2", "NPY_DISABLE_CPU_FEATURES": ""}
        set_now = expcommon.pin_cpu_kernels(env)
        assert env["MKL_CBWR"] == "AVX2" and env["NPY_DISABLE_CPU_FEATURES"] == ""
        assert "MKL_CBWR" not in set_now and "NPY_DISABLE_CPU_FEATURES" not in set_now
        assert env["ATEN_CPU_CAPABILITY"] == "avx2"

    def test_opt_out_sets_nothing(self, expcommon, x86):
        env = {"VTS_KERNEL_PINS": "0"}
        assert expcommon.pin_cpu_kernels(env) == {}
        assert env == {"VTS_KERNEL_PINS": "0"}

    def test_not_on_another_architecture(self, expcommon, monkeypatch):
        monkeypatch.setattr(expcommon.platform, "machine", lambda: "arm64")
        env: dict[str, str] = {}
        assert expcommon.pin_cpu_kernels(env) == {}

    def test_refuses_once_numpy_is_loaded(self, expcommon, x86, monkeypatch):
        """Set after the import, a pin would do nothing and say nothing."""
        import numpy  # noqa: F401

        for k in expcommon.CPU_KERNEL_PINS:
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(RuntimeError, match="before numpy"):
            expcommon.pin_cpu_kernels()

    def test_the_report_names_every_pin(self, expcommon):
        line = expcommon.kernel_report()
        assert all(f"{k}=" in line for k in expcommon.CPU_KERNEL_PINS)


def _is_call(node: ast.stmt, name: str) -> bool:
    """*node* is a bare ``<...>.name()`` statement."""
    return (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func).endswith(name)
    )


class TestRunCellsPinsFirst:
    """Source order, because a pin set after numpy loads is silently ignored."""

    @staticmethod
    def _module() -> ast.Module:
        return ast.parse(_RUN_CELLS.read_text(encoding="utf-8"))

    def test_pins_before_setup_env_and_numpy(self):
        body = self._module().body
        pin_line = next(
            node.lineno
            for node in body
            if isinstance(node, ast.If)
            and "__main__" in ast.unparse(node.test)
            and any(_is_call(n, "pin_cpu_kernels") for n in node.body)
        )
        numpy_line = next(
            node.lineno for node in body if isinstance(node, ast.Import) and any(a.name == "numpy" for a in node.names)
        )
        setup_line = next(node.lineno for node in body if _is_call(node, "setup_env"))
        assert pin_line < setup_line < numpy_line

    def test_only_as_a_script(self):
        """A test that imports run_cells must not switch its own process to the pinned kernels."""
        body = self._module().body
        bare = [n for n in body if _is_call(n, "pin_cpu_kernels")]
        assert not bare, "pin_cpu_kernels() must sit under `if __name__ == '__main__':`"


class TestGridenvAgrees:
    def test_gridenv_exports_the_same_numpy_pin(self, expcommon):
        """Two homes, one value: every GRID job gets the numpy pin, the cell runner all four."""
        src = (_ROOT / "gridenv.sh").read_text(encoding="utf-8")
        m = re.search(r'export NPY_DISABLE_CPU_FEATURES="\$\{NPY_DISABLE_CPU_FEATURES-([^}]*)\}"', src)
        assert m, "gridenv.sh must default NPY_DISABLE_CPU_FEATURES (#4110)"
        assert m.group(1) == expcommon.CPU_KERNEL_PINS["NPY_DISABLE_CPU_FEATURES"]
        assert 'export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"' in src


@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"), reason="the pin is x86-64 only")
class TestThisNumpyAcceptsThePin:
    def test_imports_without_a_warning_and_turns_the_targets_off(self, expcommon):
        """A numpy that dropped either target from its dispatch list would warn at every import.

        That warning is an ``ImportWarning``, hidden by default, so a renamed target
        would leave the pin quietly half-applied; ``-W error`` makes it fail here instead.
        """
        env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=expcommon.CPU_KERNEL_PINS["NPY_DISABLE_CPU_FEATURES"])
        code = (
            "try:\n"
            "    from numpy._core._multiarray_umath import __cpu_features__ as F\n"
            "except ImportError:\n"
            "    from numpy.core._multiarray_umath import __cpu_features__ as F\n"
            "assert not F['AVX512F'] and not F['AVX512_SKX'], F\n"
        )
        proc = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
            [sys.executable, "-W", "error", "-c", code], env=env, capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr
