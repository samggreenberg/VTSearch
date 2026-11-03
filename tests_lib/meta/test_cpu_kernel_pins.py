"""A calibration cell runs on one set of numeric kernels on every node (#4110).

The GRID mixes AVX-512 CPUs with AVX2-only ones, and OpenBLAS, numpy, torch's
MKL and torch's ATen each pick kernels by CPU.  Their last-bit differences moved
a cell's threshold by 1e-6 and, votes later, a pick, so the same arm on the same
commit was not the same cell on an Intel and an AMD node (#3840).  The fix is
four env pins (``_expcommon.cpu_kernel_pins``) that only work if they are set
before numpy and torch load, and that fail silently if they are not -- so these
tests pin the order as well as the values.  numpy's pin names dispatch targets,
which numpy 2.4 renamed, so it is chosen by numpy's version, and the last class
checks that the numpy installed here takes it (#4930).
"""

from __future__ import annotations

import ast
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS = _ROOT / "scripts" / "experiments"
_GRIDENV = _ROOT / "scripts" / "grid" / "gridenv.sh"
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
    def test_defaults_all_of_them(self, expcommon, x86):
        env: dict[str, str] = {}
        assert expcommon.pin_cpu_kernels(env) == expcommon.cpu_kernel_pins()
        assert env == expcommon.cpu_kernel_pins()
        assert env["NPY_DISABLE_CPU_FEATURES"] == expcommon.numpy_kernel_pin()
        assert env["OPENBLAS_CORETYPE"] == "Haswell" and env["OPENBLAS_L2_SIZE"] == "2048"

    @pytest.mark.parametrize("coretype", ["Haswell", "HASWELL"])
    def test_the_l2_size_goes_with_an_exported_haswell(self, expcommon, x86, coretype):
        """#4938: the L2 size is what keeps the Haswell blocking on a Zen 4 CPU."""
        env = {"OPENBLAS_CORETYPE": coretype}
        assert expcommon.pin_cpu_kernels(env)["OPENBLAS_L2_SIZE"] == "2048"

    def test_but_not_with_another_core(self, expcommon, x86):
        """Another core's blocking may follow the L2 size, so it is left to report the real one."""
        env = {"OPENBLAS_CORETYPE": "SkylakeX"}
        set_now = expcommon.pin_cpu_kernels(env)
        assert "OPENBLAS_L2_SIZE" not in env and "OPENBLAS_L2_SIZE" not in set_now

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

        for k in expcommon.cpu_kernel_pins():
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(RuntimeError, match="before numpy"):
            expcommon.pin_cpu_kernels()

    def test_the_report_names_every_pin(self, expcommon):
        line = expcommon.kernel_report()
        assert all(f"{k}=" in line for k in expcommon.cpu_kernel_pins())

    def test_the_report_names_the_kernels_numpy_runs(self, expcommon):
        """A pin numpy ignored still reads as set, so the report also says what numpy dispatched to (#4930)."""
        import numpy

        line = expcommon.kernel_report()
        assert f" numpy={numpy.__version__} numpy_targets=" in line


class TestNumpyKernelPin:
    @pytest.mark.parametrize(
        ("version", "pin"),
        [
            ("1.26.4", "AVX512F AVX512_SKX"),
            ("2.2.6", "AVX512F AVX512_SKX"),
            ("2.3.5", "AVX512F AVX512_SKX"),
            ("2.4.0rc1", "X86_V4 AVX512_ICL AVX512_SPR"),
            ("2.4.0", "X86_V4 AVX512_ICL AVX512_SPR"),
            ("2.5.3", "X86_V4 AVX512_ICL AVX512_SPR"),
        ],
    )
    def test_by_version(self, expcommon, version, pin):
        """numpy 2.4 renamed its x86 dispatch targets; each value is ignored by the other numpy."""
        assert expcommon.numpy_kernel_pin(version) == pin

    def test_the_installed_numpy_without_importing_it(self, expcommon):
        """numpy reads the pin when it loads, so choosing the pin must not load it."""
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import _expcommon\n"
            "print(_expcommon.numpy_kernel_pin()); assert 'numpy' not in sys.modules\n"
        )
        proc = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
            [sys.executable, "-c", code, str(_EXPERIMENTS)], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == expcommon.numpy_kernel_pin(importlib.metadata.version("numpy"))


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


_GRIDENV_PINS = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "OPENBLAS_L2_SIZE")


def _source_gridenv(
    tmp_path: Path, python: str | None, var: str = "NPY_DISABLE_CPU_FEATURES", **env: str
) -> subprocess.CompletedProcess[str]:
    """Source a copy of ``gridenv.sh`` in a scratch tree; print *var* as it left it.

    A copy, so the ``.shadow/`` it creates lands in *tmp_path*, not the checkout.
    *python* is what a bare ``python`` runs (a venv's), or None for one that fails.
    The pins start unset unless *env* exports them.
    """
    tree = tmp_path / "wt"
    (tree / "scripts" / "experiments").mkdir(parents=True)
    (tree / "scripts" / "grid").mkdir()
    (tree / "scripts" / "grid" / "gridenv.sh").write_text(_GRIDENV.read_text(encoding="utf-8"), encoding="utf-8")
    (tree / "scripts" / "experiments" / "_expcommon.py").write_text(
        (_EXPERIMENTS / "_expcommon.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # A wrapper, not a symlink: a venv's python finds its venv from where it was invoked.
    (bin_dir / "python").write_text("#!/bin/sh\n" + ("exit 1\n" if python is None else f'exec "{python}" "$@"\n'))
    (bin_dir / "python").chmod(0o755)
    base = {k: v for k, v in os.environ.items() if k not in _GRIDENV_PINS}
    base.update(VTS_PYTHON_MODULE="", VTS_VENV="", PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", **env)
    show = f'printf %s "${{{var}-unset}}"'
    return subprocess.run(  # noqa: S603  # fixed argv, a scratch copy of the repo's script
        ["bash", "-c", f'source "$1/scripts/grid/gridenv.sh" && {show}', "_", str(tree)],  # noqa: S607
        capture_output=True,
        text=True,
        env=base,
        timeout=60,
    )


class TestGridenvAgrees:
    def test_openblas_pin(self):
        src = _GRIDENV.read_text(encoding="utf-8")
        assert 'export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"' in src

    def test_the_l2_size_goes_with_haswell(self, expcommon, tmp_path):
        """#4938: the same L2 size the cell runner sets, so the Haswell blocking holds on a Zen 4 CPU."""
        proc = _source_gridenv(tmp_path, sys.executable, var="OPENBLAS_L2_SIZE")
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == expcommon.cpu_kernel_pins()["OPENBLAS_L2_SIZE"]

    def test_but_not_with_another_core(self, tmp_path):
        proc = _source_gridenv(tmp_path, sys.executable, var="OPENBLAS_L2_SIZE", OPENBLAS_CORETYPE="SkylakeX")
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == "unset"

    def test_an_exported_l2_size_wins(self, tmp_path):
        """An empty one included: OpenBLAS then detects the cache, the opt-out."""
        proc = _source_gridenv(tmp_path, sys.executable, var="OPENBLAS_L2_SIZE", OPENBLAS_L2_SIZE="")
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""

    @pytest.mark.skipif(platform.machine() != "x86_64", reason="gridenv.sh pins numpy on x86-64 only")
    def test_gridenv_exports_the_same_numpy_pin(self, expcommon, tmp_path):
        """Two homes, one value: every GRID job gets the numpy pin, the cell runner all four."""
        proc = _source_gridenv(tmp_path, sys.executable)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == expcommon.numpy_kernel_pin()

    def test_an_exported_value_wins(self, tmp_path):
        proc = _source_gridenv(tmp_path, sys.executable, NPY_DISABLE_CPU_FEATURES="")
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""

    @pytest.mark.skipif(platform.machine() != "x86_64", reason="gridenv.sh pins numpy on x86-64 only")
    def test_says_so_when_it_cannot_choose(self, tmp_path):
        """No numpy to read, no pin: said on stderr rather than a guess numpy might ignore."""
        proc = _source_gridenv(tmp_path, None)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == "unset"
        assert "#4930" in proc.stderr


#: The dispatch targets an AVX2-only node (EPYC 7713, Xeon E5-2698 v4) can run, in
#: numpy's names before 2.4 and from 2.4 on.  A pinned numpy runs nothing else.
_AVX2_LEVEL = ("SSSE3", "SSE41", "POPCNT", "SSE42", "AVX", "F16C", "FMA3", "AVX2", "X86_V2", "X86_V3")


@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"), reason="the pin is x86-64 only")
class TestThisNumpyAcceptsThePin:
    def test_imports_without_a_warning_and_turns_the_targets_off(self, expcommon):
        """The installed numpy takes its pin, and no optimized ufunc runs above AVX2 under it.

        numpy skips a name its build does not dispatch, with an ``ImportWarning``
        hidden by default, which is how numpy 2.4's renamed targets left the pre-2.4
        pin disabling nothing (#4930); ``-W error`` makes that fail here.
        ``__cpu_features__`` reports the CPU's flags as well as the targets turned
        off, so a target the pin does not name (``AVX512_ICL`` before numpy 2.4) can
        read as on while no kernel uses it.  The ufunc check reads
        what runs instead.  It can only fail on an AVX-512 CPU; an AVX2 one runs
        nothing above AVX2 either way.
        """
        pin = expcommon.numpy_kernel_pin()
        env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=pin)
        code = (
            "import sys\n"
            "try:\n"
            "    from numpy._core._multiarray_umath import __cpu_features__ as F\n"
            "except ImportError:\n"
            "    from numpy.core._multiarray_umath import __cpu_features__ as F\n"
            "on = [t for t in sys.argv[1].split() if F.get(t)]\n"
            "assert not on, f'pinned targets still on: {on}'\n"
            "from numpy.lib import introspect\n"
            "run = {d['current'] for s in introspect.opt_func_info().values() for d in s.values()}\n"
            "above = sorted(t for t in run if not t.startswith('baseline') and t not in sys.argv[2].split())\n"
            "assert not above, f'ufuncs run above AVX2: {above}'\n"
        )
        proc = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
            [sys.executable, "-W", "error", "-c", code, pin, " ".join(_AVX2_LEVEL)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
