"""Tests for ``scripts/setuptools_spec.py``.

``install.sh`` used to pin ``setuptools<82`` for torch 2.12.0's sake; a venv
rebuilt with that pin landed on setuptools 81, which carries PYSEC-2026-3447 and
turned every GRID suite run red at pip-audit (#3905). The helper now reads the
installed torch's own requirement instead. The ``Requires-Dist`` lines below are
copied from the real torch wheels' metadata on PyPI.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location("setuptools_spec", REPO_ROOT / "scripts" / "setuptools_spec.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


setuptools_spec = _load_module()

TORCH_2_6 = ["filelock", 'setuptools; python_version >= "3.12"', "sympy==1.13.1"]
TORCH_2_12 = ["filelock", "setuptools<82", "sympy>=1.13.3"]
TORCH_2_14 = ["filelock", "setuptools>=77.0.3", "sympy>=1.13.3"]


class TestSelectSetuptoolsSpec:
    def test_no_torch_gets_the_fixed_floor(self):
        assert setuptools_spec.select_setuptools_spec(None) == ("setuptools>=83.0.0", None)

    def test_unversioned_requirement_gets_the_fixed_floor(self):
        """torch <= 2.10 (the GRID's 2.6.0+cu124) says plain ``setuptools``."""
        assert setuptools_spec.select_setuptools_spec(TORCH_2_6) == ("setuptools>=83.0.0", None)

    def test_a_floor_that_admits_the_fix_gets_the_fixed_floor(self):
        """torch 2.13 / 2.14 ask for >=77.0.3, which 83 satisfies."""
        assert setuptools_spec.select_setuptools_spec(TORCH_2_14) == ("setuptools>=83.0.0", None)

    def test_a_torch_cap_below_the_fix_wins_and_says_so(self):
        """torch 2.11-2.12 cap setuptools<82: upgrading past it is the resolver
        ERROR the old pin silenced, so follow torch and warn about the audit."""
        spec, note = setuptools_spec.select_setuptools_spec(TORCH_2_12)
        assert spec == "setuptools<82"
        assert note and "PYSEC-2026-3447" in note

    def test_a_cap_behind_a_false_marker_is_ignored(self):
        spec, note = setuptools_spec.select_setuptools_spec(['setuptools<82; python_version < "3.0"'])
        assert (spec, note) == ("setuptools>=83.0.0", None)

    def test_a_cap_on_an_extra_is_ignored(self):
        spec, _ = setuptools_spec.select_setuptools_spec(['setuptools<81; extra == "export"'])
        assert spec == "setuptools>=83.0.0"

    def test_other_packages_and_junk_lines_are_ignored(self):
        spec, _ = setuptools_spec.select_setuptools_spec(["setuptools-rust<1", "not a requirement !!", "numpy<2"])
        assert spec == "setuptools>=83.0.0"


class TestMain:
    def test_prints_only_the_spec_on_stdout(self, capsys):
        with mock.patch.object(setuptools_spec, "requires", return_value=TORCH_2_12):
            assert setuptools_spec.main() == 0
        out, err = capsys.readouterr()
        assert out == "setuptools<82\n"
        assert "warning:" in err

    def test_missing_torch_is_not_an_error(self, capsys):
        with mock.patch.object(setuptools_spec, "requires", side_effect=PackageNotFoundError("torch")):
            assert setuptools_spec.main() == 0
        assert capsys.readouterr().out == "setuptools>=83.0.0\n"

    def test_never_fails_the_install(self, capsys):
        with mock.patch.object(setuptools_spec, "select_setuptools_spec", side_effect=RuntimeError("boom")):
            assert setuptools_spec.main() == 0
        assert capsys.readouterr().out == "setuptools>=83.0.0\n"

    def test_install_sh_no_longer_hard_pins_below_the_fix(self):
        text = (REPO_ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
        assert '"setuptools<82" wheel' not in text
        assert "setuptools_spec.py" in text
