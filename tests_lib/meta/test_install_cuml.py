"""Tests for ``vts_install_cuml`` in ``scripts/install.sh`` (#4390).

The CUDA tag decides whether cuML is installed and which spec. RAPIDS >= 26.8
(the first whose cudf takes pandas 3) goes beside the cu129 torch, which pins
the CUDA 12.9 set it is built on, and beside the cu126 torch, which leaves
nvjitlink open so pip lifts it to 12.9 (checked on a V100S and an A100). The
untested tags skip the step. The function runs here against a fake ``vts_run``
that records the pip argv.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run(tag: str, env: dict[str, str] | None = None) -> tuple[list[str], str]:
    script = (
        "eval \"$(sed -n '/^vts_install_cuml() {/,/^}/p' scripts/install.sh)\"\n"
        'vts_run() { shift; echo "run $*"; }\n'
        f"vts_install_cuml {tag}\n"
    )
    out = subprocess.run(  # noqa: S603 - fixed argv, no user input
        ["bash", "-c", script],  # noqa: S607
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", **(env or {})},
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line.startswith("run ")], out.stdout


PIP = "run pip install --extra-index-url https://pypi.nvidia.com --prefer-binary"


@pytest.mark.parametrize("tag", ["cu126", "cu129"])
def test_cu126_and_cu129_install_rapids_26_8(tag):
    assert _run(tag)[0] == [f"{PIP} cuml-cu12>=26.8"]


def test_cu13_installs_the_cu13_wheel():
    assert _run("cu130")[0] == [f"{PIP} cuml-cu13>=26.8"]


@pytest.mark.parametrize("tag", ["cu118", "cu128"])
def test_untested_tags_skip_with_a_message(tag):
    calls, stdout = _run(tag)
    assert calls == []
    assert f"this install is {tag}" in stdout


def test_spec_override_unskips():
    calls, _ = _run("cu128", {"VTS_CUML_CU12_SPEC": "cuml-cu12<26"})
    assert calls == [f"{PIP} cuml-cu12<26"]


def test_skip_env_installs_nothing():
    assert _run("cu129", {"VTSEARCH_SKIP_CUML": "1"})[0] == []
