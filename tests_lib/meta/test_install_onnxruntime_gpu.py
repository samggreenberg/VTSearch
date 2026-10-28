"""Tests for ``vts_install_onnxruntime_gpu`` in ``scripts/install.sh`` (#5332).

onnxruntime and onnxruntime-gpu install the same ``onnxruntime`` module, so the
CPU build must be uninstalled *before* the GPU build goes in: uninstalling it
afterwards deletes files the GPU build shares. The function runs here against a
fake ``pip`` that records its argv.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run(env: dict[str, str], fail_install: str = "") -> list[str]:
    script = (
        "eval \"$(sed -n '/^vts_install_onnxruntime_gpu() {/,/^}/p' scripts/install.sh)\"\n"
        f'pip() {{ echo "pip $*"; [ -n "{fail_install}" ] && [ "$1" = install ] && [ "$2" = "{fail_install}" ] && return 1; return 0; }}\n'
        "vts_install_onnxruntime_gpu\n"
    )
    out = subprocess.run(  # noqa: S603 - fixed argv, no user input
        ["bash", "-c", script],  # noqa: S607
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line.startswith("pip ")]


def test_uninstalls_the_cpu_build_before_installing_the_pin():
    assert _run({}) == [
        "pip uninstall -y onnxruntime",
        "pip install onnxruntime-gpu==1.26.0 --progress-bar on",
    ]


def test_spec_override():
    calls = _run({"VTS_ONNXRUNTIME_GPU_SPEC": "onnxruntime-gpu==1.22.0"})
    assert calls[-1] == "pip install onnxruntime-gpu==1.22.0 --progress-bar on"


def test_skip_leaves_the_cpu_build_alone():
    assert _run({"VTSEARCH_SKIP_ORT_GPU": "1"}) == []


def test_a_failed_gpu_install_puts_the_cpu_build_back():
    calls = _run({}, fail_install="onnxruntime-gpu==1.26.0")
    assert calls[-1] == "pip install onnxruntime --progress-bar on"


def test_runs_inside_the_gpu_install():
    text = (REPO_ROOT / "scripts" / "install.sh").read_text()
    body = text[text.index("\nvts_install_gpu() {") :]
    assert "vts_install_onnxruntime_gpu" in body[: body.index("\n}\n")]
