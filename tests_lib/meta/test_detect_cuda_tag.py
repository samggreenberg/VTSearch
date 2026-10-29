"""Tests for ``scripts/detect_cuda_tag.py``.

The helper picks the right PyTorch CUDA wheel tag from the GPU's compute
capability so ``install.sh`` doesn't make the user know their hardware (and
doesn't fall into the "newest tag is wrong for an old GPU" trap: cu128 dropped
Volta, so a V100 needs cu126). The anchor is cu129, the one tag whose torch
shares its whole CUDA 12.9 library set with RAPIDS >= 26.8; install.sh also
installs cuML beside cu126 (#4390). The selection logic is pure; the nvidia-smi parsing is
text-only. Both are covered here without a GPU.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location("detect_cuda_tag", REPO_ROOT / "scripts" / "detect_cuda_tag.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


detect_cuda_tag = _load_module()


class TestSelectCudaTag:
    def test_volta_v100_picks_cu126_not_cu129(self):
        """The motivating case: a V100 (cc 7.0) must NOT get the anchor (cu128+
        drop Volta); cu126 is the newest wheel that still covers it."""
        assert detect_cuda_tag.select_cuda_tag([(7, 0)]) == "cu126"

    def test_grid_v100s_on_cuda13_driver_picks_cu126(self):
        """The GRID's V100S nodes run driver 580 (CUDA 13.0): still cu126, the
        tag their venvs were rebuilt on (#4838)."""
        assert detect_cuda_tag.select_cuda_tag([(7, 0)], driver_cuda=(13, 0)) == "cu126"

    def test_no_candidate_tops_out_below_the_torch_floor(self):
        """cu121 and cu124 stop at torch 2.5.1 and 2.6.0, below pyproject's
        torch>=2.7. On torch 2.6, transformers >= 5.18 cannot import a model
        class when no GPU is visible (#4838), so those tags must not come back."""
        tags = {tag for tag, *_ in detect_cuda_tag._CANDIDATES}
        assert not tags & {"cu121", "cu124"}

    def test_ampere_a100_gets_default(self):
        assert detect_cuda_tag.select_cuda_tag([(8, 0)]) == "cu129"

    def test_hopper_h100_gets_default(self):
        assert detect_cuda_tag.select_cuda_tag([(9, 0)]) == "cu129"

    def test_turing_gets_default(self):
        assert detect_cuda_tag.select_cuda_tag([(7, 5)]) == "cu129"

    def test_blackwell_gets_default(self):
        """cc 10.0/12.0 is covered by cu128 and cu129; the anchor wins."""
        assert detect_cuda_tag.select_cuda_tag([(10, 0)]) == "cu129"
        assert detect_cuda_tag.select_cuda_tag([(12, 0)]) == "cu129"

    def test_old_driver_steps_down_from_cu126(self):
        """A V100 on a driver capped at CUDA 12.4 can't run cu126 -> cu118."""
        assert detect_cuda_tag.select_cuda_tag([(7, 0)], driver_cuda=(12, 4)) == "cu118"

    def test_very_old_driver_steps_down_to_cu118(self):
        assert detect_cuda_tag.select_cuda_tag([(7, 0)], driver_cuda=(11, 8)) == "cu118"

    def test_new_enough_driver_keeps_default(self):
        assert detect_cuda_tag.select_cuda_tag([(8, 0)], driver_cuda=(12, 9)) == "cu129"

    def test_driver_below_default_steps_down_to_newest_it_runs(self):
        """An A100 on a CUDA 12.6 driver can't run cu129 (or cu128) -> cu126; on
        a 12.8 driver -> cu128; on 12.4 -> cu118. Only cu126 of these gets cuML."""
        assert detect_cuda_tag.select_cuda_tag([(8, 0)], driver_cuda=(12, 6)) == "cu126"
        assert detect_cuda_tag.select_cuda_tag([(8, 0)], driver_cuda=(12, 8)) == "cu128"
        assert detect_cuda_tag.select_cuda_tag([(8, 0)], driver_cuda=(12, 4)) == "cu118"

    def test_driver_too_old_for_any_covering_wheel_still_returns_best(self):
        """If even the oldest covering wheel out-runs the driver, return it
        anyway (the caller warns); never return None just for an old driver."""
        assert detect_cuda_tag.select_cuda_tag([(7, 0)], driver_cuda=(11, 0)) == "cu118"

    def test_blackwell_on_old_driver_still_returns_cu128(self):
        """Only cu128/cu129 cover Blackwell; an old driver can't change that, so
        the oldest of the two (the one most likely to run) comes back."""
        assert detect_cuda_tag.select_cuda_tag([(12, 0)], driver_cuda=(12, 4)) == "cu128"

    def test_mixed_fleet_volta_and_blackwell_unsatisfiable(self):
        """No single wheel covers both sm_70 and sm_120 -> None (caller falls back)."""
        assert detect_cuda_tag.select_cuda_tag([(7, 0), (12, 0)]) is None

    def test_mixed_fleet_with_volta_picks_cu126(self):
        assert detect_cuda_tag.select_cuda_tag([(7, 0), (8, 6), (9, 0)]) == "cu126"

    def test_empty_caps_returns_none(self):
        assert detect_cuda_tag.select_cuda_tag([]) is None


class TestParseComputeCaps:
    def test_single_gpu(self):
        assert detect_cuda_tag.parse_compute_caps("7.0\n") == [(7, 0)]

    def test_multiple_gpus(self):
        assert detect_cuda_tag.parse_compute_caps("8.0\n8.0\n9.0\n") == [(8, 0), (8, 0), (9, 0)]

    def test_skips_unparseable_lines(self):
        # Older drivers emit "[N/A]"; blank lines also appear.
        assert detect_cuda_tag.parse_compute_caps("[N/A]\n\n8.6\n") == [(8, 6)]

    def test_empty_output(self):
        assert detect_cuda_tag.parse_compute_caps("") == []


class TestParseDriverCuda:
    def test_parses_banner(self):
        banner = "| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4 |"
        assert detect_cuda_tag.parse_driver_cuda(banner) == (12, 4)

    def test_missing_returns_none(self):
        assert detect_cuda_tag.parse_driver_cuda("no cuda banner here") is None


class TestDetect:
    def test_detect_happy_path(self):
        def fake_run(args):
            if "--query-gpu=compute_cap" in args:
                return "7.0\n"
            return "CUDA Version: 13.0\n"

        with mock.patch.object(detect_cuda_tag, "_run", side_effect=fake_run):
            tag, explanation = detect_cuda_tag.detect()
        assert tag == "cu126"
        assert "7.0" in explanation
        assert "cu126" in explanation

    def test_detect_no_nvidia_smi(self):
        with mock.patch.object(detect_cuda_tag, "_run", return_value=None):
            tag, explanation = detect_cuda_tag.detect()
        assert tag is None
        assert "nvidia-smi" in explanation

    def test_detect_no_usable_caps(self):
        def fake_run(args):
            if "--query-gpu=compute_cap" in args:
                return "[N/A]\n"
            return ""

        with mock.patch.object(detect_cuda_tag, "_run", side_effect=fake_run):
            tag, explanation = detect_cuda_tag.detect()
        assert tag is None
        assert "no usable compute capability" in explanation

    def test_main_prints_tag_on_success(self, capsys):
        with mock.patch.object(detect_cuda_tag, "detect", return_value=("cu126", "why")):
            rc = detect_cuda_tag.main()
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip() == "cu126"
        assert "why" in captured.err

    def test_main_no_stdout_on_failure(self, capsys):
        with mock.patch.object(detect_cuda_tag, "detect", return_value=(None, "nope")):
            rc = detect_cuda_tag.main()
        captured = capsys.readouterr()
        assert rc == 1
        assert captured.out.strip() == ""
        assert "nope" in captured.err


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestInstallShRefusesRetiredTags:
    """``install.sh cu124`` would install torch 2.6.0+cu124 and then let the
    requirements pass swap in PyPI's newest torch to meet the torch>=2.7 floor,
    usually a build without the GPU's kernels. The guard at the top of
    ``vts_install_gpu`` stops first (#4838).

    Only that function runs here, and the first helper it calls after the guard
    is stubbed to exit 99, so a tag that gets past the guard installs nothing.
    """

    @staticmethod
    def _run_install_gpu(tag: str) -> subprocess.CompletedProcess[str]:
        script = (
            "eval \"$(sed -n '/^vts_install_gpu() {/,/^}/p' scripts/install.sh)\"\n"
            "vts_progress_init() { exit 99; }\n"
            f"vts_install_gpu {tag}\n"
        )
        return subprocess.run(  # noqa: S603 - fixed argv, no user input
            ["bash", "-c", script],  # noqa: S607 - bash from PATH, like the installer itself
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    @pytest.mark.parametrize("tag", ["cu121", "cu124", "cu117"])
    def test_tag_below_the_torch_floor_is_refused(self, tag):
        proc = self._run_install_gpu(tag)
        assert proc.returncode == 2, proc.stderr
        assert "cu126" in proc.stderr

    @pytest.mark.parametrize("tag", ["cu118", "cu126", "cu129", "cu130"])
    def test_supported_tag_gets_past_the_guard(self, tag):
        assert self._run_install_gpu(tag).returncode == 99
