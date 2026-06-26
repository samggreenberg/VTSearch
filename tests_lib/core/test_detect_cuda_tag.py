"""Tests for ``scripts/detect_cuda_tag.py``.

The helper picks the right PyTorch CUDA wheel tag from the GPU's compute
capability so ``install-gpu.sh`` doesn't make the user know their hardware (and
doesn't fall into the "newest tag is wrong for an old GPU" trap: cu128 dropped
Volta, so a V100 needs cu124). The selection logic is pure; the nvidia-smi
parsing is text-only. Both are covered here without a GPU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location("detect_cuda_tag", REPO_ROOT / "scripts" / "detect_cuda_tag.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


detect_cuda_tag = _load_module()


class TestSelectCudaTag:
    def test_volta_v100_picks_cu124_not_cu128(self):
        """The motivating case: a V100 (cc 7.0) must NOT get cu128 (drops Volta)."""
        assert detect_cuda_tag.select_cuda_tag([(7, 0)]) == "cu124"

    def test_ampere_a100_gets_default(self):
        assert detect_cuda_tag.select_cuda_tag([(8, 0)]) == "cu124"

    def test_hopper_h100_gets_default(self):
        assert detect_cuda_tag.select_cuda_tag([(9, 0)]) == "cu124"

    def test_turing_gets_default(self):
        assert detect_cuda_tag.select_cuda_tag([(7, 5)]) == "cu124"

    def test_blackwell_requires_cu128(self):
        """cc 10.0/12.0 is outside cu124's range, so it must bump to cu128."""
        assert detect_cuda_tag.select_cuda_tag([(10, 0)]) == "cu128"
        assert detect_cuda_tag.select_cuda_tag([(12, 0)]) == "cu128"

    def test_old_driver_steps_down_from_cu124(self):
        """A V100 on a driver capped at CUDA 12.2 can't run cu124 -> cu121."""
        assert detect_cuda_tag.select_cuda_tag([(7, 0)], driver_cuda=(12, 2)) == "cu121"

    def test_very_old_driver_steps_down_to_cu118(self):
        assert detect_cuda_tag.select_cuda_tag([(7, 0)], driver_cuda=(11, 8)) == "cu118"

    def test_new_enough_driver_keeps_default(self):
        assert detect_cuda_tag.select_cuda_tag([(8, 0)], driver_cuda=(12, 4)) == "cu124"

    def test_driver_too_old_for_any_covering_wheel_still_returns_best(self):
        """If even the oldest covering wheel out-runs the driver, return it
        anyway (the caller warns); never return None just for an old driver."""
        assert detect_cuda_tag.select_cuda_tag([(7, 0)], driver_cuda=(11, 0)) == "cu118"

    def test_blackwell_on_old_driver_still_returns_cu128(self):
        """Only cu128 covers Blackwell; an old driver can't change that."""
        assert detect_cuda_tag.select_cuda_tag([(12, 0)], driver_cuda=(12, 4)) == "cu128"

    def test_mixed_fleet_volta_and_blackwell_unsatisfiable(self):
        """No single wheel covers both sm_70 and sm_120 -> None (caller falls back)."""
        assert detect_cuda_tag.select_cuda_tag([(7, 0), (12, 0)]) is None

    def test_mixed_fleet_within_one_wheel_picks_default(self):
        assert detect_cuda_tag.select_cuda_tag([(7, 0), (8, 6), (9, 0)]) == "cu124"

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
            return "CUDA Version: 12.4\n"

        with mock.patch.object(detect_cuda_tag, "_run", side_effect=fake_run):
            tag, explanation = detect_cuda_tag.detect()
        assert tag == "cu124"
        assert "7.0" in explanation
        assert "cu124" in explanation

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
        with mock.patch.object(detect_cuda_tag, "detect", return_value=("cu124", "why")):
            rc = detect_cuda_tag.main()
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip() == "cu124"
        assert "why" in captured.err

    def test_main_no_stdout_on_failure(self, capsys):
        with mock.patch.object(detect_cuda_tag, "detect", return_value=(None, "nope")):
            rc = detect_cuda_tag.main()
        captured = capsys.readouterr()
        assert rc == 1
        assert captured.out.strip() == ""
        assert "nope" in captured.err
