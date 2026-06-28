"""GPU/CPU pacing profiles for the dataset-load progress bar.

``load_step_weights`` picks a heavier finalize slice on a GPU host, where the
fast embed phase no longer dominates wall-clock and the un-accelerated finalize
(registry serialize/write + CPU k-means) does. See
``vtscore/datasets/stages/_common.py``.
"""

from __future__ import annotations

from vtscore.datasets.stages._common import (
    _LOAD_STEP_WEIGHTS_CPU,
    _LOAD_STEP_WEIGHTS_GPU,
    _TOTAL_LOAD_STEPS,
    load_step_weights,
)


def test_profiles_are_well_formed():
    for weights in (_LOAD_STEP_WEIGHTS_CPU, _LOAD_STEP_WEIGHTS_GPU):
        assert len(weights) == _TOTAL_LOAD_STEPS
        assert all(w >= 0 for w in weights)
        assert sum(weights) == 1.0


def test_gpu_profile_weights_finalize_more_than_cpu():
    # The whole point of the GPU profile: finalize (index 3) gets a bigger slice
    # and embed (index 2) a smaller one, because embedding is GPU-accelerated
    # but the finalize work is not.
    assert _LOAD_STEP_WEIGHTS_GPU[3] > _LOAD_STEP_WEIGHTS_CPU[3]
    assert _LOAD_STEP_WEIGHTS_GPU[2] < _LOAD_STEP_WEIGHTS_CPU[2]


def test_selects_cpu_profile_on_cpu_host(monkeypatch):
    # ``load_step_weights`` imports ``resolve_device`` lazily from
    # ``vtscore.config``, so patch it there.
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cpu")
    assert load_step_weights() == _LOAD_STEP_WEIGHTS_CPU


def test_selects_gpu_profile_on_cuda_host(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cuda")
    assert load_step_weights() == _LOAD_STEP_WEIGHTS_GPU


def test_selects_gpu_profile_on_indexed_cuda_device(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cuda:1")
    assert load_step_weights() == _LOAD_STEP_WEIGHTS_GPU


def test_falls_back_to_cpu_when_device_resolution_raises(monkeypatch):
    def _boom():
        raise RuntimeError("no torch")

    monkeypatch.setattr("vtscore.config.resolve_device", _boom)
    assert load_step_weights() == _LOAD_STEP_WEIGHTS_CPU
