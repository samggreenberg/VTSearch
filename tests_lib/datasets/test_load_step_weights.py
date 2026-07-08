"""GPU/CPU pacing profiles for the dataset-load progress bar.

``load_step_weights`` picks a heavier finalize slice on a GPU host, where the
fast embed phase no longer dominates wall-clock and the un-accelerated finalize
(registry serialize/write + CPU k-means) does. See
``vtscore/datasets/stages/_common.py``.
"""

from __future__ import annotations

from vtscore.datasets.stages._common import (
    _LOAD_STEP_WEIGHTS_CPU,
    _LOAD_STEP_WEIGHTS_CPU_IMAGE,
    _LOAD_STEP_WEIGHTS_GPU,
    _TOTAL_LOAD_STEPS,
    load_step_weights,
)


def test_profiles_are_well_formed():
    for weights in (_LOAD_STEP_WEIGHTS_CPU, _LOAD_STEP_WEIGHTS_CPU_IMAGE, _LOAD_STEP_WEIGHTS_GPU):
        assert len(weights) == _TOTAL_LOAD_STEPS
        assert all(w >= 0 for w in weights)
        assert sum(weights) == 1.0


def test_image_cpu_profile_shifts_off_embed_onto_download_and_finalize():
    # An image embed is a single cheap ViT forward per item, so the image-on-CPU
    # profile gives embed (index 2) a smaller slice than the generic CPU profile
    # and download (0) + finalize (3) larger ones.
    assert _LOAD_STEP_WEIGHTS_CPU_IMAGE[2] < _LOAD_STEP_WEIGHTS_CPU[2]
    assert _LOAD_STEP_WEIGHTS_CPU_IMAGE[0] > _LOAD_STEP_WEIGHTS_CPU[0]
    assert _LOAD_STEP_WEIGHTS_CPU_IMAGE[3] > _LOAD_STEP_WEIGHTS_CPU[3]


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


def test_selects_image_cpu_profile_for_image_media_on_cpu_host(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cpu")
    assert load_step_weights("image") == _LOAD_STEP_WEIGHTS_CPU_IMAGE
    # Non-image media types keep the generic CPU profile.
    assert load_step_weights("audio") == _LOAD_STEP_WEIGHTS_CPU


def test_image_media_type_does_not_override_gpu_profile(monkeypatch):
    # The GPU profile is media-agnostic; image on a CUDA host still gets it.
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cuda")
    assert load_step_weights("image") == _LOAD_STEP_WEIGHTS_GPU


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


# --- n-aware affine cost model -------------------------------------------------

import pytest  # noqa: E402

from vtscore.datasets.stages import _load_cost_model as _cm  # noqa: E402


def _inject_row(monkeypatch, key, coeffs, *, bandwidth=10.0):
    monkeypatch.setattr(_cm, "LOAD_COST_MODEL", {key: coeffs})
    monkeypatch.setattr(_cm, "DOWNLOAD_MB_PER_S", bandwidth)


def test_n_aware_weights_come_from_cost_model(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cuda")
    _inject_row(
        monkeypatch,
        ("cuda", "image", "siglip"),
        {"a_model": 0.3, "a_embed": 1.0, "b_embed": 0.01, "a_fin": 1.0, "b_fin": 0.002},
        bandwidth=10.0,
    )
    w = load_step_weights("image", n=1000, download_size_mb=100.0, embedder="siglip")
    # T = [download 100/10=10, model 0.3, embed 1+10=11, finalize 1+2=3] -> /24.3
    assert w == pytest.approx([10 / 24.3, 0.3 / 24.3, 11 / 24.3, 3 / 24.3])
    assert sum(w) == pytest.approx(1.0)


def test_small_n_weights_model_heavier_and_large_n_weights_embed_heavier(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cuda")
    _inject_row(
        monkeypatch,
        ("cuda", "image", "siglip"),
        {"a_model": 5.0, "a_embed": 1.0, "b_embed": 0.01, "a_fin": 1.0, "b_fin": 0.001},
    )
    small = load_step_weights("image", n=100, download_size_mb=0, embedder="siglip")
    large = load_step_weights("image", n=10000, download_size_mb=0, embedder="siglip")
    # model (index 1) is a fixed cost -> a larger *fraction* at small n.
    assert small[1] > large[1]
    # embed (index 2) grows with n -> a larger fraction at large n.
    assert large[2] > small[2]


def test_download_slice_collapses_when_size_zero(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cuda")
    _inject_row(
        monkeypatch,
        ("cuda", "image", "siglip"),
        {"a_model": 0.3, "a_embed": 1.0, "b_embed": 0.01, "a_fin": 1.0, "b_fin": 0.002},
    )
    w = load_step_weights("image", n=1000, download_size_mb=0, embedder="siglip")
    assert w[0] == 0.0  # no archive fetched (local import / cached re-add)


def test_unknown_n_returns_static_profile(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cuda")
    _inject_row(
        monkeypatch,
        ("cuda", "image", "siglip"),
        {"a_model": 0.3, "a_embed": 1.0, "b_embed": 0.01, "a_fin": 1.0, "b_fin": 0.002},
    )
    # n unknown (folder importer) -> static asymptote, not the affine model.
    assert load_step_weights("image", embedder="siglip") == _LOAD_STEP_WEIGHTS_GPU


def test_no_matching_cost_model_row_returns_static(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cpu")
    _inject_row(monkeypatch, ("cuda", "image", "siglip"), {"a_model": 0.3, "a_embed": 1.0, "b_embed": 0.01, "a_fin": 1.0, "b_fin": 0.002})
    # device cpu -> no ("cpu", image, x) row -> static image-CPU profile.
    assert load_step_weights("image", n=1000, download_size_mb=100.0, embedder="siglip") == _LOAD_STEP_WEIGHTS_CPU_IMAGE


def test_shipped_cost_model_table_is_well_formed():
    # Every checked-in coefficient row has the affine keys and yields a
    # normalized weight vector; empty table (pre-calibration) trivially passes.
    for key, coeffs in _cm.LOAD_COST_MODEL.items():
        assert set(coeffs) >= {"a_model", "a_embed", "b_embed", "a_fin", "b_fin"}
        device, media, embedder = key
        w = _cm.cost_model_weights(device, media, embedder, n=1000, download_size_mb=100.0)
        assert w is not None
        assert len(w) == _TOTAL_LOAD_STEPS
        assert all(x >= 0 for x in w)
        assert sum(w) == pytest.approx(1.0)
