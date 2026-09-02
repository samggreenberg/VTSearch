"""Tests for ``VTSEARCH_EMBED_PRECISION`` (issue #3143).

The knob decides whether an image embedder's forward runs in fp32 or half
precision.  What matters most here is the **default**: the entire pre-embedded
pile and every published calibration result are fp32, half precision shifts
cosine similarities by ~1e-3, and the studies resolve effects of 0.005 — so a
mode that silently turned itself on would invalidate results months later.
These tests pin that default, pin the degradations (no CUDA, no bf16, a typo),
and pin the contract that only *compute* is half while stored vectors stay fp32.

Every test re-reads ``vtscore.config`` because the mode is read at import time;
``restore_reloaded_modules`` puts the session's module state back afterwards
(see :mod:`tests_lib.core.test_torch_config` for why that matters).

The mode and its resolvers live in :mod:`vtscore.config.device`, so that is what
:func:`_config_with` reloads and returns: a stub installed on the ``vtscore.config``
package would be a copy the resolvers never read.  Tests that stub for a
*consumer* (``vtscore.media.embedder``) still patch the package, because a
consumer outside the package does resolve the name there.
"""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def restore_reloaded_modules():
    """Undo the reload of the ``vtscore.config`` package for the rest of the session."""
    import vtscore.config as config

    modules = [config, *(sys.modules[f"vtscore.config.{name}"] for name in config._RELOAD_ORDER)]
    snapshots = [(module, dict(module.__dict__)) for module in modules]
    yield
    for module, snapshot in snapshots:
        module.__dict__.clear()
        module.__dict__.update(snapshot)


def _config_with(env: dict[str, str], device: str = "cuda"):
    """Re-read the precision mode under *env*, with ``resolve_device`` pinned to *device*.

    Returns :mod:`vtscore.config.device` rather than the package: the resolvers
    under test read their own module globals, so that is where the pin has to go.
    """
    import vtscore.config as config
    from vtscore.config import device as device_mod

    with mock.patch.dict("os.environ", env, clear=False):
        config._reload_all()
    device_mod.resolve_device = lambda: device  # type: ignore[assignment]
    return device_mod


class TestDefaultIsFp32:
    """The shipped default must be full precision, and must stay that way."""

    def test_unset_env_is_fp32(self):
        config = _config_with({}, device="cuda")
        assert config.EMBED_PRECISION == "fp32"
        assert config.embed_precision() == "fp32"

    def test_default_casts_nothing(self):
        config = _config_with({}, device="cuda")
        assert config.embed_weight_dtype() is None
        assert config.embed_autocast_dtype() is None

    def test_explicit_fp32_is_the_escape_hatch(self):
        config = _config_with({"VTSEARCH_EMBED_PRECISION": "fp32"}, device="cuda")
        assert config.embed_precision() == "fp32"
        assert config.embed_weight_dtype() is None


class TestModeResolution:
    def test_fp16_casts_weights_not_autocast(self):
        config = _config_with({"VTSEARCH_EMBED_PRECISION": "fp16"}, device="cuda")
        assert config.embed_precision() == "fp16"
        assert config.embed_weight_dtype() is torch.float16
        assert config.embed_autocast_dtype() is None

    def test_autocast_fp16_casts_per_op_not_weights(self):
        config = _config_with({"VTSEARCH_EMBED_PRECISION": "autocast_fp16"}, device="cuda")
        assert config.embed_precision() == "autocast_fp16"
        assert config.embed_weight_dtype() is None
        assert config.embed_autocast_dtype() is torch.float16

    def test_bf16_needs_hardware_support(self):
        config = _config_with({"VTSEARCH_EMBED_PRECISION": "bf16"}, device="cuda")
        config._bf16_supported = lambda: True  # type: ignore[assignment]
        assert config.embed_precision() == "bf16"
        assert config.embed_weight_dtype() is torch.bfloat16

    def test_bf16_on_sm70_degrades_to_fp32_not_fp16(self):
        """A V100 is sm_70.  Substituting fp16 would put a format nobody asked
        for into a study that named one, so it degrades all the way to fp32."""
        config = _config_with({"VTSEARCH_EMBED_PRECISION": "bf16"}, device="cuda")
        config._bf16_supported = lambda: False  # type: ignore[assignment]
        assert config.embed_precision() == "fp32"

    def test_auto_prefers_bf16_then_fp16(self):
        config = _config_with({"VTSEARCH_EMBED_PRECISION": "auto"}, device="cuda")
        config._bf16_supported = lambda: True  # type: ignore[assignment]
        assert config.embed_precision() == "bf16"
        config._bf16_supported = lambda: False  # type: ignore[assignment]
        assert config.embed_precision() == "fp16"

    def test_case_and_whitespace_tolerated(self):
        config = _config_with({"VTSEARCH_EMBED_PRECISION": "  FP16 "}, device="cuda")
        assert config.embed_precision() == "fp16"


class TestDegradations:
    def test_half_off_cuda_degrades_to_fp32(self):
        """Half on CPU is emulated and slower than the fp32 it replaced, so the
        knob must not become a performance trap on CPU-only hosts."""
        for mode in ("fp16", "bf16", "auto", "autocast_fp16"):
            config = _config_with({"VTSEARCH_EMBED_PRECISION": mode}, device="cpu")
            assert config.embed_precision() == "fp32", mode
            assert config.embed_weight_dtype() is None, mode
            assert config.embed_autocast_dtype() is None, mode

    def test_unknown_mode_warns_and_degrades(self, caplog):
        config = _config_with({"VTSEARCH_EMBED_PRECISION": "fp8_maybe"}, device="cuda")
        with caplog.at_level("WARNING"):
            assert config.embed_precision() == "fp32"
        assert "fp8_maybe" in caplog.text


class TestAutocastContext:
    def test_fp32_yields_without_entering_autocast(self):
        from vtscore.media.embedder import embed_autocast

        with mock.patch("vtscore.config.embed_autocast_dtype", return_value=None):
            with embed_autocast():
                entered = True
        assert entered

    def test_autocast_is_a_plain_context_manager(self):
        """It must compose in the ``with torch.no_grad(), embed_autocast():``
        shape every forward now uses, fp32 included."""
        from vtscore.media.embedder import embed_autocast

        with mock.patch("vtscore.config.embed_autocast_dtype", return_value=None):
            with torch.no_grad(), embed_autocast():
                assert not torch.is_grad_enabled()

    def test_autocast_actually_casts_the_op(self):
        """Asserted through the op's output dtype rather than
        ``torch.is_autocast_enabled``, whose signature moved between torch
        versions.  ``torch.autocast`` takes a CPU device type, so this is
        exercisable without a GPU even though the resolver never asks for half
        there."""
        from vtscore.media.embedder import embed_autocast

        with (
            mock.patch("vtscore.config.embed_autocast_dtype", return_value=torch.bfloat16),
            mock.patch("vtscore.config.resolve_device", return_value="cpu"),
        ):
            with torch.no_grad(), embed_autocast():
                out = torch.zeros(2, 4) @ torch.zeros(4, 3)
        assert out.dtype is torch.bfloat16

    def test_no_cast_at_the_default(self):
        from vtscore.media.embedder import embed_autocast

        with mock.patch("vtscore.config.embed_autocast_dtype", return_value=None):
            with torch.no_grad(), embed_autocast():
                out = torch.zeros(2, 4) @ torch.zeros(4, 3)
        assert out.dtype is torch.float32


class TestStoredVectorsStayFp32:
    def test_to_float32_upcasts_half(self):
        from vtscore.media.embedder import to_float32

        half = torch.zeros(4, dtype=torch.float16)
        assert to_float32(half).dtype is torch.float32

    def test_to_float32_leaves_fp32_alone(self):
        from vtscore.media.embedder import to_float32

        full = torch.zeros(4, dtype=torch.float32)
        assert to_float32(full).dtype is torch.float32

    def test_to_float32_tolerates_a_non_tensor(self):
        from vtscore.media.embedder import to_float32

        sentinel = object()
        assert to_float32(sentinel) is sentinel


class TestModelInputMarshalling:
    """``to_model_inputs`` is what makes a weight cast work at all."""

    @staticmethod
    def _model(dtype: torch.dtype):
        model = torch.nn.Linear(2, 2)
        return model.to(dtype)

    def test_floating_inputs_follow_the_weight_dtype(self):
        from vtscore.media.embedder import to_model_inputs

        out = to_model_inputs({"pixel_values": torch.zeros(1, 3, 4, 4)}, self._model(torch.float16))
        assert out["pixel_values"].dtype is torch.float16

    def test_integer_inputs_are_never_cast(self):
        """Casting ``input_ids`` to half corrupts every id above 2048 - quietly,
        which is most of a real vocabulary."""
        from vtscore.media.embedder import to_model_inputs

        ids = torch.tensor([[101, 40311, 999]], dtype=torch.int64)
        out = to_model_inputs(
            {"input_ids": ids, "attention_mask": torch.ones(1, 3, dtype=torch.int64)},
            self._model(torch.float16),
        )
        assert out["input_ids"].dtype is torch.int64
        assert out["attention_mask"].dtype is torch.int64
        assert torch.equal(out["input_ids"], ids)

    def test_fp32_model_leaves_inputs_fp32(self):
        from vtscore.media.embedder import to_model_inputs

        out = to_model_inputs({"pixel_values": torch.zeros(1, 3, 4, 4)}, self._model(torch.float32))
        assert out["pixel_values"].dtype is torch.float32

    def test_non_tensor_values_pass_through(self):
        from vtscore.media.embedder import to_model_inputs

        out = to_model_inputs({"pixel_values": torch.zeros(1, 2), "meta": "keep"}, self._model(torch.float32))
        assert out["meta"] == "keep"


class TestToComputeDeviceOptIn:
    """The cast is opt-in per embedder: the audio/video/face backbones share
    ``to_compute_device`` and their numerics were never measured (#3143)."""

    def test_no_cast_without_allow_half(self):
        from vtscore.media.embedder import to_compute_device

        with (
            mock.patch("vtscore.config.resolve_device", return_value="cpu"),
            mock.patch("vtscore.config.embed_weight_dtype", return_value=torch.float16),
        ):
            model = to_compute_device(torch.nn.Linear(2, 2))
        assert next(model.parameters()).dtype is torch.float32

    def test_cast_applied_with_allow_half(self):
        from vtscore.media.embedder import to_compute_device

        with (
            mock.patch("vtscore.config.resolve_device", return_value="cpu"),
            mock.patch("vtscore.config.embed_weight_dtype", return_value=torch.float16),
        ):
            model = to_compute_device(torch.nn.Linear(2, 2), allow_half=True)
        assert next(model.parameters()).dtype is torch.float16

    def test_allow_half_is_a_noop_at_the_default(self):
        from vtscore.media.embedder import to_compute_device

        with (
            mock.patch("vtscore.config.resolve_device", return_value="cpu"),
            mock.patch("vtscore.config.embed_weight_dtype", return_value=None),
        ):
            model = to_compute_device(torch.nn.Linear(2, 2), allow_half=True)
        assert next(model.parameters()).dtype is torch.float32


class TestEveryImageEmbedderOptedIn:
    """A new image embedder that forgets ``allow_half=True`` would silently
    ignore the knob, which reads as "fp16 did nothing here" rather than as a
    wiring bug.  Cheaper to assert than to discover in a run."""

    def test_image_load_sites_pass_allow_half(self):
        from pathlib import Path

        import vtscore.media.image as image_pkg

        root = Path(image_pkg.__file__).parent
        offenders = []
        for path in sorted(root.glob("*.py")):
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if "to_compute_device(" in line and "allow_half=True" not in line and "def " not in line:
                    offenders.append(f"{path.name}:{lineno}")
        assert not offenders, f"image embedders not honouring VTSEARCH_EMBED_PRECISION: {offenders}"


class TestNumpyContract:
    def test_half_forward_still_yields_fp32_vectors(self):
        """End-to-end on the shape the embedders use: a half forward, upcast,
        then ``.cpu().numpy()`` must produce a float32 array - a float16 array
        here would halve the precision of every stored vector and matrix."""
        from vtscore.media.embedder import to_float32

        model = torch.nn.Linear(4, 3).to(torch.float16)
        with torch.no_grad():
            out = model(torch.zeros(2, 4, dtype=torch.float16))
            arr = to_float32(out.detach()).cpu().numpy()
        assert arr.dtype == np.float32
        assert arr.shape == (2, 3)
