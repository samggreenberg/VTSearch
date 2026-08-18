"""Tests for ``VTSEARCH_IMAGE_PROCESSOR_BACKEND`` / ``_DEVICE`` (issue #3146).

These knobs decide which implementation resizes and normalises an image before
the encoder sees it, and on which device.  The reason they exist is a silent
change rather than a wanted feature: ``transformers`` 5 removed the ``Fast``
suffix on image processors, so ``SiglipImageProcessor`` went from meaning the
PIL implementation to meaning the torchvision one **without the name changing**,
and ``requirements/image-embedders.txt`` pins only ``transformers>=4.49`` — a
range that spans the flip.  Two hosts can therefore produce different pixels
from identical code and weights.

So the tests that matter most here are the ones about *defaults* and *silence*:

* the default must pass **nothing**, because the entire pre-embedded pile was
  built that way and any other default would silently invalidate it;
* a typo must not quietly change what gets embedded;
* the ``backend``/``use_fast`` spelling must follow the installed transformers
  rather than being assumed, because an unknown kwarg is swallowed by several
  processor classes instead of raising.

Every test reloads ``vtscore.config`` because the modes are read at import time.
"""

from __future__ import annotations

import importlib
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def restore_reloaded_modules():
    """Undo the ``importlib.reload`` of ``vtscore.config`` for the rest of the session."""
    import vtscore.config as config

    snapshot = dict(config.__dict__)
    yield
    config.__dict__.clear()
    config.__dict__.update(snapshot)


def _config_with(env: dict[str, str], device: str = "cuda"):
    """Reload ``vtscore.config`` under *env*, with ``resolve_device`` pinned to *device*."""
    import vtscore.config as config

    with mock.patch.dict("os.environ", env, clear=False):
        config = importlib.reload(config)
    config.resolve_device = lambda: device  # type: ignore[assignment]
    return config


class TestDefaultPassesNothing:
    """The default must reproduce the pile byte-for-byte, which means passing nothing."""

    def test_unset_env_is_auto(self):
        config = _config_with({}, device="cuda")
        assert config.IMAGE_PROCESSOR_BACKEND == "auto"
        assert config.IMAGE_PROCESSOR_DEVICE == "auto"

    def test_auto_sends_no_load_kwargs(self):
        config = _config_with({}, device="cuda")
        assert config.image_processor_load_kwargs() == {}

    def test_auto_sends_no_call_kwargs(self):
        config = _config_with({}, device="cuda")
        assert config.image_processor_call_kwargs() == {}


class TestBackendSelection:
    def test_pil_is_requested_by_name(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "pil"})
        kwargs = config.image_processor_load_kwargs()
        assert kwargs in ({"backend": "pil"}, {"use_fast": False})

    def test_torchvision_is_requested_by_name(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "torchvision"})
        kwargs = config.image_processor_load_kwargs()
        assert kwargs in ({"backend": "torchvision"}, {"use_fast": True})

    def test_case_and_whitespace_are_tolerated(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "  PIL  "})
        assert config.IMAGE_PROCESSOR_BACKEND == "pil"
        assert config.image_processor_load_kwargs() != {}

    def test_a_typo_degrades_to_passing_nothing(self):
        """A misspelled mode must not silently change what gets embedded."""
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "torchvsion"})
        assert config.image_processor_load_kwargs() == {}


class TestKwargSpellingFollowsTheInstalledLibrary:
    """``use_fast`` on v4, ``backend`` on v5 — resolved, never assumed.

    Guessing wrong here fails *silently*: several processor classes swallow an
    unknown kwarg into ``**kwargs`` rather than raising, so the request would be
    dropped and the arm would quietly be the default one.
    """

    def test_v5_uses_backend(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "pil"})
        with mock.patch.object(config, "_transformers_backend_kwarg", lambda: "backend"):
            assert config.image_processor_load_kwargs() == {"backend": "pil"}

    def test_v4_uses_use_fast(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "torchvision"})
        with mock.patch.object(config, "_transformers_backend_kwarg", lambda: "use_fast"):
            assert config.image_processor_load_kwargs() == {"use_fast": True}

    def test_v4_maps_pil_to_use_fast_false(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "pil"})
        with mock.patch.object(config, "_transformers_backend_kwarg", lambda: "use_fast"):
            assert config.image_processor_load_kwargs() == {"use_fast": False}

    def test_no_transformers_passes_nothing(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "pil"})
        with mock.patch.object(config, "_transformers_backend_kwarg", lambda: ""):
            assert config.image_processor_load_kwargs() == {}

    def test_the_resolver_reads_the_installed_version(self):
        config = _config_with({})
        assert config._transformers_backend_kwarg() in ("backend", "use_fast", "")


class TestDeviceSelection:
    def test_cuda_is_requested_on_cuda(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_DEVICE": "cuda"}, device="cuda")
        assert config.image_processor_call_kwargs() == {"device": "cuda"}

    def test_cuda_degrades_off_cuda_rather_than_raising(self):
        """An escape hatch that crashes on a laptop is not an escape hatch."""
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_DEVICE": "cuda"}, device="cpu")
        assert config.image_processor_call_kwargs() == {}

    def test_cpu_is_explicit_and_survives_off_cuda(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_DEVICE": "cpu"}, device="cpu")
        assert config.image_processor_call_kwargs() == {"device": "cpu"}

    def test_a_typo_degrades_to_passing_nothing(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_DEVICE": "gpu"}, device="cuda")
        assert config.image_processor_call_kwargs() == {}


class TestTheKnobsAreIndependent:
    """The backend is a load-time choice and the device a call-time one.

    They are separate kwargs on separate calls, so setting one must not disturb
    the other -- a study varies them independently.
    """

    def test_backend_alone_leaves_the_call_untouched(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "pil"}, device="cuda")
        assert config.image_processor_call_kwargs() == {}

    def test_device_alone_leaves_the_load_untouched(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_DEVICE": "cuda"}, device="cuda")
        assert config.image_processor_load_kwargs() == {}
