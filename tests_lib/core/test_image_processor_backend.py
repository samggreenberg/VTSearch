"""Tests for ``VTSEARCH_IMAGE_PROCESSOR_BACKEND`` / ``_DEVICE`` (issue #3146).

These knobs decide which implementation resizes and normalises an image before
the encoder sees it, and on which device.  The reason they exist is a silent
change rather than a wanted feature: ``transformers`` 5 removed the ``Fast``
suffix on image processors, so ``SiglipImageProcessor`` went from meaning the
PIL implementation to meaning the torchvision one **without the name changing**,
and ``requirements/image-embedders.txt`` pins only ``transformers>=4.49`` — a
range that spans the flip.  Two hosts can therefore produce different pixels
from identical code and weights.

Issue #3173 then changed the default from ``auto`` (pass nothing, inherit
whatever the resolver picked) to ``torchvision`` (name it).  The pile is
torchvision-built, so on a transformers 5 host that is a no-op; on a 4.x host it
is a real behaviour change, and the intended one.

So the tests that matter most here are the ones about *defaults* and *silence*:

* the default must **name** ``torchvision``, because ``auto`` means "whatever
  this host resolved" and that is not a property of this repository;
* a typo must land on the default rather than on ``auto`` — answering a
  misspelling with the one host-dependent mode is the failure #3173 removes;
* the ``backend``/``use_fast`` spelling must follow the installed transformers
  rather than being assumed, because an unknown kwarg is swallowed by several
  processor classes instead of raising;
* a request that transformers *cannot honour* must be read back off the loaded
  class, not assumed — it warns and falls back rather than raising, so the
  default outcome of an impossible request is a mislabelled processor.

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


class TestTheDefaultNamesTheBackend:
    """Unset must mean ``torchvision`` outright, not "whatever this host picked" (#3173)."""

    def test_unset_backend_is_torchvision(self):
        config = _config_with({}, device="cuda")
        assert config.IMAGE_PROCESSOR_BACKEND == "torchvision"

    def test_unset_device_is_still_auto(self):
        """Only the backend flipped; the device knob is a perf choice still gated on #3146."""
        config = _config_with({}, device="cuda")
        assert config.IMAGE_PROCESSOR_DEVICE == "auto"

    def test_the_default_sends_a_backend_request(self):
        config = _config_with({}, device="cuda")
        assert config.image_processor_load_kwargs() in ({"backend": "torchvision"}, {"use_fast": True})

    def test_the_default_still_sends_no_call_kwargs(self):
        config = _config_with({}, device="cuda")
        assert config.image_processor_call_kwargs() == {}

    def test_auto_remains_reachable_as_the_opt_out(self):
        """The pre-#3173 behaviour has to stay available, or the change is untestable."""
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "auto"}, device="cuda")
        assert config.image_processor_load_kwargs() == {}


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

    def test_a_typo_degrades_to_the_default_not_to_auto(self):
        """A misspelled mode must land where an *unset* one lands, which is no longer ``auto``.

        Degrading to "pass nothing" would answer a typo with the single mode
        whose meaning depends on the host — the opposite of what #3173 is for.
        """
        typo = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "torchvsion"}).image_processor_load_kwargs()
        unset = _config_with({}).image_processor_load_kwargs()
        assert typo == unset != {}

    def test_a_typo_resolves_to_torchvision(self):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "torchvsion"})
        assert config.image_processor_load_kwargs() in ({"backend": "torchvision"}, {"use_fast": True})


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
        """Not "no load kwargs" — the *same* load kwargs an unset device would give."""
        with_device = _config_with(
            {"VTSEARCH_IMAGE_PROCESSOR_DEVICE": "cuda"}, device="cuda"
        ).image_processor_load_kwargs()
        baseline = _config_with({}, device="cuda").image_processor_load_kwargs()
        assert with_device == baseline


class _FakeProcessor:
    """A stand-in whose *class name* is the whole payload.

    The backend is read off the constructed class, so the only thing a double
    needs is to be named like one.  Built with ``type()`` per test rather than
    declared, because a declared class cannot vary its own name.
    """


def _processor_named(name: str):
    return type(name, (_FakeProcessor,), {})()


def _composite_wrapping(name: str):
    """A ``CLIPProcessor``-shaped wrapper: the image half hangs off ``.image_processor``."""
    return type("FakeProcessor", (_FakeProcessor,), {"image_processor": _processor_named(name)})()


class TestReadingTheBackendOffTheLoadedClass:
    """``…Pil`` and ``…Fast`` name themselves; a bare name only means something with a version."""

    def test_pil_suffix_is_pil(self):
        config = _config_with({})
        assert config.resolved_processor_backend(_processor_named("SiglipImageProcessorPil")) == "pil"

    def test_fast_suffix_is_torchvision(self):
        config = _config_with({})
        assert config.resolved_processor_backend(_processor_named("SiglipImageProcessorFast")) == "torchvision"

    def test_a_bare_name_is_torchvision_on_v5(self):
        config = _config_with({})
        with mock.patch.object(config, "_transformers_major", lambda: 5):
            assert config.resolved_processor_backend(_processor_named("SiglipImageProcessor")) == "torchvision"

    def test_a_bare_name_is_pil_on_v4(self):
        """The rename is the entire confusion, so it is resolved here and nowhere else."""
        config = _config_with({})
        with mock.patch.object(config, "_transformers_major", lambda: 4):
            assert config.resolved_processor_backend(_processor_named("SiglipImageProcessor")) == "pil"

    def test_a_composite_wrapper_is_unwrapped(self):
        config = _config_with({})
        assert config.resolved_processor_backend(_composite_wrapping("CLIPImageProcessorPil")) == "pil"

    def test_a_non_processor_abstains_rather_than_guessing(self):
        config = _config_with({})
        assert config.resolved_processor_backend(_processor_named("SiglipTokenizer")) is None

    def test_a_bare_name_with_no_transformers_abstains(self):
        config = _config_with({})
        with mock.patch.object(config, "_transformers_major", lambda: None):
            assert config.resolved_processor_backend(_processor_named("SiglipImageProcessor")) is None


class TestTheRequestIsReadBackNotAssumed:
    """transformers warns and falls back, so an impossible request must be caught here.

    DINOv3 is the concrete case: it ships no PIL implementation, so asking for
    one hands back torchvision with a log line nobody reads.
    """

    def test_a_fallback_warns_and_names_the_embedder(self, caplog):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "pil"})
        with caplog.at_level("WARNING"):
            got = config.verify_image_processor_backend(_processor_named("DINOv3ImageProcessorFast"), embedder="DINOv3")
        assert got == "torchvision"
        assert "DINOv3" in caplog.text
        assert "DINOv3ImageProcessorFast" in caplog.text

    def test_an_honoured_request_is_silent(self, caplog):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "torchvision"})
        with caplog.at_level("WARNING"):
            got = config.verify_image_processor_backend(_processor_named("SiglipImageProcessorFast"), embedder="SigLIP")
        assert got == "torchvision"
        assert caplog.text == ""

    def test_the_default_request_is_checked_too(self, caplog):
        """The point of #3173 is that *every* load now makes a request worth verifying."""
        config = _config_with({})
        with caplog.at_level("WARNING"):
            got = config.verify_image_processor_backend(_processor_named("SiglipImageProcessorPil"), embedder="SigLIP")
        assert got == "pil"
        assert "torchvision" in caplog.text

    def test_auto_has_asked_for_nothing_so_nothing_can_contradict_it(self, caplog):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "auto"})
        with caplog.at_level("WARNING"):
            got = config.verify_image_processor_backend(_processor_named("SiglipImageProcessorPil"), embedder="SigLIP")
        assert got == "pil"
        assert caplog.text == ""

    def test_an_unreadable_processor_is_not_reported_as_a_mismatch(self, caplog):
        """Abstention must not be laundered into a warning; the class simply said nothing."""
        config = _config_with({})
        with caplog.at_level("WARNING"):
            got = config.verify_image_processor_backend(_processor_named("SiglipTokenizer"), embedder="SigLIP")
        assert got is None
        assert caplog.text == ""

    def test_a_typo_is_verified_against_the_default(self, caplog):
        config = _config_with({"VTSEARCH_IMAGE_PROCESSOR_BACKEND": "torchvsion"})
        with caplog.at_level("WARNING"):
            got = config.verify_image_processor_backend(_processor_named("SiglipImageProcessorPil"), embedder="SigLIP")
        assert got == "pil"
        assert "torchvision" in caplog.text
