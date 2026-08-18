"""The embedding-stack stamp: present, honest, and never a reason to fail a save."""

from __future__ import annotations

from vtscore.datasets.loader import _loaded_embedder
from vtscore.embedding.stack import embedding_stack


class TestEmbeddingStack:
    def test_reports_the_versions_that_decide_the_pixels(self):
        stack = embedding_stack()
        # transformers and torch decide which resampler runs and how it rounds;
        # the whole point of the field is that these are recorded, not assumed.
        for key in ("torch", "transformers", "torchvision", "pillow", "cpu_capability", "device"):
            assert key in stack

    def test_missing_information_is_null_not_guessed(self):
        stack = embedding_stack()
        # `device` is None on a CPU-only host; that is a fact about the run, and
        # a wrong guess would be worse than a gap.
        assert stack["device"] is None or isinstance(stack["device"], str)
        assert stack["cpu_capability"] is None or isinstance(stack["cpu_capability"], str)

    def test_processor_classes_only_when_an_embedder_is_given(self):
        assert "image_processor_class" not in embedding_stack()
        stack = embedding_stack(object())
        assert stack["processor_class"] is None
        assert stack["image_processor_class"] is None

    def test_a_broken_embedder_does_not_sink_the_stamp(self):
        class Exploding:
            @property
            def _processor(self):
                raise RuntimeError("boom")

        stack = embedding_stack(Exploding())
        assert stack["processor_class"] is None
        assert stack["torch"] == embedding_stack()["torch"]


class TestLoadedEmbedder:
    def test_unknown_name_is_none_rather_than_an_error(self):
        assert _loaded_embedder("no-such-embedder") is None

    def test_blank_name_is_none(self):
        assert _loaded_embedder("") is None
        assert _loaded_embedder(None) is None

    def test_an_unloaded_embedder_is_not_forced_to_load(self):
        # Saving a dataset must not pay for a model load just to describe itself:
        # an embedder with no resolved processor records nothing.
        from vtscore.media import all_embedders

        unloaded = [e for e in all_embedders() if getattr(e, "_processor", None) is None]
        if not unloaded:
            return
        assert _loaded_embedder(unloaded[0].name) is None
