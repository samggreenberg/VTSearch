"""Characterization test: a clipped demo import runs the clipper twice.

When a demo dataset is created through the importer flow, the clipper is
dispatched into **two** places:

1. The demo importer's own internal clipping inside ``load_demo_dataset``
   (``DemoDatasetImporter.run`` forwards ``field_values['clipper']``).
2. The shared load pipeline's ``_apply_clipper_stage``, because
   ``_run_importer_in_background`` re-adds ``clipper`` to ``field_values``
   *and* forwards it to ``_run_origin_load_in_background``.

So for a clipped audio demo (e.g. GTZAN, whose pre-selected default is the
real ``sound_tiling`` clipper) the clipper runs on the media, then runs again
on the already-clipped media.  These tests pin that current behavior so the
redundancy is verifiable; if it is fixed, they should be flipped to assert a
single dispatch.
"""

from unittest.mock import patch


class TestDemoImportDoubleClip:
    def test_clipper_dispatched_to_both_importer_and_pipeline(self):
        """One clipped demo import fans the clipper out to importer + pipeline."""
        from vtscore.datasets import load_pipeline
        from vtscore.datasets.importers.demo import IMPORTER as demo_importer

        captured: dict = {}

        def fake_origin_load(load_fn, origin, **kwargs):
            # Record the clipper the pipeline would hand to _apply_clipper_stage.
            captured["pipeline_clipper"] = kwargs.get("clipper")
            # Drive the importer synchronously the way the real background
            # thread would, so we also observe the importer's own clipping.
            load_fn({})
            return "task-1"

        demo_calls: list = []

        def fake_load_demo(dataset_name, medias, **kwargs):
            demo_calls.append(kwargs.get("clipper_name"))

        with (
            patch.object(load_pipeline, "_run_origin_load_in_background", side_effect=fake_origin_load),
            patch("vtscore.datasets.importers.demo.load_demo_dataset", side_effect=fake_load_demo),
        ):
            load_pipeline._run_importer_in_background(
                demo_importer,
                {"name": "gtzan_a", "clipper": "sound_tiling", "media_type": "audio"},
            )

        # The demo importer clipped internally with sound_tiling...
        assert demo_calls == ["sound_tiling"], demo_calls
        # ...and the pipeline received the SAME clipper for _apply_clipper_stage.
        assert captured["pipeline_clipper"] == "sound_tiling"

    def test_apply_clipper_stage_reclips_already_clipped_media(self):
        """The pipeline stage has no guard for already-clipped media."""
        from vtscore.datasets.stages import clipper as clipper_stage

        class DummyTracker:
            def check_cancelled(self):
                pass

            def update(self, *a, **k):
                pass

        class DummyCtx:
            def __init__(self, medias):
                self.medias = medias

        # A media that already looks like a finished clip (carries an embedding).
        ctx = DummyCtx({1: {"id": 1, "media_type": "audio", "embeddings": {"clap": [0.0]}}})

        calls: list = []

        def spy_apply(clips_dict, clipper_name, clipper_params=None, on_progress=None, chain_steps=None, embedder=None):
            calls.append(clipper_name)

        with (
            patch.object(clipper_stage, "_apply_clipper", side_effect=spy_apply),
            patch.object(clipper_stage, "invalidate_embedding_matrix", lambda ctx: None),
        ):
            clipper_stage._apply_clipper_stage(ctx, DummyTracker(), "sound_tiling", None, None)

        # No early-return for a non-empty clipper: the stage re-runs the clipper
        # on media that the importer already clipped.
        assert calls == ["sound_tiling"]
