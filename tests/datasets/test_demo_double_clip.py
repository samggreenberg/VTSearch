"""Regression: a clipped demo import must clip exactly once.

A demo import runs through ``_run_importer_in_background``.  The demo importer
clips (and embeds + caches) the dataset itself inside ``load_demo_dataset``, so
the shared load pipeline must **not** run its ``_apply_clipper_stage`` on top —
that would clip the already-clipped media a second time (re-decoding audio,
re-tiling, recomputing MD5s, regenerating thumbnails).

The fix routes clipping through the importer's ``handles_own_clipping`` flag:
the dispatch keeps the full clipper config in ``field_values`` for the importer
and suppresses the pipeline-level clipper.  These tests pin that contract.
"""

from unittest.mock import patch


class TestDemoImportSingleClip:
    def test_demo_import_clips_once_via_importer_only(self):
        """The clipper goes to the importer with its params; the pipeline is suppressed."""
        from vtscore.datasets import load_pipeline
        from vtscore.datasets.importers.demo import IMPORTER as demo_importer

        captured: dict = {}

        def fake_origin_load(load_fn, origin, **kwargs):
            # Record the clipper the pipeline would hand to _apply_clipper_stage.
            captured["pipeline_clipper"] = kwargs.get("clipper")
            captured["pipeline_clipper_params"] = kwargs.get("clipper_params")
            # Drive the importer synchronously the way the real background
            # thread would, so we observe the importer's own clipping.
            load_fn({})
            return "task-1"

        demo_calls: list = []

        def fake_load_demo(dataset_name, medias, **kwargs):
            demo_calls.append((kwargs.get("clipper_name"), kwargs.get("clipper_params")))

        with (
            patch.object(load_pipeline, "_run_origin_load_in_background", side_effect=fake_origin_load),
            patch("vtscore.datasets.importers.demo.load_demo_dataset", side_effect=fake_load_demo),
        ):
            load_pipeline._run_importer_in_background(
                demo_importer,
                {
                    "name": "gtzan_a",
                    "clipper": "sound_tiling",
                    "clipper_params": {"duration": 5.0},
                    "media_type": "audio",
                },
            )

        # The demo importer clipped once, with the clipper AND its real params
        # (previously the params were dropped before run(), so it silently used
        # the clipper's defaults).
        assert demo_calls == [("sound_tiling", {"duration": 5.0})], demo_calls
        # The pipeline's clipper stage is suppressed: no second clip.
        assert captured["pipeline_clipper"] == ""
        assert captured["pipeline_clipper_params"] is None

    def test_non_self_clipping_importer_still_uses_pipeline_clipper(self):
        """Importers without ``handles_own_clipping`` still clip via the pipeline."""
        from vtscore.datasets import load_pipeline
        from vtscore.datasets.importers.base import ImporterBase

        class DummyImporter(ImporterBase):
            name = "dummy_clip_test"
            display_name = "Dummy"
            fields: list = []

            def run(self, field_values, medias, thin=False):
                pass

        captured: dict = {}

        def fake_origin_load(load_fn, origin, **kwargs):
            captured["pipeline_clipper"] = kwargs.get("clipper")
            captured["pipeline_clipper_params"] = kwargs.get("clipper_params")
            return "task-2"

        with patch.object(load_pipeline, "_run_origin_load_in_background", side_effect=fake_origin_load):
            load_pipeline._run_importer_in_background(
                DummyImporter(),
                {"clipper": "sound_tiling", "clipper_params": {"duration": 5.0}, "media_type": "audio"},
            )

        assert captured["pipeline_clipper"] == "sound_tiling"
        assert captured["pipeline_clipper_params"] == {"duration": 5.0}

    def test_apply_clipper_stage_has_no_guard_for_clipped_media(self):
        """Rationale check: the stage clips whatever non-empty clipper it's given.

        The stage itself has no "already clipped?" guard, so the double-clip can
        only be prevented upstream at dispatch (see the suppression above).
        """
        from vtscore.datasets.stages import clipper as clipper_stage
        from vtscore.state import DatasetContext

        class DummyTracker:
            def check_cancelled(self):
                pass

            def update(self, *a, **k):
                pass

        ctx = DatasetContext("test-double-clip")
        ctx.medias = {1: {"id": 1, "media_type": "audio", "embeddings": {"clap": [0.0]}}}

        calls: list = []

        def spy_apply(clips_dict, clipper_name, clipper_params=None, on_progress=None, chain_steps=None, embedder=None):
            calls.append(clipper_name)

        with (
            patch.object(clipper_stage, "_apply_clipper", side_effect=spy_apply),
            patch.object(clipper_stage, "invalidate_embedding_matrix", lambda ctx: None),
        ):
            clipper_stage._apply_clipper_stage(ctx, DummyTracker(), "sound_tiling", None, None)

        assert calls == ["sound_tiling"]


class TestDemoDatasetIdThreading:
    """A demo load must forward its dataset id to the load pipeline so the
    ``VTSEARCH_PROFILE_LOAD`` recorder stamps ``dataset_id`` (and can resolve the
    archive size) instead of writing empty rows.  Regression for #2614.
    """

    def test_demo_id_forwarded_to_origin_load(self):
        """The demo importer's ``name`` field reaches ``_run_origin_load_in_background``."""
        from vtscore.datasets import load_pipeline
        from vtscore.datasets.importers.demo import IMPORTER as demo_importer

        captured: dict = {}

        def fake_origin_load(load_fn, origin, **kwargs):
            captured.update(kwargs)
            return "task-id"

        with patch.object(load_pipeline, "_run_origin_load_in_background", side_effect=fake_origin_load):
            load_pipeline._run_importer_in_background(
                demo_importer,
                {"name": "gtzan_a", "media_type": "audio"},
            )

        assert captured["dataset_id"] == "gtzan_a"

    def test_non_demo_import_forwards_empty_id(self):
        """A non-demo importer supplies no dataset id (empty, not the field's ``name``)."""
        from vtscore.datasets import load_pipeline
        from vtscore.datasets.importers.base import ImporterBase

        class DummyImporter(ImporterBase):
            name = "dummy_id_test"
            display_name = "Dummy"
            fields: list = []

            def run(self, field_values, medias, thin=False):
                pass

        captured: dict = {}

        def fake_origin_load(load_fn, origin, **kwargs):
            captured.update(kwargs)
            return "task-id"

        with patch.object(load_pipeline, "_run_origin_load_in_background", side_effect=fake_origin_load):
            load_pipeline._run_importer_in_background(
                DummyImporter(),
                {"name": "not-a-demo-id", "media_type": "audio"},
            )

        assert captured["dataset_id"] == ""

    def test_demo_load_hints_returns_id_count_and_size(self):
        """``_demo_load_hints`` reports the id alongside the n / size hints."""
        from vtscore.datasets import load_pipeline
        from vtscore.datasets.importers.demo import IMPORTER as demo_importer

        dataset_id, n, size = load_pipeline._demo_load_hints(demo_importer, {"name": "gtzan_a"})

        assert dataset_id == "gtzan_a"
        assert n is not None and n > 0
        assert size is not None and size > 0
