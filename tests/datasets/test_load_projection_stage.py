"""Tests for the opt-in inline projection stage of the dataset-load pipeline.

When a user ticks "Build 2-D Browse projection now" in any importer form,
``_run_origin_load_in_background`` runs ``_build_projection_stage`` after the
dataset is registered: it fits UMAP on the cached embedding matrix, builds the
hex-tile pyramid, caches both on the context, and persists them into the
dataset container.  The stage is best-effort — the dataset is already saved
and usable before it runs, so a failure (or empty/embedding-less dataset)
leaves the dataset intact and just defers the projection to the lazy
Browse-time build.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from vtscore.concurrency.progress import LoadingTasksTracker
from vtscore.datasets.load_pipeline import _parse_bool
from vtscore.datasets.stages.projection import _build_projection_stage
from vtscore.projection import Projection
from vtscore.state.core import DatasetContext


def _make_tracker():
    return LoadingTasksTracker().create_task("proj_task", "test")


def _fake_fit_projection(matrix, ids, on_progress=None, **kwargs):
    """Cheap stand-in for UMAP: a seeded random 2-D layout.

    Calls ``on_progress`` once so the stage's progress wiring is exercised.
    """
    if on_progress is not None:
        on_progress("projecting", "fitting", 1, 1)
    rng = np.random.default_rng(42)
    coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
    return Projection("fake-pid", list(ids), coords, "pca")


class TestParseBool:
    def test_true_string(self):
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("  TRUE  ") is True

    def test_false_and_none(self):
        assert _parse_bool("false") is False
        assert _parse_bool("0") is False
        assert _parse_bool("") is False
        assert _parse_bool(None) is False

    def test_native_bool(self):
        assert _parse_bool(True) is True
        assert _parse_bool(False) is False


class TestBuildProjectionStage:
    def _ctx_with_embeddings(self, name: str) -> DatasetContext:
        ctx = DatasetContext(name)
        rng = np.random.default_rng(0)
        for cid in (1, 2, 3, 4):
            ctx.medias[cid] = {
                "id": cid,
                "media_type": "audio",
                "embedder": "clap",
                "embeddings": {"clap": rng.standard_normal(8).astype(np.float32)},
            }
        return ctx

    def test_caches_and_persists(self):
        ctx = self._ctx_with_embeddings("proj_stage_ok")
        with (
            patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection),
            patch("vtscore.projection.store.persist_projection") as mock_persist,
        ):
            _build_projection_stage(ctx, _make_tracker())

        assert ctx._projection is not None
        assert ctx._projection.projection_id == "fake-pid"
        # The fixtures are audio, which tiles as squares (waveform thumbnails).
        assert ctx._pyramids.get("square") is not None
        assert ctx._pyramids["square"].projection_id == "fake-pid"
        mock_persist.assert_called_once()
        # The context's own id and the freshly-built artifacts go to the store.
        args = mock_persist.call_args.args
        assert args[0] == "proj_stage_ok"
        assert args[1] is ctx._projection
        assert args[2] is ctx._pyramids["square"]

    def test_threads_the_resolved_params_into_the_fit(self):
        """The ingest fit uses the same knobs the route would resolve (#3056).

        Not ``fit_projection``'s signature defaults: a layout fit under
        different params than the route resolves is discarded on the first
        Browse open (so the opt-in pre-build bought nothing), or — when the
        mismatch happens to be invisible — served under knobs nobody chose.
        """
        ctx = self._ctx_with_embeddings("proj_stage_params")
        captured: dict = {}

        def _capturing(matrix, ids, on_progress=None, **kwargs):
            captured.update(kwargs)
            return _fake_fit_projection(matrix, ids, on_progress=on_progress)

        with (
            patch("vtscore.projection.fit_projection", side_effect=_capturing),
            patch("vtscore.projection.store.persist_projection"),
        ):
            _build_projection_stage(ctx, _make_tracker())

        # The fixtures are CLAP-embedded, whose swept defaults are (15, 0.10).
        assert captured["n_neighbors"] == 15
        assert captured["min_dist"] == 0.10
        # And compaction is off, matching PROJECTION_COMPACT_DEFAULT — the
        # signature default used to be True here.
        assert captured["compact"] is False

    def test_tuned_embedder_params_reach_the_fit(self):
        """A SigLIP dataset is fit under SigLIP's swept knobs, not the globals."""
        ctx = DatasetContext("proj_stage_siglip")
        rng = np.random.default_rng(3)
        for cid in (1, 2, 3, 4):
            ctx.medias[cid] = {
                "id": cid,
                "media_type": "image",
                "embedder": "siglip",
                "embeddings": {"siglip": rng.standard_normal(8).astype(np.float32)},
            }
        captured: dict = {}

        def _capturing(matrix, ids, on_progress=None, **kwargs):
            captured.update(kwargs)
            return _fake_fit_projection(matrix, ids, on_progress=on_progress)

        with (
            patch("vtscore.projection.fit_projection", side_effect=_capturing),
            patch("vtscore.projection.store.persist_projection"),
        ):
            _build_projection_stage(ctx, _make_tracker())

        assert (captured["n_neighbors"], captured["min_dist"]) == (10, 0.05)

    def test_pre_built_layout_survives_the_routes_freshness_check(self):
        """An ingest-built layout is kept — and kept uncompacted — on first Browse.

        The end-to-end shape of #3056: the ingest stage stamps what it fit
        under, and the route's persisted-layout guard asks for exactly those
        knobs.  An untuned embedder is the interesting case, because there the
        old mismatch was *invisible* — the stale layout passed the guard and
        the user silently got the compacted arrangement the sweep measured as
        worse.
        """
        from vtscore.projection.store import projection_params_match

        ctx = DatasetContext("proj_stage_untuned")
        rng = np.random.default_rng(11)
        for cid in (1, 2, 3, 4):
            ctx.medias[cid] = {
                "id": cid,
                "media_type": "text",
                "embedder": "e5",
                "embeddings": {"e5": rng.standard_normal(8).astype(np.float32)},
            }

        def _stamping(matrix, ids, on_progress=None, **kwargs):
            """A fake fit that records its knobs the way the real one does."""
            coords = np.random.default_rng(1).standard_normal((len(ids), 2)).astype(np.float32)
            return Projection(
                "untuned-pid",
                list(ids),
                coords,
                "umap",
                kwargs.get("n_neighbors"),
                kwargs.get("min_dist"),
                kwargs.get("compact"),
            )

        with (
            patch("vtscore.projection.fit_projection", side_effect=_stamping),
            patch("vtscore.projection.store.persist_projection"),
        ):
            _build_projection_stage(ctx, _make_tracker())

        assert ctx._projection.compact is False
        assert projection_params_match(ctx._projection, ctx) is True

    def test_empty_dataset_is_noop(self):
        ctx = DatasetContext("proj_stage_empty")
        with (
            patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection) as mock_fit,
            patch("vtscore.projection.store.persist_projection") as mock_persist,
        ):
            _build_projection_stage(ctx, _make_tracker())

        assert ctx._projection is None
        assert ctx._pyramids == {}
        mock_fit.assert_not_called()
        mock_persist.assert_not_called()
