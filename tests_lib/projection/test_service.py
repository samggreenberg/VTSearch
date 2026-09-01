"""The Browse layout lifecycle, driven directly (``vtscore.projection.service``).

Every case here used to need a Flask test client: the build/re-bin/reset/
subset state machine lived inside ``vtsearch/routes/projection.py`` and could
only be reached through an HTTP request.  It is library code now, so these
exercise it as what it is — functions over an explicit ``DatasetContext``.

The UMAP fit is faked throughout (a seeded random layout): what is under test
is *which* of the four sources answers a build and what it leaves on the
context, not the arrangement itself.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import patch

import numpy as np
import pytest

from vtscore.concurrency.async_jobs import projection_jobs
from vtscore.projection import Projection, build_pyramid
from vtscore.projection import service as svc
from vtscore.state.core import DatasetContext


def _fake_fit(matrix, ids, **kwargs):
    """Cheap stand-in for UMAP: a seeded random 2-D layout with a fresh id."""
    rng = np.random.default_rng(len(ids))
    coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
    return Projection(uuid.uuid4().hex, list(ids), coords, "fake")


def _faked_fit():
    return patch("vtscore.projection.fit_projection", side_effect=_fake_fit)


def _ctx(name: str = "svc", n: int = 6, media_type: str = "text") -> DatasetContext:
    ctx = DatasetContext(name)
    rng = np.random.default_rng(0)
    for cid in range(1, n + 1):
        ctx.medias[cid] = {
            "id": cid,
            "media_type": media_type,
            "embedder": "clap",
            "embeddings": {"clap": rng.standard_normal(8).astype(np.float32)},
        }
    return ctx


def _layout(ctx: DatasetContext, bin_shape: str = "hex", pid: str = "pid"):
    """A projection + pyramid over *ctx*'s ids, built without any fit."""
    ids = sorted(ctx.medias.keys())
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
    proj = Projection(pid, ids, coords, "fake")
    return proj, build_pyramid(proj, bin_shape=bin_shape, n_levels=2)


def _await_build(timeout: float = 30.0) -> None:
    """Block until the projection runner is idle."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = projection_jobs.current()
        if job is None:
            return
        if job.status in ("running", "pending"):
            job.done_event.wait(timeout=0.05)
            continue
        return
    raise TimeoutError("projection job did not finish")


class TestShapeFor:
    """The bin shape is read off the dataset, never off a request."""

    def test_text_tiles_as_hexes_and_audio_as_squares(self):
        assert svc.shape_for(_ctx(media_type="text")) == "hex"
        # Audio has a browsable waveform thumbnail, so it packs as squares.
        assert svc.shape_for(_ctx(media_type="audio")) == "square"

    def test_empty_dataset_falls_back_to_hex(self):
        assert svc.media_type_for(DatasetContext("empty")) == ""
        assert svc.shape_for(DatasetContext("empty")) == "hex"

    def test_a_media_without_a_declared_type_reads_as_audio(self):
        """Which is what the ingest pre-build must agree with.

        The stage used to read a bare ``.get("media_type")`` (``None`` → hex)
        while the serve path defaulted to audio (→ square), so such a dataset
        was pre-built under a shape the serve path would never ask for.
        """
        ctx = DatasetContext("untyped")
        ctx.medias[1] = {"id": 1}
        assert svc.media_type_for(ctx) == "audio"


class TestBuildLayoutShortCircuits:
    def test_a_cached_pyramid_answers_without_a_fit(self):
        ctx = _ctx()
        proj, pyr = _layout(ctx)
        svc.install_layout(ctx, proj, pyr)

        with _faked_fit() as fit:
            assert svc.build_layout(ctx) == {"status": "ready", "projection_id": "pid"}
        fit.assert_not_called()

    def test_an_empty_dataset_has_nothing_to_project(self):
        with pytest.raises(svc.NothingToProject, match="Dataset is empty"):
            svc.build_layout(DatasetContext("empty"))

    def test_a_dataset_without_embeddings_has_nothing_to_project(self):
        ctx = DatasetContext("no-vectors")
        ctx.medias[1] = {"id": 1, "media_type": "text"}
        with pytest.raises(ValueError):
            svc.build_layout(ctx)

    def test_nothing_to_project_is_a_value_error(self):
        """So one ``except ValueError`` at the HTTP edge covers both it and the
        embedding-matrix builder's own signal."""
        assert issubclass(svc.NothingToProject, ValueError)


class TestBuildLayoutFits:
    def test_first_build_starts_a_job_and_installs_the_result(self):
        ctx = _ctx("svc-first")
        with _faked_fit():
            started = svc.build_layout(ctx)
            assert started["status"] == "building"
            assert ctx._full_job_id == started["job_id"]
            _await_build()

        assert ctx._projection is not None
        assert ctx._pyramids.get("hex") is not None
        assert svc.layout_meta(ctx, "hex", subset=False)["status"] == "ready"

    def test_the_other_shape_re_bins_the_frozen_layout_without_re_fitting(self):
        """The hex/square toggle costs a re-bin, never a second UMAP fit."""
        ctx = _ctx("svc-rebin")
        proj, pyr = _layout(ctx, "hex")
        svc.install_layout(ctx, proj, pyr)

        with _faked_fit() as fit:
            ready = svc.rebin_from_existing_layout(ctx, sorted(ctx.medias), "square")
        fit.assert_not_called()
        assert ready == {"status": "ready", "projection_id": "pid"}
        # Same coordinates, both binnings cached side by side.
        assert ctx._pyramids["square"].projection_id == "pid"
        assert ctx._pyramids["hex"] is pyr

    def test_a_stale_id_set_is_not_re_binned(self):
        """A layout fit over different items can't be re-tiled for these ones."""
        ctx = _ctx("svc-stale")
        proj, pyr = _layout(ctx, "hex")
        svc.install_layout(ctx, proj, pyr)
        assert svc.rebin_from_existing_layout(ctx, [*sorted(ctx.medias), 999], "square") is None

    def test_force_drops_the_frozen_layout_and_re_fits(self):
        ctx = _ctx("svc-force")
        proj, pyr = _layout(ctx, "hex")
        svc.install_layout(ctx, proj, pyr)

        with _faked_fit() as fit:
            assert svc.build_layout(ctx, force=True)["status"] == "building"
            _await_build()
        assert fit.call_count == 1
        assert ctx._projection.projection_id != "pid"

    def test_reset_drops_the_layout_its_pyramids_and_its_signs(self):
        ctx = _ctx("svc-reset")
        proj, pyr = _layout(ctx, "hex")
        svc.install_layout(ctx, proj, pyr)
        ctx._region_labels = object()
        ctx._full_job_id = "stale-job"

        svc.reset_full_projection(ctx)

        assert ctx._projection is None
        assert ctx._pyramids == {}
        assert ctx._region_labels is None
        assert ctx._full_job_id is None

    def test_reset_leaves_the_subset_layout_standing(self):
        """It is an independent fit the user may still be browsing."""
        ctx = _ctx("svc-reset-subset")
        proj, pyr = _layout(ctx, "hex")
        svc.install_layout(ctx, proj, pyr, subset=True)

        svc.reset_full_projection(ctx)
        assert ctx._subset_projection is proj
        assert ctx._subset_pyramids["hex"] is pyr


class TestSubsetLayout:
    def test_an_empty_selection_has_nothing_to_project(self):
        with pytest.raises(svc.NothingToProject, match="No items selected"):
            svc.build_layout(_ctx(), ids=[])

    def test_a_selection_against_an_empty_dataset_has_nothing_to_project(self):
        with pytest.raises(svc.NothingToProject, match="Dataset is empty"):
            svc.build_layout(DatasetContext("empty"), ids=[1, 2])

    def test_the_subset_fit_lands_in_the_subset_slots_only(self):
        ctx = _ctx("svc-sub")
        with _faked_fit():
            assert svc.build_layout(ctx, ids=[1, 2, 3])["status"] == "building"
            _await_build()

        assert ctx._subset_projection is not None
        assert sorted(ctx._subset_projection.ids) == [1, 2, 3]
        assert ctx._subset_pyramids.get("hex") is not None
        # The full-dataset layout is untouched by a subset browse.
        assert ctx._projection is None
        assert ctx._pyramids == {}

    def test_the_same_subset_re_bins_instead_of_re_fitting(self):
        ctx = _ctx("svc-sub-same")
        proj, pyr = _layout(ctx, "hex", pid="sub-pid")
        ctx._subset_projection = proj
        ctx._subset_pyramids = {"hex": pyr}
        ctx._subset_ids = list(proj.ids)

        with _faked_fit() as fit:
            same = svc.build_layout(ctx, ids=list(reversed(proj.ids)))
            other = svc.build_subset_layout(ctx, list(proj.ids), "square")
        fit.assert_not_called()
        assert same == {"status": "ready", "projection_id": "sub-pid"}
        assert other == {"status": "ready", "projection_id": "sub-pid"}

    def test_a_different_subset_drops_the_stale_layout_before_fitting(self):
        ctx = _ctx("svc-sub-diff")
        proj, pyr = _layout(ctx, "hex", pid="sub-pid")
        ctx._subset_projection = proj
        ctx._subset_pyramids = {"hex": pyr}
        ctx._subset_ids = list(proj.ids)
        ctx._subset_content_version = 4
        ctx._subset_region_labels = object()

        with _faked_fit():
            assert svc.build_layout(ctx, ids=[1, 2])["status"] == "building"
            # The old layout is gone the moment the new fit is dispatched — the
            # tile token resets and the signs anchored in it are dropped.
            assert ctx._subset_content_version == 0
            assert ctx._subset_region_labels is None
            _await_build()

        assert sorted(ctx._subset_projection.ids) == [1, 2]


class TestRemoveFromSubset:
    def _built_subset(self, ctx, shapes=("hex",)):
        ids = sorted(ctx.medias.keys())
        rng = np.random.default_rng(2)
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("cull-pid", ids, coords, "fake")
        ctx._subset_projection = proj
        ctx._subset_ids = list(ids)
        ctx._subset_pyramids = {s: build_pyramid(proj, bin_shape=s, n_levels=2) for s in shapes}
        return proj

    def test_the_cull_preserves_layout_identity_and_busts_the_tile_cache(self):
        ctx = _ctx("svc-cull")
        self._built_subset(ctx)

        svc.remove_subset_ids(ctx, [1, 2])

        assert ctx._subset_ids == [3, 4, 5, 6]
        # Same layout — the canvas keeps its viewport — but a new tile token.
        assert ctx._subset_pyramids["hex"].projection_id == "cull-pid"
        assert ctx._subset_content_version == 1
        assert svc.layout_meta(ctx, "hex", subset=True)["content_version"] == 1

    def test_every_built_shape_is_re_binned(self):
        ctx = _ctx("svc-cull-shapes")
        self._built_subset(ctx, shapes=("hex", "square"))
        svc.remove_subset_ids(ctx, [1])
        for shape, pyr in ctx._subset_pyramids.items():
            assert pyr.bin_shape == shape
            assert pyr.point_count == 5

    def test_the_bounds_shrink_to_the_survivors(self):
        """So the canvas re-frames to what is left instead of keeping dead space."""
        ctx = _ctx("svc-cull-bounds")
        self._built_subset(ctx)
        svc.remove_subset_ids(ctx, [1, 2, 3])
        assert ctx._subset_pyramids["hex"].bounds == ctx._subset_projection.bounds

    def test_a_cull_without_a_subset_is_a_refusal(self):
        with pytest.raises(svc.NothingToProject, match="build it first"):
            svc.remove_subset_ids(_ctx("svc-cull-none"), [1])


class TestLayoutMeta:
    def test_idle_without_a_layout_or_a_job(self):
        ctx = _ctx("svc-meta-idle")
        assert svc.layout_meta(ctx, "hex", subset=False) == {"status": "idle"}
        assert svc.layout_meta(ctx, "hex", subset=True) == {"status": "idle"}

    def test_ready_reports_the_shape_the_media_type_resolved_to(self):
        ctx = _ctx("svc-meta-ready", media_type="audio")
        proj, pyr = _layout(ctx, "square")
        svc.install_layout(ctx, proj, pyr)

        meta = svc.layout_meta(ctx, "square", subset=False)
        assert meta["status"] == "ready"
        assert meta["bin_shape"] == "square"
        assert meta["media_type"] == "audio"
        assert meta["method"] == "fake"
        # Full-dataset layouts are never edited in place.
        assert meta["content_version"] == 0

    def test_an_errored_job_is_reported_rather_than_read_as_idle(self):
        ctx = _ctx("svc-meta-err")

        def _boom(job):
            raise RuntimeError("umap exploded")

        job = projection_jobs.start(("svc-meta-err", "hex"), _boom, dataset_id=ctx.dataset_id)
        ctx._full_job_id = job.job_id
        job.done_event.wait(timeout=10)

        meta = svc.layout_meta(ctx, "hex", subset=False)
        assert meta["status"] == "error"
        assert "umap exploded" in meta["error"]

    def test_a_job_id_pointing_at_nothing_reads_as_idle(self):
        """The runner is app-wide and its history is bounded, so a stale id is
        expected rather than exceptional."""
        ctx = _ctx("svc-meta-stale")
        ctx._full_job_id = "no-such-job"
        assert svc.layout_meta(ctx, "hex", subset=False) == {"status": "idle"}


class TestBuildProgress:
    class _Job:
        job_id = "j"
        message = "arranging items"
        error = None
        status = "running"

        def __init__(self, step, total_steps, current, total):
            self.step, self.total_steps = step, total_steps
            self.current, self.total = current, total

    def test_stitches_within_phase_counts_into_a_whole_job_fraction(self):
        body = svc.build_progress(self._Job(2, 3, 1, 4))
        # Phase 2 of 3, a quarter of the way in: (2 - 1 + 0.25) / 3.
        assert body["overall"] == pytest.approx(1.25 / 3)
        assert body["overall_step_end"] == pytest.approx(2 / 3)

    def test_an_uncountable_phase_parks_at_its_slice_start(self):
        """The UMAP fit reports no fraction, so the bar must not invent one."""
        body = svc.build_progress(self._Job(1, 3, 0, 0))
        assert body["overall"] == pytest.approx(0.0)
        assert body["overall_step_end"] == pytest.approx(1 / 3)

    def test_a_phaseless_job_reports_no_overall(self):
        body = svc.build_progress(self._Job(0, 0, 3, 10))
        assert body["overall"] is None
        assert body["overall_step_end"] is None
        assert (body["step"], body["total_steps"]) == (None, None)


class TestTilePayload:
    def test_an_unbuilt_layout_has_no_tile(self):
        assert svc.tile_payload(_ctx("svc-tile-none"), "hex", 0, 0, 0, subset=False) is None

    def test_a_tile_off_the_grid_is_empty_rather_than_missing(self):
        ctx = _ctx("svc-tile-off")
        proj, pyr = _layout(ctx, "hex")
        svc.install_layout(ctx, proj, pyr)
        payload = svc.tile_payload(ctx, "hex", 0, 9999, 9999, subset=False)
        assert payload == {"level": 0, "tx": 9999, "ty": 9999, "cells": []}

    def test_cells_carry_their_member_ids(self):
        """Re-derived from the frozen layout; the pyramid stores only counts."""
        ctx = _ctx("svc-tile-members")
        proj, pyr = _layout(ctx, "hex")
        svc.install_layout(ctx, proj, pyr)

        seen: list[int] = []
        for level, tx, ty in pyr.tiles:
            payload = svc.tile_payload(ctx, "hex", level, tx, ty, subset=False)
            assert payload is not None
            if level == 0:
                for cell in payload["cells"]:
                    assert "member_ids" in cell
                    assert len(cell["member_ids"]) == cell["count"]
                    seen.extend(cell["member_ids"])
        assert sorted(seen) == sorted(ctx.medias)
