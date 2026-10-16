"""The coverage-atlas builders fit outside ``_state_lock`` (#4977).

``_state_lock`` serialises votes, media reads and every request's state sync,
and the hierarchical k-means fit can take minutes at scale, so the builders
snapshot the medias under the lock, fit unlocked, and re-take the lock only to
install the atlas. These tests pin that lock scope and the ``media_revision``
re-check that stops a fit over since-changed medias from being installed.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import vtscore.coverage.atlas as atlas_mod
from vtscore.state.core import (
    DatasetContext,
    DetectorContext,
    _state_lock,
    thread_dataset_context,
    thread_detector_context,
)
from vtscore.state.coverage import build_coverage_atlas, build_coverage_atlas_for_context

_RealAtlas = atlas_mod.CoverageAtlas


def _media(cid: int, rng: np.random.Generator) -> dict:
    return {"id": cid, "embedder": "siglip", "embeddings": {"siglip": rng.standard_normal(16).astype(np.float32)}}


def _ctx(n: int = 25) -> DatasetContext:
    ctx = DatasetContext("_atlas_lock_scope")
    rng = np.random.default_rng(42)
    for cid in range(1, n + 1):
        ctx.medias[cid] = _media(cid, rng)
    return ctx


def _substitute_atlas(monkeypatch, during_fit) -> list[int]:
    """Make every fit call ``during_fit(fit_number)`` before fitting; return the fit counter."""
    fits: list[int] = []

    class _ProbeAtlas(_RealAtlas):
        def __init__(self, vectors, *args, **kwargs):
            fits.append(len(vectors))
            during_fit(len(fits))
            super().__init__(vectors, *args, **kwargs)

    monkeypatch.setattr(atlas_mod, "CoverageAtlas", _ProbeAtlas)
    return fits


def _lock_free_from_another_thread() -> bool:
    # ``_state_lock`` is an RLock, so only a second thread can tell whether
    # the building thread holds it.
    got = threading.Event()

    def grab():
        if _state_lock.acquire(timeout=2.0):
            got.set()
            _state_lock.release()

    t = threading.Thread(target=grab)
    t.start()
    t.join(3.0)
    return got.is_set()


class TestFitRunsOutsideStateLock:
    def test_build_coverage_atlas(self, monkeypatch):
        seen: list[bool] = []
        _substitute_atlas(monkeypatch, lambda _n: seen.append(_lock_free_from_another_thread()))
        ctx = _ctx()
        with thread_dataset_context(ctx), thread_detector_context(DetectorContext("det")):
            build_coverage_atlas()
        assert seen == [True], "_state_lock was held across the k-means fit"
        assert isinstance(ctx.coverage_atlas, _RealAtlas)

    def test_build_coverage_atlas_for_context(self, monkeypatch):
        seen: list[bool] = []
        _substitute_atlas(monkeypatch, lambda _n: seen.append(_lock_free_from_another_thread()))
        ctx = _ctx()
        build_coverage_atlas_for_context(ctx)
        assert seen == [True], "_state_lock was held across the k-means fit"
        assert isinstance(ctx.coverage_atlas, _RealAtlas)


class TestMediaChangeDuringFit:
    def test_stale_fit_is_rebuilt_from_the_changed_medias(self, monkeypatch):
        ctx = _ctx()
        det = DetectorContext("det")
        det.good_votes[1] = None
        rng = np.random.default_rng(7)

        def add_media_during_first_fit(n: int) -> None:
            if n == 1:
                ctx.medias[99] = _media(99, rng)

        fits = _substitute_atlas(monkeypatch, add_media_during_first_fit)
        with thread_dataset_context(ctx), thread_detector_context(det):
            build_coverage_atlas()

        assert fits == [25, 26]
        assert 99 in ctx.coverage_atlas.vector_to_leaf
        # The votes are replayed into the atlas that was installed.
        assert ctx.coverage_atlas.labeled_ids == {1}

    def test_clear_during_fit_leaves_no_atlas(self, monkeypatch):
        ctx = _ctx()

        def clear_during_first_fit(n: int) -> None:
            if n == 1:
                with _state_lock:  # what clear_medias does
                    ctx.medias.clear()
                    ctx.coverage_atlas = None

        fits = _substitute_atlas(monkeypatch, clear_during_first_fit)
        with thread_dataset_context(ctx), thread_detector_context(DetectorContext("det")):
            build_coverage_atlas()

        # The retry snapshots no vectors, so there is nothing to fit.
        assert fits == [25]
        assert ctx.coverage_atlas is None

    @pytest.mark.parametrize("builder", ["active", "for_context"])
    def test_change_during_both_fits_keeps_the_current_atlas(self, monkeypatch, builder):
        ctx = _ctx()
        current = object()
        ctx.coverage_atlas = current  # type: ignore[assignment]

        fits = _substitute_atlas(monkeypatch, lambda _n: ctx.bump_media_revision())
        if builder == "active":
            with thread_dataset_context(ctx), thread_detector_context(DetectorContext("det")):
                build_coverage_atlas()
        else:
            build_coverage_atlas_for_context(ctx)

        assert len(fits) == 2
        assert ctx.coverage_atlas is current

    def test_explicit_media_dict_installs_without_the_check(self, monkeypatch):
        ctx = _ctx()
        explicit = {cid: ctx.medias[cid] for cid in range(1, 21)}

        fits = _substitute_atlas(monkeypatch, lambda _n: ctx.bump_media_revision())
        with thread_dataset_context(ctx), thread_detector_context(DetectorContext("det")):
            build_coverage_atlas(explicit)

        assert fits == [20]
        assert set(ctx.coverage_atlas.vector_to_leaf) == set(explicit)
