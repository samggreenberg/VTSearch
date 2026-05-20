"""Tests for `vtscore.detectors.dataset_sync`.

Covers the embedder-switch invalidation helpers introduced for H5.  The
full ``ensure_votes_match_active_dataset`` flow (with on-disk detector
JSON and the request lifecycle) is exercised in the app-tier
``tests/detectors/test_detectors.py``.
"""

from __future__ import annotations

import numpy as np

from vtscore.detectors.dataset_sync import (
    first_media_embedder,
    invalidate_model_on_embedder_switch,
)
from vtscore.state.core import DatasetContext, DetectorContext


class _DummyModel:
    """Stand-in for ``nn.Module`` — only its presence matters here."""


def _build_det_ctx(*, model=None, embedder: str = "") -> DetectorContext:
    ctx = DetectorContext("det", name="det", media_type="audio", embedder=embedder)
    ctx.model = model
    ctx.threshold = 0.7
    ctx.calibration_cache = ("fingerprint", 0.7)
    ctx.label_embeddings["eid-a"] = np.zeros(8, dtype=np.float32)
    return ctx


def _ds_with_embedder(embedder: str, *, dataset_id: str = "ds") -> DatasetContext:
    ds = DatasetContext(dataset_id)
    ds.medias[1] = {"id": 1, "embedder": embedder, "type": "audio"}
    return ds


class TestFirstMediaEmbedder:
    def test_empty_dict_returns_blank(self):
        assert first_media_embedder({}) == ""

    def test_returns_first_entry(self):
        medias = {
            1: {"embedder": "clap"},
            2: {"embedder": "siglip"},  # not consulted — same-dataset invariant
        }
        assert first_media_embedder(medias) == "clap"

    def test_missing_field_returns_blank(self):
        assert first_media_embedder({1: {"type": "audio"}}) == ""

    def test_blank_value_returns_blank(self):
        assert first_media_embedder({1: {"embedder": ""}}) == ""


class TestInvalidateModelOnEmbedderSwitch:
    def test_no_model_is_noop(self):
        det_ctx = _build_det_ctx(model=None, embedder="clap")
        assert invalidate_model_on_embedder_switch(det_ctx, "siglip") is False
        assert det_ctx.embedder == "clap"
        assert det_ctx.label_embeddings  # untouched

    def test_unknown_old_embedder_is_noop(self):
        """Empty stamp on det_ctx means we can't be confident it's stale."""
        model = _DummyModel()
        det_ctx = _build_det_ctx(model=model, embedder="")
        assert invalidate_model_on_embedder_switch(det_ctx, "siglip") is False
        assert det_ctx.model is model
        assert det_ctx.calibration_cache is not None

    def test_unknown_new_embedder_is_noop(self):
        """Empty active embedder means we can't be confident either."""
        model = _DummyModel()
        det_ctx = _build_det_ctx(model=model, embedder="clap")
        assert invalidate_model_on_embedder_switch(det_ctx, "") is False
        assert det_ctx.model is model
        assert det_ctx.embedder == "clap"

    def test_matching_embedder_is_noop(self):
        model = _DummyModel()
        det_ctx = _build_det_ctx(model=model, embedder="clap")
        assert invalidate_model_on_embedder_switch(det_ctx, "clap") is False
        assert det_ctx.model is model
        assert det_ctx.embedder == "clap"
        assert det_ctx.calibration_cache is not None
        assert det_ctx.label_embeddings  # cache preserved

    def test_mismatch_clears_all_caches(self):
        """Mismatch clears the MLP, threshold, calibration, label-embeddings,
        and the stamp itself so the next train re-stamps it."""
        model = _DummyModel()
        det_ctx = _build_det_ctx(model=model, embedder="clap")
        cleared = invalidate_model_on_embedder_switch(det_ctx, "siglip")
        assert cleared is True
        assert det_ctx.model is None
        assert det_ctx.threshold == 0.5
        assert det_ctx.calibration_cache is None
        assert det_ctx.label_embeddings == {}
        assert det_ctx.embedder == ""


class TestFirstMediaEmbedderOnContext:
    """Sanity check: the helper integrates cleanly with DatasetContext."""

    def test_active_dataset_with_embedder(self):
        ds = _ds_with_embedder("clap")
        assert first_media_embedder(ds.medias) == "clap"

    def test_empty_dataset(self):
        ds = DatasetContext("ds")
        assert first_media_embedder(ds.medias) == ""
