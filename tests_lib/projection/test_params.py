"""Tests for the shared Browse-projection parameter resolver.

Both fit paths — the on-demand ``POST /api/projection/build`` route and the
opt-in ingest-time ``build_projection`` load stage — resolve their UMAP knobs
here, so this is the single place the per-embedder tuned defaults, the operator
override, and the compaction default are decided (issue #3056).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vtscore import config
from vtscore.config import (
    PROJECTION_COMPACT_DEFAULT,
    PROJECTION_MIN_DIST,
    PROJECTION_N_NEIGHBORS,
)
from vtscore.projection.params import projection_embedder_for, resolve_projection_params
from vtscore.state.core import DatasetContext


def _ctx(embedder: str) -> DatasetContext:
    """A tiny context whose medias carry a single *embedder* vector."""
    ctx = DatasetContext(f"params_{embedder}")
    rng = np.random.default_rng(0)
    for cid in (1, 2, 3):
        ctx.medias[cid] = {
            "id": cid,
            "media_type": "audio",
            "embedder": embedder,
            "embeddings": {embedder: rng.standard_normal(8).astype(np.float32)},
        }
    return ctx


@pytest.fixture
def override_settings(monkeypatch):
    """Install a ``CoreConfig`` builder with overridden projection knobs."""

    def _apply(**fields):
        base = config.CoreConfig.from_settings()
        replaced = dataclasses.replace(base, **fields)
        monkeypatch.setattr(config, "_core_config_builder", lambda _path=None: replaced)

    return _apply


class TestProjectionEmbedderFor:
    def test_reads_the_medias_primary_embedder(self):
        assert projection_embedder_for(_ctx("siglip")) == "siglip"

    def test_no_context_or_no_medias_is_none(self):
        assert projection_embedder_for(None) is None
        assert projection_embedder_for(DatasetContext("params_empty")) is None


class TestResolveProjectionParams:
    def test_tuned_embedder_gets_its_swept_defaults(self):
        params = resolve_projection_params(_ctx("siglip"))
        assert (params.n_neighbors, params.min_dist) == (10, 0.05)

    def test_audio_embedder_gets_its_own_tuned_pair(self):
        params = resolve_projection_params(_ctx("clap"))
        assert (params.n_neighbors, params.min_dist) == (15, 0.10)

    def test_default_audio_embedder_gets_the_same_tuned_pair(self):
        params = resolve_projection_params(_ctx("clap_general"))
        assert (params.n_neighbors, params.min_dist) == (15, 0.10)

    def test_untuned_embedder_falls_back_to_the_globals(self):
        params = resolve_projection_params(_ctx("siglip2"))
        assert params.n_neighbors == PROJECTION_N_NEIGHBORS
        assert params.min_dist == PROJECTION_MIN_DIST

    def test_no_context_falls_back_to_the_globals(self):
        params = resolve_projection_params()
        assert params.n_neighbors == PROJECTION_N_NEIGHBORS
        assert params.min_dist == PROJECTION_MIN_DIST

    def test_compaction_follows_the_shipped_default(self):
        assert resolve_projection_params(_ctx("clip")).compact is PROJECTION_COMPACT_DEFAULT

    def test_explicit_override_beats_the_tuned_default(self, override_settings):
        override_settings(projection_n_neighbors=42, projection_min_dist=0.33)
        params = resolve_projection_params(_ctx("siglip"))
        assert (params.n_neighbors, params.min_dist) == (42, 0.33)

    def test_setting_left_at_the_global_default_is_not_an_override(self, override_settings):
        # Only ``min_dist`` is genuinely overridden; ``n_neighbors`` sits at the
        # global default, which means "unset" and lets the tuned value apply.
        override_settings(projection_n_neighbors=PROJECTION_N_NEIGHBORS, projection_min_dist=0.33)
        params = resolve_projection_params(_ctx("siglip"))
        assert (params.n_neighbors, params.min_dist) == (10, 0.33)

    def test_unavailable_core_config_falls_back_to_the_tuned_defaults(self, monkeypatch):
        # A library-only process with no builder installed must still fit, not
        # raise: no override is readable, so the tuned defaults stand.
        monkeypatch.setattr(config, "_core_config_builder", None)
        params = resolve_projection_params(_ctx("clip"))
        assert (params.n_neighbors, params.min_dist) == (10, 0.05)
        assert params.compact is PROJECTION_COMPACT_DEFAULT
