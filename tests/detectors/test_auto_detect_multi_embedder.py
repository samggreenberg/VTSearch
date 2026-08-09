"""``POST /api/auto-detect`` on a dataset whose score precedence nobody keys to.

Regression for the ``KeyError`` at the top of ``auto_detect``: the route built
one embedding matrix per *distinct keying embedder* of the Auto-Find detectors,
then unconditionally read ``matrices[default_score]`` to size the worker cap and
the achievement count.  On a multi-embedder dataset the score precedence
(structural ▸ patch ▸ text) can name an embedder no detector keys to - the
normal case when every Auto-Find detector is type-locked to ``semantic`` while
the dataset also binds a patch embedder - so that key was never built and the
whole request 500'd even though every per-detector scoring path would have
succeeded.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

import app as app_module  # noqa: F401  (ensures routes are registered)
from vtsearch.settings import get_detectors_dir

DIM = 4


@pytest.fixture(autouse=True)
def clean_detectors_dir():
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)
    yield
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)


def _basis(i: int) -> np.ndarray:
    return np.eye(DIM, dtype=np.float32)[i]


def _trio_medias() -> dict[int, dict]:
    """Four image medias bound to siglip (text slot) + dinov3_patch (patch slot).

    The score precedence therefore resolves to ``dinov3_patch``, while a
    ``semantic`` detector keys to ``siglip``.
    """
    medias = {}
    for i in range(1, 5):
        on = i <= 2
        medias[i] = {
            "id": i,
            "media_type": "image",
            "embedder": "siglip",
            "embeddings": {
                "siglip": _basis(0) if on else _basis(1),
                "dinov3_patch": _basis(1) if on else _basis(0),
            },
            "filename": f"m{i}.png",
            "md5": f"m{i}",
            "origin_name": f"m{i}.png",
            "origin": {"importer": "test", "params": {}},
        }
    return medias


def _activate_trio_dataset():
    from vtscore.state.core import DatasetContext, register_context, set_thread_dataset_context

    ctx = DatasetContext("ds-auto-detect-trio")
    ctx.medias.update(_trio_medias())
    register_context(ctx)
    set_thread_dataset_context(ctx)
    return ctx


class TestAutoDetectOnMultiEmbedderDataset:
    def test_semantic_only_autofind_does_not_500(self, client):
        """No detector keys to the score-precedence embedder → must still run."""
        from tests.helpers import setup_trainable_model_in_registry
        from vtsearch.settings import add_autofind_detector

        ctx = _activate_trio_dataset()
        assert ctx.routed_embedder("score") == "dinov3_patch"

        snap = dict(ctx.medias)
        setup_trainable_model_in_registry(
            "semantic-autofind",
            good_ids=[1, 2],
            bad_ids=[3, 4],
            snap=snap,
            media_type="image",
            embedder_type="semantic",
        )
        add_autofind_detector("semantic-autofind")

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["media_type"] == "image"
        assert data["detectors_run"] == 1, data
        result = data["results"]["semantic-autofind"]
        assert len(result["hits"]) + len(result["negative_hits"]) == len(snap)
