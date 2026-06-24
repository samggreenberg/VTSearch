"""Per-detector primary embedder: create-time validation, persistence, reload.

A detector binds one *primary* embedder at create time (its scoring vector
space).  These cover the route + state plumbing:

* a single-embedder active dataset auto-resolves the empty pick;
* a multi-embedder active dataset rejects an empty / unbound pick and accepts a
  bound one, persisting it on the detector JSON;
* loading a detector stamps ``DetectorContext.primary_embedder`` from the JSON;
* a legacy detector with no field loads fine (empty primary).

See docs/plans/patch-embedder.md → "Per-detector primary embedder".
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from vtsearch.settings import get_detectors_dir


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
    return np.eye(4, dtype=np.float32)[i]


def _activate_dataset(embedder_names: list[str]):
    """Register + thread-activate a dataset whose one media binds *embedder_names*.

    The test client injects this context's id as ``X-Dataset-Id`` on each call,
    so ``get_active_context()`` in the create route sees these bound embedders.
    """
    from vtscore.state.core import (
        DatasetContext,
        register_context,
        set_thread_dataset_context,
    )

    ctx = DatasetContext("ds-primary-route")
    ctx.medias[1] = {
        "id": 1,
        "media_type": "image",
        "embedder": embedder_names[0] if embedder_names else "",
        "embeddings": {name: _basis(0) for name in embedder_names},
    }
    register_context(ctx)
    set_thread_dataset_context(ctx)
    return ctx


def _read_detector_json(name: str) -> dict:
    from vtscore.detectors.store import _detector_path, _read_detector

    return _read_detector(_detector_path(name)) or {}


class TestCreateValidation:
    def test_single_embedder_auto_resolves(self, client):
        _activate_dataset(["clap"])
        res = client.post(
            "/api/detectors/registry",
            json={"name": "single-emb", "media_type": "image", "text_query": "x"},
        )
        assert res.status_code == 201, res.get_json()
        assert _read_detector_json("single-emb")["primary_embedder"] == "clap"

    def test_multi_embedder_empty_pick_rejected(self, client):
        _activate_dataset(["siglip", "dinov3_patch"])
        res = client.post(
            "/api/detectors/registry",
            json={"name": "multi-empty", "media_type": "image", "text_query": "x"},
        )
        assert res.status_code == 400
        assert "multiple" in res.get_json()["message"].lower()

    def test_multi_embedder_unbound_pick_rejected(self, client):
        _activate_dataset(["siglip", "dinov3_patch"])
        res = client.post(
            "/api/detectors/registry",
            json={
                "name": "multi-bad",
                "media_type": "image",
                "text_query": "x",
                "primary_embedder": "not_bound",
            },
        )
        assert res.status_code == 400
        assert "not bound" in res.get_json()["message"].lower()

    def test_multi_embedder_explicit_pick_persists(self, client):
        _activate_dataset(["siglip", "dinov3_patch"])
        res = client.post(
            "/api/detectors/registry",
            json={
                "name": "multi-ok",
                "media_type": "image",
                "text_query": "x",
                "primary_embedder": "dinov3_patch",
            },
        )
        assert res.status_code == 201, res.get_json()
        assert _read_detector_json("multi-ok")["primary_embedder"] == "dinov3_patch"

    def test_crud_route_also_persists(self, client):
        _activate_dataset(["siglip", "dinov3_patch"])
        res = client.post(
            "/api/detectors",
            json={
                "name": "crud-ok",
                "media_type": "image",
                "text_query": "x",
                "primary_embedder": "siglip",
            },
        )
        assert res.status_code == 201, res.get_json()
        assert _read_detector_json("crud-ok")["primary_embedder"] == "siglip"


class TestReloadStampsPrimary:
    def test_loaded_context_carries_primary(self, client):
        from tests import load_detector_and_wait
        from vtscore.detectors.registry import find_by_name
        from vtscore.state.core import get_detector_context

        _activate_dataset(["siglip", "dinov3_patch"])
        client.post(
            "/api/detectors/registry",
            json={
                "name": "reload-primary",
                "media_type": "image",
                "text_query": "x",
                "primary_embedder": "dinov3_patch",
            },
        )
        entry = find_by_name("reload-primary")
        assert entry is not None
        load_detector_and_wait(client, entry["id"])
        ctx = get_detector_context(entry["id"])
        assert ctx is not None
        assert ctx.primary_embedder == "dinov3_patch"

    def test_legacy_detector_without_field_loads(self, client):
        """A detector JSON with no ``primary_embedder`` loads with empty primary."""
        from tests import load_detector_and_wait
        from vtscore.detectors.registry import register_detector, reset_for_tests
        from vtscore.detectors.store import _detector_path, _write_detector
        from vtscore.state.core import get_detector_context

        reset_for_tests()
        _write_detector(
            _detector_path("legacy-det"),
            {
                "name": "legacy-det",
                "text_query": "",
                "media_type": "image",
                "examples": [],
                "labelset": {"labels": []},
            },
        )
        entry = register_detector(name="legacy-det", media_type="image")
        load_detector_and_wait(client, entry["id"])
        ctx = get_detector_context(entry["id"])
        assert ctx is not None
        assert ctx.primary_embedder == ""
