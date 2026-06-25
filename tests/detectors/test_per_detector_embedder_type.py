"""Per-detector embedder *type*: create-time validation, persistence, reload, gate.

A detector locks one embedder **type** at create time (its scoring vector-space
*kind*).  These cover the route + state plumbing:

* a single-type active dataset auto-resolves the empty pick (and two same-type
  embedders, e.g. SigLIP+CLIP, still count as one type → still auto-resolve);
* a multi-type active dataset rejects an empty / unsupplied pick and accepts a
  supplied one, persisting it on the detector JSON;
* loading a detector stamps ``DetectorContext.embedder_type`` from the JSON,
  migrating a legacy ``primary_embedder`` name;
* find-label refuses a detector whose type the active dataset can't supply.

See docs/plans/patch-embedder.md → "Per-detector embedder type".
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
    so ``get_active_context()`` in the create / find routes sees these bound
    embedders.
    """
    from vtscore.state.core import (
        DatasetContext,
        register_context,
        set_thread_dataset_context,
    )

    ctx = DatasetContext("ds-type-route")
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
    def test_single_type_auto_resolves(self, client):
        _activate_dataset(["clap"])
        res = client.post(
            "/api/detectors/registry",
            json={"name": "single-type", "media_type": "image", "text_query": "x"},
        )
        assert res.status_code == 201, res.get_json()
        assert _read_detector_json("single-type")["embedder_type"] == "semantic"

    def test_two_semantic_embedders_still_auto_resolve(self, client):
        # SigLIP + CLIP are both semantic → one supplied type → no pick needed.
        _activate_dataset(["siglip", "clip"])
        res = client.post(
            "/api/detectors/registry",
            json={"name": "two-semantic", "media_type": "image", "text_query": "x"},
        )
        assert res.status_code == 201, res.get_json()
        assert _read_detector_json("two-semantic")["embedder_type"] == "semantic"

    def test_multi_type_empty_pick_rejected(self, client):
        _activate_dataset(["siglip", "dinov3_patch"])
        res = client.post(
            "/api/detectors/registry",
            json={"name": "multi-empty", "media_type": "image", "text_query": "x"},
        )
        assert res.status_code == 400
        assert "multiple" in res.get_json()["message"].lower()

    def test_multi_type_unsupplied_pick_rejected(self, client):
        _activate_dataset(["siglip", "dinov3_patch"])
        res = client.post(
            "/api/detectors/registry",
            json={
                "name": "multi-bad",
                "media_type": "image",
                "text_query": "x",
                "embedder_type": "structural",
            },
        )
        assert res.status_code == 400
        assert "not bound" in res.get_json()["message"].lower()

    def test_multi_type_explicit_pick_persists(self, client):
        _activate_dataset(["siglip", "dinov3_patch"])
        res = client.post(
            "/api/detectors/registry",
            json={
                "name": "multi-ok",
                "media_type": "image",
                "text_query": "x",
                "embedder_type": "patch_semantic",
            },
        )
        assert res.status_code == 201, res.get_json()
        assert _read_detector_json("multi-ok")["embedder_type"] == "patch_semantic"

    def test_crud_route_also_persists(self, client):
        _activate_dataset(["siglip", "dinov3_patch"])
        res = client.post(
            "/api/detectors",
            json={
                "name": "crud-ok",
                "media_type": "image",
                "text_query": "x",
                "embedder_type": "semantic",
            },
        )
        assert res.status_code == 201, res.get_json()
        assert _read_detector_json("crud-ok")["embedder_type"] == "semantic"


class TestReloadStampsType:
    def test_loaded_context_carries_type(self, client):
        from tests import load_detector_and_wait
        from vtscore.detectors.registry import find_by_name
        from vtscore.state.core import get_detector_context

        _activate_dataset(["siglip", "dinov3_patch"])
        client.post(
            "/api/detectors/registry",
            json={
                "name": "reload-type",
                "media_type": "image",
                "text_query": "x",
                "embedder_type": "patch_semantic",
            },
        )
        entry = find_by_name("reload-type")
        assert entry is not None
        load_detector_and_wait(client, entry["id"])
        ctx = get_detector_context(entry["id"])
        assert ctx is not None
        assert ctx.embedder_type == "patch_semantic"

    def test_legacy_primary_name_migrates_to_type(self, client):
        """A detector JSON with a legacy ``primary_embedder`` name loads as a type."""
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
                "primary_embedder": "dinov3_patch",  # legacy name field
                "labelset": {"labels": []},
            },
        )
        entry = register_detector(name="legacy-det", media_type="image")
        load_detector_and_wait(client, entry["id"])
        ctx = get_detector_context(entry["id"])
        assert ctx is not None
        assert ctx.embedder_type == "patch_semantic"

    def test_legacy_detector_without_field_loads(self, client):
        """A detector JSON with neither type nor primary loads with empty type."""
        from tests import load_detector_and_wait
        from vtscore.detectors.registry import register_detector, reset_for_tests
        from vtscore.detectors.store import _detector_path, _write_detector
        from vtscore.state.core import get_detector_context

        reset_for_tests()
        _write_detector(
            _detector_path("bare-det"),
            {
                "name": "bare-det",
                "text_query": "",
                "media_type": "image",
                "examples": [],
                "labelset": {"labels": []},
            },
        )
        entry = register_detector(name="bare-det", media_type="image")
        load_detector_and_wait(client, entry["id"])
        ctx = get_detector_context(entry["id"])
        assert ctx is not None
        assert ctx.embedder_type == ""


class TestCompatibilityGate:
    def test_find_label_refuses_incompatible_type(self, client):
        """A structural detector on a semantic-only dataset is gated (409)."""
        from vtscore.detectors.registry import register_detector, reset_for_tests
        from vtscore.detectors.store import _detector_path, _write_detector

        reset_for_tests()
        _write_detector(
            _detector_path("structural-det"),
            {
                "name": "structural-det",
                "text_query": "",
                "media_type": "image",
                "examples": [],
                "embedder_type": "structural",
                "labelset": {"labels": []},
            },
        )
        entry = register_detector(name="structural-det", media_type="image", embedder_type="structural")
        # Active dataset binds only a semantic embedder → no structural slot.
        _activate_dataset(["siglip"])
        res = client.post("/api/find-label", json={"detector_id": entry["id"]})
        assert res.status_code == 409, res.get_json()
        assert "structural" in res.get_json()["message"].lower()
