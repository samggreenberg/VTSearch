"""Tests for ``GET /api/find/training-domain`` (#4961).

The training-domain overlap check, with the reference worked out rather than
picked: the loaded dataset holding most of the active detector's Good / Bad
examples.  The SPA calls it the moment a detector meets a dataset, so it must
answer ``domain_shift: null`` with a reason, never a 4xx, whenever there is
nothing to compare.
"""

from __future__ import annotations

import numpy as np

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.state.core import (
    DatasetContext,
    DetectorContext,
    register_context,
    register_detector_context,
    set_thread_dataset_context,
    set_thread_detector_context,
)
from vtscore.utils.hashing import content_md5


def _media(cid: int, dataset: str, *, seed_offset: int, shifted: bool, embedder: str) -> dict:
    rng = np.random.default_rng(cid + seed_offset)
    emb = np.zeros(64, dtype=np.float32)
    if shifted:
        # Directions confined to the second half of the coordinates: unlike
        # anything an in-domain dataset (first half) contains.
        emb[32:] = rng.standard_normal(32).astype(np.float32)
        emb[32] += 4.0
    else:
        emb[:32] = rng.standard_normal(32).astype(np.float32)
        emb[cid % 2] += 4.0
    emb /= np.linalg.norm(emb)
    name = f"{dataset}_{cid}.wav"
    return {
        "id": cid,
        "media_type": "audio",
        "embedder": embedder,
        "md5": content_md5(f"{dataset}/{cid}".encode()),
        "embeddings": {embedder: emb},
        "filename": name,
        "origin": {"importer": "test", "params": {"dataset": dataset}},
        "origin_name": name,
    }


def _dataset(
    name: str,
    n: int,
    *,
    seed_offset: int,
    shifted: bool = False,
    atlas: bool = False,
    embedder: str = "clap",
):
    """Register + load a dataset of *n* synthetic medias, with a coverage atlas when asked."""
    from vtscore.coverage.atlas import CoverageAtlas
    from vtscore.datasets.registry import add_loaded_id, register_dataset

    entry = register_dataset(name=name, media_type="audio", num_items=n, pkl_path=f"/tmp/{name}.pkl", embedder=embedder)
    ctx = DatasetContext(entry["id"])
    for cid in range(n):
        ctx.medias[cid] = _media(cid, name, seed_offset=seed_offset, shifted=shifted, embedder=embedder)
    if atlas:
        ctx.coverage_atlas = CoverageAtlas({cid: m["embeddings"][embedder] for cid, m in ctx.medias.items()}, k=3)
    register_context(ctx)
    add_loaded_id(entry["id"])
    return entry, ctx


def _labels_from(ctx: DatasetContext, ids) -> list[LabeledElement]:
    """Good / Bad examples naming *ctx*'s medias, alternating."""
    return [
        LabeledElement(
            md5=ctx.medias[cid]["md5"],
            label="good" if i % 2 == 0 else "bad",
            origin=ctx.medias[cid]["origin"],
            origin_name=ctx.medias[cid]["origin_name"],
        )
        for i, cid in enumerate(ids)
    ]


def _detector(elements: list[LabeledElement]) -> str:
    det = DetectorContext("det-training-domain", name="training-domain")
    det.cached_labelset = LabelSet(elements)
    register_detector_context(det)
    return det.detector_id


def _check(client, dataset_id: str, detector_id: str = "") -> dict:
    headers = {"X-Dataset-Id": dataset_id, "X-Detector-Id": detector_id}
    resp = client.get("/api/find/training-domain", headers=headers)
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


class TestTrainingDomain:
    def test_flags_a_dataset_unlike_the_training_one(self, client):
        """Trained on one kind of data, met with another: the report names the training dataset and says shifted."""
        ref_entry, ref_ctx = _dataset("photos", 120, seed_offset=0, atlas=True)
        target_entry, target_ctx = _dataset("screens", 40, seed_offset=2000, shifted=True)
        set_thread_dataset_context(target_ctx)
        detector_id = _detector(_labels_from(ref_ctx, range(10)))

        data = _check(client, target_entry["id"], detector_id)
        assert data["reference_dataset_id"] == ref_entry["id"]
        assert data["reference_dataset_name"] == "photos"
        assert data["labels_total"] == 10
        assert data["labels_in_reference"] == 10
        assert data["labels_in_active"] == 0
        assert data["reason"] == ""
        report = data["domain_shift"]
        assert report["reference_dataset_id"] == ref_entry["id"]
        assert report["n_items"] == 40
        assert report["shifted"] is True
        assert report["frac_atypical"] > 0.5

    def test_quiet_on_a_dataset_like_the_training_one(self, client):
        ref_entry, ref_ctx = _dataset("photos", 120, seed_offset=0, atlas=True)
        target_entry, target_ctx = _dataset("more-photos", 40, seed_offset=1000)
        set_thread_dataset_context(target_ctx)
        detector_id = _detector(_labels_from(ref_ctx, range(10)))

        data = _check(client, target_entry["id"], detector_id)
        assert data["reference_dataset_id"] == ref_entry["id"]
        assert data["domain_shift"]["shifted"] is False

    def test_nothing_to_compare_on_the_training_dataset(self, client):
        """Used on the dataset its examples came from: no report, and the reference says so."""
        ref_entry, ref_ctx = _dataset("photos", 120, seed_offset=0, atlas=True)
        _dataset("screens", 40, seed_offset=2000, shifted=True, atlas=True)
        set_thread_dataset_context(ref_ctx)
        detector_id = _detector(_labels_from(ref_ctx, range(10)))

        data = _check(client, ref_entry["id"], detector_id)
        assert data["domain_shift"] is None
        assert data["reference_dataset_id"] == ref_entry["id"]
        assert data["labels_in_active"] == 10
        assert "from this dataset" in data["reason"]

    def test_the_active_dataset_wins_a_tie(self, client):
        """Half the examples here, half there: the detector has seen this dataset, so nothing is flagged."""
        ref_entry, ref_ctx = _dataset("photos", 120, seed_offset=0, atlas=True)
        target_entry, target_ctx = _dataset("screens", 40, seed_offset=2000, shifted=True)
        set_thread_dataset_context(target_ctx)
        detector_id = _detector(_labels_from(ref_ctx, range(4)) + _labels_from(target_ctx, range(4)))

        data = _check(client, target_entry["id"], detector_id)
        assert data["domain_shift"] is None
        assert data["reference_dataset_id"] == target_entry["id"]

    def test_most_examples_pick_the_reference(self, client):
        """Examples in two other datasets: the one holding more of them is the reference."""
        _few_entry, few_ctx = _dataset("few", 60, seed_offset=500, atlas=True)
        many_entry, many_ctx = _dataset("many", 120, seed_offset=0, atlas=True)
        target_entry, target_ctx = _dataset("screens", 40, seed_offset=2000, shifted=True)
        set_thread_dataset_context(target_ctx)
        detector_id = _detector(_labels_from(few_ctx, range(2)) + _labels_from(many_ctx, range(8)))

        data = _check(client, target_entry["id"], detector_id)
        assert data["reference_dataset_id"] == many_entry["id"]
        assert data["labels_in_reference"] == 8
        assert data["domain_shift"]["reference_dataset_id"] == many_entry["id"]

    def test_training_dataset_not_loaded(self, client):
        target_entry, target_ctx = _dataset("screens", 40, seed_offset=2000, shifted=True)
        set_thread_dataset_context(target_ctx)
        stray = DatasetContext("unloaded")
        stray.medias[0] = _media(0, "unloaded", seed_offset=0, shifted=False, embedder="clap")
        detector_id = _detector(_labels_from(stray, [0]))

        data = _check(client, target_entry["id"], detector_id)
        assert data["domain_shift"] is None
        assert data["reference_dataset_id"] == ""
        assert data["labels_total"] == 1
        assert "none of the loaded datasets" in data["reason"].lower()

    def test_reference_without_atlas(self, client):
        """The check reads the reference's vectors, not its atlas (#3348), so an unbuilt atlas does not stop it."""
        _ref_entry, ref_ctx = _dataset("photos", 60, seed_offset=0)
        target_entry, target_ctx = _dataset("screens", 40, seed_offset=2000, shifted=True)
        set_thread_dataset_context(target_ctx)
        detector_id = _detector(_labels_from(ref_ctx, range(6)))

        data = _check(client, target_entry["id"], detector_id)
        assert ref_ctx.coverage_atlas is None
        assert data["domain_shift"]["shifted"] is True

    def test_embedder_mismatch(self, client):
        _ref_entry, ref_ctx = _dataset("photos", 120, seed_offset=0, atlas=True, embedder="clap")
        target_entry, target_ctx = _dataset("screens", 40, seed_offset=2000, shifted=True, embedder="other")
        set_thread_dataset_context(target_ctx)
        detector_id = _detector(_labels_from(ref_ctx, range(6)))

        data = _check(client, target_entry["id"], detector_id)
        assert data["domain_shift"] is None
        assert "mismatch" in data["reason"].lower()

    def test_patch_embedder_is_answered(self, client):
        """Refused until #3348, when the atlas guard (silent in a patch space, #3329) gave way to the kNN report."""
        _ref_entry, ref_ctx = _dataset("photos", 120, seed_offset=0, embedder="dinov3_patch")
        target_entry, target_ctx = _dataset("screens", 40, seed_offset=2000, shifted=True, embedder="dinov3_patch")
        set_thread_dataset_context(target_ctx)
        detector_id = _detector(_labels_from(ref_ctx, range(6)))

        data = _check(client, target_entry["id"], detector_id)
        assert data["domain_shift"]["shifted"] is True

    def test_detector_without_examples(self, client):
        target_entry, target_ctx = _dataset("screens", 40, seed_offset=2000, shifted=True)
        set_thread_dataset_context(target_ctx)
        detector_id = _detector([])

        data = _check(client, target_entry["id"], detector_id)
        assert data["domain_shift"] is None
        assert data["labels_total"] == 0
        assert "no labelled examples" in data["reason"]

    def test_no_active_detector(self, client):
        target_entry, target_ctx = _dataset("screens", 40, seed_offset=2000, shifted=True)
        set_thread_dataset_context(target_ctx)
        # The fixtures install a default detector; a request with none has none.
        set_thread_detector_context(None)

        data = _check(client, target_entry["id"])
        assert data["domain_shift"] is None
        assert data["reason"] == "No detector is active"
