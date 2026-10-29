"""An item missing one of its dataset's vectors is dropped, not saved (issue #3798).

A dataset holds one vector space per embedder it binds, and everything that
reads a space - Browse, sort, Train - needs every media's vector in it.  The
finalize stage dropped only the items with no vector at all, so in a load that
runs two embedders (or one embedder over an importer's shipped vectors) an item
one of them could not read kept the other's vector, was saved, and left the
dataset raising ``has no embedding for embedder`` on every request for the
missing space, on every load after.

The embed stage now empties such an item so the finalize stage drops it like
any unreadable item; a dataset already saved with one is repaired when it next
loads (:func:`drop_incomplete_medias`), and either way the items are kept on
the registry entry for the Stats window.

The embedders are MagicMocks standing in for plugins, registered under their
names for the duration of a test, and the medias are shaped the way an importer
hands them to the embed stage.
"""

from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pytest

from vtscore.concurrency.notifications import Notification, notifications
from vtscore.concurrency.progress import LoadingTasksTracker
from vtscore.datasets.loader import export_dataset_to_file, load_dataset_from_pickle
from vtscore.datasets.stages import finalize
from vtscore.datasets.stages.embedding import _embed_missing_stage
from vtscore.datasets.stages.finalize import (
    _drop_none_embeddings_stage,
    _dropped_groups,
    drop_incomplete_medias,
    merge_dropped_groups,
)
from vtscore.datasets.stages.registry import resave_registered_dataset
from vtscore.embedding.matrix import get_embedding_matrix
from vtscore.embedding.media_vectors import init_embeddings, media_embedder_names
from vtscore.state.core import DatasetContext

DIM = 4
FIRST = "first_embedder"
SECOND = "second_embedder"


def _fake_embedder(name: str, *, fail_ids: frozenset[int] = frozenset(), fail_all: bool = False):
    """A MagicMock image embedder that declines *fail_ids* (or the whole batch)."""
    emb = mock.MagicMock()
    emb.name = name
    emb.display_name = name.replace("_", " ").title()
    emb.media_type_id = "image"
    emb._model = True
    emb.supports_text = False
    emb.supports_patch_regions = False
    emb.supports_geometric_verification = False
    emb.embedding_dim = None

    def _bulk(medias):
        if fail_all:
            raise RuntimeError("model crashed")
        return [None if m["id"] in fail_ids else np.full(DIM, float(m["id"]), dtype=np.float32) for m in medias]

    emb.embed_media_bulk.side_effect = _bulk
    return emb


def _media(mid: int, embeddings: dict | None = None) -> dict:
    return {
        "id": mid,
        "media_type": "image",
        "filename": f"{mid}.jpg",
        "origin_name": f"photo_{mid}.jpg",
        "md5": f"md5-{mid}",
        "embedder": next(iter(embeddings), "") if embeddings else "",
        "embeddings": init_embeddings("", None) if embeddings is None else dict(embeddings),
        "media_path": f"/tmp/{mid}.jpg",
        "duration": 0.0,
        "file_size": 10,
    }


def _vec(mid: int) -> np.ndarray:
    return np.full(DIM, float(mid), dtype=np.float32)


@pytest.fixture
def register(monkeypatch):
    """Register fake embedders by name, as a plugin's would be."""
    from vtscore import media

    def _register(*embs):
        for emb in embs:
            monkeypatch.setitem(media._embedder_registry, emb.name, emb)

    return _register


@pytest.fixture
def collected():
    received: list[Notification] = []
    notifications.subscribe(received.append)
    try:
        yield received
    finally:
        notifications.unsubscribe(received.append)


def _tracker():
    return LoadingTasksTracker().create_task("test_incomplete_vectors", "test")


def _load(ctx: DatasetContext, requested: list[str]):
    """The GUI import's embed + drop stages, as ``_finish_dataset_load`` runs them."""
    failures = _embed_missing_stage(ctx, _tracker(), requested)
    return _drop_none_embeddings_stage(ctx, _tracker(), failures)


def _ctx(medias: dict[int, dict]) -> DatasetContext:
    ctx = DatasetContext("test_incomplete_vectors_ctx")
    ctx.medias.update(medias)
    return ctx


class TestImportDropsIncompleteItems:
    def test_an_item_the_first_embedder_declines_is_dropped_though_the_second_read_it(self, register, collected):
        register(_fake_embedder(FIRST, fail_ids=frozenset({2})), _fake_embedder(SECOND))
        ctx = _ctx({i: _media(i) for i in (1, 2, 3)})

        dropped = _load(ctx, [FIRST, SECOND])

        assert sorted(ctx.medias) == [1, 3]
        assert dropped == [{"reason": mock.ANY, "count": 1, "items": ["photo_2.jpg"]}]
        assert "First Embedder" in dropped[0]["reason"]
        (note,) = collected
        assert note.items == ("photo_2.jpg",)

    def test_an_item_the_second_embedder_declines_is_dropped(self, register):
        register(_fake_embedder(FIRST), _fake_embedder(SECOND, fail_ids=frozenset({3})))
        ctx = _ctx({i: _media(i) for i in (1, 2, 3)})

        dropped = _load(ctx, [FIRST, SECOND])

        assert sorted(ctx.medias) == [1, 2]
        assert "Second Embedder" in dropped[0]["reason"]

    def test_every_space_of_what_is_kept_builds(self, register):
        """The calls Browse, sort and Train make: the ones that raised before."""
        register(_fake_embedder(FIRST, fail_ids=frozenset({2})), _fake_embedder(SECOND, fail_ids=frozenset({4})))
        ctx = _ctx({i: _media(i) for i in (1, 2, 3, 4, 5)})

        _load(ctx, [FIRST, SECOND])

        assert sorted(ctx.medias) == [1, 3, 5]
        for name in (FIRST, SECOND):
            matrix, ids = get_embedding_matrix(ctx, embedder_name=name)[:2]
            assert list(ids) == [1, 3, 5]
            assert matrix.shape == (3, DIM)

    def test_a_shipped_vector_does_not_save_an_item_the_load_cannot_read(self, register):
        """The issue's case: the importer shipped a vector the load's embedder cannot match."""
        register(_fake_embedder(FIRST, fail_ids=frozenset({2})), _fake_embedder(SECOND))
        ctx = _ctx({i: _media(i, {SECOND: _vec(i)}) for i in (1, 2, 3)})

        _load(ctx, [FIRST])

        assert sorted(ctx.medias) == [1, 3]
        for media in ctx.medias.values():
            assert set(media_embedder_names(media)) == {FIRST, SECOND}

    def test_an_embedder_that_read_nothing_costs_no_item(self, register):
        """A model that fails every item is not one of the dataset's spaces."""
        register(_fake_embedder(FIRST), _fake_embedder(SECOND, fail_all=True))
        ctx = _ctx({i: _media(i) for i in (1, 2, 3)})

        assert _load(ctx, [FIRST, SECOND]) == []
        assert sorted(ctx.medias) == [1, 2, 3]

    def test_a_vector_the_load_never_ran_marks_nobody_incomplete(self, register):
        """A stray shipped vector on a few items is not a space the rest must have."""
        register(_fake_embedder(FIRST))
        medias = {i: _media(i) for i in (1, 2, 3)}
        medias[3] = _media(3, {SECOND: _vec(3)})
        ctx = _ctx(medias)

        assert _load(ctx, [FIRST]) == []
        assert sorted(ctx.medias) == [1, 2, 3]

    def test_a_single_embedder_load_is_unchanged(self, register):
        register(_fake_embedder(FIRST, fail_ids=frozenset({2})))
        ctx = _ctx({i: _media(i) for i in (1, 2, 3)})

        dropped = _load(ctx, [FIRST])

        assert sorted(ctx.medias) == [1, 3]
        assert "returned no vector" in dropped[0]["reason"]


class TestDroppedGroups:
    def test_groups_are_commonest_first_with_exact_counts(self):
        groups = _dropped_groups(["a", "b", "c"], ["B", "B", "A"])
        assert groups == [
            {"reason": "B", "count": 2, "items": ["a", "b"]},
            {"reason": "A", "count": 1, "items": ["c"]},
        ]

    def test_names_stop_at_the_cap_but_counts_do_not(self, monkeypatch):
        monkeypatch.setattr(finalize, "MAX_RECORDED_DROPS", 2)
        groups = _dropped_groups(["a", "b", "c"], ["B", "B", "A"])
        assert groups == [
            {"reason": "B", "count": 2, "items": ["a", "b"]},
            {"reason": "A", "count": 1, "items": []},
        ]

    def test_merge_folds_a_known_reason_and_appends_a_new_one(self):
        recorded = [{"reason": "A", "count": 1, "items": ["a"]}]
        new = [{"reason": "A", "count": 2, "items": ["b", "c"]}, {"reason": "B", "count": 1, "items": ["d"]}]

        merged = merge_dropped_groups(recorded, new)

        assert merged == [
            {"reason": "A", "count": 3, "items": ["a", "b", "c"]},
            {"reason": "B", "count": 1, "items": ["d"]},
        ]
        assert recorded == [{"reason": "A", "count": 1, "items": ["a"]}]

    def test_merge_keeps_the_cap(self, monkeypatch):
        monkeypatch.setattr(finalize, "MAX_RECORDED_DROPS", 2)
        merged = merge_dropped_groups(
            [{"reason": "A", "count": 1, "items": ["a"]}], [{"reason": "B", "count": 3, "items": ["b", "c", "d"]}]
        )
        assert merged == [
            {"reason": "A", "count": 1, "items": ["a"]},
            {"reason": "B", "count": 3, "items": ["b"]},
        ]


class TestRegistryKeepsTheList:
    def test_the_entry_records_what_the_import_dropped(self):
        from vtscore.datasets.registry import register_dataset

        groups = [{"reason": "A", "count": 3, "items": ["a", "b", "c"]}]
        entry = register_dataset(
            name="x", media_type="image", num_items=1, pkl_path="/nonexistent.pkl", dropped_groups=groups
        )
        assert entry["num_dropped"] == 3
        assert entry["dropped_groups"] == groups

    def test_an_entry_that_dropped_nothing_says_so(self):
        from vtscore.datasets.registry import register_dataset

        entry = register_dataset(name="x", media_type="image", num_items=1, pkl_path="/nonexistent.pkl")
        assert entry["num_dropped"] == 0
        assert entry["dropped_groups"] == []


def _saved_shape(n: int, *, holes: dict[int, str]) -> dict[int, dict]:
    """*n* media carrying both vectors, except each id in *holes* lacks the named one."""
    medias = {}
    for i in range(1, n + 1):
        embs = {FIRST: _vec(i), SECOND: _vec(i)}
        embs.pop(holes.get(i, ""), None)
        medias[i] = _media(i, embs)
    return medias


class TestRepairOnLoad:
    def test_items_missing_a_vector_most_carry_are_removed(self, register):
        register(_fake_embedder(FIRST), _fake_embedder(SECOND))
        medias = _saved_shape(5, holes={2: FIRST, 4: SECOND})

        dropped = drop_incomplete_medias(medias)

        assert sorted(medias) == [1, 3, 5]
        assert sorted(item for g in dropped for item in g["items"]) == ["photo_2.jpg", "photo_4.jpg"]
        reasons = {g["reason"] for g in dropped}
        assert reasons == {
            "Missing the First Embedder vector the rest of the dataset has",
            "Missing the Second Embedder vector the rest of the dataset has",
        }

    def test_a_complete_dataset_is_left_alone(self):
        medias = _saved_shape(3, holes={})
        assert drop_incomplete_medias(medias) == []
        assert sorted(medias) == [1, 2, 3]

    def test_a_vector_only_a_few_carry_is_not_required(self):
        medias = {i: _media(i, {FIRST: _vec(i)}) for i in (1, 2, 3)}
        medias[3]["embeddings"][SECOND] = _vec(3)

        assert drop_incomplete_medias(medias) == []
        assert sorted(medias) == [1, 2, 3]

    def test_it_never_removes_half_the_dataset(self):
        medias = _saved_shape(4, holes={1: FIRST, 2: SECOND})

        assert drop_incomplete_medias(medias) == []
        assert sorted(medias) == [1, 2, 3, 4]

    def test_the_rewritten_file_reloads_repaired(self, tmp_path, register):
        register(_fake_embedder(FIRST), _fake_embedder(SECOND))
        path = tmp_path / "ds.pkl"
        path.write_bytes(export_dataset_to_file(_saved_shape(5, holes={2: SECOND}), name="Photos"))
        loaded: dict[int, dict] = {}
        load_dataset_from_pickle(path, loaded)
        assert sorted(loaded) == [1, 2, 3, 4, 5]

        drop_incomplete_medias(loaded)
        resave_registered_dataset({"pkl_path": str(path), "name": "Photos", "media_type": "image"}, loaded)

        reloaded: dict[int, dict] = {}
        load_dataset_from_pickle(path, reloaded)
        assert sorted(reloaded) == [1, 3, 4, 5]
        assert drop_incomplete_medias(reloaded) == []
        assert [p.name for p in tmp_path.iterdir()] == ["ds.pkl"]
