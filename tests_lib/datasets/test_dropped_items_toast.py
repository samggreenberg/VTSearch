"""The "Dropped N item(s)" toast says which items, and why (issue #4232).

An import whose embedder produced nothing for some items drops them at the end
of the load and raises a warning toast.  That toast used to end with "See the
server log for which embedder declined" - advice a GUI user cannot follow, since
the app has no log viewer.  The embed stage now records a reason per item, and
the drop stage puts those reasons in the toast's detail line and every dropped
item's name in its expandable, copyable item list.

The embedder here is a MagicMock standing in for a plugin's, and the medias
dict is shaped the way an importer hands it to the embed stage.
"""

from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pytest

from vtscore.concurrency.notifications import Notification, notifications
from vtscore.concurrency.progress import LoadingTasksTracker
from vtscore.datasets.stages.embedding import _embed_missing_stage, embed_missing
from vtscore.datasets.stages.finalize import _drop_none_embeddings_stage, _dropped_item_label
from vtscore.embedding.media_vectors import init_embeddings
from vtscore.state.core import DatasetContext

DIM = 4
EMBEDDER = "custom_embedder"
DISPLAY = "Custom Embedder"


def _fake_embedder(*, fail_ids: frozenset[int] = frozenset()):
    """A MagicMock embedder whose bulk pass returns ``None`` for *fail_ids*."""
    emb = mock.MagicMock()
    emb.name = EMBEDDER
    emb.display_name = DISPLAY
    emb.media_type_id = "audio"
    emb._model = True
    emb.supports_patch_regions = False
    emb.supports_geometric_verification = False
    emb.embedding_dim = None

    def _bulk(medias):
        return [None if m["id"] in fail_ids else np.full(DIM, float(m["id"]), dtype=np.float32) for m in medias]

    emb.embed_media_bulk.side_effect = _bulk
    return emb


def _medias(*ids: int) -> dict[int, dict]:
    return {
        i: {
            "id": i,
            "media_type": "audio",
            "filename": f"{i}.wav",
            "origin_name": f"take_{i}.wav",
            "md5": f"md5-{i}",
            "embedder": "",
            "embeddings": init_embeddings("", None),
            "media_path": f"/tmp/{i}.wav",
        }
        for i in ids
    }


def _tracker():
    return LoadingTasksTracker().create_task("test_dropped_items_toast", "test")


@pytest.fixture
def collected():
    """Subscribe a collector to the process-wide notification broker."""
    received: list[Notification] = []
    notifications.subscribe(received.append)
    try:
        yield received
    finally:
        notifications.unsubscribe(received.append)


class TestEmbedMissingRecordsWhy:
    """``embed_missing(failures=...)`` maps each vector-less item to a reason."""

    def test_items_the_embedder_declines_are_recorded_by_name(self):
        emb = _fake_embedder(fail_ids=frozenset({2, 3}))
        medias = _medias(1, 2, 3)
        failures: dict[int, str] = {}

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias, "", failures=failures)

        assert set(failures) == {2, 3}
        assert DISPLAY in failures[2] and "returned no vector" in failures[2]

    def test_a_crashing_batch_records_the_exception_for_every_item(self):
        emb = _fake_embedder()
        emb.embed_media_bulk.side_effect = RuntimeError("CUDA out of memory")
        medias = _medias(1, 2)
        failures: dict[int, str] = {}

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias, "", failures=failures)

        assert set(failures) == {1, 2}
        assert "RuntimeError: CUDA out of memory" in failures[1]

    def test_a_wrong_length_answer_is_recorded_for_every_item(self):
        emb = _fake_embedder()
        emb.embed_media_bulk.side_effect = lambda medias: [np.ones(DIM, dtype=np.float32)]
        medias = _medias(1, 2)
        failures: dict[int, str] = {}

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias, "", failures=failures)

        assert set(failures) == {1, 2}
        assert "1 vector(s) for 2 item(s)" in failures[1]

    def test_no_embedder_for_the_media_type_is_recorded(self):
        medias = _medias(1, 2)
        failures: dict[int, str] = {}

        with mock.patch("vtscore.media.embedders_for_type", return_value=[]):
            embed_missing(medias, "", failures=failures)

        assert failures == {
            1: "No embedder is installed for audio media",
            2: "No embedder is installed for audio media",
        }

    def test_the_slug_stands_in_when_there_is_no_display_name(self):
        emb = _fake_embedder(fail_ids=frozenset({1}))
        emb.display_name = None
        failures: dict[int, str] = {}

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(_medias(1), "", failures=failures)

        assert EMBEDDER in failures[1]

    def test_failures_is_optional(self):
        emb = _fake_embedder(fail_ids=frozenset({1}))
        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(_medias(1, 2), "")  # must not raise without a failures dict


class TestDropToast:
    """The drop stage's toast names the reason and lists the dropped items."""

    def _load(self, emb, *ids: int) -> DatasetContext:
        ctx = DatasetContext("test_dropped_items_toast_ctx")
        ctx.medias.update(_medias(*ids))
        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            failures = _embed_missing_stage(ctx, _tracker(), [""])
        _drop_none_embeddings_stage(ctx, _tracker(), failures)
        return ctx

    def test_toast_states_the_reason_and_lists_every_dropped_item(self, collected):
        ctx = self._load(_fake_embedder(fail_ids=frozenset({2, 3})), 1, 2, 3)

        assert sorted(ctx.medias) == [1]
        (note,) = collected
        assert note.level == "warning"
        assert note.message == "Dropped 2 item(s) whose embedding failed"
        assert "2 of 3 imported item(s)" in note.detail
        assert DISPLAY in note.detail and "returned no vector" in note.detail
        # One reason is said once, in the detail; the list is just the names.
        assert note.items == ("take_2.wav", "take_3.wav")

    def test_toast_never_points_at_the_server_log(self, collected):
        self._load(_fake_embedder(fail_ids=frozenset({1})), 1, 2)

        (note,) = collected
        assert "log" not in note.detail.lower()

    def test_mixed_reasons_are_grouped_and_said_per_item(self, collected):
        ctx = DatasetContext("test_dropped_items_toast_ctx")
        ctx.medias.update(_medias(1, 2, 3))
        failures = {1: "Reason A", 2: "Reason B", 3: "Reason B"}

        _drop_none_embeddings_stage(ctx, _tracker(), failures)

        (note,) = collected
        assert "Reason B (2 item(s))." in note.detail
        assert "Reason A (1 item(s))." in note.detail
        # Commonest reason first, each item carrying its own.
        assert note.items == (
            "take_2.wav — Reason B",
            "take_3.wav — Reason B",
            "take_1.wav — Reason A",
        )

    def test_an_item_with_no_recorded_reason_still_gets_one(self, collected):
        ctx = DatasetContext("test_dropped_items_toast_ctx")
        ctx.medias.update(_medias(1))

        _drop_none_embeddings_stage(ctx, _tracker())

        (note,) = collected
        assert "No embedder produced a vector" in note.detail
        assert note.items == ("take_1.wav",)

    def test_nothing_dropped_means_no_toast(self, collected):
        self._load(_fake_embedder(), 1, 2)
        assert collected == []


class TestDroppedItemLabel:
    def test_prefers_origin_name_then_filename_then_path_then_id(self):
        assert _dropped_item_label(1, {"origin_name": "a.wav", "filename": "b.wav"}) == "a.wav"
        assert _dropped_item_label(1, {"filename": "b.wav", "media_path": "/x/c.wav"}) == "b.wav"
        assert _dropped_item_label(1, {"media_path": "/x/c.wav"}) == "/x/c.wav"
        assert _dropped_item_label(7, {}) == "item 7"

    def test_a_time_clip_says_where_in_the_file_it_sits(self):
        media = {"origin_name": "song.wav", "clip_start": 2.0, "clip_end": 4.5, "clip_index": 1}
        assert _dropped_item_label(1, media) == "song.wav [2–4.5s]"

    def test_an_untimed_clip_falls_back_to_its_index(self):
        assert _dropped_item_label(1, {"origin_name": "doc.txt", "clip_index": 3}) == "doc.txt (clip 3)"
