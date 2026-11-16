"""Per-dataset blur state: overrides, the blurred-id set, and the sidecar (#5692).

An item's effective blur state is its Blur / Unblur override when it has one,
else whether any blur rule matches its raw ``custom_metadata``. Only the
overrides are stored, in a ``<stem>.blur.json`` sidecar beside the dataset's
container, so a rules change re-decides every item nobody overruled. The
rules' own semantics are pinned in ``test_blur_rules.py``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import vtscore.datasets.blur_state as blur_state
from vtscore.datasets.blur_state import (
    blur_overrides,
    blur_sidecar_path,
    blurred_ids,
    effective_blurred,
    flush_blur_writes,
    read_blur_overrides,
    set_blur_override,
)
from vtscore.datasets.registry import register_dataset, unregister_dataset
from vtscore.state.core import DatasetContext

PORN = {"field": "Porn", "op": ">=", "value": 0.8}
GORE = {"field": "Gore", "op": "truthy"}

SENSITIVE = {"Porn": 0.93}
SAFE = {"Porn": 0.1}


def _ctx(dataset_id: str, metadata: dict[int, dict]) -> DatasetContext:
    ctx = DatasetContext(dataset_id)
    ctx.medias.update(
        {mid: {"id": mid, "media_type": "image", "custom_metadata": dict(md)} for mid, md in metadata.items()}
    )
    return ctx


def _registered(tmp_path: Path, name: str = "blur") -> tuple[str, Path]:
    """A registered dataset whose container is a real file; returns ``(dataset_id, pkl_path)``."""
    pkl = tmp_path / f"ds_{name}.pkl"
    pkl.write_bytes(b"container")
    entry = register_dataset(name=name, media_type="image", num_items=3, pkl_path=str(pkl))
    return entry["id"], pkl


@pytest.fixture
def parked_writes(monkeypatch):
    """Keep queued sidecar writes queued until a test flushes them."""
    monkeypatch.setattr(blur_state, "WRITE_DELAY_SECONDS", 3600.0)


class TestEffectiveState:
    """The truth table: an override wins in both directions; without one, the rules decide."""

    @pytest.mark.parametrize(
        ("override", "metadata", "expected"),
        [
            (None, SENSITIVE, True),
            (None, SAFE, False),
            (True, SENSITIVE, True),
            (True, SAFE, True),
            (False, SENSITIVE, False),
            (False, SAFE, False),
        ],
    )
    def test_override_else_rules(self, override, metadata, expected):
        assert effective_blurred(override, [PORN], metadata) is expected

    def test_no_rules_blur_nothing_without_an_override(self):
        assert effective_blurred(None, [], SENSITIVE) is False

    def test_the_set_follows_the_same_table(self):
        ctx = _ctx("truth", {1: SENSITIVE, 2: SAFE, 3: SENSITIVE, 4: SAFE, 5: SENSITIVE, 6: SAFE})
        set_blur_override(ctx, 3, True)
        set_blur_override(ctx, 4, True)
        set_blur_override(ctx, 5, False)
        set_blur_override(ctx, 6, False)
        assert blurred_ids(ctx, [PORN]) == {1, 3, 4}


class TestRulesChange:
    def test_redecides_items_without_an_override_and_leaves_overridden_ones(self):
        ctx = _ctx("rules", {1: {"Porn": 0.9}, 2: {"Gore": "yes"}, 3: {"Porn": 0.9}, 4: {"Gore": "yes"}})
        set_blur_override(ctx, 3, False)  # sensitive by Porn, overruled
        set_blur_override(ctx, 4, True)  # pinned blurred
        assert blurred_ids(ctx, [PORN]) == {1, 4}
        # Swap Porn for Gore: 1 and 2 are re-decided, 3 and 4 keep their overrides.
        assert blurred_ids(ctx, [GORE]) == {2, 4}
        assert blurred_ids(ctx, []) == {4}

    def test_an_override_on_an_item_no_longer_present_is_ignored(self):
        ctx = _ctx("gone", {1: SAFE, 2: SAFE})
        set_blur_override(ctx, 2, True)
        del ctx.medias[2]
        assert blurred_ids(ctx, []) == frozenset()
        assert blurred_ids(ctx, [PORN]) == frozenset()


class TestTheCachedSet:
    """``_blurred_ids`` is reused until the medias, the overrides or the rules move."""

    def test_reused_while_nothing_changes(self):
        ctx = _ctx("cache", {1: SENSITIVE, 2: SAFE})
        first = blurred_ids(ctx, [PORN])
        assert blurred_ids(ctx, [dict(PORN)]) is first

    def test_an_override_write_invalidates_it(self):
        ctx = _ctx("cache-override", {1: SENSITIVE, 2: SAFE})
        assert blurred_ids(ctx, [PORN]) == {1}
        set_blur_override(ctx, 2, True)
        assert blurred_ids(ctx, [PORN]) == {1, 2}

    def test_a_media_revision_invalidates_it(self):
        ctx = _ctx("cache-medias", {1: SENSITIVE})
        assert blurred_ids(ctx, [PORN]) == {1}
        ctx.medias[2] = {"id": 2, "custom_metadata": dict(SENSITIVE)}
        assert blurred_ids(ctx, [PORN]) == {1, 2}

    def test_an_in_place_edit_counts_once_the_revision_is_bumped(self):
        ctx = _ctx("cache-inplace", {1: SAFE})
        assert blurred_ids(ctx, [PORN]) == frozenset()
        ctx.medias[1]["custom_metadata"] = dict(SENSITIVE)
        ctx.bump_media_revision()
        assert blurred_ids(ctx, [PORN]) == {1}

    def test_a_rules_change_invalidates_it(self):
        ctx = _ctx("cache-rules", {1: SENSITIVE, 2: {"Gore": True}})
        assert blurred_ids(ctx, [PORN]) == {1}
        assert blurred_ids(ctx, [PORN, GORE]) == {1, 2}

    def test_a_derived_cache_reset_drops_it_and_keeps_the_overrides(self):
        ctx = _ctx("cache-reset", {1: SAFE})
        set_blur_override(ctx, 1, True)
        blurred_ids(ctx, [])
        ctx.reset_derived_caches()
        assert ctx._blurred_ids is None
        assert dict(blur_overrides(ctx)) == {1: True}


class TestOverrideWrites:
    def test_setting_the_state_an_item_has_changes_nothing(self):
        ctx = _ctx("noop", {1: SAFE})
        assert set_blur_override(ctx, 1, True) is True
        revision = ctx._blur_overrides_revision
        assert set_blur_override(ctx, 1, True) is False
        assert ctx._blur_overrides_revision == revision

    def test_a_write_replaces_the_mapping_rather_than_mutating_it(self):
        ctx = _ctx("cow", {1: SAFE, 2: SAFE})
        set_blur_override(ctx, 1, True)
        before = blur_overrides(ctx)
        set_blur_override(ctx, 2, False)
        assert dict(before) == {1: True}
        assert dict(blur_overrides(ctx)) == {1: True, 2: False}

    def test_a_dataset_with_no_container_keeps_its_overrides_in_memory(self, parked_writes):
        ctx = _ctx("in-memory-only", {1: SAFE})
        set_blur_override(ctx, 1, True)
        assert dict(blur_overrides(ctx)) == {1: True}
        assert not blur_state.pending_blur_writes()

    def test_concurrent_writes_lose_none_of_them(self, tmp_path, parked_writes):
        dataset_id, pkl = _registered(tmp_path, "concurrent")
        ctx = _ctx(dataset_id, {i: SAFE for i in range(40)})
        start = threading.Event()

        def click(media_id: int) -> None:
            start.wait()
            set_blur_override(ctx, media_id, True)

        threads = [threading.Thread(target=click, args=(i,)) for i in range(40)]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join()
        assert dict(blur_overrides(ctx)) == {i: True for i in range(40)}
        flush_blur_writes()
        assert read_blur_overrides(pkl) == {i: True for i in range(40)}


class TestPersistence:
    def test_the_sidecar_sits_beside_the_container_under_its_stem(self, tmp_path):
        assert blur_sidecar_path(tmp_path / "ds_abc.pkl") == tmp_path / "ds_abc.blur.json"

    def test_a_write_lands_atomically_as_media_id_to_bool(self, tmp_path):
        dataset_id, pkl = _registered(tmp_path, "format")
        ctx = _ctx(dataset_id, {3: SAFE, 12: SENSITIVE})
        set_blur_override(ctx, 12, False)
        set_blur_override(ctx, 3, True)
        flush_blur_writes()
        sidecar = blur_sidecar_path(pkl)
        assert json.loads(sidecar.read_text()) == {"3": True, "12": False}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["ds_format.blur.json", "ds_format.pkl"]

    def test_the_writer_lands_a_queued_write_on_its_own(self, tmp_path, monkeypatch):
        monkeypatch.setattr(blur_state, "WRITE_DELAY_SECONDS", 0.0)
        dataset_id, pkl = _registered(tmp_path, "timer")
        landed = threading.Event()
        real_write = blur_state.atomic_write_json

        def write(path, obj, **kw):
            real_write(path, obj, **kw)
            landed.set()

        monkeypatch.setattr(blur_state, "atomic_write_json", write)
        set_blur_override(_ctx(dataset_id, {1: SAFE}), 1, True)
        assert landed.wait(10.0)
        assert json.loads(blur_sidecar_path(pkl).read_text()) == {"1": True}

    def test_a_burst_of_clicks_is_one_write_of_the_newest_state(self, tmp_path, parked_writes, monkeypatch):
        dataset_id, pkl = _registered(tmp_path, "burst")
        writes: list[dict] = []
        monkeypatch.setattr(blur_state, "atomic_write_json", lambda path, obj, **kw: writes.append(obj))
        ctx = _ctx(dataset_id, {1: SAFE, 2: SAFE})
        set_blur_override(ctx, 1, True)
        set_blur_override(ctx, 2, True)
        set_blur_override(ctx, 1, False)
        flush_blur_writes()
        assert writes == [{"1": False, "2": True}]

    def test_an_override_survives_closing_and_reopening_the_dataset(self, tmp_path):
        dataset_id, _pkl = _registered(tmp_path, "reopen")
        first = _ctx(dataset_id, {1: SENSITIVE, 2: SAFE})
        set_blur_override(first, 1, False)
        set_blur_override(first, 2, True)
        flush_blur_writes()
        # A load builds a fresh context for the same dataset id.
        reopened = _ctx(dataset_id, {1: SENSITIVE, 2: SAFE})
        assert dict(blur_overrides(reopened)) == {1: False, 2: True}
        assert blurred_ids(reopened, [PORN]) == {2}

    def test_a_reopen_before_the_write_lands_still_sees_it(self, tmp_path, parked_writes):
        dataset_id, pkl = _registered(tmp_path, "early-reopen")
        set_blur_override(_ctx(dataset_id, {1: SAFE}), 1, True)
        assert not blur_sidecar_path(pkl).exists()
        assert dict(blur_overrides(_ctx(dataset_id, {1: SAFE}))) == {1: True}

    def test_no_sidecar_is_no_overrides(self, tmp_path):
        assert read_blur_overrides(tmp_path / "ds_none.pkl") == {}

    @pytest.mark.parametrize("text", ["not json", "[1, 2]", '"x"'])
    def test_an_unreadable_sidecar_reads_as_none(self, tmp_path, text):
        blur_sidecar_path(tmp_path / "ds_bad.pkl").write_text(text)
        assert read_blur_overrides(tmp_path / "ds_bad.pkl") == {}

    def test_a_malformed_entry_is_skipped_and_the_rest_kept(self, tmp_path):
        blur_sidecar_path(tmp_path / "ds_mixed.pkl").write_text(
            json.dumps({"1": True, "x": True, "2": "yes", "3": False})
        )
        assert read_blur_overrides(tmp_path / "ds_mixed.pkl") == {1: True, 3: False}


class TestRemovedWithTheDataset:
    def test_unregister_deletes_the_sidecar(self, tmp_path):
        dataset_id, pkl = _registered(tmp_path, "delete")
        set_blur_override(_ctx(dataset_id, {1: SAFE}), 1, True)
        flush_blur_writes()
        assert blur_sidecar_path(pkl).is_file()
        assert unregister_dataset(dataset_id) is True
        assert list(tmp_path.iterdir()) == []

    def test_a_write_still_queued_does_not_put_it_back(self, tmp_path, parked_writes):
        dataset_id, pkl = _registered(tmp_path, "resurrect")
        ctx = _ctx(dataset_id, {1: SAFE, 2: SAFE})
        set_blur_override(ctx, 1, True)
        flush_blur_writes()
        set_blur_override(ctx, 2, True)  # queued, not yet landed
        assert unregister_dataset(dataset_id) is True
        flush_blur_writes()
        assert list(tmp_path.iterdir()) == []

    def test_a_write_queued_after_the_delete_is_skipped(self, tmp_path, parked_writes):
        dataset_id, pkl = _registered(tmp_path, "late")
        ctx = _ctx(dataset_id, {1: SAFE})
        blur_overrides(ctx)  # read while the dataset is still registered
        unregister_dataset(dataset_id)
        blur_state._queue_write(pkl, {1: True})
        flush_blur_writes()
        assert list(tmp_path.iterdir()) == []
