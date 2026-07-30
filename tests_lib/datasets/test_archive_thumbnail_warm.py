"""Library-tier tests for the archive-member thumbnail warm-up (issue #2738).

An archive-member import reads **no** member bytes by design, so its media leave
the load with no ``thumbnail_bytes``, no ``width``/``height``, and no file to
decode -- and every browse tile fell back to streaming a tar member and
decoding it on the request thread.  The background pass in
:mod:`vtscore.datasets.thumbnail_warm` fixes that by streaming each member once
off the request path.

These tests lock the properties that make the pass safe to run over a
multi-terabyte corpus: it produces the same thumbnails the ingest path would,
it never retains member payloads, it is bounded in in-flight bytes, it is
cancellable at item granularity, it survives unresolvable members, and it is
idempotent so every load can re-kick it.
"""

from __future__ import annotations

import io
import tarfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from vtscore.concurrency.async_jobs import AsyncJob
from vtscore.datasets.archive_stream import build_archive_member_origin
from vtscore.datasets.thumbnail_warm import (
    _ByteBudget,
    pending_archive_thumbnail_ids,
    start_archive_thumbnail_warm,
    warm_archive_thumbnails,
    warm_order,
)
from vtscore.media.image.media_type import ImageMediaType
from vtscore.media.image.thumbnail import make_image_thumbnail
from vtscore.state.core import DatasetContext, register_context, unregister_context

DIM = 8


def _jpeg_bytes(size: tuple[int, int] = (600, 400), color: tuple[int, int, int] = (10, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_image_tar(tmp_path: Path, names_to_bytes: dict[str, bytes], name: str = "shard_0.tar") -> Path:
    archive = tmp_path / name
    with tarfile.open(archive, "w") as tf:
        for member, payload in names_to_bytes.items():
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return archive


def _archive_media(media_id: int, archive: Path, member: str, payload: bytes, media_type: str = "image") -> dict:
    """A media shaped exactly like ``local_archive_member`` produces one.

    No ``media_bytes``, no ``media_path``, no ``thumbnail_bytes``, no
    dimensions -- only the ``{archive, member}`` reference plus the member size
    the importer reads from the tar header.
    """
    return {
        "id": media_id,
        "media_type": media_type,
        "filename": Path(member).name,
        "category": "custom",
        "file_size": len(payload),
        "media_bytes": None,
        "media_string": None,
        "duration": 0,
        "origin": build_archive_member_origin(archive, member, media_type),
        "origin_name": f"{archive}::{member}",
        "archive_member": {"path": str(archive), "member": member},
        "embeddings": {"clip": np.zeros(DIM, dtype=np.float32)},
    }


def _ctx_with(medias: dict[int, dict], dataset_id: str = "_warm_test") -> DatasetContext:
    ctx = DatasetContext(dataset_id)
    ctx.medias.update(medias)
    register_context(ctx)
    return ctx


def _image_corpus(tmp_path: Path, n: int = 6) -> tuple[Path, dict[str, bytes], dict[int, dict]]:
    payloads = {f"img{i}.jpg": _jpeg_bytes(color=(10 * i, 60, 200 - 10 * i)) for i in range(n)}
    archive = _make_image_tar(tmp_path, payloads)
    medias = {
        i + 1: _archive_media(i + 1, archive, member, payload) for i, (member, payload) in enumerate(payloads.items())
    }
    return archive, payloads, medias


class TestPendingSelection:
    def test_selects_archive_members_without_thumbnails(self, tmp_path: Path):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=3)
        assert sorted(pending_archive_thumbnail_ids(medias)) == [1, 2, 3]

    def test_skips_media_that_already_have_a_thumbnail(self, tmp_path: Path):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=3)
        medias[2]["thumbnail_bytes"] = b"already warm"
        assert sorted(pending_archive_thumbnail_ids(medias)) == [1, 3]

    def test_skips_non_archive_media(self, tmp_path: Path):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=2)
        # A plain in-memory image: nothing to stream, thumbnail already handled
        # at ingest by the folder loader.
        medias[99] = {
            "id": 99,
            "media_type": "image",
            "media_bytes": _jpeg_bytes(),
            "origin": {"importer": "server_folder", "params": {}},
        }
        assert 99 not in pending_archive_thumbnail_ids(medias)

    def test_skips_types_without_a_browsable_thumbnail(self, tmp_path: Path):
        payload = b"some text content"
        archive = _make_image_tar(tmp_path, {"a.txt": payload})
        medias = {1: _archive_media(1, archive, "a.txt", payload, media_type="text")}
        # Text tiles render the words themselves; there is no thumbnail to warm.
        assert pending_archive_thumbnail_ids(medias) == []


class TestWarmPass:
    def test_warms_every_member_and_matches_the_ingest_thumbnail(self, tmp_path: Path):
        _archive, payloads, medias = _image_corpus(tmp_path, n=5)
        ctx = _ctx_with(medias)
        try:
            warmed = warm_archive_thumbnails(ctx)
        finally:
            unregister_context(ctx.dataset_id)

        assert warmed == 5
        for media_id, member in zip(sorted(ctx.medias), payloads):
            media = ctx.medias[media_id]
            assert media["thumbnail_bytes"], f"media {media_id} was not warmed"
            # Byte-identical to what the loader would have produced from the
            # same source bytes: a warmed thumbnail is not a second-class one.
            expected = make_image_thumbnail(payloads[member])
            assert expected is not None
            assert media["thumbnail_bytes"] == expected[0]

    def test_records_dimensions_and_retains_no_payload(self, tmp_path: Path):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=3)
        ctx = _ctx_with(medias)
        try:
            warm_archive_thumbnails(ctx)
        finally:
            unregister_context(ctx.dataset_id)

        for media in ctx.medias.values():
            assert (media["width"], media["height"]) == (600, 400)
            # The whole point of the importer: member bytes never stay resident.
            assert media["media_bytes"] is None
            assert not media.get("media_path")

    def test_is_idempotent(self, tmp_path: Path):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=4)
        ctx = _ctx_with(medias)
        try:
            assert warm_archive_thumbnails(ctx) == 4
            # Second pass finds nothing pending, so it neither re-streams nor
            # re-decodes -- which is what makes kicking it on every load free.
            assert warm_archive_thumbnails(ctx) == 0
        finally:
            unregister_context(ctx.dataset_id)

    def test_unresolvable_members_are_skipped_not_fatal(self, tmp_path: Path):
        archive, _payloads, medias = _image_corpus(tmp_path, n=3)
        # A manifest row pointing at a shard that has since moved away, and one
        # naming a member that isn't in the shard.
        missing_archive = tmp_path / "gone.tar"
        medias[10] = _archive_media(10, missing_archive, "img0.jpg", b"x" * 10)
        medias[11] = _archive_media(11, archive, "not-in-shard.jpg", b"x" * 10)
        ctx = _ctx_with(medias)
        try:
            warmed = warm_archive_thumbnails(ctx)
        finally:
            unregister_context(ctx.dataset_id)

        # The three resolvable members still warmed; the two broken rows keep
        # the request-time fallback they had before.
        assert warmed == 3
        assert not ctx.medias[10].get("thumbnail_bytes")
        assert not ctx.medias[11].get("thumbnail_bytes")

    def test_undecodable_member_is_skipped(self, tmp_path: Path):
        payload = b"this is not an image"
        archive = _make_image_tar(tmp_path, {"broken.jpg": payload})
        ctx = _ctx_with({1: _archive_media(1, archive, "broken.jpg", payload)})
        try:
            assert warm_archive_thumbnails(ctx) == 0
        finally:
            unregister_context(ctx.dataset_id)
        assert not ctx.medias[1].get("thumbnail_bytes")

    def test_empty_dataset_is_a_no_op(self):
        ctx = _ctx_with({}, dataset_id="_warm_empty")
        try:
            assert warm_archive_thumbnails(ctx) == 0
        finally:
            unregister_context(ctx.dataset_id)


class TestCancellation:
    def test_a_cancelled_job_warms_nothing(self, tmp_path: Path):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=8)
        ctx = _ctx_with(medias)
        job = AsyncJob(job_id="cancelled-up-front")
        job.cancel()
        try:
            warmed = warm_archive_thumbnails(ctx, job=job, max_workers=2)
        finally:
            unregister_context(ctx.dataset_id)

        # The pass checks the cancel event before each batch, so a job
        # cancelled up front does no streaming at all.
        assert warmed == 0
        assert all(not m.get("thumbnail_bytes") for m in ctx.medias.values())

    def test_cancel_midway_stops_early_and_keeps_what_it_warmed(self, tmp_path: Path, monkeypatch):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=12)
        ctx = _ctx_with(medias)
        job = AsyncJob(job_id="cancel-midway")

        # Cancel from *inside* the pass once three members have been warmed, so
        # the stop point is deterministic rather than timing-dependent.
        real = ImageMediaType.ensure_thumbnail_bytes
        lock = threading.Lock()
        seen = {"n": 0}

        def cancel_after_three(self, media):
            result = real(self, media)
            with lock:
                seen["n"] += 1
                if seen["n"] == 3:
                    job.cancel()
            return result

        monkeypatch.setattr(ImageMediaType, "ensure_thumbnail_bytes", cancel_after_three)
        try:
            warmed = warm_archive_thumbnails(ctx, job=job, max_workers=1)
        finally:
            unregister_context(ctx.dataset_id)

        # Partial coverage is a valid outcome: what warmed stays warm, the rest
        # degrades to the per-request decode.
        assert warmed == 3
        assert sum(1 for m in ctx.medias.values() if m.get("thumbnail_bytes")) == 3

    def test_unloading_the_dataset_stops_the_pass(self, tmp_path: Path):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=6)
        ctx = _ctx_with(medias)
        # Never registered under its own id any more: the pass must notice that
        # ``get_context`` no longer resolves to *this* context and bail rather
        # than keep streaming for a dataset the user removed.
        unregister_context(ctx.dataset_id)
        assert warm_archive_thumbnails(ctx) == 0


class _FakeCell:
    def __init__(self, rep_id: int) -> None:
        self.rep_id = rep_id


class _FakeTile:
    def __init__(self, rep_ids: list[int]) -> None:
        self.cells = [_FakeCell(r) for r in rep_ids]


class _FakePyramid:
    """Minimal stand-in for a built pyramid: ``{level: [rep_id, ...]}``.

    Avoids fitting a real UMAP just to assert the ordering contract.
    """

    def __init__(self, reps_by_level: dict[int, list[int]]) -> None:
        self.tiles = {(level, 0, 0): _FakeTile(reps) for level, reps in reps_by_level.items()}


class TestWarmOrder:
    def test_representatives_come_first_coarse_levels_before_fine(self, tmp_path: Path):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=6)
        ctx = _ctx_with(medias)
        try:
            ctx._pyramids = {"square": _FakePyramid({0: [5], 1: [3, 6]})}
            order = warm_order(ctx, list(ctx.medias))
        finally:
            unregister_context(ctx.dataset_id)

        # Level 0's representative is what the canvas paints when Browse opens,
        # so it must be warmed before level 1's, and both before the tail.
        assert order[0] == 5
        assert set(order[1:3]) == {3, 6}
        assert sorted(order) == sorted(ctx.medias)

    def test_tail_is_grouped_by_member_so_windows_share_one_decode(self, tmp_path: Path):
        # Two shards, each with two members; ids deliberately interleave the
        # shards so a naive id-order sweep would alternate archives.
        payloads = {"m0.jpg": _jpeg_bytes(), "m1.jpg": _jpeg_bytes(color=(1, 2, 3))}
        shard_a = _make_image_tar(tmp_path, payloads, name="a.tar")
        shard_b = _make_image_tar(tmp_path, payloads, name="b.tar")
        medias = {
            1: _archive_media(1, shard_a, "m0.jpg", payloads["m0.jpg"]),
            2: _archive_media(2, shard_b, "m0.jpg", payloads["m0.jpg"]),
            3: _archive_media(3, shard_a, "m1.jpg", payloads["m1.jpg"]),
            4: _archive_media(4, shard_b, "m1.jpg", payloads["m1.jpg"]),
        }
        ctx = _ctx_with(medias)
        try:
            order = warm_order(ctx, list(medias))
        finally:
            unregister_context(ctx.dataset_id)

        # Grouped by (archive, member): both of shard a's members, then both of
        # shard b's -- so a windowed member's rows land adjacent and reuse the
        # single cached decode instead of evicting each other.
        assert order == [1, 3, 2, 4]

    def test_no_projection_falls_back_to_a_plain_sweep(self, tmp_path: Path):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=4)
        ctx = _ctx_with(medias)
        try:
            assert sorted(warm_order(ctx, list(medias))) == [1, 2, 3, 4]
        finally:
            unregister_context(ctx.dataset_id)


class TestByteBudget:
    def test_bounds_in_flight_bytes(self):
        budget = _ByteBudget(100)
        assert budget.acquire(60, should_stop=lambda: False)
        blocked = threading.Event()
        acquired = threading.Event()

        def second() -> None:
            blocked.set()
            if budget.acquire(60, should_stop=lambda: False):
                acquired.set()

        t = threading.Thread(target=second, daemon=True)
        t.start()
        blocked.wait(timeout=5)
        # 60 + 60 > 100, so the second reader must wait rather than double the
        # peak: this is what keeps 8-way streaming over multi-GB shards safe.
        assert not acquired.wait(timeout=0.5)
        budget.release(60)
        assert acquired.wait(timeout=5)
        t.join(timeout=5)

    def test_oversized_item_is_admitted_alone(self):
        budget = _ByteBudget(100)
        # A member bigger than the whole budget must still get through once
        # nothing else is in flight -- refusing it would skip exactly the
        # members whose per-request decode hurts most.
        assert budget.acquire(10_000, should_stop=lambda: False)

    def test_should_stop_releases_a_waiting_reader(self):
        budget = _ByteBudget(100)
        assert budget.acquire(100, should_stop=lambda: False)
        stop = threading.Event()
        result: list[bool] = []
        started = threading.Event()

        def waiter() -> None:
            started.set()
            result.append(budget.acquire(100, should_stop=stop.is_set))

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        started.wait(timeout=5)
        stop.set()
        t.join(timeout=5)
        # A cancel is honoured even while every worker is parked on a full
        # budget, so the pass unwinds instead of hanging.
        assert result == [False]

    def test_release_never_goes_negative(self):
        budget = _ByteBudget(100)
        budget.release(500)
        assert budget.acquire(100, should_stop=lambda: False)


class TestKick:
    def test_no_kick_when_there_is_nothing_to_warm(self):
        ctx = _ctx_with(
            {1: {"id": 1, "media_type": "image", "media_bytes": _jpeg_bytes(), "thumbnail_bytes": b"warm"}},
            dataset_id="_warm_nokick",
        )
        try:
            assert start_archive_thumbnail_warm(ctx) is None
        finally:
            unregister_context(ctx.dataset_id)

    def test_kick_runs_the_sweep_to_completion(self, tmp_path: Path):
        _archive, _payloads, medias = _image_corpus(tmp_path, n=4)
        ctx = _ctx_with(medias, dataset_id="_warm_kick")
        try:
            job_id = start_archive_thumbnail_warm(ctx)
            assert job_id is not None
            from vtscore.concurrency.async_jobs import archive_thumbnail_jobs

            job = archive_thumbnail_jobs.get(job_id)
            assert job is not None
            assert job.done_event.wait(timeout=30), "warm-up job did not finish"
            assert job.status == "done"
            assert job.result == {"warmed": 4}
            assert all(m["thumbnail_bytes"] for m in ctx.medias.values())
        finally:
            unregister_context(ctx.dataset_id)
