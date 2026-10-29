"""The multi-dataset load pipeline (#4703): one importer run, N datasets.

``_run_multi_output_load_in_background`` owes its caller one dataset, one
loading task, one ``DatasetImported`` event and one AutoFind ``post_load``
per output; a shared acquire that runs once; and isolation between the
outputs — one empty or failing output does not take the others down, and a
cancel on one row during the shared phase drops that row alone.

These drive a real load on the calling thread, the way
``test_load_post_load.py`` does.
"""

from __future__ import annotations

import io
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from vtscore.concurrency.progress import loading_tasks
from vtscore.datasets.import_event import FAILED, SUCCEEDED, DatasetImported
from vtscore.datasets.importers.base import ImporterBase, OutputSpec, PluginField, SourceSpec
from vtscore.embedding.binding import expected_dim_for_embedder
from vtscore.embedding.media_vectors import init_embeddings


def _sync_thread_factory():
    """Run the load body inline so the assertions see a finished load."""

    def fake_thread(target, daemon=True, name=None):
        t = mock.MagicMock()
        t.start = lambda: target()
        return t

    return fake_thread


def _default_embedder(media_type: str) -> str:
    from vtscore.media import embedders_for_type

    return embedders_for_type(media_type)[0].name


def _media(media_id: int, media_type: str, embedder: str) -> dict:
    """A pre-embedded media, so the load's embed stage has nothing to compute.

    The suite's embedder stubs cover ``embed_media``; an image embedder's bulk
    path still declines a media with no bytes, so carrying the vector in is
    what keeps the fixture about the pipeline rather than about decoding.
    """
    return {
        "id": media_id,
        "media_type": media_type,
        "duration": 1.0,
        "file_size": 100,
        "md5": f"{media_type}-{media_id}",
        "embedder": embedder,
        "embeddings": {embedder: np.full(expected_dim_for_embedder(embedder) or 512, media_id, dtype=np.float32)},
        "filename": f"{media_type}_{media_id}.bin",
        "category": "unknown",
        "origin": None,
        "origin_name": f"{media_type}_{media_id}.bin",
        "media_bytes": None,
        "media_string": None,
        "media_path": None,
    }


class _MultiImporter(ImporterBase):
    """Produces ``per_type[media_type]`` items per output through the default hooks."""

    name = "test_multi"
    display_name = "Multi test"
    description = "test double"
    fields = [
        PluginField("url", "URL", "url"),
        PluginField(
            "media_type", "Media type", "select", options=["audio", "image", "text"], default="audio", required=False
        ),
    ]

    def __init__(self, per_type: dict[str, int] | None = None) -> None:
        super().__init__()
        self.per_type = per_type or {}
        self.run_calls: list[str] = []

    def run(self, field_values, medias, thin=False):
        media_type = field_values["media_type"]
        self.run_calls.append(media_type)
        embedder = field_values.get("embedder") or _default_embedder(media_type)
        for i in range(1, self.per_type.get(media_type, 2) + 1):
            medias[i] = _media(i, media_type, embedder)


class _AcquireOnce(_MultiImporter):
    """Overrides the multi hook: one shared acquire, then every output."""

    name = "test_multi_once"

    def __init__(self) -> None:
        super().__init__()
        self.acquires = 0

    def run_outputs(self, field_values, outputs, thin=False):
        self.acquires += 1
        for output in outputs:
            medias: dict = {}
            self.run(output.narrow(field_values), medias, thin=thin)
            yield output, medias


class _OneUnreadableAudio(_MultiImporter):
    """:class:`_MultiImporter`, but audio item 2 arrives with no vector, for the embedder to decline."""

    name = "test_multi_unreadable"

    def run(self, field_values, medias, thin=False):
        super().run(field_values, medias, thin=thin)
        if field_values["media_type"] == "audio":
            medias[2].update(embedder="", embeddings=init_embeddings("", None))


class _Exploding(_MultiImporter):
    name = "test_multi_boom"

    def run_outputs(self, field_values, outputs, thin=False):
        yield from ()  # a generator, so it raises on the first ``next`` like a real importer
        raise RuntimeError("archive corrupt")


def _run(importer, outputs, field_values=None, *, post_load=None, on_finished=None, **kwargs) -> list[str]:
    from vtscore.datasets.load_multi import _run_multi_output_load_in_background

    with mock.patch("vtscore.datasets.load_multi.threading.Thread", side_effect=_sync_thread_factory()):
        return _run_multi_output_load_in_background(
            importer,
            dict(field_values or {"url": "http://example.com/holiday.zip", "dataset_name": "holiday"}),
            outputs,
            post_load=post_load,
            on_finished=on_finished,
            **kwargs,
        )


def _snapshot(task_id: str) -> dict:
    tracker = loading_tasks.get_tracker(task_id)
    assert tracker is not None, "the finished load should still be listed"
    return tracker.get()


def _registered() -> dict[str, dict]:
    from vtscore.datasets.registry import list_datasets

    return {e["name"]: e for e in list_datasets()}


class TestHappyPath:
    def test_one_dataset_per_output(self, isolated_settings):
        importer = _MultiImporter({"audio": 2, "image": 3})
        events: list[DatasetImported] = []
        seen_ctx = []

        task_ids = _run(
            importer,
            [OutputSpec("audio", embedder="clap"), OutputSpec("image", embedder="siglip")],
            post_load=lambda ctx: seen_ctx.append(ctx),
            on_finished=events.append,
        )

        assert len(task_ids) == 2 and len(set(task_ids)) == 2
        assert importer.run_calls == ["audio", "image"], "the default hook runs the single path once per output"

        registered = _registered()
        assert set(registered) == {"holiday – Audio", "holiday – Image"}
        assert registered["holiday – Audio"]["media_type"] == "audio"
        assert registered["holiday – Audio"]["num_items"] == 2
        assert registered["holiday – Image"]["media_type"] == "image"
        assert registered["holiday – Image"]["num_items"] == 3
        assert registered["holiday – Audio"]["embedder"] == "clap"
        assert registered["holiday – Image"]["embedder"] == "siglip"

        for task_id in task_ids:
            snap = _snapshot(task_id)
            assert snap["status"] == "idle" and snap["error"] is None, snap

        assert [e.outcome for e in events] == [SUCCEEDED, SUCCEEDED]
        assert [e.name for e in events] == ["holiday – Audio", "holiday – Image"]
        assert [e.media_type for e in events] == ["audio", "image"]
        assert [e.n_media for e in events] == [2, 3]
        assert {e.dataset_id for e in events} == {e["id"] for e in registered.values()}

        assert len(seen_ctx) == 2 and seen_ctx[0] is not seen_ctx[1], "AutoFind runs once per dataset"
        assert {c.dataset_id for c in seen_ctx} == {e["id"] for e in registered.values()}

    def test_each_output_reports_its_own_drops(self, isolated_settings, monkeypatch):
        """The import-finished event of each dataset counts only what that dataset dropped (#5449)."""
        from tests.helpers import register_declining_embedder

        register_declining_embedder(monkeypatch, "clap", "audio", decline={"audio_2.bin"})
        events: list[DatasetImported] = []
        _run(
            _OneUnreadableAudio({"audio": 3, "image": 3}),
            [OutputSpec("audio", embedder="clap"), OutputSpec("image", embedder="siglip")],
            on_finished=events.append,
        )

        audio, image = events
        assert (audio.n_imported, audio.n_media, audio.n_dropped) == (3, 2, 1)
        assert [g["items"] for g in audio.dropped] == [["audio_2.bin"]]
        assert (image.n_imported, image.n_media, image.n_dropped, image.dropped) == (3, 3, 0, [])

    def test_each_output_records_its_own_origin(self, isolated_settings):
        _run(_MultiImporter(), [OutputSpec("audio"), OutputSpec("image", category="document")])

        registered = _registered()
        audio_origin = registered["holiday – Audio"]["source"]
        document_origin = registered["holiday – Document"]["source"]
        assert audio_origin["importer"] == "test_multi"
        assert audio_origin["params"]["media_type"] == "audio"
        assert document_origin["params"]["media_type"] == "image", "the dataset's type, not its category"
        assert audio_origin["params"]["url"] == "http://example.com/holiday.zip"

    def test_rows_are_published_before_the_run_starts(self, isolated_settings):
        """Each output has a loading task from the request's return, named after its dataset."""
        task_ids = _run(_MultiImporter(), [OutputSpec("audio"), OutputSpec("image")])
        names = {t["task_id"]: t["name"] for t in loading_tasks.list_tasks()}
        assert [names[t] for t in task_ids] == ["holiday – Audio", "holiday – Image"]

    def test_acquire_once_override_runs_once(self, isolated_settings):
        importer = _AcquireOnce()
        _run(importer, [OutputSpec("audio"), OutputSpec("image"), OutputSpec("text")])

        assert importer.acquires == 1
        assert set(_registered()) == {"holiday – Audio", "holiday – Image", "holiday – Text"}

    def test_explicit_output_name_and_importer_default_base(self, isolated_settings):
        _run(
            _MultiImporter(),
            [OutputSpec("audio", dataset_name="Birdsong"), OutputSpec("image")],
            field_values={"url": "http://example.com/holiday.zip"},
        )
        # No typed name: the importer derives the base from its URL field.
        assert set(_registered()) == {"Birdsong", "holiday – Image"}

    def test_requires_at_least_one_output(self, isolated_settings):
        with pytest.raises(ValueError, match="at least one output"):
            _run(_MultiImporter(), [])


class TestImportGroup:
    """Every dataset of one run records the run's group id and its own category (#4747)."""

    def test_every_output_of_a_run_shares_one_group(self, isolated_settings):
        _run(_MultiImporter(), [OutputSpec("audio"), OutputSpec("image", category="document")])

        registered = _registered()
        audio, document = registered["holiday – Audio"], registered["holiday – Document"]
        assert audio["import_group"], "a multi-dataset import mints a group id"
        assert audio["import_group"] == document["import_group"]
        assert audio["output_category"] == "audio"
        assert document["output_category"] == "document"
        assert document["media_type"] == "image", "the category is recorded beside the type, not instead of it"

    def test_each_run_mints_its_own_group(self, isolated_settings):
        _run(_MultiImporter(), [OutputSpec("audio"), OutputSpec("image")])
        _run(
            _MultiImporter(),
            [OutputSpec("audio"), OutputSpec("image")],
            field_values={"url": "http://example.com/trip.zip", "dataset_name": "trip"},
        )

        registered = _registered()
        holiday = {registered[f"holiday – {c}"]["import_group"] for c in ("Audio", "Image")}
        trip = {registered[f"trip – {c}"]["import_group"] for c in ("Audio", "Image")}
        assert len(holiday) == 1 and len(trip) == 1
        assert holiday != trip

    def test_a_single_dataset_import_has_no_group(self, isolated_settings):
        from vtscore.datasets.load_pipeline import _run_importer_in_background

        with mock.patch("vtscore.datasets.load_pipeline.threading.Thread", side_effect=_sync_thread_factory()):
            _run_importer_in_background(
                _MultiImporter(),
                {"url": "http://example.com/solo.zip", "dataset_name": "solo", "media_type": "audio"},
            )

        entry = _registered()["solo"]
        assert entry["import_group"] is None
        assert entry["output_category"] == ""

    def test_deleting_one_sibling_leaves_the_others(self, isolated_settings):
        from vtscore.datasets.registry import unregister_dataset

        _run(_MultiImporter(), [OutputSpec("audio"), OutputSpec("image")])
        before = _registered()

        assert unregister_dataset(before["holiday – Audio"]["id"])

        after = _registered()
        assert set(after) == {"holiday – Image"}
        assert after["holiday – Image"] == before["holiday – Image"], "the group is informational"


class TestIsolation:
    def test_an_empty_output_fails_alone(self, isolated_settings):
        events: list[DatasetImported] = []
        task_ids = _run(
            _MultiImporter({"audio": 2, "image": 0}),
            [OutputSpec("audio"), OutputSpec("image")],
            on_finished=events.append,
        )

        assert set(_registered()) == {"holiday – Audio"}
        assert _snapshot(task_ids[0])["error"] is None
        assert "Image" in _snapshot(task_ids[1])["error"]
        assert [e.outcome for e in events] == [SUCCEEDED, FAILED]
        assert events[1].name == "holiday – Image" and events[1].dataset_id == ""

    def test_a_failing_acquire_fails_every_output(self, isolated_settings):
        events: list[DatasetImported] = []
        task_ids = _run(_Exploding(), [OutputSpec("audio"), OutputSpec("image")], on_finished=events.append)

        assert _registered() == {}
        for task_id in task_ids:
            assert _snapshot(task_id)["error"] == "archive corrupt"
        assert [e.outcome for e in events] == [FAILED, FAILED]
        assert all(e.error == "archive corrupt" for e in events)

    def test_a_failing_finalize_does_not_take_the_others_down(self, isolated_settings):
        """A stage raising for one dataset leaves the next dataset to finish normally."""
        from vtscore.datasets import load_multi

        real = load_multi._finish_dataset_load
        calls = {"n": 0}

        def flaky(ctx, tracker, pacer, spec, ids):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("disk full")
            real(ctx, tracker, pacer, spec, ids)

        with mock.patch.object(load_multi, "_finish_dataset_load", side_effect=flaky):
            task_ids = _run(_MultiImporter(), [OutputSpec("audio"), OutputSpec("image")])

        assert _snapshot(task_ids[0])["error"] == "disk full"
        assert _snapshot(task_ids[1])["error"] is None
        assert set(_registered()) == {"holiday – Image"}


class _CancelsOneRow(_MultiImporter):
    """Cancels the *Image* row from inside the shared acquire, then keeps going."""

    name = "test_multi_cancel"

    def run_outputs(self, field_values, outputs, thin=False):
        from vtscore.concurrency.progress import update_progress

        for task in loading_tasks.list_tasks():
            if task["name"].endswith("Image"):
                loading_tasks.cancel_task(task["task_id"])
        # The progress tick is where the pipeline notices the cancel.
        update_progress("loading", "Scanning…", 1, 2)
        for output in outputs:
            medias: dict = {}
            self.run(output.narrow(field_values), medias, thin=thin)
            yield output, medias


class TestCancellation:
    def test_cancelling_one_row_during_acquire_drops_that_output_only(self, isolated_settings):
        events: list[DatasetImported] = []
        task_ids = _run(_CancelsOneRow(), [OutputSpec("audio"), OutputSpec("image")], on_finished=events.append)

        assert set(_registered()) == {"holiday – Audio"}
        assert _snapshot(task_ids[0])["error"] is None
        assert _snapshot(task_ids[1])["error"] == "Cancelled"
        assert [e.name for e in events] == ["holiday – Audio"], "a cancelled dataset is not reported"
        assert all(loading_tasks.is_finished(t) for t in task_ids)


class TestEmbedderMemory:
    def test_each_output_remembers_its_embedder_pick(self, isolated_settings, monkeypatch):
        """The per-media-type "last embedder" memory sees the hook the app installs after import."""
        from vtscore.datasets import load_pipeline

        remembered: list[tuple[str, str]] = []
        monkeypatch.setattr(
            load_pipeline, "_last_embedder_persistence_hook", lambda mt, emb: remembered.append((mt, emb))
        )

        _run(_MultiImporter(), [OutputSpec("audio", embedder="clap"), OutputSpec("image", embedder="siglip")])

        assert remembered == [("audio", "clap"), ("image", "siglip")]


class TestCleanupAndOriginHooks:
    def test_cleanup_runs_after_the_importer_before_finalizing(self, isolated_settings):
        from vtscore.datasets import load_multi

        order: list[str] = []
        real = load_multi._finish_dataset_load

        def record_finish(ctx, tracker, pacer, spec, ids):
            order.append("finish")
            real(ctx, tracker, pacer, spec, ids)

        with mock.patch.object(load_multi, "_finish_dataset_load", side_effect=record_finish):
            _run(_MultiImporter(), [OutputSpec("audio"), OutputSpec("image")], cleanup=lambda: order.append("cleanup"))

        assert order == ["cleanup", "finish", "finish"]

    def test_origin_for_replaces_the_importer_origin(self, isolated_settings):
        def synthetic(output, narrowed):
            return {
                "importer": "server_folder",
                "params": {"path": "<browser_upload>", "media_type": narrowed["media_type"]},
            }

        _run(_MultiImporter(), [OutputSpec("audio"), OutputSpec("image")], origin_for=synthetic)

        registered = _registered()
        assert registered["holiday – Audio"]["source"] == {
            "importer": "server_folder",
            "params": {"path": "<browser_upload>", "media_type": "audio"},
        }
        assert registered["holiday – Image"]["source"]["params"]["media_type"] == "image"


def _thumb(seed: int, *, size: int = 128, quality: int | None = None) -> bytes:
    """A photo-like thumbnail; *quality* re-encodes it as a lossy JPEG copy."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    img = Image.fromarray(small).resize((size, size), Image.Resampling.BICUBIC).convert("RGB")
    buf = io.BytesIO()
    if quality is None:
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class _PhotosAndFaces(_MultiImporter):
    """An Image output of three photos, the second a smaller copy of the first, and a Face output of one face each."""

    name = "test_multi_faces"

    #: ``(file name, thumbnail, file size)`` per photo.
    PHOTOS = (("a.jpg", _thumb(7), 900), ("a_small.jpg", _thumb(7, quality=70), 300), ("b.jpg", _thumb(8), 900))

    def run(self, field_values, medias, thin=False):
        media_type = field_values["media_type"]
        self.run_calls.append(media_type)
        embedder = field_values.get("embedder") or _default_embedder(media_type)
        for i, (name, thumb, size) in enumerate(self.PHOTOS, start=1):
            media = _media(i, media_type, embedder)
            if media_type == "image":
                media.update(filename=name, origin_name=name, file_size=size, thumbnail_bytes=thumb)
            else:
                face = f"{name}\u2192face_0.jpg"
                params = {"converter": "image2face", "source_file": name, "converter_out_index": "0"}
                media.update(
                    filename=face,
                    origin_name=face,
                    origin={"importer": "converter", "params": params},
                    source_box=[0.2, 0.2, 0.4, 0.5],
                )
            medias[i] = media


class TestDerivedOutputsFollowTheirSource:
    """A Face output keeps one face per photo the Image output kept, not one per copy."""

    _OUTPUTS = (
        # Face first: the pipeline must still finish the Image dataset before it.
        OutputSpec("face", [SourceSpec("image", "image2face")]),
        OutputSpec("image", [SourceSpec("image")]),
    )

    def test_faces_of_merged_photos_merge(self, isolated_settings):
        importer = _PhotosAndFaces()
        events: list[DatasetImported] = []
        _run(
            importer,
            list(self._OUTPUTS),
            {"url": "http://example.com/holiday.zip", "dataset_name": "holiday", "merge_near_duplicates": "true"},
            on_finished=events.append,
        )

        registered = _registered()
        assert registered["holiday – Image"]["num_items"] == 2, "a.jpg and its smaller copy are one photo"
        assert registered["holiday – Face"]["num_items"] == 2, "so their faces are one face"
        assert [e.name for e in events] == ["holiday – Face", "holiday – Image"], "events keep the user's order"

    def test_unmerged_photos_keep_every_face(self, isolated_settings):
        _run(_PhotosAndFaces(), list(self._OUTPUTS))

        registered = _registered()
        assert registered["holiday – Image"]["num_items"] == 3
        assert registered["holiday – Face"]["num_items"] == 3
