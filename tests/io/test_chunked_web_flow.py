"""Tests for chunked dataset loading via the web (modal) flow.

Three pieces:

1. ``consume_chunks_into``; the helper that drains an importer's
   ``run_chunked()`` iterator into a target medias dict, renumbering IDs
   so chunks (which each restart at 1) don't collide.
2. ``auto_chunk_size``; picks a chunk size from the media type so the
   user is never asked for one.
3. The two web import routes;
   ``POST /api/dataset/import/<importer_name>`` and
   ``POST /api/dataset/import-local-folder``; must dispatch to
   ``run_chunked`` automatically when the importer supports it, with the
   chunk size derived from the field's ``media_type``.
"""

import io
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from vtscore.datasets.importers.base import DatasetImporter, ImporterField
from vtscore.datasets.load_pipeline import auto_chunk_size, consume_chunks_into


# ===========================================================================
# consume_chunks_into
# ===========================================================================


class TestConsumeChunksInto:
    def test_renumbers_into_empty_target(self):
        target: dict[int, dict[str, Any]] = {}
        chunks = [{1: {"id": 1, "x": "a"}, 2: {"id": 2, "x": "b"}}, {1: {"id": 1, "x": "c"}}]
        consume_chunks_into(target, iter(chunks))
        assert sorted(target.keys()) == [1, 2, 3]
        assert target[1]["x"] == "a"
        assert target[2]["x"] == "b"
        assert target[3]["x"] == "c"
        # IDs on the media dicts themselves are rewritten to match.
        assert target[1]["id"] == 1
        assert target[2]["id"] == 2
        assert target[3]["id"] == 3

    def test_continues_from_existing_ids(self):
        target: dict[int, dict[str, Any]] = {5: {"id": 5, "x": "pre"}}
        chunks = [{1: {"id": 1, "x": "a"}}]
        consume_chunks_into(target, iter(chunks))
        assert sorted(target.keys()) == [5, 6]
        assert target[5]["x"] == "pre"
        assert target[6]["x"] == "a"
        assert target[6]["id"] == 6

    def test_empty_iterator_no_ops(self):
        target: dict[int, dict[str, Any]] = {}
        consume_chunks_into(target, iter([]))
        assert target == {}


# ===========================================================================
# auto_chunk_size
# ===========================================================================


class TestAutoChunkSize:
    def test_returns_positive_int_for_each_known_media_type(self):
        for mt in ("audio", "image", "text", "video", "document"):
            assert auto_chunk_size(mt) > 0

    def test_text_chunks_larger_than_video(self):
        # Text embeddings are tiny; videos are heavy.  The auto sizer
        # should give text far more headroom per chunk.
        assert auto_chunk_size("text") > auto_chunk_size("video")

    def test_unknown_media_type_falls_back_to_default(self):
        assert auto_chunk_size("") > 0
        assert auto_chunk_size("not-a-real-type") > 0


# ===========================================================================
# _DummyChunkedImporter: supports_chunked=True, no real I/O
# ===========================================================================


class _DummyChunkedImporter(DatasetImporter):
    name = "_dummy_chunked"
    display_name = "Dummy Chunked"
    description = "Test importer"
    icon = ""
    fields: list[ImporterField] = []

    def __init__(self) -> None:
        super().__init__()
        self.run_called = False
        self.run_chunked_called = False
        self.last_chunk_size: int | None = None

    @property
    def supports_chunked(self) -> bool:
        return True

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        self.run_called = True
        medias[1] = {"id": 1, "media_type": "audio", "embedding": np.zeros(4)}
        medias[2] = {"id": 2, "media_type": "audio", "embedding": np.ones(4)}

    def run_chunked(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        self.run_chunked_called = True
        self.last_chunk_size = chunk_size
        yield {1: {"id": 1, "media_type": "audio", "embedding": np.zeros(4)}}
        yield {1: {"id": 1, "media_type": "audio", "embedding": np.ones(4)}}


class _DummyNonChunkedImporter(DatasetImporter):
    name = "_dummy_nonchunked"
    display_name = "Dummy Non-Chunked"
    description = "Test importer (no chunked support)"
    icon = ""
    fields: list[ImporterField] = []

    def __init__(self) -> None:
        super().__init__()
        self.run_called = False
        self.run_chunked_called = False

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        self.run_called = True
        medias[1] = {"id": 1, "media_type": "audio", "embedding": np.zeros(4)}

    def run_chunked(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        # The base class default delegates to run().  Track whether we
        # were called so the test can assert the pipeline did *not* call
        # us when the importer doesn't advertise chunked support.
        self.run_chunked_called = True
        yield from super().run_chunked(field_values, chunk_size, thin=thin)


# ===========================================================================
# _run_importer_in_background: chunked-vs-whole dispatch
# ===========================================================================


class TestRunImporterChunkedDispatch:
    """Verify the load_fn produced by ``_run_importer_in_background``
    routes to ``run_chunked`` whenever the importer supports it, with a
    chunk size derived from the field's ``media_type``.

    We patch ``_run_origin_load_in_background`` so the load_fn is invoked
    synchronously against a throwaway dict instead of being scheduled on
    a background thread.
    """

    def _invoke(self, importer, field_values: dict | None = None) -> dict[int, dict[str, Any]]:
        from vtscore.datasets import load_pipeline

        captured: dict[str, Any] = {}

        def _fake_origin_load(load_fn, origin, **kwargs):
            target: dict[int, dict[str, Any]] = {}
            load_fn(target)
            captured["target"] = target
            return "task-fake"

        with patch.object(
            load_pipeline,
            "_run_origin_load_in_background",
            side_effect=_fake_origin_load,
        ):
            load_pipeline._run_importer_in_background(importer, dict(field_values or {}))

        return captured["target"]

    def test_uses_run_chunked_when_supported(self):
        imp = _DummyChunkedImporter()
        target = self._invoke(imp, {"media_type": "audio"})
        assert imp.run_chunked_called is True
        assert imp.run_called is False
        # Chunk size was auto-picked for audio.
        assert imp.last_chunk_size == auto_chunk_size("audio")
        # Two single-media chunks (each with id=1) renumbered to 1,2.
        assert sorted(target.keys()) == [1, 2]

    def test_chunk_size_varies_with_media_type(self):
        imp_audio = _DummyChunkedImporter()
        self._invoke(imp_audio, {"media_type": "audio"})

        imp_text = _DummyChunkedImporter()
        self._invoke(imp_text, {"media_type": "text"})

        # Text gets a much larger chunk than audio.
        assert imp_text.last_chunk_size is not None
        assert imp_audio.last_chunk_size is not None
        assert imp_text.last_chunk_size > imp_audio.last_chunk_size

    def test_falls_back_to_run_when_importer_unsupported(self):
        imp = _DummyNonChunkedImporter()
        target = self._invoke(imp, {"media_type": "audio"})
        assert imp.run_chunked_called is False
        assert imp.run_called is True
        assert sorted(target.keys()) == [1]


# ===========================================================================
# Route plumbing: POST /api/dataset/import/<importer_name>
# ===========================================================================


class TestImportRouteAutoChunked:
    """The public API no longer accepts a user-supplied ``chunk_size``;
    the load pipeline picks one from the media type.  These tests verify
    the route still wires through to the pipeline correctly.
    """

    def test_route_invokes_pipeline(self, client):
        with patch("vtsearch.routes.datasets.staging._run_importer_in_background") as mock_run:
            resp = client.post(
                "/api/dataset/import/server_folder",
                json={"path": "/tmp/test", "media_type": "image"},
            )
            assert resp.status_code == 200
            mock_run.assert_called_once()
            # The pipeline takes (importer, field_values); no chunk_size kwarg.
            assert "chunk_size" not in mock_run.call_args.kwargs

    def test_user_supplied_chunk_size_is_ignored(self, client):
        with patch("vtsearch.routes.datasets.staging._run_importer_in_background") as mock_run:
            resp = client.post(
                "/api/dataset/import/server_folder",
                json={"path": "/tmp/test", "media_type": "image", "chunk_size": 99},
            )
            assert resp.status_code == 200
            assert "chunk_size" not in mock_run.call_args.kwargs
            field_values = mock_run.call_args.args[1]
            # The bogus client-supplied chunk_size doesn't bleed into the
            # field_values dict either.
            assert "chunk_size" not in field_values


class TestImportRouteClipperParams:
    """``clipper_params`` from the modal must reach the load pipeline so
    user-tuned values (e.g. tiling duration) override the clipper's
    registry default.  Regression: the route used to drop them silently,
    leaving e.g. a 1s tiling clip selection running with the registered
    default of 2s; a no-op for 2s synthetic videos.
    """

    def test_clipper_params_passed_through_json(self, client):
        with patch("vtsearch.routes.datasets.staging._run_importer_in_background") as mock_run:
            resp = client.post(
                "/api/dataset/import/server_folder",
                json={
                    "path": "/tmp/test",
                    "media_type": "video",
                    "clipper": "video_tiling",
                    "clipper_params": {"duration": 1.0, "min_overlap": 0.0},
                },
            )
            assert resp.status_code == 200
            field_values = mock_run.call_args.args[1]
            assert field_values["clipper"] == "video_tiling"
            assert field_values["clipper_params"] == {"duration": 1.0, "min_overlap": 0.0}

    def test_no_clipper_params_when_omitted(self, client):
        with patch("vtsearch.routes.datasets.staging._run_importer_in_background") as mock_run:
            resp = client.post(
                "/api/dataset/import/server_folder",
                json={"path": "/tmp/test", "media_type": "image"},
            )
            assert resp.status_code == 200
            field_values = mock_run.call_args.args[1]
            assert "clipper_params" not in field_values

    def test_invalid_clipper_params_returns_400(self, client):
        with patch("vtsearch.routes.datasets.staging._run_importer_in_background") as mock_run:
            resp = client.post(
                "/api/dataset/import/server_folder",
                json={
                    "path": "/tmp/test",
                    "media_type": "image",
                    "clipper_params": "not-a-dict",
                },
            )
            assert resp.status_code == 400
            mock_run.assert_not_called()


# ===========================================================================
# Route plumbing: POST /api/dataset/import-local-folder
# ===========================================================================


class TestLocalFolderRouteChunked:
    """The browser-upload route must auto-pick a chunk size from the
    declared media type and route through ``run_chunked`` whenever the
    underlying importer supports it.
    """

    def test_auto_chunked_dispatch_for_audio(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR", tmp_path / "uploads")

        captured: dict[str, Any] = {}

        # Patch the server_folder importer registered globally so we
        # observe whether run_chunked was invoked, without doing any
        # real audio embedding.
        from vtscore.datasets.importers import get_importer

        real_importer = get_importer("server_folder")
        assert real_importer is not None

        chunked_calls: list[int] = []

        def _stub_run_chunked(field_values, chunk_size, thin=False):
            chunked_calls.append(chunk_size)
            yield {1: {"id": 1, "media_type": "audio", "embedding": np.zeros(4), "filename": "a.wav"}}

        def _stub_run(field_values, medias, thin=False):
            captured["run_called"] = True

        monkeypatch.setattr(real_importer, "run_chunked", _stub_run_chunked)
        monkeypatch.setattr(real_importer, "run", _stub_run)

        def _fake_origin_load(load_fn, origin, **kwargs):
            target: dict[int, dict[str, Any]] = {}
            load_fn(target)
            captured["target"] = target
            return "task-fake-chunked"

        with patch(
            "vtsearch.routes.datasets.load._run_origin_load_in_background",
            side_effect=_fake_origin_load,
        ):
            resp = client.post(
                "/api/dataset/import-local-folder",
                data={
                    "media_type": "audio",
                    "files": [(io.BytesIO(b"AAA"), "myfolder/one.wav")],
                },
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        # Chunk size auto-picked from media_type=audio, not from any
        # client-supplied value.
        assert chunked_calls == [auto_chunk_size("audio")]
        assert "run_called" not in captured
        assert sorted(captured["target"].keys()) == [1]
