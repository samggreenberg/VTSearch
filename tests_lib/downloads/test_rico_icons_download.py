"""Tests for the Rico UI-semantics demo — the boxed-icon screenshot dataset.

``Voxel51/rico`` is a FiftyOne export (a ``samples.json`` manifest plus a
``data/data_<k>/`` media tree) whose detections carry a normalized
``[x, y, w, h]`` box, a 25-way component ``label`` and, for icons, a
``content_or_function`` naming the icon's semantics.  It is the only demo in the
tree that boxes *elements inside* a screenshot, so these tests pin the two things
that make it work: the icon-only label mapping, and the two-phase download that
fetches the manifest first and then only the image shards the slice lands in.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _icon(function, box, label="Icon"):
    return {"label": label, "bounding_box": box, "content_or_function": function}


class TestIterFiftyoneSamples:
    """The streaming manifest reader."""

    def test_yields_samples_without_materializing_the_document(self, tmp_path):
        """Samples come out one at a time, in file order."""
        from vtscore.media.image._demo_sources import _iter_fiftyone_samples

        p = tmp_path / "samples.json"
        p.write_text(json.dumps({"samples": [{"filepath": f"data/data_0/{i}.jpg"} for i in range(5)]}))

        out = list(_iter_fiftyone_samples(p))
        assert [s["filepath"] for s in out] == [f"data/data_0/{i}.jpg" for i in range(5)]

    def test_handles_braces_and_brackets_inside_strings(self, tmp_path):
        """A naive brace-counting splitter would mis-slice these; raw_decode does not.

        Rico's ``content_or_function`` and ``resource_id`` fields are free text
        straight out of an Android view hierarchy, so they really do contain
        brackets, braces, quotes and backslashes.
        """
        from vtscore.media.image._demo_sources import _iter_fiftyone_samples

        p = tmp_path / "samples.json"
        p.write_text(
            json.dumps(
                {
                    "samples": [
                        {"filepath": "a.jpg", "resource_id": 'x{"]},[ \\" y'},
                        {"filepath": "b.jpg", "resource_id": "]}"},
                    ]
                }
            )
        )

        assert [s["filepath"] for s in _iter_fiftyone_samples(p)] == ["a.jpg", "b.jpg"]

    def test_empty_sample_list(self, tmp_path):
        from vtscore.media.image._demo_sources import _iter_fiftyone_samples

        p = tmp_path / "samples.json"
        p.write_text('{"samples": []}')
        assert list(_iter_fiftyone_samples(p)) == []

    def test_falls_back_to_whole_document_parse(self, tmp_path):
        """A bare top-level array (no ``samples`` envelope) still loads."""
        from vtscore.media.image._demo_sources import _iter_fiftyone_samples

        p = tmp_path / "samples.json"
        p.write_text(json.dumps([{"filepath": "a.jpg"}]))
        assert [s["filepath"] for s in _iter_fiftyone_samples(p)] == ["a.jpg"]


class TestRelpathNormalisation:
    def test_repo_relative_path_passes_through(self):
        from vtscore.media.image._demo_sources import _rico_icons_relpath

        assert _rico_icons_relpath("data/data_17/8022.jpg") == "data/data_17/8022.jpg"

    def test_absolute_export_path_is_rebuilt(self):
        """FiftyOne manifests can carry the exporting machine's absolute path."""
        from vtscore.media.image._demo_sources import _rico_icons_relpath

        assert _rico_icons_relpath("/home/someone/fiftyone/rico/data/data_3/99.jpg") == "data/data_3/99.jpg"


class TestDownloadRicoIcons:
    def test_manifest_phase_requests_only_samples_json(self, tmp_path):
        """Phase one must not pull the ~7.7 GB of screenshots."""
        from vtscore.datasets import downloader as dl_module

        captured: dict = {}

        def fake_snapshot(*, repo_id, repo_type, local_dir, allow_patterns, token, tqdm_class=None, max_workers=8):
            captured["repo_id"] = repo_id
            captured["allow_patterns"] = allow_patterns
            captured["max_workers"] = max_workers
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "samples.json").write_text('{"samples": []}')

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch("huggingface_hub.snapshot_download", fake_snapshot),
        ):
            result = dl_module.download_rico_icons_manifest(on_progress=lambda *a: None)

        assert result.name == "rico_icons"
        assert captured["allow_patterns"] == ["samples.json"]
        assert captured["repo_id"] == dl_module.core.RICO_ICONS_REPO_ID
        assert captured["max_workers"] == dl_module.core.RICO_ICONS_DOWNLOAD_WORKERS

    def test_cached_manifest_skips_download(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        d = tmp_path / "rico_icons"
        d.mkdir()
        (d / "samples.json").write_text('{"samples": []}')

        called = []
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch("huggingface_hub.snapshot_download", lambda **kw: called.append(True)),
        ):
            dl_module.download_rico_icons_manifest(on_progress=lambda *a: None)

        assert not called

    def test_shard_phase_requests_only_the_named_folders(self, tmp_path):
        """Only the shards the slice lands in are fetched, as ``<folder>/*``."""
        from vtscore.datasets import downloader as dl_module

        captured: dict = {}

        def fake_snapshot(*, repo_id, repo_type, local_dir, allow_patterns, token, tqdm_class=None, max_workers=8):
            captured["allow_patterns"] = allow_patterns
            for pattern in allow_patterns:
                folder = Path(local_dir) / pattern[: -len("/*")]
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "1.jpg").write_bytes(b"\xff\xd8")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch("huggingface_hub.snapshot_download", fake_snapshot),
        ):
            dl_module.download_rico_icons_shards(["data/data_3", "data/data_1"], on_progress=lambda *a: None)

        assert captured["allow_patterns"] == ["data/data_1/*", "data/data_3/*"]

    def test_shards_already_on_disk_are_not_refetched(self, tmp_path):
        """Loading (M) after (S) costs only the shards it adds."""
        from vtscore.datasets import downloader as dl_module

        captured: dict = {}

        def fake_snapshot(*, repo_id, repo_type, local_dir, allow_patterns, token, tqdm_class=None, max_workers=8):
            captured["allow_patterns"] = allow_patterns

        have = tmp_path / "rico_icons" / "data" / "data_1"
        have.mkdir(parents=True)
        (have / "1.jpg").write_bytes(b"\xff\xd8")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch("huggingface_hub.snapshot_download", fake_snapshot),
        ):
            dl_module.download_rico_icons_shards(["data/data_1", "data/data_2"], on_progress=lambda *a: None)

        assert captured["allow_patterns"] == ["data/data_2/*"]

    def test_empty_shard_folder_does_not_masquerade_as_complete(self, tmp_path):
        """An interrupted fetch leaves an empty folder; it must not block re-download."""
        from vtscore.datasets import downloader as dl_module

        captured: dict = {}

        def fake_snapshot(*, repo_id, repo_type, local_dir, allow_patterns, token, tqdm_class=None, max_workers=8):
            captured["allow_patterns"] = allow_patterns

        (tmp_path / "rico_icons" / "data" / "data_1").mkdir(parents=True)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch("huggingface_hub.snapshot_download", fake_snapshot),
        ):
            dl_module.download_rico_icons_shards(["data/data_1"], on_progress=lambda *a: None)

        assert captured["allow_patterns"] == ["data/data_1/*"]

    def test_reports_measurable_file_progress(self, tmp_path):
        """The tqdm_class forwards a live file count, not a static sentence."""
        from vtscore.datasets import downloader as dl_module

        events: list = []

        def fake_snapshot(*, repo_id, repo_type, local_dir, allow_patterns, token, tqdm_class, max_workers=8):
            folder = Path(local_dir) / "data" / "data_0"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "1.jpg").write_bytes(b"\xff\xd8")
            bar = tqdm_class(total=3)
            for _ in range(3):
                bar.update(1)
            bar.close()

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch("huggingface_hub.snapshot_download", fake_snapshot),
        ):
            dl_module.download_rico_icons_shards(["data/data_0"], on_progress=lambda *a: events.append(a))

        measurable = [e for e in events if e[0] == "downloading" and e[3] > 0]
        assert measurable, f"expected a measurable download event, got {events}"
        assert measurable[-1] == ("downloading", "Downloading Rico UI screenshots", 3, 3)


class TestLoadDemoSourceRicoIcons:
    """ImageMediaType.load_demo_source with source='rico_icons'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "siglip"
        mock_emb.media_type_id = "image"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(768, dtype=np.float32))
        return mock_emb

    def _prepare(self, tmp_path: Path, samples: list) -> Path:
        from PIL import Image

        ds_dir = tmp_path / "rico_icons"
        for s in samples:
            img = ds_dir / s["filepath"]
            img.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 200), (12, 34, 56)).save(img)
        (ds_dir / "samples.json").write_text(json.dumps({"samples": samples}))
        return ds_dir

    def _load(self, ds_dir, categories, clips, **kwargs):
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        shard_calls: list = []
        with (
            patch.object(dl_module, "download_rico_icons_manifest", return_value=ds_dir),
            patch.object(
                dl_module,
                "download_rico_icons_shards",
                side_effect=lambda dirs, on_progress=None: shard_calls.append(list(dirs)) or ds_dir,
            ),
        ):
            ImageMediaType().load_demo_source(
                source="rico_icons",
                categories=categories,
                slice_start=kwargs.pop("slice_start", 0),
                slice_end=kwargs.pop("slice_end", None),
                clips=clips,
                on_progress=lambda *a: None,
                embedder=self._make_mock_embedder(),
                **kwargs,
            )
        return shard_calls

    def test_icon_semantics_become_multilabel_categories_and_boxes(self, tmp_path):
        """Icons map to display categories; each instance contributes a region."""
        samples = [
            {
                "filepath": "data/data_0/1.jpg",
                "detections": {
                    "detections": [
                        _icon("arrow_backward", [0.0, 0.03, 0.14, 0.07]),
                        _icon("search", [0.5, 0.1, 0.1, 0.05]),
                    ]
                },
            },
            # Two instances of one class -> one category, two regions.
            {
                "filepath": "data/data_0/2.jpg",
                "detections": {"detections": [_icon("star", [0.1, 0.1, 0.05, 0.05]), _icon("star", [0.3, 0.1, 0.05, 0.05])]},
            },
        ]
        ds_dir = self._prepare(tmp_path, samples)
        clips: dict = {}
        self._load(ds_dir, ["Back arrow", "Search", "Star"], clips)

        assert len(clips) == 2
        by_primary = {c["category"]: c for c in clips.values()}

        first = by_primary["Back arrow"]
        assert first["categories"] == ["Back arrow", "Search"]
        # [x, y, w, h] -> [x0, y0, x1, y1]
        assert first["regions"][0] == {"box": [0.0, 0.03, 0.14, 0.1], "label": "Back arrow"}

        star = by_primary["Star"]
        assert star["categories"] == ["Star"]
        assert len(star["regions"]) == 2

    def test_non_icon_elements_are_ignored(self, tmp_path):
        """Every element type carries content_or_function; only Icon means an icon.

        On a ``Text`` element the field holds the element's *text*, so a screen
        whose only match is a text run reading "search" must not be labelled.
        """
        samples = [
            {
                "filepath": "data/data_0/1.jpg",
                "detections": {
                    "detections": [
                        _icon("search", [0.1, 0.1, 0.1, 0.1], label="Text"),
                        _icon("search", [0.2, 0.2, 0.1, 0.1], label="Text Button"),
                    ]
                },
            },
            {"filepath": "data/data_0/2.jpg", "detections": {"detections": [_icon("search", [0.1, 0.1, 0.1, 0.1])]}},
        ]
        ds_dir = self._prepare(tmp_path, samples)
        clips: dict = {}
        self._load(ds_dir, ["Search"], clips)

        assert len(clips) == 1
        assert next(iter(clips.values()))["filename"] == "Search/2.jpg"

    def test_screens_without_an_in_vocab_icon_are_dropped(self, tmp_path):
        samples = [
            {"filepath": "data/data_0/1.jpg", "detections": {"detections": [_icon("national_flag", [0.1, 0.1, 0.1, 0.1])]}},
            {"filepath": "data/data_0/2.jpg", "detections": {"detections": [_icon(None, [0.1, 0.1, 0.1, 0.1])]}},
            {"filepath": "data/data_0/3.jpg", "detections": {"detections": []}},
            {"filepath": "data/data_0/4.jpg"},
            {"filepath": "data/data_0/5.jpg", "detections": {"detections": [_icon("search", [0.1, 0.1, 0.1, 0.1])]}},
        ]
        ds_dir = self._prepare(tmp_path, samples)
        clips: dict = {}
        self._load(ds_dir, ["Search"], clips)

        assert len(clips) == 1

    def test_degenerate_box_drops_the_region_but_keeps_the_label(self, tmp_path):
        """A zero-area box is unusable as a region; the icon is still present."""
        samples = [
            {
                "filepath": "data/data_0/1.jpg",
                "detections": {"detections": [_icon("search", [0.5, 0.5, 0.0, 0.0]), _icon("star", "nope")]},
            }
        ]
        ds_dir = self._prepare(tmp_path, samples)
        clips: dict = {}
        self._load(ds_dir, ["Search", "Star"], clips)

        clip = next(iter(clips.values()))
        assert set(clip["categories"]) == {"Search", "Star"}
        assert clip["regions"] == []

    def test_only_shards_the_slice_lands_in_are_downloaded(self, tmp_path):
        """The point of the two-phase download: (S) must not pull all 67 shards."""
        samples = [
            {
                "filepath": f"data/data_{shard}/{i}.jpg",
                "detections": {"detections": [_icon("search", [0.1, 0.1, 0.1, 0.1])]},
            }
            for shard in range(4)
            for i in range(5)
        ]
        ds_dir = self._prepare(tmp_path, samples)
        clips: dict = {}
        # First quarter of the 20 path-sorted records -> data_0 only.
        shard_calls = self._load(
            ds_dir, ["Search"], clips, slice_start=None, slice_end=None, slice_frac_start=0.0, slice_frac_end=0.25
        )

        assert shard_calls == [["data/data_0"]]
        assert len(clips) == 5

    def test_slice_is_applied(self, tmp_path):
        samples = [
            {
                "filepath": f"data/data_0/img_{i:03d}.jpg",
                "detections": {"detections": [_icon("search", [0.1, 0.1, 0.2, 0.2])]},
            }
            for i in range(10)
        ]
        ds_dir = self._prepare(tmp_path, samples)
        clips: dict = {}
        self._load(ds_dir, ["Search"], clips, slice_start=0, slice_end=4)

        assert len(clips) == 4
