"""Tests for the Celebrities in Context demo (Celebrity Together photos).

Covers ``download_celebrity_together`` (pinned loose files from the HF mirror,
only those not already on disk), the ``source="celebrity_together"`` branch of
``ImageMediaType.load_demo_source`` (slice the manifest, then download only the
slice), and the shipped manifest's shape.
"""

import io
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vtscore.media.image._demo_categories import CELEBRITY_TOGETHER_CATEGORIES


def _photo(i: int, folder: int = 1) -> str:
    """A mirror path of the real ``n<id>/<md5>.jpg`` shape."""
    return f"n{folder}/{i:032x}.jpg"


def _tiny_jpeg() -> bytes:
    """A real (decodable) 4x4 RGB JPEG, so PIL-based loaders don't choke."""
    from PIL import Image  # noqa: PLC0415

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (12, 34, 56)).save(buf, format="JPEG")
    return buf.getvalue()


def _fake_atomic(fetched: list[str], fail: frozenset[str] = frozenset()):
    """A ``download_file_atomic`` stand-in: records the URL, writes a JPEG, or raises for *fail*."""

    def fetch(url, final_path, expected_size, on_progress):
        fetched.append(url)
        if any(url.endswith(p) for p in fail):
            raise OSError("boom")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(_tiny_jpeg())

    return fetch


class TestDownloadCelebrityTogether:
    def test_fetches_each_missing_photo_from_the_mirror(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        fetched: list[str] = []
        photos = [_photo(1), _photo(2, folder=7)]
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_atomic", _fake_atomic(fetched)),
        ):
            result = dl_module.download_celebrity_together(photos, on_progress=lambda *a: None)

        assert result == tmp_path / "celebrity_together" / "images"
        assert sorted(fetched) == sorted(
            f"https://huggingface.co/datasets/Juoguessr/celebritytogether/resolve/main/images/{p}" for p in photos
        )
        assert all((result / p).is_file() for p in photos)

    def test_photos_already_on_disk_are_not_fetched_again(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        have = tmp_path / "celebrity_together" / "images" / _photo(1)
        have.parent.mkdir(parents=True)
        have.write_bytes(_tiny_jpeg())

        fetched: list[str] = []
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_atomic", _fake_atomic(fetched)),
        ):
            dl_module.download_celebrity_together([_photo(1), _photo(2)], on_progress=lambda *a: None)

        assert len(fetched) == 1
        assert fetched[0].endswith(_photo(2))

    def test_reports_progress_in_photos_not_bytes(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        calls: list[tuple] = []
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_atomic", _fake_atomic([])),
        ):
            dl_module.download_celebrity_together([_photo(i) for i in range(3)], on_progress=lambda *a: calls.append(a))

        assert calls[-1][2:] == (3, 3)
        assert {c[3] for c in calls} == {3}

    def test_a_path_outside_the_mirror_layout_is_refused(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        fetched: list[str] = []
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_atomic", _fake_atomic(fetched)),
            pytest.raises(ValueError, match="Not a Celebrity Together photo path"),
        ):
            dl_module.download_celebrity_together([_photo(1), "../../escape.jpg"], on_progress=lambda *a: None)
        assert not fetched

    def test_failed_photos_raise_after_the_rest_arrive(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        fetched: list[str] = []
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_atomic", _fake_atomic(fetched, fail=frozenset({_photo(2)}))),
            pytest.raises(RuntimeError, match="1 of 3 celebrity photos failed"),
        ):
            dl_module.download_celebrity_together([_photo(i) for i in range(3)], on_progress=lambda *a: None)

        images = tmp_path / "celebrity_together" / "images"
        assert (images / _photo(0)).is_file()
        assert (images / _photo(1)).is_file()
        assert not (images / _photo(2)).exists()


class TestLoadDemoSourceCelebrityTogether:
    """ImageMediaType.load_demo_source with source='celebrity_together'."""

    _PEOPLE = ("Taylor Swift", "Idris Elba")

    def _manifest(self) -> dict[str, list[str]]:
        return {person: [_photo(10 * k + i) for i in range(3)] for k, person in enumerate(self._PEOPLE)}

    def _load(self, tmp_path, slice_end: int) -> tuple[dict, list[str]]:
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image import _demo_sources
        from vtscore.media.image.media_type import ImageMediaType

        requested: list[str] = []

        def fake_download(photos, on_progress=None):
            for photo in photos:
                requested.append(photo)
                (tmp_path / photo).parent.mkdir(parents=True, exist_ok=True)
                (tmp_path / photo).write_bytes(_tiny_jpeg())
            return tmp_path

        embedder = MagicMock()
        embedder.name = "siglip"
        embedder.media_type_id = "image"
        embedder._model = True
        embedder.embed_media = MagicMock(return_value=np.zeros(768))

        clips: dict = {}
        with (
            patch.object(_demo_sources, "celebrity_together_photos", self._manifest),
            patch.object(dl_module, "download_celebrity_together", fake_download),
        ):
            ImageMediaType().load_demo_source(
                source="celebrity_together",
                categories=list(self._PEOPLE),
                slice_start=0,
                slice_end=slice_end,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=embedder,
            )
        return clips, requested

    def test_each_photo_is_labelled_with_the_person_it_was_picked_for(self, tmp_path):
        clips, _ = self._load(tmp_path, slice_end=3)

        assert len(clips) == 6
        by_person: dict[str, int] = {}
        for clip in clips.values():
            by_person[clip["category"]] = by_person.get(clip["category"], 0) + 1
        assert by_person == dict.fromkeys(self._PEOPLE, 3)

    def test_only_the_slice_is_downloaded(self, tmp_path):
        """A smaller variant never pays for a larger one's photos."""
        clips, requested = self._load(tmp_path, slice_end=2)

        manifest = self._manifest()
        assert sorted(requested) == sorted(p for person in self._PEOPLE for p in manifest[person][:2])
        assert len(clips) == 4


class TestShippedManifest:
    """The pinned photo list the demo slices (``vtscore/media/assets/celebrity_together.json``)."""

    def test_covers_exactly_the_demo_people(self):
        from vtscore.media.image._demo_sources import celebrity_together_photos

        assert sorted(celebrity_together_photos()) == sorted(CELEBRITY_TOGETHER_CATEGORIES)

    def test_every_variant_fits_in_every_persons_list(self):
        from vtscore.media.image._demo_sources import build_demo_datasets, celebrity_together_photos

        variants = [d for d in build_demo_datasets() if d.source == "celebrity_together"]
        assert {d.id for d in variants} == {"celebrity_together_s", "celebrity_together_m"}
        shortest = min(len(v) for v in celebrity_together_photos().values())
        assert all(d.slice_end is not None and d.slice_end <= shortest for d in variants)

    def test_photos_are_mirror_paths_and_never_repeat(self):
        from vtscore.media.image._demo_sources import celebrity_together_photos

        photos = [p for listed in celebrity_together_photos().values() for p in listed]
        assert all(re.fullmatch(r"n\d+/[0-9a-f]{32}\.jpg", p) for p in photos)
        assert len(photos) == len(set(photos))

    def test_names_the_repo_the_downloader_reads(self):
        from vtscore.datasets.downloader.core import CELEBRITY_TOGETHER_REPO_ID
        from vtscore.media.image import _demo_sources

        manifest = json.loads(Path(_demo_sources._CELEBRITY_TOGETHER_MANIFEST).read_text())
        assert manifest["repo_id"] == CELEBRITY_TOGETHER_REPO_ID


def test_people_are_unique_and_ascii():
    """The curated people have unique, UI-safe ASCII labels."""
    assert len(CELEBRITY_TOGETHER_CATEGORIES) == len(set(CELEBRITY_TOGETHER_CATEGORIES))
    assert all(name.isascii() and name.strip() == name for name in CELEBRITY_TOGETHER_CATEGORIES)
