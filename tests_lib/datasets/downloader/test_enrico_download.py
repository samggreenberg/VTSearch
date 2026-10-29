"""Tests for the Enrico (Enhanced Rico) mobile-UI screenshot demo.

Enrico is VTSearch's born-digital *screenshot* image demo: ~1,460 Android UI
screenshots (a curated Rico subset) each labeled with one of 20 "design topic"
categories.  It ships as ``screenshots.zip`` (JPEGs keyed on the Rico screen
id, unpacking as either flat ``<screen_id>.jpg`` or the older
``<screen_id>-screenshot.jpg``) plus a separate ``design_topics.csv``
(``screen_id,topic``) carrying the labels — both fetched by ``download_enrico``.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class TestDownloadEnrico:
    def test_fetches_screenshots_and_labels(self, tmp_path):
        """download_enrico extracts the screenshots zip and pulls design_topics.csv."""
        from vtscore.datasets import downloader as dl_module

        def fake_extract(
            *, url, archive_name, extract_to, check_path, download_size_mb, dataset_name, on_progress, is_complete=None
        ):
            shots = Path(extract_to) / "screenshots"
            shots.mkdir(parents=True, exist_ok=True)
            (shots / "50245-screenshot.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        def fake_file(url, dest, expected=0, on_progress=None):
            Path(dest).write_text("screen_id,topic\n50245,login\n")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(dl_module.core, "_download_and_extract", fake_extract),
            patch.object(dl_module.core, "download_file_with_progress", fake_file),
        ):
            result = dl_module.download_enrico(on_progress=lambda *a: None)

        assert result == tmp_path / "enrico"
        assert (result / "design_topics.csv").exists()
        assert next(result.rglob("*-screenshot.jpg"), None) is not None

    def test_cached_skips_both_downloads(self, tmp_path):
        """With screenshots and the CSV already present, nothing is fetched."""
        from vtscore.datasets import downloader as dl_module

        enrico = tmp_path / "enrico"
        (enrico / "screenshots").mkdir(parents=True)
        (enrico / "screenshots" / "1-screenshot.jpg").write_bytes(b"\xff\xd8")
        (enrico / "design_topics.csv").write_text("screen_id,topic\n1,chat\n")

        extract_called, file_called = [], []
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(dl_module.core, "_download_and_extract", lambda **k: extract_called.append(True)),
            patch.object(dl_module.core, "download_file_with_progress", lambda *a, **k: file_called.append(True)),
        ):
            result = dl_module.download_enrico(on_progress=lambda *a: None)

        assert not extract_called
        assert not file_called
        assert result == enrico

    def test_flat_jpg_layout_counts_as_cached(self, tmp_path):
        """The new flat ``<screen_id>.jpg`` layout is recognized as complete."""
        from vtscore.datasets import downloader as dl_module

        enrico = tmp_path / "enrico"
        enrico.mkdir(parents=True)
        (enrico / "50245.jpg").write_bytes(b"\xff\xd8")
        (enrico / "design_topics.csv").write_text("screen_id,topic\n50245,login\n")

        extract_called, file_called = [], []
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(dl_module.core, "_download_and_extract", lambda **k: extract_called.append(True)),
            patch.object(dl_module.core, "download_file_with_progress", lambda *a, **k: file_called.append(True)),
        ):
            dl_module.download_enrico(on_progress=lambda *a: None)

        assert not extract_called
        assert not file_called

    def test_empty_screenshots_dir_does_not_block_download(self, tmp_path):
        """A partially-extracted empty ``screenshots/`` folder still re-downloads.

        The completion probe requires an actual ``*.jpg`` to be present, so a
        bare directory left behind by a failed run can't masquerade as done and
        block the fetch (the RVL-CDIP / Enrico partial-state trap).
        """
        from vtscore.datasets import downloader as dl_module

        enrico = tmp_path / "enrico"
        (enrico / "screenshots").mkdir(parents=True)  # empty: no jpgs

        def fake_extract(*, is_complete=None, **k):
            # is_complete must reflect actual image presence, not dir existence.
            assert is_complete is not None
            assert is_complete() is False
            (enrico / "1.jpg").write_bytes(b"\xff\xd8")

        def fake_file(url, dest, expected=0, on_progress=None):
            Path(dest).write_text("screen_id,topic\n1,login\n")

        extract_called = []
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(
                dl_module.core,
                "_download_and_extract",
                lambda **k: (extract_called.append(True), fake_extract(**k))[0],
            ),
            patch.object(dl_module.core, "download_file_with_progress", fake_file),
        ):
            dl_module.download_enrico(on_progress=lambda *a: None)

        assert extract_called


class TestLoadDemoSourceEnrico:
    """ImageMediaType.load_demo_source with source='enrico'."""

    def _make_mock_embedder(self):
        emb = MagicMock()
        emb.name = "siglip"
        emb.media_type_id = "image"
        emb._model = True
        emb.embed_media = MagicMock(return_value=np.zeros(768, dtype=np.float32))
        return emb

    def _prepare_dir(self, tmp_path: Path, rows: list[tuple[str, str]], *, flat: bool = False) -> Path:
        """rows are (screen_id, topic) pairs; writes a screenshot per id + CSV.

        ``flat`` toggles the JPEG naming: the new flat ``<screen_id>.jpg`` in a
        ``screenshots/`` folder vs. the older ``<screen_id>-screenshot.jpg``.
        """
        from PIL import Image

        enrico = tmp_path / "enrico"
        shots = enrico / "screenshots"
        shots.mkdir(parents=True)
        for sid, _topic in rows:
            name = f"{sid}.jpg" if flat else f"{sid}-screenshot.jpg"
            Image.new("RGB", (40, 80), (10, 20, 30)).save(shots / name)
        csv_lines = ["screen_id,topic"] + [f"{sid},{topic}" for sid, topic in rows]
        (enrico / "design_topics.csv").write_text("\n".join(csv_lines) + "\n")
        return enrico

    def test_buckets_by_display_category_and_folds_case(self, tmp_path):
        """Topics fold case-insensitively to display categories; out-of-vocab is dropped."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        rows = [
            ("1", "login"),
            ("2", "chat"),
            ("3", "mediaplayer"),  # -> "MediaPlayer"
            ("4", "maps"),  # not in the requested category subset -> dropped
        ]
        enrico = self._prepare_dir(tmp_path, rows)

        clips: dict = {}
        with patch.object(dl_module, "download_enrico", return_value=enrico):
            ImageMediaType().load_demo_source(
                source="enrico",
                categories=["Login", "Chat", "MediaPlayer"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=self._make_mock_embedder(),
            )

        assert len(clips) == 3
        assert {c["category"] for c in clips.values()} == {"Login", "Chat", "MediaPlayer"}

    def test_flat_jpg_naming_buckets_by_category(self, tmp_path):
        """Flat ``<screen_id>.jpg`` names recover the id from the stem."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        rows = [("1", "login"), ("2", "chat"), ("3", "mediaplayer")]
        enrico = self._prepare_dir(tmp_path, rows, flat=True)

        clips: dict = {}
        with patch.object(dl_module, "download_enrico", return_value=enrico):
            ImageMediaType().load_demo_source(
                source="enrico",
                categories=["Login", "Chat", "MediaPlayer"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=self._make_mock_embedder(),
            )

        assert len(clips) == 3
        assert {c["category"] for c in clips.values()} == {"Login", "Chat", "MediaPlayer"}

    def test_slice_is_applied(self, tmp_path):
        """Fractional slicing limits how many screenshots load per category."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        rows = [(str(i), "login") for i in range(10)]
        enrico = self._prepare_dir(tmp_path, rows)

        clips: dict = {}
        with patch.object(dl_module, "download_enrico", return_value=enrico):
            ImageMediaType().load_demo_source(
                source="enrico",
                categories=["Login"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=self._make_mock_embedder(),
                slice_frac_start=0.0,
                slice_frac_end=0.3,
            )

        assert len(clips) == 3


class TestEnricoCategoriesList:
    def test_has_20_unique_topics(self):
        from vtscore.media.image._demo_categories import ENRICO_CATEGORIES

        assert len(ENRICO_CATEGORIES) == 20
        assert len(set(ENRICO_CATEGORIES)) == 20
        assert "MediaPlayer" in ENRICO_CATEGORIES
        assert "Login" in ENRICO_CATEGORIES
