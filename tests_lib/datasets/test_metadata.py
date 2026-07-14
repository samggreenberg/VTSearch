"""Tests for :mod:`vtscore.datasets.metadata` extraction loaders.

Each loader turns an on-disk metadata layout (a CSV, a MAT file, a CIFAR-10
pickle batch, or a category-folder tree) into a ``{key: metadata}`` dict.
These are pure filesystem functions, so the tests build tiny fixtures in a
``tmp_path`` and assert the returned shape — no network, no real datasets.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from vtscore.datasets import metadata


# ---------------------------------------------------------------------------
# ESC-50 CSV
# ---------------------------------------------------------------------------


class TestEsc50Metadata:
    def _write_csv(self, esc50_dir: Path) -> None:
        meta = esc50_dir / "meta"
        meta.mkdir(parents=True)
        (meta / "esc50.csv").write_text(
            "filename,fold,target,category,esc10,src_file,take\n"
            "1-100032-A-0.wav,1,0,dog,True,100032,A\n"
            "1-100038-A-14.wav,2,14,chirping_birds,False,100038,A\n",
            encoding="utf-8",
        )

    def test_parses_rows(self, tmp_path):
        self._write_csv(tmp_path)
        md = metadata.load_esc50_metadata(tmp_path)
        assert set(md) == {"1-100032-A-0.wav", "1-100038-A-14.wav"}
        dog = md["1-100032-A-0.wav"]
        assert dog == {"category": "dog", "esc10": True, "target": 0, "fold": 1}

    def test_esc10_flag_is_bool(self, tmp_path):
        self._write_csv(tmp_path)
        md = metadata.load_esc50_metadata(tmp_path)
        assert md["1-100032-A-0.wav"]["esc10"] is True
        assert md["1-100038-A-14.wav"]["esc10"] is False

    def test_numeric_fields_are_ints(self, tmp_path):
        self._write_csv(tmp_path)
        md = metadata.load_esc50_metadata(tmp_path)
        entry = md["1-100038-A-14.wav"]
        assert entry["target"] == 14
        assert entry["fold"] == 2
        assert isinstance(entry["target"], int)


# ---------------------------------------------------------------------------
# UrbanSound8K CSV
# ---------------------------------------------------------------------------


class TestUrbanSound8kMetadata:
    def _write_csv(self, us8k_dir: Path) -> None:
        meta = us8k_dir / "metadata"
        meta.mkdir(parents=True)
        (meta / "UrbanSound8K.csv").write_text(
            "slice_file_name,fsID,start,end,salience,fold,classID,class\n"
            "100032-3-0-0.wav,100032,0,4,1,5,3,dog_bark\n"
            "100263-2-0-3.wav,100263,0,4,1,10,2,children_playing\n",
            encoding="utf-8",
        )

    def test_parses_rows_and_builds_path(self, tmp_path):
        self._write_csv(tmp_path)
        md = metadata.load_urbansound8k_metadata(tmp_path)
        entry = md["100032-3-0-0.wav"]
        assert entry["category"] == "dog_bark"
        assert entry["fold"] == 5
        assert entry["class_id"] == 3
        assert entry["path"] == tmp_path / "audio" / "fold5" / "100032-3-0-0.wav"

    def test_second_row_uses_its_own_fold(self, tmp_path):
        self._write_csv(tmp_path)
        md = metadata.load_urbansound8k_metadata(tmp_path)
        entry = md["100263-2-0-3.wav"]
        assert entry["path"] == tmp_path / "audio" / "fold10" / "100263-2-0-3.wav"


# ---------------------------------------------------------------------------
# Oxford Flowers 102 (MAT file, scipy)
# ---------------------------------------------------------------------------


class TestOxfordFlowersMetadata:
    def test_maps_labels_to_categories(self, tmp_path):
        scipy_io = pytest.importorskip("scipy.io")
        (tmp_path / "jpg").mkdir()
        # 1-indexed labels: category index = label - 1.
        labels = np.array([[1, 3, 2]], dtype=np.int64)
        scipy_io.savemat(str(tmp_path / "imagelabels.mat"), {"labels": labels})
        categories = ["pink primrose", "hard-leaved pocket orchid", "canterbury bells"]
        md = metadata.load_oxford_flowers_metadata(tmp_path, categories)
        assert md["image_00001.jpg"]["category"] == "pink primrose"  # label 1 -> idx 0
        assert md["image_00002.jpg"]["category"] == "canterbury bells"  # label 3 -> idx 2
        assert md["image_00003.jpg"]["category"] == "hard-leaved pocket orchid"  # label 2 -> idx 1
        assert md["image_00001.jpg"]["path"] == tmp_path / "jpg" / "image_00001.jpg"

    def test_out_of_range_label_is_skipped(self, tmp_path):
        scipy_io = pytest.importorskip("scipy.io")
        (tmp_path / "jpg").mkdir()
        labels = np.array([[1, 99]], dtype=np.int64)  # 99 exceeds the 1-long category list
        scipy_io.savemat(str(tmp_path / "imagelabels.mat"), {"labels": labels})
        md = metadata.load_oxford_flowers_metadata(tmp_path, ["only category"])
        assert "image_00001.jpg" in md
        assert "image_00002.jpg" not in md


# ---------------------------------------------------------------------------
# Places365 label file
# ---------------------------------------------------------------------------


class TestPlaces365Metadata:
    def _write(self, places_dir: Path, lines: str) -> None:
        (places_dir / "val_256").mkdir(parents=True)
        (places_dir / "places365_val.txt").write_text(lines, encoding="utf-8")

    def test_maps_index_to_category(self, tmp_path):
        self._write(
            tmp_path,
            "Places365_val_00000001.jpg 0\nPlaces365_val_00000002.jpg 2\n",
        )
        categories = ["airfield", "airplane_cabin", "airport_terminal"]
        md = metadata.load_places365_metadata(tmp_path, categories)
        assert md["Places365_val_00000001.jpg"]["category"] == "airfield"
        assert md["Places365_val_00000002.jpg"]["category"] == "airport_terminal"
        assert md["Places365_val_00000001.jpg"]["path"] == tmp_path / "val_256" / "Places365_val_00000001.jpg"

    def test_blank_lines_and_bad_indices_are_skipped(self, tmp_path):
        self._write(
            tmp_path,
            "\n"  # blank line
            "Places365_val_00000001.jpg 0\n"
            "Places365_val_00000002.jpg notanint\n"  # non-numeric index
            "Places365_val_00000003.jpg 5\n",  # out-of-range index
        )
        md = metadata.load_places365_metadata(tmp_path, ["airfield"])
        assert set(md) == {"Places365_val_00000001.jpg"}


# ---------------------------------------------------------------------------
# CIFAR-10 pickle batch
# ---------------------------------------------------------------------------


class TestCifar10Batch:
    def test_reshapes_and_returns_labels(self, tmp_path):
        rng = np.random.default_rng(0)
        # 4 images, 3072 = 32*32*3 flat, in the original (C, H, W) row order.
        data = rng.integers(0, 256, size=(4, 3072), dtype=np.uint8)
        labels = [0, 5, 9, 3]
        batch_path = tmp_path / "data_batch_1"
        with open(batch_path, "wb") as f:
            pickle.dump({b"data": data, b"labels": labels}, f)

        images, out_labels, label_names = metadata.load_cifar10_batch(batch_path)
        assert images.shape == (4, 32, 32, 3)
        assert images.dtype == np.uint8
        assert out_labels == labels
        assert label_names[0] == "airplane"
        assert label_names[9] == "truck"
        assert len(label_names) == 10

    def test_channel_transpose_is_correct(self, tmp_path):
        # First image all red (R=255) in the (3,32,32) layout -> channel 0 hot.
        img = np.zeros((1, 3, 32, 32), dtype=np.uint8)
        img[0, 0] = 255
        data = img.reshape(1, 3072)
        batch_path = tmp_path / "data_batch_1"
        with open(batch_path, "wb") as f:
            pickle.dump({b"data": data, b"labels": [0]}, f)

        images, _, _ = metadata.load_cifar10_batch(batch_path)
        # After transpose to (H, W, C), the red channel is index 0.
        assert (images[0, :, :, 0] == 255).all()
        assert (images[0, :, :, 1] == 0).all()
        assert (images[0, :, :, 2] == 0).all()


# ---------------------------------------------------------------------------
# Category-folder loaders (audio / video / image / paragraph)
# ---------------------------------------------------------------------------


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


class TestAudioFolderMetadata:
    def test_collects_only_listed_categories(self, tmp_path):
        _touch(tmp_path / "dog" / "a.wav")
        _touch(tmp_path / "cat" / "b.mp3")
        _touch(tmp_path / "ignored" / "c.flac")
        md = metadata.load_audio_metadata_from_folders(tmp_path, ["dog", "cat"])
        assert set(md) == {"dog/a.wav", "cat/b.mp3"}
        assert md["dog/a.wav"]["category"] == "dog"
        assert md["dog/a.wav"]["path"] == tmp_path / "dog" / "a.wav"

    def test_skips_appledouble_sidecars(self, tmp_path):
        _touch(tmp_path / "dog" / "real.wav")
        _touch(tmp_path / "dog" / "._real.wav")  # macOS resource fork
        md = metadata.load_audio_metadata_from_folders(tmp_path, ["dog"])
        assert set(md) == {"dog/real.wav"}

    def test_non_directory_entries_ignored(self, tmp_path):
        _touch(tmp_path / "dog" / "a.wav")
        (tmp_path / "loose.wav").write_bytes(b"x")  # a file at the root, not a category
        md = metadata.load_audio_metadata_from_folders(tmp_path, ["dog"])
        assert set(md) == {"dog/a.wav"}


class TestVideoFolderMetadata:
    def test_collects_video_extensions(self, tmp_path):
        _touch(tmp_path / "run" / "a.mp4")
        _touch(tmp_path / "jump" / "b.mkv")
        _touch(tmp_path / "run" / "notvideo.txt")
        md = metadata.load_video_metadata_from_folders(tmp_path, ["run", "jump"])
        assert set(md) == {"run/a.mp4", "jump/b.mkv"}

    def test_skips_appledouble_sidecars(self, tmp_path):
        _touch(tmp_path / "run" / "clip.mp4")
        _touch(tmp_path / "run" / "._clip.mp4")
        md = metadata.load_video_metadata_from_folders(tmp_path, ["run"])
        assert set(md) == {"run/clip.mp4"}


class TestImageFolderMetadata:
    def test_collects_image_extensions_keyed_by_category(self, tmp_path):
        _touch(tmp_path / "cats" / "image_0001.jpg")
        _touch(tmp_path / "dogs" / "image_0001.jpg")  # same basename, different category
        _touch(tmp_path / "cats" / "readme.md")  # non-image ignored
        md = metadata.load_image_metadata_from_folders(tmp_path, ["cats", "dogs"])
        assert set(md) == {"cats/image_0001.jpg", "dogs/image_0001.jpg"}
        assert md["cats/image_0001.jpg"]["category"] == "cats"


class TestParagraphFolderMetadata:
    def test_collects_txt_and_md(self, tmp_path):
        _touch(tmp_path / "sport" / "a.txt")
        _touch(tmp_path / "tech" / "b.md")
        _touch(tmp_path / "sport" / "skip.pdf")
        md = metadata.load_paragraph_metadata_from_folders(tmp_path, ["sport", "tech"])
        assert set(md) == {"sport/a.txt", "tech/b.md"}
        assert md["tech/b.md"]["path"] == tmp_path / "tech" / "b.md"

    def test_unlisted_category_excluded(self, tmp_path):
        _touch(tmp_path / "sport" / "a.txt")
        _touch(tmp_path / "politics" / "b.txt")
        md = metadata.load_paragraph_metadata_from_folders(tmp_path, ["sport"])
        assert set(md) == {"sport/a.txt"}
