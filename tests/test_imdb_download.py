"""Tests for IMDB dataset download and load_demo_source integration."""

import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_imdb_tar(tmp_path: Path) -> Path:
    """Create a minimal IMDB tar.gz fixture with 3 reviews per sentiment per split."""
    tar_path = tmp_path / "aclImdb_v1.tar.gz"

    # Build a temporary directory tree that mirrors the real dataset layout.
    tree_root = tmp_path / "tar_staging" / "aclImdb"
    for split in ("train", "test"):
        for sentiment in ("pos", "neg"):
            d = tree_root / split / sentiment
            d.mkdir(parents=True)
            for i in range(3):
                (d / f"{i}_{7 + i}.txt").write_text(
                    f"IMDB {sentiment} review {i} from {split}. This is sample text."
                )

    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(tree_root, arcname="aclImdb")

    return tar_path


# ---------------------------------------------------------------------------
# download_imdb
# ---------------------------------------------------------------------------

class TestDownloadImdb:
    def test_returns_reviews_by_category(self, tmp_path):
        """download_imdb returns a dict of sentiment -> list[str] from a tar.gz."""
        from vtsearch.datasets import downloader as dl_module

        tar_path = _make_imdb_tar(tmp_path)

        progress_calls = []

        def fake_progress(status, msg, cur, tot):
            progress_calls.append((status, msg))

        with (
            patch.object(dl_module, "DATA_DIR", tmp_path),
            patch.object(dl_module, "download_file_with_progress", lambda url, dest, size, cb: tar_path.rename(dest)),
        ):
            result = dl_module.download_imdb(on_progress=fake_progress)

        assert set(result.keys()) == {"pos", "neg"}
        # 3 reviews per split × 2 splits = 6 per sentiment
        for reviews in result.values():
            assert len(reviews) == 6
            assert all(isinstance(r, str) and r for r in reviews)

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the extract directory already exists, no download is triggered."""
        from vtsearch.datasets import downloader as dl_module

        # Pre-create the extract directory with one sentiment category.
        extract_dir = tmp_path / "aclImdb"
        pos_dir = extract_dir / "train" / "pos"
        pos_dir.mkdir(parents=True)
        (pos_dir / "0_7.txt").write_text("A positive review.")

        download_called = []

        with (
            patch.object(dl_module, "DATA_DIR", tmp_path),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_imdb(on_progress=lambda *a: None)

        assert not download_called, "download should be skipped when cache exists"
        assert "pos" in result


# ---------------------------------------------------------------------------
# load_demo_source — imdb branch
# ---------------------------------------------------------------------------

class TestLoadDemoSourceImdb:
    """TextMediaType.load_demo_source with source='imdb'."""

    def _make_text_media_type(self):
        from vtsearch.media.text.media_type import TextMediaType

        mt = TextMediaType()
        # Provide a stub embedding model so we never hit the real network.
        stub_model = MagicMock()
        stub_model.encode.return_value = [0.1] * 768
        mt._model = stub_model
        return mt

    def test_imdb_source_populates_clips(self):
        """load_demo_source with source='imdb' fills the clips dict."""
        from vtsearch.datasets import downloader as dl_module

        fake_reviews = {
            "pos": ["Great movie!", "Loved it!"],
            "neg": ["Terrible film.", "Hated it."],
        }

        mt = self._make_text_media_type()
        clips: dict = {}

        with patch.object(dl_module, "download_imdb", return_value=fake_reviews):
            mt.load_demo_source(
                source="imdb",
                categories=["pos", "neg"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
            )

        assert len(clips) == 4
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"pos", "neg"}

    def test_imdb_slice_is_applied(self):
        """slice_start/slice_end limits reviews per category."""
        from vtsearch.datasets import downloader as dl_module

        fake_reviews = {
            "pos": [f"Positive review {i}." for i in range(10)],
        }

        mt = self._make_text_media_type()
        clips: dict = {}

        with patch.object(dl_module, "download_imdb", return_value=fake_reviews):
            mt.load_demo_source(
                source="imdb",
                categories=["pos"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
            )

        # Only reviews[2:5] = 3 reviews.
        assert len(clips) == 3
