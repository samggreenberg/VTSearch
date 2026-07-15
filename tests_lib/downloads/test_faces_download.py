"""Tests for the VGGFace2 Faces demo download + load_demo_source integration.

Covers ``download_vggface2`` (tarball → ``test/n######/*.jpg`` tree) and the
``source="vggface2"`` branch of ``ImageMediaType.load_demo_source``, which maps
the curated identity subset to human-readable ``category`` (person) labels.
"""

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from vtscore.media.image._demo_categories import VGGFACE2_IDENTITIES


# Two identities from the curated subset, used across the fixtures below.
_EWAN = ("n002684", "Ewan McGregor")
_KATIE = ("n004652", "Katie Holmes")


def _tiny_jpeg() -> bytes:
    """A real (decodable) 4x4 RGB JPEG, so PIL-based loaders don't choke."""
    from PIL import Image  # noqa: PLC0415

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (123, 45, 67)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_vggface2_test_tar(tmp_path: Path, per_identity: int = 6) -> Path:
    """Build a minimal ``vggface2_test.tar.gz`` with a ``test/n######/*.jpg`` tree."""
    tar_path = tmp_path / "vggface2_test.tar.gz"
    staging = tmp_path / "tar_staging" / "test"
    for class_id, _name in (_EWAN, _KATIE):
        d = staging / class_id
        d.mkdir(parents=True)
        for i in range(per_identity):
            (d / f"{i:04d}_01.jpg").write_bytes(_tiny_jpeg())
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(staging, arcname="test")
    return tar_path


class TestDownloadVggface2:
    def test_returns_test_directory(self, tmp_path):
        """download_vggface2 extracts the tarball and returns the test/ dir."""
        import shutil  # noqa: PLC0415

        from vtscore.datasets import downloader as dl_module

        tar_path = _make_vggface2_test_tar(tmp_path)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(tar_path), str(dest)),
            ),
        ):
            result = dl_module.download_vggface2(on_progress=lambda *a: None)

        assert result == tmp_path / "vggface2" / "test"
        assert (result / _EWAN[0]).is_dir()
        assert (result / _KATIE[0]).is_dir()
        assert len(list((result / _EWAN[0]).glob("*.jpg"))) == 6

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the test/ dir already exists, no download is triggered."""
        from vtscore.datasets import downloader as dl_module

        test_dir = tmp_path / "vggface2" / "test" / _EWAN[0]
        test_dir.mkdir(parents=True)
        (test_dir / "0000_01.jpg").write_bytes(_tiny_jpeg())

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_vggface2(on_progress=lambda *a: None)

        assert not download_called
        assert result == tmp_path / "vggface2" / "test"


class TestLoadDemoSourceVggface2:
    """ImageMediaType.load_demo_source with source='vggface2'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "siglip"
        mock_emb.media_type_id = "image"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(768))
        return mock_emb

    def test_vggface2_populates_clips_with_person_categories(self, tmp_path):
        """Photos are grouped under the person's display name, not the n-id."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        # Real on-disk fixture the collector will glob.
        test_dir = tmp_path / "test"
        for class_id, _name in (_EWAN, _KATIE):
            d = test_dir / class_id
            d.mkdir(parents=True)
            for i in range(5):
                (d / f"{i:04d}_01.jpg").write_bytes(_tiny_jpeg())

        mt = ImageMediaType()
        clips: dict = {}
        with patch.object(dl_module, "download_vggface2", return_value=test_dir):
            mt.load_demo_source(
                source="vggface2",
                categories=[_EWAN[1], _KATIE[1]],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=self._make_mock_embedder(),
            )

        assert len(clips) == 10
        assert {c["category"] for c in clips.values()} == {_EWAN[1], _KATIE[1]}

    def test_vggface2_items_per_category_slice_is_applied(self, tmp_path):
        """slice_end caps the number of photos taken per person."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        test_dir = tmp_path / "test"
        d = test_dir / _EWAN[0]
        d.mkdir(parents=True)
        for i in range(8):
            (d / f"{i:04d}_01.jpg").write_bytes(_tiny_jpeg())

        mt = ImageMediaType()
        clips: dict = {}
        with patch.object(dl_module, "download_vggface2", return_value=test_dir):
            mt.load_demo_source(
                source="vggface2",
                categories=[_EWAN[1]],
                slice_start=0,
                slice_end=3,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=self._make_mock_embedder(),
            )

        assert len(clips) == 3
        assert all(c["category"] == _EWAN[1] for c in clips.values())


def test_curated_identities_are_unique_and_ascii():
    """The curated subset has unique ids/names and UI-safe ASCII labels."""
    ids = [cid for cid, _ in VGGFACE2_IDENTITIES]
    names = [name for _, name in VGGFACE2_IDENTITIES]
    assert len(ids) == len(set(ids))
    assert len(names) == len(set(names))
    assert all(name.isascii() and name.strip() == name for name in names)
