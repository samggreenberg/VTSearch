"""Tests for TUT Sound Events 2017 download and load_demo_source integration."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# download_tut_sound_events_2017
# ---------------------------------------------------------------------------


def _fake_download(url, dest, size, cb):
    """Stand in for download_file_with_progress: write a per-archive zip.

    The destination filename encodes the archive slug (``.dl_<id>_tut_<slug>.zip``),
    so we can give each archive distinct wav members plus a non-wav member that
    must be ignored on extraction.
    """
    slug = Path(dest).stem.split("_tut_")[-1]
    with zipfile.ZipFile(dest, "w") as zf:
        zf.writestr(f"audio/street/{slug}_a.wav", b"RIFF" + b"\x00" * 40)
        zf.writestr(f"audio/street/{slug}_b.wav", b"RIFF" + b"\x00" * 40)
        # Annotations are deliberately not extracted.
        zf.writestr(f"meta/{slug}.ann", b"0.0\t1.0\tcar\n")


class TestDownloadTutSoundEvents2017:
    def test_extracts_wavs_into_per_archive_dirs(self, tmp_path):
        """Each archive's wavs land in their own subdir; non-wavs are skipped."""
        from vtscore.datasets import downloader as dl_module

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", _fake_download),
        ):
            result = dl_module.download_tut_sound_events_2017(on_progress=lambda *a: None)

        assert result == tmp_path / "tut_sound_events_2017"
        assert result.exists()

        # One subdir per archive (development_1, development_2, evaluation).
        slugs = [slug for _url, slug in dl_module.core.TUT_SOUND_EVENTS_2017_ARCHIVES]
        for slug in slugs:
            assert (result / slug).is_dir()
            assert any((result / slug).glob("*.wav"))

        # 2 wavs per archive, flattened; the .ann members are excluded.
        wavs = list(result.rglob("*.wav"))
        assert len(wavs) == 2 * len(slugs)
        assert not list(result.rglob("*.ann"))

    def test_cached_extraction_skips_download(self, tmp_path):
        """If every archive subdir already has a wav, no download is triggered."""
        from vtscore.datasets import downloader as dl_module

        base = tmp_path / "tut_sound_events_2017"
        for _url, slug in dl_module.core.TUT_SOUND_EVENTS_2017_ARCHIVES:
            d = base / slug
            d.mkdir(parents=True)
            (d / f"{slug}.wav").write_bytes(b"RIFF" + b"\x00" * 40)

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_tut_sound_events_2017(on_progress=lambda *a: None)

        assert not download_called, "download should be skipped when cache exists"
        assert result.exists()


# ---------------------------------------------------------------------------
# load_demo_source: tut_sound_events_2017
# ---------------------------------------------------------------------------


class TestLoadDemoSourceTut:
    """AudioMediaType.load_demo_source with source='tut_sound_events_2017'."""

    def _make_mock_embedder(self):
        import numpy as np

        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb.media_type_id = "audio"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(512))
        return mock_emb

    def _make_recordings(self, tmp_path, n):
        audio_dir = tmp_path / "tut"
        (audio_dir / "development_1").mkdir(parents=True)
        for i in range(n):
            (audio_dir / "development_1" / f"a{i:03d}.wav").write_bytes(b"RIFF" + b"\x00" * 40)
        return audio_dir

    def test_single_street_bucket_populates_clips(self, tmp_path):
        """All recordings load under one 'street' category."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.audio.media_type import AudioMediaType

        audio_dir = self._make_recordings(tmp_path, 7)

        mt = AudioMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 1.0})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with patch.object(dl_module, "download_tut_sound_events_2017", return_value=audio_dir):
            mt.load_demo_source(
                source="tut_sound_events_2017",
                categories=["street"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 7
        assert {c["category"] for c in clips.values()} == {"street"}

    def test_fractional_slice_is_applied(self, tmp_path):
        """slice_frac bounds limit how many recordings load (the S/M/L/A axis)."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.audio.media_type import AudioMediaType

        audio_dir = self._make_recordings(tmp_path, 7)

        mt = AudioMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 1.0})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with patch.object(dl_module, "download_tut_sound_events_2017", return_value=audio_dir):
            mt.load_demo_source(
                source="tut_sound_events_2017",
                categories=["street"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
            )

        # int(7 * 1/7) == 1 recording in the small slice.
        assert len(clips) == 1
