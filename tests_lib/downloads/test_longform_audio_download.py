"""Tests for the long-form audio demos (Apollo 11, BirdVox-full-night, Nixon tapes).

These three share a download contract the older audio demos don't have: the
media type slices a *manifest* first and the downloader fetches only that
selection, so a size variant never pulls the whole multi-GB source.  The tests
below pin both halves of that -- manifest ordering/filtering, and the
selection actually reaching the network layer.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_embedder():
    mock_emb = MagicMock()
    mock_emb.name = "clap"
    mock_emb.media_type_id = "audio"
    mock_emb._model = True
    mock_emb.embed_media = MagicMock(return_value=np.zeros(512))
    return mock_emb


def _media_type():
    from vtscore.media.audio.media_type import AudioMediaType

    mt = AudioMediaType()
    mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 1.0})
    return mt


def _writing_download(payload=b"ID3" + b"\x00" * 40):
    """Stand in for download_file_with_progress: write *payload* to the dest."""

    def _download(url, dest, expected_size=0, on_progress=None):
        Path(dest).write_bytes(payload)

    return _download


# ---------------------------------------------------------------------------
# Apollo 11
# ---------------------------------------------------------------------------

_APOLLO_METADATA = {
    "files": [
        {"name": "11-03302.mp3", "format": "VBR MP3", "size": "200"},
        {"name": "11-03301.mp3", "format": "VBR MP3", "size": "100"},
        # Non-MP3 derivatives of the same tracks must not enter the manifest.
        {"name": "11-03301.flac", "format": "Flac", "size": "900"},
        {"name": "11-03301_spectrogram.png", "format": "Spectrogram", "size": "5"},
        {"name": "155-AAA.mp3", "format": "VBR MP3", "size": "300"},
    ]
}


class TestApollo11Manifest:
    def test_keeps_only_mp3s_and_sorts_by_name(self):
        from vtscore.datasets.downloader import audio as audio_module

        with patch.object(audio_module, "_fetch_text", return_value=json.dumps(_APOLLO_METADATA)):
            manifest = audio_module.apollo11_audio_manifest()

        assert manifest == [("11-03301.mp3", 100), ("11-03302.mp3", 200), ("155-AAA.mp3", 300)]

    def test_sizes_are_ints_even_when_absent(self):
        from vtscore.datasets.downloader import audio as audio_module

        payload = {"files": [{"name": "x.mp3", "format": "VBR MP3"}]}
        with patch.object(audio_module, "_fetch_text", return_value=json.dumps(payload)):
            assert audio_module.apollo11_audio_manifest() == [("x.mp3", 0)]

    def test_the_fetch_shares_the_download_retry_budget(self, monkeypatch):
        """A one-shot GET for the track list would be strictly more fragile
        than the multi-GB transfer it precedes (issue #3216)."""
        from vtscore.datasets.downloader import audio as audio_module
        from vtscore.datasets.downloader import core as dl_core

        seen = {}

        def fake_fetch(url, label="", on_progress=None):
            seen["url"], seen["label"] = url, label
            return json.dumps(_APOLLO_METADATA)

        monkeypatch.setattr(dl_core, "fetch_text_with_retry", fake_fetch)
        audio_module.apollo11_audio_manifest()

        assert seen["url"].startswith(dl_core.ARCHIVE_ORG_METADATA_URL)
        assert seen["label"], "the retry notice needs a name for this fetch"


class TestDownloadApollo11Audio:
    def test_downloads_only_the_given_tracks(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        requested = []

        def _download(url, dest, expected_size=0, on_progress=None):
            requested.append(url)
            Path(dest).write_bytes(b"ID3")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", _download),
        ):
            result = dl_module.download_apollo11_audio([("11-03301.mp3", 100)], on_progress=lambda *a: None)

        assert result == tmp_path / "apollo11_audio"
        assert [p.name for p in result.glob("*.mp3")] == ["11-03301.mp3"]
        assert requested == [
            f"{dl_module.core.ARCHIVE_ORG_DOWNLOAD_URL}/{dl_module.core.APOLLO11_AUDIO_ITEM}/11-03301.mp3"
        ]
        # The .part staging file is renamed away, never left behind.
        assert not list(result.glob("*.part"))

    def test_cached_track_is_not_refetched(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        cached = tmp_path / "apollo11_audio"
        cached.mkdir(parents=True)
        (cached / "11-03301.mp3").write_bytes(b"ID3")
        calls = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: calls.append(True),
            ),
        ):
            dl_module.download_apollo11_audio([("11-03301.mp3", 100)], on_progress=lambda *a: None)

        assert not calls, "an already-downloaded track should be skipped"


class TestLoadDemoSourceApollo11:
    def test_slice_selects_tracks_before_downloading(self, tmp_path):
        """The manifest is sliced first, so only the slice's tracks are fetched."""
        from vtscore.datasets import downloader as dl_module

        audio_dir = tmp_path / "apollo11_audio"
        audio_dir.mkdir()
        manifest = [(f"t{i:02d}.mp3", 10) for i in range(12)]
        for name, _size in manifest:
            (audio_dir / name).write_bytes(b"ID3")

        passed = {}

        def _download(tracks, on_progress=None):
            passed["tracks"] = tracks
            return audio_dir

        clips: dict = {}
        with (
            patch.object(dl_module, "apollo11_audio_manifest", return_value=manifest),
            patch.object(dl_module, "download_apollo11_audio", _download),
        ):
            _media_type().load_demo_source(
                source="apollo11_audio",
                categories=["mission_audio"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=_mock_embedder(),
                slice_frac_start=0.0,
                slice_frac_end=1 / 12,
            )

        assert passed["tracks"] == [("t00.mp3", 10)]
        assert len(clips) == 1
        assert {c["category"] for c in clips.values()} == {"mission_audio"}

    def test_missing_track_file_is_skipped(self, tmp_path):
        """A track the download couldn't produce doesn't abort the load."""
        from vtscore.datasets import downloader as dl_module

        audio_dir = tmp_path / "apollo11_audio"
        audio_dir.mkdir()
        (audio_dir / "present.mp3").write_bytes(b"ID3")
        manifest = [("present.mp3", 10), ("absent.mp3", 10)]

        clips: dict = {}
        with (
            patch.object(dl_module, "apollo11_audio_manifest", return_value=manifest),
            patch.object(dl_module, "download_apollo11_audio", lambda tracks, on_progress=None: audio_dir),
        ):
            _media_type().load_demo_source(
                source="apollo11_audio",
                categories=["mission_audio"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=_mock_embedder(),
            )

        assert [c["filename"] for c in clips.values()] == ["mission_audio/present.mp3"]


class TestPerFileFailureTolerance:
    """One file the host won't serve must not sink a hundred-file download.

    The Internet Archive serves each file of an item from a data node that can
    answer HTTP 500 for minutes while its siblings stay healthy; that took out a
    whole Apollo 11 load over a single track (issue #3227).
    """

    @staticmethod
    def _downloader(failing: dict[str, int], payload=b"ID3"):
        """Return a ``download_file_with_progress`` stand-in that fails for each
        name in *failing* the given number of times, then succeeds.

        Returns ``(fn, attempts)`` where *attempts* records every requested
        filename in order.
        """
        from vtscore.datasets.downloader import core as dl_core

        attempts: list[str] = []
        remaining = dict(failing)

        def _download(url, dest, expected_size=0, on_progress=None):
            name = url.rsplit("/", 1)[-1]
            attempts.append(name)
            if remaining.get(name, 0) > 0:
                remaining[name] -= 1
                raise dl_core.RemoteUnreachableError("archive.org kept returning …", url=url)
            Path(dest).write_bytes(payload)

        return _download, attempts

    def _run(self, tmp_path, tracks, downloader, on_progress=None):
        from vtscore.datasets import downloader as dl_module

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", downloader),
        ):
            return dl_module.download_apollo11_audio(tracks, on_progress=on_progress or (lambda *a: None))

    def test_a_dead_track_is_retried_at_the_end_and_then_skipped(self, tmp_path):
        tracks = [(f"t{i:02d}.mp3", 10) for i in range(8)]
        download, attempts = self._downloader({"t02.mp3": 99})

        result = self._run(tmp_path, tracks, download)

        # The other seven tracks are on disk; the load is not lost.
        assert sorted(p.name for p in result.glob("*.mp3")) == [f"t{i:02d}.mp3" for i in range(8) if i != 2]
        # The failure was retried once, after the rest of the set had run.
        assert attempts.count("t02.mp3") == 2
        assert attempts[-1] == "t02.mp3"

    def test_the_end_of_set_retry_recovers_a_transient_failure(self, tmp_path):
        """A node that wobbles for one file gets the whole rest of the download
        as backoff before we write the track off."""
        tracks = [(f"t{i:02d}.mp3", 10) for i in range(4)]
        download, attempts = self._downloader({"t01.mp3": 1})

        result = self._run(tmp_path, tracks, download)

        assert sorted(p.name for p in result.glob("*.mp3")) == [f"t{i:02d}.mp3" for i in range(4)]
        assert attempts.count("t01.mp3") == 2

    def test_a_hard_http_status_on_one_file_is_tolerated_too(self, tmp_path):
        """A single 404 (a track dropped from the item) is still just one file."""
        import requests

        def _download(url, dest, expected_size=0, on_progress=None):
            if url.endswith("t03.mp3"):
                raise requests.HTTPError("404 Client Error")
            Path(dest).write_bytes(b"ID3")

        tracks = [(f"t{i:02d}.mp3", 10) for i in range(8)]
        result = self._run(tmp_path, tracks, _download)

        assert len(list(result.glob("*.mp3"))) == 7

    def test_too_many_failures_still_fail_the_load(self, tmp_path):
        """Tolerating one bad file must not quietly hand back a gutted dataset."""
        from vtscore.datasets.downloader import core as dl_core

        tracks = [(f"t{i:02d}.mp3", 10) for i in range(8)]
        download, _ = self._downloader({f"t{i:02d}.mp3": 99 for i in range(3)})

        with pytest.raises(dl_core.RemoteUnreachableError) as excinfo:
            self._run(tmp_path, tracks, download)
        assert "3 of 8 files could not be downloaded" in str(excinfo.value)

    def test_a_single_file_set_that_fails_is_a_failed_load(self, tmp_path):
        from vtscore.datasets.downloader import core as dl_core

        download, _ = self._downloader({"only.mp3": 99})
        with pytest.raises(dl_core.RemoteUnreachableError):
            self._run(tmp_path, [("only.mp3", 10)], download)

    def test_cancellation_is_never_swallowed(self, tmp_path):
        """A cancelled load is about the run, not the file: it must stop the
        download rather than be counted as one more skippable track."""
        from vtscore.concurrency.progress import CancelledError

        def _download(url, dest, expected_size=0, on_progress=None):
            raise CancelledError("Operation cancelled by user")

        with pytest.raises(CancelledError):
            self._run(tmp_path, [(f"t{i:02d}.mp3", 10) for i in range(8)], _download)

    def test_a_gated_dataset_is_never_swallowed(self, tmp_path):
        from vtscore.security.hf_auth import GatedResourceError

        def _download(url, dest, expected_size=0, on_progress=None):
            raise GatedResourceError("Sign in with HuggingFace", url=url, status=403)

        with pytest.raises(GatedResourceError):
            self._run(tmp_path, [(f"t{i:02d}.mp3", 10) for i in range(8)], _download)

    def test_the_skip_is_reported_not_silent(self, tmp_path):
        tracks = [(f"t{i:02d}.mp3", 10) for i in range(8)]
        download, _ = self._downloader({"t02.mp3": 99})
        reported: list = []

        self._run(tmp_path, tracks, download, on_progress=lambda *a: reported.append(a))

        messages = [a[1] for a in reported]
        assert any("skipped 1 of 8 files" in m for m in messages)


# ---------------------------------------------------------------------------
# BirdVox-full-night
# ---------------------------------------------------------------------------


class TestSegmentAudioFile:
    def test_splits_into_fixed_length_chunks_keeping_rate_and_depth(self, tmp_path):
        soundfile = pytest.importorskip("soundfile")
        from vtscore.datasets.downloader import audio as audio_module

        sr = 8000
        src = tmp_path / "src.flac"
        samples = (np.sin(np.arange(25 * sr) / 40) * 20000).astype("int16")
        soundfile.write(str(src), samples, sr, format="FLAC")

        written = audio_module._segment_audio_file(src, tmp_path / "out", "unit01", 10.0)

        chunks = sorted((tmp_path / "out").glob("*.flac"))
        assert written == 3
        assert [c.name for c in chunks] == [
            "unit01_0000.flac",
            "unit01_0001.flac",
            "unit01_0002.flac",
        ]
        durations = [soundfile.info(str(c)).duration for c in chunks]
        assert durations == pytest.approx([10.0, 10.0, 5.0])
        assert {soundfile.info(str(c)).samplerate for c in chunks} == {sr}
        assert {soundfile.info(str(c)).subtype for c in chunks} == {"PCM_16"}


class TestDownloadBirdVoxFullNight:
    def test_segments_each_unit_and_removes_the_source_flac(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        def _segment(src, dest_dir, stem, seconds):
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / f"{stem}_0000.flac").write_bytes(b"fLaC")
            return 1

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", _writing_download(b"fLaC")),
            patch.object(dl_module.audio, "_segment_audio_file", _segment),
        ):
            base = dl_module.download_birdvox_full_night(["unit01"], on_progress=lambda *a: None)

        assert base == tmp_path / "birdvox_full_night"
        assert [p.name for p in (base / "unit01").glob("*.flac")] == ["unit01_0000.flac"]
        # The ~1 GB source and the staging dir are both cleaned up.
        assert not (base / "unit01.flac").exists()
        assert not (base / "unit01.partial").exists()

    def test_cached_unit_skips_download_and_segmentation(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        base = tmp_path / "birdvox_full_night"
        (base / "unit01").mkdir(parents=True)
        (base / "unit01" / "unit01_0000.flac").write_bytes(b"fLaC")
        calls = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: calls.append("download"),
            ),
            patch.object(
                dl_module.audio,
                "_segment_audio_file",
                lambda *a, **kw: calls.append("segment"),
            ),
        ):
            dl_module.download_birdvox_full_night(["unit01"], on_progress=lambda *a: None)

        assert not calls


class TestLoadDemoSourceBirdVox:
    def test_slice_selects_units_and_loads_their_chunks(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        base = tmp_path / "birdvox"
        for unit in ("unit01", "unit02"):
            (base / unit).mkdir(parents=True)
            for i in range(3):
                (base / unit / f"{unit}_{i:04d}.flac").write_bytes(b"fLaC")

        passed = {}

        def _download(units, on_progress=None):
            passed["units"] = units
            return base

        clips: dict = {}
        with (
            patch.object(dl_module, "birdvox_full_night_manifest", return_value=["unit01", "unit02"]),
            patch.object(dl_module, "download_birdvox_full_night", _download),
        ):
            _media_type().load_demo_source(
                source="birdvox_full_night",
                categories=["night_recording"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=_mock_embedder(),
                slice_frac_start=0.0,
                slice_frac_end=0.5,
            )

        assert passed["units"] == ["unit01"]
        assert len(clips) == 3, "only the selected unit's chunks load"
        assert {c["category"] for c in clips.values()} == {"night_recording"}


# ---------------------------------------------------------------------------
# Nixon White House Tapes
# ---------------------------------------------------------------------------

_NIXON_PAGE = """
<audio controls>
  <source src="https://catalog.archives.gov/medialz/x/y/37-wht-conversation-001-002-pa.mp3">
</audio>
<audio controls>
  <source src="https://catalog.archives.gov/medialz/x/y/37-wht-conversation-001-001-pa.mp3">
</audio>
<a href="https://catalog.archives.gov/medialz/x/y/37-wht-conversation-001-001-pa.mp3">dupe</a>
<a href="/sites/default/files/37-wht-audiotape-001-log.pdf">tape log</a>
"""


class TestNixonTapeConversationUrls:
    def test_extracts_dedupes_and_sorts_conversation_mp3s(self):
        from vtscore.datasets.downloader import audio as audio_module

        with patch.object(audio_module, "_fetch_text", return_value=_NIXON_PAGE):
            urls = audio_module.nixon_tape_conversation_urls("001")

        assert urls == [
            "https://catalog.archives.gov/medialz/x/y/37-wht-conversation-001-001-pa.mp3",
            "https://catalog.archives.gov/medialz/x/y/37-wht-conversation-001-002-pa.mp3",
        ]


class TestDownloadNixonTapes:
    def test_files_land_in_a_per_tape_directory(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        urls = ["https://catalog.archives.gov/medialz/x/y/37-wht-conversation-001-001-pa.mp3"]

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", _writing_download()),
            patch.object(dl_module.audio, "nixon_tape_conversation_urls", lambda tape: urls),
        ):
            base = dl_module.download_nixon_tapes(["001"], on_progress=lambda *a: None)

        assert base == tmp_path / "nixon_tapes"
        assert [p.name for p in (base / "001").glob("*.mp3")] == ["37-wht-conversation-001-001-pa.mp3"]
        assert not (base / "001.partial").exists()

    def test_tape_with_no_audio_online_is_skipped_not_fatal(self, tmp_path):
        """NARA is still releasing tapes; a quiet number contributes nothing."""
        from vtscore.datasets import downloader as dl_module

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.audio, "nixon_tape_conversation_urls", lambda tape: []),
        ):
            base = dl_module.download_nixon_tapes(["005"], on_progress=lambda *a: None)

        assert base.exists()
        assert not (base / "005").exists()

    def test_cached_tape_skips_the_page_scrape(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        base = tmp_path / "nixon_tapes"
        (base / "001").mkdir(parents=True)
        (base / "001" / "conv.mp3").write_bytes(b"ID3")
        calls = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.audio,
                "nixon_tape_conversation_urls",
                lambda tape: calls.append(tape) or [],
            ),
        ):
            dl_module.download_nixon_tapes(["001"], on_progress=lambda *a: None)

        assert not calls


class TestLoadDemoSourceNixon:
    def test_slice_selects_tapes_and_loads_their_conversations(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        base = tmp_path / "nixon_tapes"
        for tape, n in (("001", 2), ("002", 3)):
            (base / tape).mkdir(parents=True)
            for i in range(n):
                (base / tape / f"conv-{tape}-{i:03d}.mp3").write_bytes(b"ID3")

        passed = {}

        def _download(tapes, on_progress=None):
            passed["tapes"] = tapes
            return base

        clips: dict = {}
        with (
            patch.object(dl_module, "nixon_tape_manifest", return_value=["001", "002"]),
            patch.object(dl_module, "download_nixon_tapes", _download),
        ):
            _media_type().load_demo_source(
                source="nixon_tapes",
                categories=["conversation"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=_mock_embedder(),
                slice_frac_start=0.5,
                slice_frac_end=None,
            )

        assert passed["tapes"] == ["002"]
        assert len(clips) == 3
        assert {c["category"] for c in clips.values()} == {"conversation"}


# ---------------------------------------------------------------------------
# Demo registry wiring
# ---------------------------------------------------------------------------


class TestDemoRegistration:
    @pytest.mark.parametrize("prefix", ["apollo11_audio", "birdvox_full_night", "nixon_tapes"])
    def test_slmla_variants_are_registered_and_partition_the_source(self, prefix):
        """S/M/L cover the source exactly once, and A covers all of it."""
        from vtscore.media.audio.media_type import AudioMediaType

        by_id = {d.id: d for d in AudioMediaType().demo_datasets}
        variants = {suffix: by_id[f"{prefix}_{suffix}"] for suffix in "smla"}

        assert variants["s"].slice_frac_start == 0.0
        assert variants["s"].slice_frac_end == variants["m"].slice_frac_start
        assert variants["m"].slice_frac_end == variants["l"].slice_frac_start
        assert variants["l"].slice_frac_end is None
        assert variants["a"].slice_frac_start == 0.0
        assert variants["a"].slice_frac_end is None
        for demo in variants.values():
            assert demo.source == prefix
            assert demo.download_size_mb > 0
            assert demo.items_per_category > 0
            assert demo.required_folder is not None
