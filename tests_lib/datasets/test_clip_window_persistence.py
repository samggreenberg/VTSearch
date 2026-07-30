"""Library-tier tests for clip windows surviving the pickle round-trip.

A clipped or windowed media's playback window lives in four *top-level* fields
(``provenance.CLIP_WINDOW_FIELDS``): every player seeks to ``clip_start`` and
loops within ``[clip_start, clip_end]``, the audio waveform is sliced by the
same pair, and ``display_metadata`` renders them as the "Clip …" rows.  The
same extents also sit in ``origin.params``, but only as a re-derivation recipe
in a different shape, so they do not stand in — ``export_dataset_to_file``
writes an explicit field list, so the window has to be named there and restored
on load or every window of a shared source reloads playing the whole source
from 0 (a manifest that windows one tar member N times reloads as N
identical-sounding items).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
from vtscore.media.audio.media_type import AudioMediaType
from vtscore.media.provenance import CLIP_WINDOW_FIELDS


def _media(media_type: str = "audio", **extra) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    return {
        "id": 1,
        "media_type": media_type,
        "duration": 10.0,
        "file_size": 64,
        "md5": "abc",
        "embedder": "clap",
        "embeddings": {"clap": rng.standard_normal(16).astype(np.float32)},
        "filename": f"a.{media_type}",
        "category": "test",
        "media_bytes": b"\x00" * 64,
        "media_string": None,
        "media_path": None,
        **extra,
    }


def _archive_member_media(**extra) -> dict[str, Any]:
    """A windowed archive member: bytes stream from a tar shard, never inline."""
    return _media(
        media_bytes=None,
        filename="shard0/chunk.m4a",
        origin={
            "importer": "local_archive_member",
            "params": {
                "archive_path": "/data/shard0.tar",
                "member": "chunk.m4a",
                "media_type": "audio",
                "clip_start": 5.0,
                "clip_end": 15.0,
            },
        },
        origin_name="shard0.tar::chunk.m4a",
        **extra,
    )


def _round_trip(media: dict[str, Any], tmp_path: Path, thin: bool = False) -> dict[str, Any]:
    container = export_dataset_to_file(
        {1: media},
        embedder="clap",
        media_type=media.get("media_type", "audio"),
    )
    pkl = tmp_path / "ds.pkl"
    pkl.write_bytes(container)
    loaded: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(pkl, loaded, thin=thin)
    return loaded[1]


class TestClipWindowSurvivesPickleRoundTrip:
    def test_archive_member_window_persists(self, tmp_path: Path):
        out = _round_trip(_archive_member_media(clip_start=5.0, clip_end=15.0), tmp_path)

        assert out["clip_start"] == 5.0
        assert out["clip_end"] == 15.0

    def test_two_windows_of_one_member_stay_distinct(self, tmp_path: Path):
        """The point of the window: N rows off one member must not collapse."""
        first = _archive_member_media(clip_start=0.0, clip_end=10.0)
        second = _archive_member_media(clip_start=10.0, clip_end=20.0)
        second["id"] = 2
        second["md5"] = "def"
        container = export_dataset_to_file({1: first, 2: second}, embedder="clap", media_type="audio")
        pkl = tmp_path / "windows.pkl"
        pkl.write_bytes(container)
        loaded: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl, loaded)

        assert [(m["clip_start"], m["clip_end"]) for m in loaded.values()] == [(0.0, 10.0), (10.0, 20.0)]

    def test_video_clip_window_and_index_persist(self, tmp_path: Path):
        out = _round_trip(
            _media("video", clip_start=1.5, clip_end=4.25, clip_index=3),
            tmp_path,
        )

        assert (out["clip_start"], out["clip_end"], out["clip_index"]) == (1.5, 4.25, 3)

    def test_image_clip_box_persists_as_a_list(self, tmp_path: Path):
        """``clip_box`` is a 4-int list top-level (comma-joined only in origin params)."""
        out = _round_trip(_media("image", clip_box=[10, 20, 110, 120]), tmp_path)

        assert out["clip_box"] == [10, 20, 110, 120]

    def test_window_persists_through_a_thin_load(self, tmp_path: Path):
        out = _round_trip(_archive_member_media(clip_start=5.0, clip_end=15.0), tmp_path, thin=True)

        assert (out["clip_start"], out["clip_end"]) == (5.0, 15.0)

    def test_display_metadata_keeps_the_clip_rows(self, tmp_path: Path):
        out = _round_trip(_archive_member_media(clip_start=5.0, clip_end=15.0), tmp_path)
        meta = AudioMediaType().display_metadata(out)

        assert meta["Clip Start"] == 5.0
        assert meta["Clip End"] == 15.0

    def test_unclipped_media_gains_no_clip_fields(self, tmp_path: Path):
        out = _round_trip(_media(), tmp_path)

        assert not [field for field in CLIP_WINDOW_FIELDS if field in out]


class TestWaveformWindowingAfterReload:
    """The restored window must not re-slice bytes that are already the clip."""

    def _captured_window(self, monkeypatch, media: dict[str, Any]) -> tuple:
        from vtscore.media.audio import media_type as audio_module

        captured: dict[str, Any] = {}

        def fake_window(_loader, clip_start=None, clip_end=None, cache_key=None):
            captured["window"] = (clip_start, clip_end)
            return b"png"

        monkeypatch.setattr(audio_module, "generate_waveform_thumbnail_window", fake_window)
        AudioMediaType()._waveform_for_media(media)
        return captured["window"]

    def test_archive_member_window_is_applied(self, monkeypatch):
        """The whole member is served, so the waveform has to window it."""
        media = _archive_member_media(clip_start=5.0, clip_end=15.0)

        assert self._captured_window(monkeypatch, media) == (5.0, 15.0)

    def test_lazy_audio_clip_is_not_windowed_twice(self, monkeypatch):
        """A byte-sliced clip resolves as its own 0-based file; windowing it again
        would render the wrong stretch."""
        media = _media(
            media_bytes=None,
            media_path="/data/soundscape.wav",
            clip_start=120.0,
            clip_end=125.0,
            origin={
                "importer": "server_folder",
                "params": {"clipper": "fixed_window", "clip_start": 120.0, "clip_end": 125.0},
            },
        )

        assert self._captured_window(monkeypatch, media) == (None, None)

    def test_materialized_clip_is_not_windowed_twice(self, monkeypatch):
        media = _media(clip_start=120.0, clip_end=125.0)

        assert self._captured_window(monkeypatch, media) == (None, None)
