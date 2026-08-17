"""Tests for the ``local_archive_member`` importer + its NPZ manifest reader."""

from __future__ import annotations

import io
import struct
import tarfile
import wave
from pathlib import Path

import numpy as np
import pytest

from vtscore.datasets.archive_stream import archive_member_ref, read_member
from vtscore.datasets.importers._npz_vectors import read_npz_archive_member_rows
from vtscore.datasets.importers.local_archive_member import IMPORTER
from vtscore.embedding.binding import expected_dim_for_embedder

MEMBERS = {
    "chunk_a.mp4": b"AAAA-payload-a" * 16,
    "sub/chunk_b.mp4": b"BBBB-payload-b" * 16,
}
#: Width used when a manifest declares no embedder (nothing to agree with).
DIM = 512


def _dim_for(embedder_name: str | None) -> int:
    """Width a manifest naming *embedder_name* must use to be self-consistent.

    Manifest reads reject an archive whose vectors contradict the embedder it
    names, so a fixture cannot pair a real name with an arbitrary toy width -
    that combination is precisely the mislabelled manifest the guard exists to
    catch. Deriving the width here keeps every fixture a *valid* manifest, so
    these tests exercise the behaviour they are about rather than the guard.
    """
    return expected_dim_for_embedder(embedder_name) or DIM


def _make_tar(tmp_path: Path) -> Path:
    archive = tmp_path / "shard_000000.tar"
    with tarfile.open(archive, "w") as tf:
        for name, payload in MEMBERS.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return archive


def _make_manifest(
    tmp_path: Path,
    archive: Path,
    *,
    scalar_archive: bool = True,
    members=None,
    embedder_name: str | None = "xclip",
) -> Path:
    members = members if members is not None else list(MEMBERS)
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((len(members), _dim_for(embedder_name))).astype(np.float32)
    manifest = tmp_path / "manifest.npz"
    archives = np.array(str(archive)) if scalar_archive else np.array([str(archive)] * len(members))
    arrays = {"vectors": vectors, "members": np.array(members), "archives": archives}
    if embedder_name is not None:
        arrays["embedder_name"] = np.array(embedder_name)
    np.savez(manifest, **arrays)
    return manifest


def _make_windowed_manifest(tmp_path: Path, archive: Path) -> Path:
    """One member fanned into three 10 s windows, plus a second whole-member row."""
    rows = [
        ("chunk_a.mp4", 0.0, 10.0),
        ("chunk_a.mp4", 10.0, 20.0),
        ("chunk_a.mp4", 20.0, 30.0),
    ]
    members = [m for m, _, _ in rows] + ["sub/chunk_b.mp4"]
    clip_starts = [s for _, s, _ in rows] + [float("nan")]  # NaN: whole-member row, no window
    clip_ends = [e for _, _, e in rows] + [float("nan")]
    rng = np.random.default_rng(11)
    vectors = rng.standard_normal((len(members), _dim_for("xclip"))).astype(np.float32)
    manifest = tmp_path / "windowed.npz"
    np.savez(
        manifest,
        vectors=vectors,
        members=np.array(members),
        archives=np.array(str(archive)),
        clip_start=np.array(clip_starts, dtype=np.float32),
        clip_end=np.array(clip_ends, dtype=np.float32),
        embedder_name=np.array("xclip"),
    )
    return manifest


class TestManifestReader:
    def test_reads_rows_with_scalar_archive(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive, scalar_archive=True)
        rows = read_npz_archive_member_rows(manifest)
        assert len(rows) == 2
        assert {r["member"] for r in rows} == set(MEMBERS)
        for r in rows:
            assert Path(r["archive"]).name == "shard_000000.tar"
            assert r["vector"].shape == (_dim_for("xclip"),)
        # filename defaults to the member basename.
        by_member = {r["member"]: r for r in rows}
        assert by_member["sub/chunk_b.mp4"]["filename"] == "chunk_b.mp4"

    def test_reads_rows_with_per_row_archives(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive, scalar_archive=False)
        rows = read_npz_archive_member_rows(manifest)
        assert len(rows) == 2

    def test_missing_required_arrays_raise(self, tmp_path):
        bad = tmp_path / "bad.npz"
        np.savez(bad, vectors=np.zeros((2, DIM), dtype=np.float32))
        with pytest.raises(ValueError):
            read_npz_archive_member_rows(bad)


class TestImporter:
    def test_imports_whole_members_without_bytes(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive)

        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)

        assert len(medias) == 2
        for media in medias.values():
            assert media["media_type"] == "video"
            # No materialized bytes / disk path: bytes re-derive from the shard.
            assert media["media_bytes"] is None
            assert "media_path" not in media or media["media_path"] is None
            # Precomputed embedding is carried, under the manifest's embedder.
            assert media["embedder"] == "xclip"
            assert media["embeddings"]["xclip"].shape == (_dim_for("xclip"),)
            assert len(media["md5"]) == 32
            ref = media["archive_member"]
            assert Path(ref["path"]).name == "shard_000000.tar"
            assert media["origin"]["importer"] == "local_archive_member"
            assert media["origin"]["params"]["manifest"] == str(manifest.resolve())

    def test_imported_media_bytes_resolve_from_shard(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)

        for media in medias.values():
            ref = archive_member_ref(media)
            assert ref is not None
            data = read_member(ref[0], ref[1])
            assert data == MEMBERS[ref[1]]

    def test_unique_md5_per_member(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        md5s = {m["md5"] for m in medias.values()}
        assert len(md5s) == len(medias)

    def test_skips_missing_members(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive, members=["chunk_a.mp4", "not_in_shard.mp4"])
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        # The missing member is skipped; the real one still imports.
        assert len(medias) == 1
        assert next(iter(medias.values()))["archive_member"]["member"] == "chunk_a.mp4"

    def test_all_missing_raises(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive, members=["nope1.mp4", "nope2.mp4"])
        medias: dict[int, dict] = {}
        with pytest.raises(ValueError):
            IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)

    def test_reload_from_origin_round_trips(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        origin = next(iter(medias.values()))["origin"]
        assert IMPORTER.can_reload_from_origin(origin)
        reloaded = IMPORTER.reload_from_origin(origin)
        assert reloaded == {"manifest": str(manifest.resolve()), "media_type": "video"}


class TestEmbedderNameValidation:
    """A manifest embedder_name that VTSearch can't route is rejected at import."""

    def test_unregistered_embedder_name_raises_with_options(self, tmp_path):
        archive = _make_tar(tmp_path)
        # The exact symptom from the field report: a name from an external
        # pipeline that isn't in VTSearch's registry.
        manifest = _make_manifest(tmp_path, archive, embedder_name="audemb_largerclapgeneral")
        medias: dict[int, dict] = {}
        with pytest.raises(ValueError) as exc:
            IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        msg = str(exc.value)
        assert "audemb_largerclapgeneral" in msg
        # The error names the media type and lists the valid registered options.
        assert "video" in msg
        assert "xclip" in msg
        # Nothing was imported.
        assert medias == {}

    def test_wrong_media_type_embedder_raises(self, tmp_path):
        archive = _make_tar(tmp_path)
        # ``clap_general`` is a registered *audio* embedder, so it can't bind
        # this ``video`` dataset's slots either.
        manifest = _make_manifest(tmp_path, archive, embedder_name="clap_general")
        medias: dict[int, dict] = {}
        with pytest.raises(ValueError) as exc:
            IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        assert "clap_general" in str(exc.value)
        assert medias == {}

    def test_registered_embedder_name_imports(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive, embedder_name="xclip")
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        assert len(medias) == 2

    def test_missing_embedder_name_is_allowed(self, tmp_path):
        # A manifest that declares no embedder imports fine (the media just
        # carries no embedder identity); validation only gates a *named* one.
        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive, embedder_name=None)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        assert len(medias) == 2
        assert next(iter(medias.values()))["embedder"] == ""


class TestWindowedManifest:
    def test_reader_emits_clip_fields(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_windowed_manifest(tmp_path, archive)
        rows = read_npz_archive_member_rows(manifest)
        assert len(rows) == 4
        windows = [r for r in rows if r["member"] == "chunk_a.mp4"]
        assert {(r["clip_start"], r["clip_end"]) for r in windows} == {(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)}
        # The NaN-sentinel whole-member row carries no window.
        whole = next(r for r in rows if r["member"] == "sub/chunk_b.mp4")
        assert whole["clip_start"] is None
        assert whole["clip_end"] is None

    def test_importer_fans_member_into_windows(self, tmp_path):
        archive = _make_tar(tmp_path)
        manifest = _make_windowed_manifest(tmp_path, archive)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)

        # 3 windows of chunk_a + 1 whole chunk_b.
        assert len(medias) == 4
        windows = [m for m in medias.values() if m["archive_member"]["member"] == "chunk_a.mp4"]
        assert len(windows) == 3
        for m in windows:
            assert m["clip_start"] is not None and m["clip_end"] is not None
            # Display-only window persisted in origin.params too.
            assert m["origin"]["params"]["clip_start"] == m["clip_start"]
            assert m["origin"]["params"]["clip_end"] == m["clip_end"]
        # Each window's synthesized md5 is unique despite sharing one member.
        assert len({m["md5"] for m in windows}) == 3
        # The whole-member row carries no clip fields.
        whole = next(m for m in medias.values() if m["archive_member"]["member"] == "sub/chunk_b.mp4")
        assert "clip_start" not in whole
        assert "clip_start" not in whole["origin"]["params"]

    def test_window_id_disambiguates_md5(self, tmp_path):
        archive = _make_tar(tmp_path)
        rng = np.random.default_rng(3)
        manifest = tmp_path / "wid.npz"
        np.savez(
            manifest,
            vectors=rng.standard_normal((2, _dim_for("xclip"))).astype(np.float32),
            members=np.array(["chunk_a.mp4", "chunk_a.mp4"]),
            archives=np.array(str(archive)),
            window_id=np.array(["w0", "w1"]),
            embedder_name=np.array("xclip"),
        )
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        assert len(medias) == 2
        assert len({m["md5"] for m in medias.values()}) == 2
        assert {m["origin"]["params"]["window_id"] for m in medias.values()} == {"w0", "w1"}


class TestArchiveMemberClipRecipe:
    def test_archive_member_audio_never_lazy_slices(self, tmp_path):
        """A windowed audio archive member must not trigger WAV byte-slicing."""
        from vtscore.media.lazy_clip import clip_recipe

        archive = _make_tar(tmp_path)
        media = {
            "media_type": "audio",
            "archive_member": {"path": str(archive), "member": "chunk_a.mp4"},
            "origin": {
                "importer": "local_archive_member",
                "params": {"archive_path": str(archive), "member": "chunk_a.mp4", "clip_start": 0.0, "clip_end": 10.0},
            },
            "clip_start": 0.0,
            "clip_end": 10.0,
        }
        assert clip_recipe(media) is None


class TestLocalArchiveMemberSource:
    def test_factory_resolves_origin(self, tmp_path):
        from vtscore.datasets.sources import get_source_for_origin

        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        origin = next(iter(medias.values()))["origin"]

        source = get_source_for_origin(origin)
        assert source is not None
        assert source.name == "local_archive_member"

    def test_fetch_item_resupplies_vector_no_path(self, tmp_path):
        from vtscore.datasets.sources import get_source_for_origin

        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        media = next(m for m in medias.values() if m["archive_member"]["member"] == "chunk_a.mp4")

        source = get_source_for_origin(media["origin"])
        assert source is not None
        fetched = source.fetch_item("chunk_a.mp4")
        assert fetched.path is None
        assert fetched.embedding is not None
        assert fetched.embedding.shape == (_dim_for("xclip"),)
        assert fetched.embedder_name == "xclip"
        # Re-supplied vector matches the in-memory embedding from import.
        np.testing.assert_array_equal(fetched.embedding, media["embeddings"]["xclip"])

    def test_fetch_item_resolves_windowed_member(self, tmp_path):
        from vtscore.datasets.sources import get_source_for_origin

        archive = _make_tar(tmp_path)
        manifest = _make_windowed_manifest(tmp_path, archive)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        window = next(
            m for m in medias.values() if m["archive_member"]["member"] == "chunk_a.mp4" and m.get("clip_start") == 10.0
        )

        source = get_source_for_origin(window["origin"])
        assert source is not None
        # The window key is "member@start"; fetch by it returns that window's vector.
        fetched = source.fetch_item("chunk_a.mp4@10")
        assert fetched.embedding is not None
        np.testing.assert_array_equal(fetched.embedding, window["embeddings"]["xclip"])

    def test_fetch_item_unknown_key_returns_pathless_empty(self, tmp_path):
        from vtscore.datasets.sources import get_source_for_origin

        archive = _make_tar(tmp_path)
        manifest = _make_manifest(tmp_path, archive)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "video"}, medias)
        source = get_source_for_origin(next(iter(medias.values()))["origin"])
        assert source is not None
        fetched = source.fetch_item("does/not/exist.mp4")
        assert fetched.path is None
        assert fetched.embedding is None


def _ramped_wav_bytes(seconds: float = 3.0, sr: int = 16000, freq: float = 220.0) -> bytes:
    """Return WAV bytes whose amplitude ramps from near-0 to full over its length.

    The ramp makes each per-second window's waveform envelope visibly distinct,
    so a window-aware thumbnail differs across windows of the same member.
    """
    n = int(seconds * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            env = 0.05 + 0.95 * (i / n)
            sample = int(env * 32767 * np.sin(2 * np.pi * freq * i / sr))
            frames += struct.pack("<h", sample)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _make_audio_tar(tmp_path: Path, member: str = "clip_a.wav") -> Path:
    archive = tmp_path / "audio_shard.tar"
    payload = _ramped_wav_bytes()
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return archive


def _make_windowed_audio_manifest(tmp_path: Path, archive: Path, member: str = "clip_a.wav") -> Path:
    """One audio member fanned into three 1 s windows."""
    members = [member, member, member]
    starts = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    ends = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vectors = np.random.default_rng(3).standard_normal((3, 512)).astype(np.float32)
    manifest = tmp_path / "audio_windowed.npz"
    np.savez(
        manifest,
        vectors=vectors,
        members=np.array(members),
        archives=np.array(str(archive)),
        clip_start=starts,
        clip_end=ends,
    )
    return manifest


class TestArchiveMemberAudioWaveform:
    """Waveform thumbnails for windowed archive-member audio (VTSBrowse tiles)."""

    def _audio_type(self):
        from vtscore.media.audio.media_type import AudioMediaType

        return AudioMediaType()

    def test_windows_get_distinct_waveforms(self, tmp_path):
        import vtscore.media.audio.media_type as mt

        mt._clear_decode_cache()
        archive = _make_audio_tar(tmp_path)
        manifest = _make_windowed_audio_manifest(tmp_path, archive)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "audio"}, medias)
        assert len(medias) == 3

        audio = self._audio_type()
        pngs = []
        for media in medias.values():
            resp = audio.image_response(media)
            assert resp is not None, "windowed archive-member audio must render a waveform"
            assert resp.mimetype == "image/png"
            assert resp.data[:8] == b"\x89PNG\r\n\x1a\n"
            # The generated PNG is memoised so repeat fetches skip the decode.
            assert media.get("thumbnail_bytes") == resp.data
            pngs.append(resp.data)

        # Each window shows its own slice, not one shared whole-member waveform.
        assert len(set(pngs)) == 3

    def test_member_decoded_once_across_windows(self, tmp_path):
        import vtscore.media.audio.media_type as mt

        mt._clear_decode_cache()
        archive = _make_audio_tar(tmp_path)
        manifest = _make_windowed_audio_manifest(tmp_path, archive)
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "audio"}, medias)

        audio = self._audio_type()
        for media in medias.values():
            audio.image_response(media)
        # All three windows share one member, so the decode cache holds one entry.
        assert len(mt._decode_cache) == 1

    def test_whole_member_audio_renders_waveform(self, tmp_path):
        import vtscore.media.audio.media_type as mt

        mt._clear_decode_cache()
        archive = _make_audio_tar(tmp_path)
        # Manifest with no clip windows: one whole-member row.
        vectors = np.random.default_rng(4).standard_normal((1, 512)).astype(np.float32)
        manifest = tmp_path / "audio_whole.npz"
        np.savez(manifest, vectors=vectors, members=np.array(["clip_a.wav"]), archives=np.array(str(archive)))
        medias: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "audio"}, medias)

        resp = self._audio_type().image_response(next(iter(medias.values())))
        assert resp is not None
        assert resp.data[:8] == b"\x89PNG\r\n\x1a\n"


class TestSliceWindow:
    def test_slices_to_bounds(self):
        from vtscore.media.audio.media_type import _slice_window

        data = np.arange(1000, dtype=np.float32)
        out = _slice_window(data, 100, 2.0, 5.0)  # sr=100 -> samples [200:500]
        assert len(out) == 300
        np.testing.assert_array_equal(out, data[200:500])

    def test_open_bounds(self):
        from vtscore.media.audio.media_type import _slice_window

        data = np.arange(1000, dtype=np.float32)
        np.testing.assert_array_equal(_slice_window(data, 100, None, 3.0), data[:300])
        np.testing.assert_array_equal(_slice_window(data, 100, 7.0, None), data[700:])

    def test_nan_bound_left_open(self):
        from vtscore.media.audio.media_type import _slice_window

        data = np.arange(1000, dtype=np.float32)
        np.testing.assert_array_equal(_slice_window(data, 100, float("nan"), float("nan")), data)

    def test_degenerate_window_falls_back_to_whole(self):
        from vtscore.media.audio.media_type import _slice_window

        data = np.arange(1000, dtype=np.float32)
        # end <= start -> whole array rather than an empty slice.
        np.testing.assert_array_equal(_slice_window(data, 100, 5.0, 2.0), data)
