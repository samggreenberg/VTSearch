"""Tests for the ``local_archive_member`` importer + its NPZ manifest reader."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from vtscore.datasets.archive_stream import archive_member_ref, read_member
from vtscore.datasets.importers._npz_vectors import read_npz_archive_member_rows
from vtscore.datasets.importers.local_archive_member import IMPORTER

MEMBERS = {
    "chunk_a.mp4": b"AAAA-payload-a" * 16,
    "sub/chunk_b.mp4": b"BBBB-payload-b" * 16,
}
DIM = 512


def _make_tar(tmp_path: Path) -> Path:
    archive = tmp_path / "shard_000000.tar"
    with tarfile.open(archive, "w") as tf:
        for name, payload in MEMBERS.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return archive


def _make_manifest(tmp_path: Path, archive: Path, *, scalar_archive: bool = True, members=None) -> Path:
    members = members if members is not None else list(MEMBERS)
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((len(members), DIM)).astype(np.float32)
    manifest = tmp_path / "manifest.npz"
    archives = np.array(str(archive)) if scalar_archive else np.array([str(archive)] * len(members))
    np.savez(
        manifest,
        vectors=vectors,
        members=np.array(members),
        archives=archives,
        embedder_name=np.array("largerclapgeneral"),
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
            assert r["vector"].shape == (DIM,)
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
            assert media["embedder"] == "largerclapgeneral"
            assert media["embeddings"]["largerclapgeneral"].shape == (DIM,)
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
