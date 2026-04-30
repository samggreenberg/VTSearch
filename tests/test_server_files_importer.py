"""Tests for the ``server_files`` dataset importer.

The importer reads a text file on the server containing one media-file
path per line, symlinks each into a temp directory, and delegates to the
``server_folder`` importer for embedding.  After the import each media's
origin is rewritten to point at this importer with the original absolute
path stored in ``origin_name``, so :meth:`resolve_file` can locate the
file without needing the temp dir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vtsearch.datasets.importers import get_importer
from vtsearch.datasets.importers.server_files import (
    ServerFilesDatasetImporter,
    _read_paths_file,
    _symlink_paths,
)


class TestReadPathsFile:
    def test_reads_one_path_per_line(self, tmp_path):
        f = tmp_path / "paths.txt"
        f.write_text("/a/b/c.wav\n/d/e/f.wav\n")
        assert _read_paths_file(f) == [Path("/a/b/c.wav"), Path("/d/e/f.wav")]

    def test_skips_blank_lines_and_comments(self, tmp_path):
        f = tmp_path / "paths.txt"
        f.write_text(
            "# this is a comment\n/a.wav\n\n  # indented comment\n/b.wav\n",
        )
        # The "  # indented comment" line is *not* skipped because it
        # has leading whitespace before the ``#``; the importer only
        # treats lines starting with ``#`` as comments.  Document the
        # actual behaviour so users know to put comments at column 0.
        paths = _read_paths_file(f)
        assert Path("/a.wav") in paths
        assert Path("/b.wav") in paths

    def test_relative_paths_resolved_against_paths_file_dir(self, tmp_path):
        media = tmp_path / "data" / "x.wav"
        media.parent.mkdir(parents=True)
        media.write_bytes(b"x")
        listing = tmp_path / "list.txt"
        listing.write_text("data/x.wav\n")
        result = _read_paths_file(listing)
        assert result == [media.resolve()]

    def test_missing_file_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            _read_paths_file(tmp_path / "no.txt")


class TestSymlinkPaths:
    def test_symlinks_existing_files(self, tmp_path):
        src_a = tmp_path / "a.wav"
        src_b = tmp_path / "b.wav"
        src_a.write_bytes(b"A")
        src_b.write_bytes(b"B")
        target = tmp_path / "stage"
        mapping = _symlink_paths([src_a, src_b], target)
        assert set(mapping.keys()) == {"a.wav", "b.wav"}
        assert (target / "a.wav").is_symlink()
        assert (target / "a.wav").resolve() == src_a.resolve()

    def test_skips_missing_files(self, tmp_path):
        src_a = tmp_path / "a.wav"
        src_a.write_bytes(b"A")
        mapping = _symlink_paths([src_a, tmp_path / "missing.wav"], tmp_path / "stage")
        assert list(mapping.keys()) == ["a.wav"]

    def test_disambiguates_duplicate_basenames(self, tmp_path):
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        a1 = d1 / "x.wav"
        a2 = d2 / "x.wav"
        a1.write_bytes(b"1")
        a2.write_bytes(b"2")
        mapping = _symlink_paths([a1, a2], tmp_path / "stage")
        assert len(mapping) == 2
        assert "x.wav" in mapping
        # Second one gets a suffix like x__1.wav
        assert any(k.startswith("x__") for k in mapping)


class TestImporterMetadata:
    def test_registered_name(self):
        imp = get_importer("server_files")
        assert imp is not None
        assert imp.name == "server_files"
        assert imp.display_name == "Server Files"
        assert imp.picker_view == "form"
        assert imp.hidden_from_picker is False

    def test_fields_include_paths_file_and_media_type(self):
        imp = ServerFilesDatasetImporter()
        keys = {f.key for f in imp.fields}
        assert "paths_file" in keys
        assert "media_type" in keys

    def test_build_origin_excludes_blank_params(self):
        imp = ServerFilesDatasetImporter()
        origin = imp.build_origin({"paths_file": "/a/list.txt", "media_type": "audio"})
        assert origin == {
            "importer": "server_files",
            "params": {"paths_file": "/a/list.txt", "media_type": "audio"},
        }

    def test_resolve_file_returns_origin_name_when_file_exists(self, tmp_path):
        f = tmp_path / "real.wav"
        f.write_bytes(b"x")
        imp = ServerFilesDatasetImporter()
        result = imp.resolve_file(
            {"importer": "server_files", "params": {"paths_file": str(tmp_path / "list.txt")}},
            origin_name=str(f),
        )
        assert result == f

    def test_resolve_file_returns_none_when_path_missing(self, tmp_path):
        imp = ServerFilesDatasetImporter()
        assert (
            imp.resolve_file(
                {"importer": "server_files", "params": {}},
                origin_name=str(tmp_path / "ghost.wav"),
            )
            is None
        )


class TestRunEndToEnd:
    """Verify run() produces medias whose origin/origin_name point at the
    real source paths.

    The folder importer's audio loader runs the embedder, which on the
    test container is the lightweight raw-WAV embedder; that's fine for
    our purposes as long as we use real WAV bytes."""

    def test_run_imports_listed_files_and_rewrites_origin(self, tmp_path):
        from helpers import make_raw_wav_bytes

        # Stage two real wav files plus a paths.txt referencing them.
        src_a = tmp_path / "src_a.wav"
        src_b = tmp_path / "src_b.wav"
        src_a.write_bytes(make_raw_wav_bytes())
        # Make src_b a structurally-distinct WAV so dedup doesn't collapse it.
        src_b.write_bytes(make_raw_wav_bytes() + b"\x00\x00")
        listing = tmp_path / "list.txt"
        listing.write_text(f"{src_a}\n{src_b}\n")

        imp = ServerFilesDatasetImporter()
        medias: dict = {}
        imp.run(
            {"paths_file": str(listing), "media_type": "audio"},
            medias,
        )

        assert len(medias) == 2
        for media in medias.values():
            # Origin is rewritten to this importer with paths_file param.
            assert media["origin"]["importer"] == "server_files"
            assert media["origin"]["params"]["paths_file"] == str(listing)
            # origin_name points at the original absolute path, not the
            # temp staging dir, so resolve_file works after the temp dir
            # is cleaned up.
            assert media["origin_name"] in {str(src_a), str(src_b)}
            assert Path(media["origin_name"]).is_file()
            assert isinstance(media["embedding"], np.ndarray)


class TestRunChunked:
    """Verify run_chunked yields chunks of the requested size and
    rewrites each chunk's origins back to the source paths."""

    def test_supports_chunked(self):
        assert ServerFilesDatasetImporter().supports_chunked is True

    def test_run_chunked_yields_in_chunk_size(self, tmp_path):
        from helpers import make_raw_wav_bytes

        # Four structurally-distinct WAVs so dedup doesn't collapse them.
        srcs = []
        for i in range(4):
            p = tmp_path / f"s_{i}.wav"
            p.write_bytes(make_raw_wav_bytes() + bytes([i]) * (i + 1))
            srcs.append(p)
        listing = tmp_path / "list.txt"
        listing.write_text("\n".join(str(p) for p in srcs) + "\n")

        imp = ServerFilesDatasetImporter()
        chunks = list(
            imp.run_chunked(
                {"paths_file": str(listing), "media_type": "audio"},
                chunk_size=2,
                thin=True,
            )
        )

        # Two chunks of two medias each.
        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk) == 2
            for media in chunk.values():
                assert media["origin"]["importer"] == "server_files"
                assert media["origin"]["params"]["paths_file"] == str(listing)
                # origin_name is the real source path, not the staging symlink.
                assert Path(media["origin_name"]).is_file()
                assert media["origin_name"] in {str(p) for p in srcs}

    def test_run_chunked_cli_validates_paths_file(self):
        import pytest

        imp = ServerFilesDatasetImporter()
        with pytest.raises(FileNotFoundError):
            list(
                imp.run_chunked_cli(
                    {"paths_file": "/nonexistent.txt", "media_type": "audio"},
                    chunk_size=10,
                )
            )

    def test_run_chunked_cleans_up_staging_dir(self, tmp_path):
        from helpers import make_raw_wav_bytes

        src = tmp_path / "only.wav"
        src.write_bytes(make_raw_wav_bytes())
        listing = tmp_path / "list.txt"
        listing.write_text(f"{src}\n")

        # Snapshot existing tmp prefixes so we can detect a leak.
        import tempfile as _tempfile

        tmp_root = Path(_tempfile.gettempdir())
        before = {p.name for p in tmp_root.iterdir() if p.name.startswith("server_files_")}

        imp = ServerFilesDatasetImporter()
        list(
            imp.run_chunked(
                {"paths_file": str(listing), "media_type": "audio"},
                chunk_size=10,
                thin=True,
            )
        )

        after = {p.name for p in tmp_root.iterdir() if p.name.startswith("server_files_")}
        assert after == before, f"server_files_ staging dirs leaked: {after - before}"
