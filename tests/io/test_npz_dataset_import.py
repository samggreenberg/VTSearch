"""Tests for ``.npz`` support in the ``server_files`` and ``local_files`` importers.

The ``server_files`` importer accepts a ``.npz`` archive (in addition to
plain text) that supplies both the media-file paths and their
pre-computed embedding vectors.  Listed files are still symlinked into a
staging dir and run through the regular folder loader, but the loader
skips re-embedding for files whose names appear in the supplied
``content_vectors`` map.

The ``/api/dataset/import-local-files`` endpoint is the browser-upload
equivalent of ``server_files``: the user POSTs a single ``paths_file``
(a ``.txt`` of paths or a ``.npz`` of paths + vectors) and the server
runs the regular ``server_files`` importer over the uploaded file.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vtscore.datasets.importers._npz_vectors import (
    read_npz_embedder_name,
    read_npz_filenames_and_vectors,
    read_npz_multi_vectors,
    write_npz_multi_vectors,
)
from vtscore.embedding.binding import expected_dim_for_embedder
from vtscore.embedding.media_vectors import media_embedding
from vtscore.datasets.importers.server_files import (
    ServerFilesDatasetImporter,
    _read_npz_paths_file,
    _read_paths_and_vectors,
    _read_paths_file,
)


def _read_multi(npz):
    """Read a per-embedder archive, asserting it is recognised (narrows None)."""
    result = read_npz_multi_vectors(npz)
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Low-level NPZ reader helper
# ---------------------------------------------------------------------------


class TestReadNpzFilenamesAndVectors:
    def test_reads_standard_layout(self, tmp_path):
        npz = tmp_path / "vecs.npz"
        filenames = np.array(["a.wav", "b.wav", "c.wav"])
        vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
        np.savez(npz, filenames=filenames, vectors=vectors)

        mapping = read_npz_filenames_and_vectors(npz)
        assert list(mapping.keys()) == ["a.wav", "b.wav", "c.wav"]
        np.testing.assert_array_equal(mapping["a.wav"], vectors[0])
        np.testing.assert_array_equal(mapping["b.wav"], vectors[1])
        np.testing.assert_array_equal(mapping["c.wav"], vectors[2])

    def test_reads_per_key_layout_fallback(self, tmp_path):
        npz = tmp_path / "vecs.npz"
        v1 = np.array([1, 2, 3], dtype=np.float32)
        v2 = np.array([4, 5, 6], dtype=np.float32)
        # numpy savez stubs mis-bind unpacked kwargs to allow_pickle.
        np.savez(npz, **{"x.wav": v1, "y.wav": v2})  # pyright: ignore[reportArgumentType]

        mapping = read_npz_filenames_and_vectors(npz)
        assert set(mapping) == {"x.wav", "y.wav"}
        np.testing.assert_array_equal(mapping["x.wav"], v1)
        np.testing.assert_array_equal(mapping["y.wav"], v2)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_npz_filenames_and_vectors(tmp_path / "missing.npz")

    def test_mismatched_lengths_raises(self, tmp_path):
        npz = tmp_path / "bad.npz"
        np.savez(
            npz,
            filenames=np.array(["a.wav", "b.wav"]),
            vectors=np.zeros((3, 4), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="mismatched"):
            read_npz_filenames_and_vectors(npz)

    def test_2d_filenames_array_rejected(self, tmp_path):
        npz = tmp_path / "bad.npz"
        np.savez(
            npz,
            filenames=np.array([["a", "b"], ["c", "d"]]),
            vectors=np.zeros((2, 4), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="1-D"):
            read_npz_filenames_and_vectors(npz)

    def test_blank_names_are_skipped(self, tmp_path):
        npz = tmp_path / "vecs.npz"
        filenames = np.array(["a.wav", "", "  ", "b.wav"])
        vectors = np.arange(16, dtype=np.float32).reshape(4, 4)
        np.savez(npz, filenames=filenames, vectors=vectors)
        mapping = read_npz_filenames_and_vectors(npz)
        assert set(mapping) == {"a.wav", "b.wav"}


# ---------------------------------------------------------------------------
# read_npz_embedder_name
# ---------------------------------------------------------------------------


class TestReadNpzEmbedderName:
    def test_reads_embedder_name_key(self, tmp_path):
        npz = tmp_path / "vecs.npz"
        np.savez(
            npz,
            filenames=np.array(["a.wav"]),
            vectors=np.zeros((1, 4), dtype=np.float32),
            embedder_name=np.array("laion-clap"),
        )
        assert read_npz_embedder_name(npz) == "laion-clap"

    def test_reads_embedder_key_as_fallback(self, tmp_path):
        npz = tmp_path / "vecs.npz"
        np.savez(
            npz,
            filenames=np.array(["a.wav"]),
            vectors=np.zeros((1, 4), dtype=np.float32),
            embedder=np.array("siglip"),
        )
        assert read_npz_embedder_name(npz) == "siglip"

    def test_returns_empty_string_when_absent(self, tmp_path):
        npz = tmp_path / "vecs.npz"
        np.savez(npz, filenames=np.array(["a.wav"]), vectors=np.zeros((1, 4), dtype=np.float32))
        assert read_npz_embedder_name(npz) == ""

    def test_returns_empty_string_for_missing_file(self, tmp_path):
        assert read_npz_embedder_name(tmp_path / "missing.npz") == ""

    def test_strips_whitespace(self, tmp_path):
        npz = tmp_path / "vecs.npz"
        np.savez(
            npz,
            filenames=np.array(["a.wav"]),
            vectors=np.zeros((1, 4), dtype=np.float32),
            embedder_name=np.array("  clap  "),
        )
        assert read_npz_embedder_name(npz) == "clap"


# ---------------------------------------------------------------------------
# server_files paths-file plumbing
# ---------------------------------------------------------------------------


def _col(embedder_name: str, n_rows: int, offset: float = 0.0) -> np.ndarray:
    """A ``vectors_<name>`` column of the width *embedder_name* actually declares.

    Manifest reads reject a column whose width contradicts the embedder its key
    names - that combination is the mislabelled-archive case the guard exists to
    catch (see ``tests_lib/io/test_precomputed_vector_validation.py``).  So a
    fixture that wants a *valid* trio archive has to use the real width; *offset*
    keeps each column's values distinguishable.
    """
    dim = expected_dim_for_embedder(embedder_name) or 4
    return np.arange(n_rows * dim, dtype=np.float32).reshape(n_rows, dim) + offset


# ---------------------------------------------------------------------------
# read_npz_multi_vectors / write_npz_multi_vectors (per-embedder trio layout)
# ---------------------------------------------------------------------------


class TestReadNpzMultiVectors:
    def test_returns_none_without_per_embedder_keys(self, tmp_path):
        # A plain single-``vectors`` archive has no vectors_<name> key, so the
        # multi reader declines it (signalling the single-vector fallback).
        npz = tmp_path / "single.npz"
        np.savez(npz, filenames=np.array(["a.wav"]), vectors=np.zeros((1, 4), dtype=np.float32))
        assert read_npz_multi_vectors(npz) is None

    def test_reads_per_embedder_columns(self, tmp_path):
        siglip = _col("siglip", 2)
        patch = _col("dinov3_patch", 2, offset=100.0)
        npz = tmp_path / "multi.npz"
        np.savez(
            npz,
            filenames=np.array(["a.jpg", "b.jpg"]),
            vectors_siglip=siglip,
            vectors_dinov3_patch=patch,
        )
        mapping, primary = _read_multi(npz)
        assert set(mapping) == {"a.jpg", "b.jpg"}
        assert set(mapping["a.jpg"]) == {"siglip", "dinov3_patch"}
        np.testing.assert_array_equal(mapping["a.jpg"]["siglip"], siglip[0])
        np.testing.assert_array_equal(mapping["b.jpg"]["dinov3_patch"], patch[1])
        # No scalar embedder_name → the first column (archive order) is primary.
        assert primary == "siglip"

    def test_embedder_name_scalar_designates_primary(self, tmp_path):
        npz = tmp_path / "multi.npz"
        np.savez(
            npz,
            filenames=np.array(["a.jpg"]),
            vectors_siglip=_col("siglip", 1),
            vectors_dinov3_patch=_col("dinov3_patch", 1),
            embedder_name=np.array("dinov3_patch"),
        )
        _mapping, primary = _read_multi(npz)
        assert primary == "dinov3_patch"

    def test_primary_falls_back_when_scalar_not_a_column(self, tmp_path):
        # A scalar naming an embedder that has no vectors_<name> column can't be
        # the primary; the leading column wins instead.
        npz = tmp_path / "multi.npz"
        np.savez(
            npz,
            filenames=np.array(["a.jpg"]),
            vectors_siglip=_col("siglip", 1),
            embedder_name=np.array("clip"),
        )
        _mapping, primary = _read_multi(npz)
        assert primary == "siglip"

    def test_embedder_name_with_underscores_preserved(self, tmp_path):
        # Only the leading ``vectors_`` prefix is stripped, so an embedder name
        # that itself contains underscores survives intact.
        npz = tmp_path / "multi.npz"
        np.savez(
            npz,
            filenames=np.array(["a.jpg"]),
            vectors_dinov3_patch=_col("dinov3_patch", 1),
        )
        mapping, _primary = _read_multi(npz)
        assert set(mapping["a.jpg"]) == {"dinov3_patch"}

    def test_mismatched_column_length_raises(self, tmp_path):
        npz = tmp_path / "multi.npz"
        np.savez(
            npz,
            filenames=np.array(["a.jpg", "b.jpg"]),
            vectors_siglip=_col("siglip", 1),  # only 1 row for 2 files
        )
        with pytest.raises(ValueError, match="mismatched lengths"):
            read_npz_multi_vectors(npz)

    def test_missing_filenames_raises(self, tmp_path):
        npz = tmp_path / "multi.npz"
        np.savez(npz, vectors_siglip=_col("siglip", 2))
        with pytest.raises(ValueError, match="filenames"):
            read_npz_multi_vectors(npz)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_npz_multi_vectors(tmp_path / "nope.npz")


class TestWriteNpzMultiVectors:
    def test_round_trips_through_reader(self, tmp_path):
        rng = np.random.default_rng(7)
        siglip_dim = expected_dim_for_embedder("siglip") or 3
        patch_dim = expected_dim_for_embedder("dinov3_patch") or 4
        mapping = {
            "a.jpg": {
                "siglip": rng.standard_normal(siglip_dim).astype(np.float32),
                "dinov3_patch": rng.standard_normal(patch_dim).astype(np.float32),
            },
            "b.jpg": {
                "siglip": rng.standard_normal(siglip_dim).astype(np.float32),
                "dinov3_patch": rng.standard_normal(patch_dim).astype(np.float32),
            },
        }
        npz = tmp_path / "out.npz"
        write_npz_multi_vectors(npz, mapping, primary_embedder="dinov3_patch")

        read_back, primary = _read_multi(npz)
        assert primary == "dinov3_patch"
        assert set(read_back) == set(mapping)
        for fname, cols in mapping.items():
            for emb, vec in cols.items():
                np.testing.assert_array_equal(read_back[fname][emb], vec)

    def test_compressed_round_trips(self, tmp_path):
        dim = expected_dim_for_embedder("siglip") or 3
        mapping = {"a.jpg": {"siglip": np.ones(dim, dtype=np.float32)}}
        npz = tmp_path / "out.npz"
        write_npz_multi_vectors(npz, mapping, compressed=True)
        read_back, _primary = _read_multi(npz)
        np.testing.assert_array_equal(read_back["a.jpg"]["siglip"], np.ones(dim, dtype=np.float32))

    def test_empty_mapping_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            write_npz_multi_vectors(tmp_path / "out.npz", {})

    def test_misaligned_columns_raise(self, tmp_path):
        mapping = {
            "a.jpg": {"siglip": np.zeros(3, dtype=np.float32), "dinov3_patch": np.zeros(4, dtype=np.float32)},
            "b.jpg": {"siglip": np.zeros(3, dtype=np.float32)},  # missing dinov3_patch column
        }
        with pytest.raises(ValueError, match="columns must align"):
            write_npz_multi_vectors(tmp_path / "out.npz", mapping)

    def test_primary_not_a_column_raises(self, tmp_path):
        mapping = {"a.jpg": {"siglip": np.zeros(3, dtype=np.float32)}}
        with pytest.raises(ValueError, match="not among"):
            write_npz_multi_vectors(tmp_path / "out.npz", mapping, primary_embedder="clip")


class TestServerFilesFieldsAdvertiseNpz:
    """The frontend file-browser uses the ``accept`` attribute to filter
    the picker.  Make sure both txt/list and npz are advertised."""

    def test_paths_file_accept_includes_npz(self):
        imp = ServerFilesDatasetImporter()
        paths_field = next(f for f in imp.fields if f.key == "paths_file")
        accept_exts = {e.strip() for e in (paths_field.accept or "").split(",")}
        assert ".txt" in accept_exts
        assert ".list" in accept_exts
        assert ".npz" in accept_exts

    def test_paths_file_hint_mentions_npz(self):
        # The format hint (shown via the (?) icon next to the label) is where
        # the user discovers that .npz is an accepted shape for the paths file.
        # The shorter ``description`` covers the field's purpose only.
        imp = ServerFilesDatasetImporter()
        paths_field = next(f for f in imp.fields if f.key == "paths_file")
        assert ".npz" in (paths_field.hint or "").lower()


class TestServerFilesNpzPathsFile:
    def test_read_paths_file_dispatches_on_suffix(self, tmp_path):
        # txt path still works
        txt = tmp_path / "list.txt"
        txt.write_text("/a.wav\n/b.wav\n")
        assert _read_paths_file(txt) == [Path("/a.wav"), Path("/b.wav")]

        # npz with absolute paths
        npz = tmp_path / "list.npz"
        np.savez(
            npz,
            filenames=np.array(["/x.wav", "/y.wav"]),
            vectors=np.zeros((2, 4), dtype=np.float32),
        )
        assert _read_paths_file(npz) == [Path("/x.wav"), Path("/y.wav")]

    def test_read_npz_paths_file_returns_paths_and_vectors(self, tmp_path):
        media_a = tmp_path / "a.wav"
        media_b = tmp_path / "b.wav"
        media_a.write_bytes(b"A")
        media_b.write_bytes(b"B")
        npz = tmp_path / "list.npz"
        vectors = np.arange(8, dtype=np.float32).reshape(2, 4)
        np.savez(npz, filenames=np.array([str(media_a), str(media_b)]), vectors=vectors)

        paths, path_to_vector, embedder_name = _read_npz_paths_file(npz)
        assert paths == [Path(str(media_a)), Path(str(media_b))]
        assert set(path_to_vector) == {str(media_a), str(media_b)}
        np.testing.assert_array_equal(path_to_vector[str(media_a)], vectors[0])
        assert embedder_name == ""

    def test_relative_npz_paths_resolved_against_npz_dir(self, tmp_path):
        media = tmp_path / "data" / "x.wav"
        media.parent.mkdir(parents=True)
        media.write_bytes(b"x")
        npz = tmp_path / "list.npz"
        np.savez(
            npz,
            filenames=np.array(["data/x.wav"]),
            vectors=np.zeros((1, 4), dtype=np.float32),
        )

        paths, vecs, embedder_name = _read_npz_paths_file(npz)
        assert paths == [media.resolve()]
        # Vector is keyed by the resolved absolute path so the staging
        # rekey step can look it up.
        assert str(media.resolve()) in vecs
        assert embedder_name == ""

    def test_read_paths_and_vectors_txt_returns_empty_vectors(self, tmp_path):
        txt = tmp_path / "list.txt"
        txt.write_text("/a.wav\n")
        paths, vecs, embedder_name = _read_paths_and_vectors(txt)
        assert paths == [Path("/a.wav")]
        assert vecs == {}
        assert embedder_name == ""

    def test_read_npz_paths_file_returns_embedder_name(self, tmp_path):
        media = tmp_path / "clip.wav"
        media.write_bytes(b"data")
        npz = tmp_path / "list.npz"
        np.savez(
            npz,
            filenames=np.array([str(media)]),
            vectors=np.zeros((1, 4), dtype=np.float32),
            embedder_name=np.array("laion-clap"),
        )
        paths, vecs, embedder_name = _read_npz_paths_file(npz)
        assert paths == [media]
        assert embedder_name == "laion-clap"


# ---------------------------------------------------------------------------
# server_files end-to-end: NPZ vectors skip re-embedding
# ---------------------------------------------------------------------------


class TestServerFilesNpzRunsEndToEnd:
    def test_npz_vectors_are_used_instead_of_embedder(self, tmp_path):
        """When the npz supplies a vector for a file, the resulting media's
        embedding is exactly that vector (the embedder is bypassed)."""
        from tests.helpers import make_raw_wav_bytes

        src_a = tmp_path / "src_a.wav"
        src_b = tmp_path / "src_b.wav"
        src_a.write_bytes(make_raw_wav_bytes())
        src_b.write_bytes(make_raw_wav_bytes() + b"\x00\x00")

        # Distinctive vectors so we can verify they make it through
        # unmodified instead of being replaced by the embedder output.
        rng = np.random.default_rng(42)
        vec_a = rng.standard_normal(512).astype(np.float32) * 100.0
        vec_b = rng.standard_normal(512).astype(np.float32) * 100.0

        npz = tmp_path / "list.npz"
        np.savez(
            npz,
            filenames=np.array([str(src_a), str(src_b)]),
            vectors=np.stack([vec_a, vec_b]),
            embedder_name=np.array("clap"),
        )

        imp = ServerFilesDatasetImporter()
        medias: dict = {}
        imp.run({"paths_file": str(npz), "media_type": "audio"}, medias)

        assert len(medias) == 2
        # Index medias by their original source path so we can pair each
        # with its expected vector.
        by_source = {m["origin_name"]: m for m in medias.values()}
        np.testing.assert_array_equal(media_embedding(by_source[str(src_a)]), vec_a)
        np.testing.assert_array_equal(media_embedding(by_source[str(src_b)]), vec_b)
        # Origin still points at the npz so subsequent reloads work.
        for media in medias.values():
            assert media["origin"]["importer"] == "server_files"
            assert media["origin"]["params"]["paths_file"] == str(npz)

    def test_npz_embedder_name_stored_in_media_and_origin(self, tmp_path):
        """Embedder name from the NPZ is recorded on media['embedder'] and origin params."""
        from tests.helpers import make_raw_wav_bytes

        src = tmp_path / "clip.wav"
        src.write_bytes(make_raw_wav_bytes())

        rng = np.random.default_rng(0)
        vec = rng.standard_normal(512).astype(np.float32)
        npz = tmp_path / "list.npz"
        np.savez(
            npz,
            filenames=np.array([str(src)]),
            vectors=vec[np.newaxis],
            embedder_name=np.array("clap"),
        )

        imp = ServerFilesDatasetImporter()
        medias: dict = {}
        imp.run({"paths_file": str(npz), "media_type": "audio"}, medias)

        assert len(medias) == 1
        media = next(iter(medias.values()))
        assert media["embedder"] == "clap"
        assert media["origin"]["params"]["embedder_name"] == "clap"

    def test_unregistered_npz_embedder_name_raises_at_import(self, tmp_path):
        """An NPZ embedder name VTSearch can't route is rejected up front, with
        the valid options listed, rather than silently disabling text search."""
        from tests.helpers import make_raw_wav_bytes

        src = tmp_path / "clip.wav"
        src.write_bytes(make_raw_wav_bytes())

        rng = np.random.default_rng(0)
        vec = rng.standard_normal(512).astype(np.float32)
        npz = tmp_path / "list.npz"
        np.savez(
            npz,
            filenames=np.array([str(src)]),
            vectors=vec[np.newaxis],
            embedder_name=np.array("audemb_largerclapgeneral"),
        )

        imp = ServerFilesDatasetImporter()
        medias: dict = {}
        with pytest.raises(ValueError) as exc:
            imp.run({"paths_file": str(npz), "media_type": "audio"}, medias)
        msg = str(exc.value)
        assert "audemb_largerclapgeneral" in msg
        assert "audio" in msg
        assert "clap" in msg  # the registered options are listed
        assert medias == {}

    def test_npz_per_key_layout_is_accepted(self, tmp_path):
        from tests.helpers import make_raw_wav_bytes

        src = tmp_path / "only.wav"
        src.write_bytes(make_raw_wav_bytes())

        # Per-key layout: each filename is a top-level key in the npz.  A
        # per-key archive carries no embedder name, and under the dict-keyed
        # contract a vector can only be stored under an embedder name, so the
        # supplied vector is not retained; the test asserts the layout is
        # accepted (the media is created from the listed file).
        vec = np.full(256, 7.0, dtype=np.float32)
        npz = tmp_path / "list.npz"
        np.savez(npz, **{str(src): vec})  # pyright: ignore[reportArgumentType]

        imp = ServerFilesDatasetImporter()
        medias: dict = {}
        imp.run({"paths_file": str(npz), "media_type": "audio"}, medias)

        assert len(medias) == 1
        media = next(iter(medias.values()))
        assert media["origin_name"] == str(src)


# ---------------------------------------------------------------------------
# server_files end-to-end: per-embedder vectors_<name> archive (V3 trio, #2669)
# ---------------------------------------------------------------------------


class TestServerFilesMultiVectorsEndToEnd:
    def _write_wavs(self, tmp_path):
        from tests.helpers import make_raw_wav_bytes

        src_a = tmp_path / "src_a.wav"
        src_b = tmp_path / "src_b.wav"
        src_a.write_bytes(make_raw_wav_bytes())
        src_b.write_bytes(make_raw_wav_bytes() + b"\x00\x00")
        return src_a, src_b

    def test_read_npz_paths_file_returns_per_embedder_dicts(self, tmp_path):
        src_a, src_b = self._write_wavs(tmp_path)
        npz = tmp_path / "multi.npz"
        np.savez(
            npz,
            filenames=np.array([str(src_a), str(src_b)]),
            vectors_clap=_col("clap", 2),
            vectors_ast=_col("ast", 2, offset=50.0),
            embedder_name=np.array("clap"),
        )
        paths, path_to_vector, embedder_name = _read_npz_paths_file(npz)
        assert paths == [src_a, src_b]
        assert embedder_name == "clap"
        # Each resolved path maps to a per-embedder dict, not a bare vector.
        assert set(path_to_vector[str(src_a)]) == {"clap", "ast"}

    def test_both_embedders_stored_on_media(self, tmp_path):
        """A vectors_<name> archive lands one vector per embedder under
        media['embeddings'], with the scalar embedder_name as the primary."""
        src_a, src_b = self._write_wavs(tmp_path)
        rng = np.random.default_rng(3)
        clap = rng.standard_normal((2, expected_dim_for_embedder("clap") or 512)).astype(np.float32)
        ast = rng.standard_normal((2, expected_dim_for_embedder("ast") or 512)).astype(np.float32)
        npz = tmp_path / "multi.npz"
        np.savez(
            npz,
            filenames=np.array([str(src_a), str(src_b)]),
            vectors_clap=clap,
            vectors_ast=ast,
            embedder_name=np.array("clap"),
        )

        imp = ServerFilesDatasetImporter()
        medias: dict = {}
        imp.run({"paths_file": str(npz), "media_type": "audio"}, medias)

        assert len(medias) == 2
        by_source = {m["origin_name"]: m for m in medias.values()}
        for src, i in ((src_a, 0), (src_b, 1)):
            media = by_source[str(src)]
            # Both embedders' vectors are present and unmodified.
            np.testing.assert_array_equal(media_embedding(media, "clap"), clap[i])
            np.testing.assert_array_equal(media_embedding(media, "ast"), ast[i])
            # The scalar embedder_name is recorded as the primary/score embedder.
            assert media["embedder"] == "clap"
            np.testing.assert_array_equal(media_embedding(media), clap[i])

    def test_binding_derives_a_slot_per_capable_embedder(self, tmp_path):
        """The dataset binding recovered from a multi-vector media's embedder
        keys fills a distinct role slot per capable embedder (siglip→text,
        dinov3_patch→patch)."""
        from vtscore.embedding.binding import derive_binding_from_names
        from vtscore.embedding.media_vectors import init_embeddings, media_embedder_names

        media = {
            "embedder": "siglip",
            "embeddings": init_embeddings(
                "siglip",
                {"siglip": np.zeros(3, dtype=np.float32), "dinov3_patch": np.zeros(4, dtype=np.float32)},
            ),
        }
        names = media_embedder_names(media)
        assert set(names) == {"siglip", "dinov3_patch"}
        text, patch, structural = derive_binding_from_names(names)
        assert text == "siglip"
        assert patch == "dinov3_patch"
        assert structural is None

    def test_unregistered_column_name_rejected_at_import(self, tmp_path):
        """Every named column binds a role slot, so an unroutable name in *any*
        column is rejected up front (not just the primary)."""
        src_a, _src_b = self._write_wavs(tmp_path)
        npz = tmp_path / "multi.npz"
        np.savez(
            npz,
            filenames=np.array([str(src_a)]),
            vectors_clap=np.zeros((1, 512), dtype=np.float32),
            vectors_not_a_real_embedder=np.zeros((1, 512), dtype=np.float32),
            embedder_name=np.array("clap"),
        )
        imp = ServerFilesDatasetImporter()
        medias: dict = {}
        with pytest.raises(ValueError) as exc:
            imp.run({"paths_file": str(npz), "media_type": "audio"}, medias)
        assert "not_a_real_embedder" in str(exc.value)
        assert medias == {}


# ---------------------------------------------------------------------------
# /api/dataset/import-local-files: paths_file upload delegates to server_files
# ---------------------------------------------------------------------------


class TestImportLocalFilesEndpoint:
    """The /api/dataset/import-local-files endpoint accepts a single
    uploaded ``paths_file`` (txt list or npz archive) and runs the
    ``server_files`` importer over the saved copy on the server.  The
    background task runner is stubbed so the test can invoke the
    importer's ``load_fn`` synchronously and inspect the field_values
    that the importer would have been called with."""

    @staticmethod
    def _stub_run_and_observe(observed: dict):
        from vtscore.datasets.importers import get_importer

        def _fake_run(load_fn, origin, **kwargs):
            importer = get_importer("server_files")
            assert importer is not None
            observed["origin"] = origin
            observed["kwargs"] = kwargs

            def _sniff(field_values, medias=None, thin=False):
                observed["field_values"] = dict(field_values)
                # Snapshot the uploaded paths_file's bytes before the loader
                # tears down the temp dir.
                pf = Path(field_values["paths_file"])
                observed["paths_file_bytes"] = pf.read_bytes()
                return iter(())

            with (
                patch.object(importer, "run", side_effect=_sniff),
                patch.object(importer, "run_chunked", side_effect=_sniff),
            ):
                load_fn({})
            return "task-fake-local-files"

        return _fake_run

    def test_no_paths_file_returns_400(self, client):
        resp = client.post(
            "/api/dataset/import-local-files",
            data={"media_type": "audio"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "paths file" in resp.get_json()["message"].lower()

    def test_missing_media_type_returns_400(self, client):
        resp = client.post(
            "/api/dataset/import-local-files",
            data={"paths_file": (io.BytesIO(b"/a.wav\n"), "list.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "media_type" in resp.get_json()["message"]

    def test_uploads_txt_paths_file_and_starts_load(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR",
            tmp_path / "uploads",
        )

        observed: dict = {}
        with patch(
            "vtsearch.routes.datasets.load._run_origin_load_in_background",
            side_effect=self._stub_run_and_observe(observed),
        ):
            resp = client.post(
                "/api/dataset/import-local-files",
                data={
                    "media_type": "audio",
                    "paths_file": (io.BytesIO(b"/a.wav\n/b.wav\n"), "list.txt"),
                },
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["ok"] is True
        assert body["task_id"] == "task-fake-local-files"

        # The uploaded paths file was saved with its original suffix
        # (so the server_files importer dispatches on extension).
        assert observed["field_values"]["paths_file"].endswith(".txt")
        assert observed["paths_file_bytes"] == b"/a.wav\n/b.wav\n"
        # Origin is synthetic (the temp paths_file is about to be deleted)
        # so reload-from-origin is naturally disabled.
        assert observed["origin"]["importer"] == "server_files"
        assert observed["origin"]["params"]["paths_file"] == "<browser_upload>"
        assert observed["origin"]["params"]["media_type"] == "audio"

    def test_uploads_npz_paths_file_preserves_suffix(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR",
            tmp_path / "uploads",
        )

        # A real npz so the suffix-based reader dispatch is meaningful.
        npz_bytes_io = io.BytesIO()
        np.savez(
            npz_bytes_io,
            filenames=np.array(["/a.wav"]),
            vectors=np.zeros((1, 4), dtype=np.float32),
        )
        npz_bytes_io.seek(0)

        observed: dict = {}
        with patch(
            "vtsearch.routes.datasets.load._run_origin_load_in_background",
            side_effect=self._stub_run_and_observe(observed),
        ):
            resp = client.post(
                "/api/dataset/import-local-files",
                data={
                    "media_type": "audio",
                    "paths_file": (npz_bytes_io, "list.npz"),
                },
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert observed["field_values"]["paths_file"].endswith(".npz")
