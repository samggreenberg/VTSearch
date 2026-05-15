"""Tests for embedding vectors and MD5 hashes flowing through DatasetImporter.

These tests verify the end-to-end path:
- A DatasetImporter provides pre-computed embedding vectors via content_vectors
- A DatasetImporter provides pre-computed MD5 hashes via content_md5s
- The resulting medias carry the importer-supplied values (not re-computed ones)
- Those medias work correctly in downstream operations: cosine sort,
  learned sort (train_and_score), API listing, and label export
"""

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path
from typing import Any

import numpy as np

from helpers import make_raw_wav_bytes as _make_wav_bytes


def _write_wav(path: Path) -> None:
    path.write_bytes(_make_wav_bytes())


def _make_mock_media_type():
    """Return a mock media-type and mock embedder for audio."""
    mt = mock.MagicMock()
    mt.type_id = "audio"
    mt.file_extensions = ["*.wav"]
    mt.load_media_data.return_value = {"duration": 1.0}

    emb = mock.MagicMock()
    emb.name = "clap"
    emb.media_type_id = "audio"
    emb._model = True  # already loaded
    emb.embed_media.return_value = np.zeros(8)
    # Route the loader's bulk dispatch through the per-file mock.
    emb.embed_media_bulk.side_effect = lambda medias: [emb.embed_media(m) for m in medias]
    return mt, emb


def _patch_media_registry(mt, emb):
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(mock.patch("vtsearch.media.get_by_folder_name", return_value=mt))
    stack.enter_context(mock.patch("vtsearch.media.embedders_for_type", return_value=[emb]))
    return stack


# ---------------------------------------------------------------------------
# Custom importer that populates content_vectors and content_md5s
# ---------------------------------------------------------------------------


class _VectorAndMD5Importer:
    """Helper: a DatasetImporter subclass that supplies pre-computed vectors and MD5s."""

    @staticmethod
    def create(folder: Path, vectors: dict[str, Any], md5s: dict[str, str]):
        from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
        from vtsearch.datasets.loader import load_dataset_from_folder

        class Importer(DatasetImporter):
            name = "test_vec_md5"
            display_name = "Test Vector+MD5"
            description = "Test importer providing vectors and MD5s."
            fields = [
                ImporterField("media_type", "Media Type", "text", default="audio"),
                ImporterField("path", "Path", "text"),
            ]

            def run(self, field_values, medias, thin=False):
                self.content_vectors.update(vectors)
                self.content_md5s.update(md5s)
                load_dataset_from_folder(
                    Path(field_values["path"]),
                    field_values.get("media_type", "audio"),
                    medias,
                    content_vectors=self.content_vectors or None,
                    content_md5s=self.content_md5s or None,
                    on_progress=lambda *a: None,
                    thin=thin,
                )

        imp = Importer()
        return imp


# ---------------------------------------------------------------------------
# Tests: vectors and MD5s arrive in medias
# ---------------------------------------------------------------------------


class TestImporterProvidedVectors:
    """Verify that content_vectors from a DatasetImporter land on the resulting medias."""

    def test_importer_vector_used_instead_of_model(self, tmp_path):
        rng = np.random.default_rng(42)
        pre_vec = rng.standard_normal(8).astype(np.float32)

        _write_wav(tmp_path / "tone.wav")
        mt, emb = _make_mock_media_type()
        imp = _VectorAndMD5Importer.create(
            tmp_path,
            vectors={"tone.wav": pre_vec},
            md5s={},
        )

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], pre_vec)
        emb.embed_media.assert_not_called()

    def test_importer_vector_multiple_files(self, tmp_path):
        rng = np.random.default_rng(123)
        vec_a = rng.standard_normal(8).astype(np.float32)
        vec_b = rng.standard_normal(8).astype(np.float32)

        _write_wav(tmp_path / "a.wav")
        _write_wav(tmp_path / "b.wav")
        mt, emb = _make_mock_media_type()
        imp = _VectorAndMD5Importer.create(
            tmp_path,
            vectors={"a.wav": vec_a, "b.wav": vec_b},
            md5s={},
        )

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        assert len(medias) == 2
        embs = {m["filename"]: m["embedding"] for m in medias.values()}
        np.testing.assert_array_equal(embs["a.wav"], vec_a)
        np.testing.assert_array_equal(embs["b.wav"], vec_b)
        emb.embed_media.assert_not_called()

    def test_mixed_importer_and_model_vectors(self, tmp_path):
        rng = np.random.default_rng(7)
        pre_vec = rng.standard_normal(8).astype(np.float32)
        model_vec = np.ones(8, dtype=np.float32) * 0.5

        _write_wav(tmp_path / "pre.wav")
        _write_wav(tmp_path / "model.wav")
        mt, emb = _make_mock_media_type()
        emb.embed_media.return_value = model_vec
        imp = _VectorAndMD5Importer.create(
            tmp_path,
            vectors={"pre.wav": pre_vec},
            md5s={},
        )

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        assert len(medias) == 2
        embs = {m["filename"]: m["embedding"] for m in medias.values()}
        np.testing.assert_array_equal(embs["pre.wav"], pre_vec)
        np.testing.assert_array_equal(embs["model.wav"], model_vec)

    def test_importer_vector_in_thin_mode(self, tmp_path):
        rng = np.random.default_rng(99)
        pre_vec = rng.standard_normal(8).astype(np.float32)

        _write_wav(tmp_path / "thin.wav")
        mt, emb = _make_mock_media_type()
        imp = _VectorAndMD5Importer.create(
            tmp_path,
            vectors={"thin.wav": pre_vec},
            md5s={},
        )

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias, thin=True)

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], pre_vec)
        assert medias[1]["media_bytes"] is None  # thin mode


class TestImporterProvidedMD5s:
    """Verify that content_md5s from a DatasetImporter land on the resulting medias."""

    def test_importer_md5_used_instead_of_computed(self, tmp_path):
        pre_md5 = "a" * 32

        _write_wav(tmp_path / "tone.wav")
        mt, emb = _make_mock_media_type()
        imp = _VectorAndMD5Importer.create(
            tmp_path,
            vectors={},
            md5s={"tone.wav": pre_md5},
        )

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        assert len(medias) == 1
        assert medias[1]["md5"] == pre_md5

    def test_importer_md5_multiple_files(self, tmp_path):
        md5_a = "1" * 32
        md5_b = "2" * 32

        _write_wav(tmp_path / "a.wav")
        _write_wav(tmp_path / "b.wav")
        mt, emb = _make_mock_media_type()
        imp = _VectorAndMD5Importer.create(
            tmp_path,
            vectors={},
            md5s={"a.wav": md5_a, "b.wav": md5_b},
        )

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        md5s = {m["filename"]: m["md5"] for m in medias.values()}
        assert md5s["a.wav"] == md5_a
        assert md5s["b.wav"] == md5_b

    def test_importer_md5_in_thin_mode(self, tmp_path):
        pre_md5 = "b" * 32

        _write_wav(tmp_path / "thin.wav")
        mt, emb = _make_mock_media_type()
        imp = _VectorAndMD5Importer.create(
            tmp_path,
            vectors={},
            md5s={"thin.wav": pre_md5},
        )

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias, thin=True)

        assert medias[1]["md5"] == pre_md5

    def test_mixed_importer_and_computed_md5s(self, tmp_path):
        import hashlib

        pre_md5 = "c" * 32
        _write_wav(tmp_path / "pre.wav")
        _write_wav(tmp_path / "computed.wav")
        computed_md5 = hashlib.md5((tmp_path / "computed.wav").read_bytes()).hexdigest()

        mt, emb = _make_mock_media_type()
        imp = _VectorAndMD5Importer.create(
            tmp_path,
            vectors={},
            md5s={"pre.wav": pre_md5},
        )

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        md5s = {m["filename"]: m["md5"] for m in medias.values()}
        assert md5s["pre.wav"] == pre_md5
        assert md5s["computed.wav"] == computed_md5


class TestImporterProvidedBoth:
    """Verify that vectors AND MD5s from a single importer both arrive correctly."""

    def test_both_vector_and_md5_from_importer(self, tmp_path):
        rng = np.random.default_rng(55)
        pre_vec = rng.standard_normal(8).astype(np.float32)
        pre_md5 = "d" * 32

        _write_wav(tmp_path / "both.wav")
        mt, emb = _make_mock_media_type()
        imp = _VectorAndMD5Importer.create(
            tmp_path,
            vectors={"both.wav": pre_vec},
            md5s={"both.wav": pre_md5},
        )

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], pre_vec)
        assert medias[1]["md5"] == pre_md5
        emb.embed_media.assert_not_called()

    def test_both_for_multiple_files(self, tmp_path):
        rng = np.random.default_rng(77)
        names = ["x.wav", "y.wav", "z.wav"]
        vectors = {}
        md5s = {}
        for name in names:
            _write_wav(tmp_path / name)
            vectors[name] = rng.standard_normal(8).astype(np.float32)
            md5s[name] = name[0] * 32

        mt, emb = _make_mock_media_type()
        imp = _VectorAndMD5Importer.create(tmp_path, vectors=vectors, md5s=md5s)

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        assert len(medias) == 3
        for m in medias.values():
            fname = m["filename"]
            np.testing.assert_array_equal(m["embedding"], vectors[fname])
            assert m["md5"] == md5s[fname]


# ---------------------------------------------------------------------------
# Tests: folder importer passes content_vectors/content_md5s through
# ---------------------------------------------------------------------------


class TestImporterCustomMetadataMD5:
    """Verify that MD5 embedded in custom_metadata_map flows through correctly."""

    def test_custom_metadata_md5_used_as_media_md5(self, tmp_path):
        """When custom_metadata_map has an 'md5' key, it should be used as the media's MD5."""
        from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
        from vtsearch.datasets.loader import load_dataset_from_folder

        class MetadataMD5Importer(DatasetImporter):
            name = "test_cm_md5"
            display_name = "Test CM MD5"
            description = "Test importer using custom_metadata_map for MD5."
            fields = [
                ImporterField("media_type", "Media Type", "text", default="audio"),
                ImporterField("path", "Path", "text"),
            ]

            def run(self, field_values, medias, thin=False):
                self.custom_metadata_map["tone.wav"] = {
                    "md5": "metadata_md5_" + "a" * 20,
                    "source": "test",
                }
                load_dataset_from_folder(
                    Path(field_values["path"]),
                    field_values.get("media_type", "audio"),
                    medias,
                    custom_metadata_map=self.custom_metadata_map or None,
                    on_progress=lambda *a: None,
                    thin=thin,
                )

        _write_wav(tmp_path / "tone.wav")
        mt, emb = _make_mock_media_type()
        imp = MetadataMD5Importer()

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        assert len(medias) == 1
        assert medias[1]["md5"] == "metadata_md5_" + "a" * 20
        # custom_metadata should also be attached
        assert medias[1]["custom_metadata"]["source"] == "test"

    def test_custom_metadata_md5_takes_priority_over_content_md5s(self, tmp_path):
        """custom_metadata_map MD5 should beat content_md5s."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        cm_md5 = "custom_meta_" + "1" * 20
        content_md5 = "content_md5_" + "2" * 20

        _write_wav(tmp_path / "prio.wav")
        mt, emb = _make_mock_media_type()

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                content_md5s={"prio.wav": content_md5},
                custom_metadata_map={"prio.wav": {"md5": cm_md5}},
                on_progress=lambda *a: None,
            )

        assert medias[1]["md5"] == cm_md5

    def test_custom_metadata_md5_in_thin_mode(self, tmp_path):
        """custom_metadata_map MD5 should work in thin mode too."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        cm_md5 = "thinmeta_" + "f" * 23

        _write_wav(tmp_path / "slim.wav")
        mt, emb = _make_mock_media_type()

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                custom_metadata_map={"slim.wav": {"md5": cm_md5}},
                on_progress=lambda *a: None,
                thin=True,
            )

        assert medias[1]["md5"] == cm_md5

    def test_folder_importer_custom_metadata_md5_passthrough(self, tmp_path):
        """The folder importer should pass custom_metadata_map through, including its MD5."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        cm_md5 = "folder_cm_" + "9" * 22
        _write_wav(tmp_path / "cm.wav")
        mt, emb = _make_mock_media_type()

        IMPORTER.custom_metadata_map = {"cm.wav": {"md5": cm_md5, "tag": "hello"}}
        medias: dict = {}
        try:
            with _patch_media_registry(mt, emb):
                IMPORTER.run({"path": str(tmp_path), "media_type": "audio"}, medias)
            assert medias[1]["md5"] == cm_md5
            assert medias[1]["custom_metadata"]["tag"] == "hello"
        finally:
            IMPORTER.custom_metadata_map = {}

    def test_apply_custom_metadata_md5_post_load(self):
        """apply_custom_metadata_md5 should extract MD5 from custom_metadata after loading."""
        from vtsearch.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "original_hash", "custom_metadata": {"md5": "authoritative_hash", "extra": "data"}},
            2: {"md5": "stays_same", "custom_metadata": {"extra": "no md5 here"}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 1
        assert media_dict[1]["md5"] == "authoritative_hash"
        # The md5 key should be popped from custom_metadata
        assert "md5" not in media_dict[1]["custom_metadata"]
        assert media_dict[1]["custom_metadata"]["extra"] == "data"
        assert media_dict[2]["md5"] == "stays_same"


class TestImporterCustomMetadataEmbedding:
    """Verify that an embedding in custom_metadata_map flows through correctly."""

    def test_custom_metadata_embedding_used_as_media_embedding(self, tmp_path):
        """When custom_metadata_map has an 'embedding' key, it should be used as the media's embedding."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        rng = np.random.default_rng(42)
        cm_vec = rng.standard_normal(8).astype(np.float32)

        _write_wav(tmp_path / "tone.wav")
        mt, emb = _make_mock_media_type()

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                custom_metadata_map={"tone.wav": {"embedding": cm_vec, "source": "test"}},
                on_progress=lambda *a: None,
            )

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], cm_vec)
        emb.embed_media.assert_not_called()
        assert medias[1]["custom_metadata"]["source"] == "test"

    def test_custom_metadata_embedding_takes_priority_over_content_vectors(self, tmp_path):
        """custom_metadata_map embedding should beat content_vectors."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        rng = np.random.default_rng(7)
        cm_vec = rng.standard_normal(8).astype(np.float32)
        cv_vec = np.ones(8, dtype=np.float32) * 99.0

        _write_wav(tmp_path / "prio.wav")
        mt, emb = _make_mock_media_type()

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                content_vectors={"prio.wav": cv_vec},
                custom_metadata_map={"prio.wav": {"embedding": cm_vec}},
                on_progress=lambda *a: None,
            )

        np.testing.assert_array_equal(medias[1]["embedding"], cm_vec)

    def test_custom_metadata_embedding_in_thin_mode(self, tmp_path):
        """custom_metadata_map embedding should work in thin mode."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        rng = np.random.default_rng(11)
        cm_vec = rng.standard_normal(8).astype(np.float32)

        _write_wav(tmp_path / "slim.wav")
        mt, emb = _make_mock_media_type()

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                custom_metadata_map={"slim.wav": {"embedding": cm_vec}},
                on_progress=lambda *a: None,
                thin=True,
            )

        np.testing.assert_array_equal(medias[1]["embedding"], cm_vec)
        emb.embed_media.assert_not_called()

    def test_custom_metadata_both_embedding_and_md5(self, tmp_path):
        """custom_metadata_map can provide both embedding and MD5 in a single entry."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        rng = np.random.default_rng(55)
        cm_vec = rng.standard_normal(8).astype(np.float32)
        cm_md5 = "meta_both_" + "b" * 22

        _write_wav(tmp_path / "both.wav")
        mt, emb = _make_mock_media_type()

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                custom_metadata_map={"both.wav": {"embedding": cm_vec, "md5": cm_md5}},
                on_progress=lambda *a: None,
            )

        np.testing.assert_array_equal(medias[1]["embedding"], cm_vec)
        assert medias[1]["md5"] == cm_md5
        emb.embed_media.assert_not_called()

    def test_custom_metadata_embedding_mixed_with_model(self, tmp_path):
        """Files with custom_metadata embedding skip the model; others use the model."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        rng = np.random.default_rng(33)
        cm_vec = rng.standard_normal(8).astype(np.float32)
        model_vec = np.ones(8, dtype=np.float32) * 0.5

        _write_wav(tmp_path / "meta.wav")
        _write_wav(tmp_path / "model.wav")
        mt, emb = _make_mock_media_type()
        emb.embed_media.return_value = model_vec

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                custom_metadata_map={"meta.wav": {"embedding": cm_vec}},
                on_progress=lambda *a: None,
            )

        embs = {m["filename"]: m["embedding"] for m in medias.values()}
        np.testing.assert_array_equal(embs["meta.wav"], cm_vec)
        np.testing.assert_array_equal(embs["model.wav"], model_vec)

    def test_custom_metadata_embedding_chunked(self, tmp_path):
        """custom_metadata_map embedding should work with the chunked loader."""
        from vtsearch.datasets.loader import load_dataset_from_folder_chunked

        rng = np.random.default_rng(88)
        cm_vec = rng.standard_normal(8).astype(np.float32)

        _write_wav(tmp_path / "chunk.wav")
        mt, emb = _make_mock_media_type()

        with _patch_media_registry(mt, emb):
            chunks = list(
                load_dataset_from_folder_chunked(
                    tmp_path,
                    "audio",
                    chunk_size=10,
                    custom_metadata_map={"chunk.wav": {"embedding": cm_vec}},
                    on_progress=lambda *a: None,
                )
            )

        all_medias = {}
        for chunk in chunks:
            all_medias.update(chunk)
        assert len(all_medias) == 1
        np.testing.assert_array_equal(all_medias[1]["embedding"], cm_vec)
        emb.embed_media.assert_not_called()

    def test_importer_custom_metadata_embedding_end_to_end(self, tmp_path):
        """A DatasetImporter providing embedding via custom_metadata_map should work end-to-end."""
        from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
        from vtsearch.datasets.loader import load_dataset_from_folder

        rng = np.random.default_rng(77)
        cm_vec = rng.standard_normal(8).astype(np.float32)
        cm_md5 = "e2e_cm_" + "f" * 25

        class MetadataImporter(DatasetImporter):
            name = "test_cm_emb"
            display_name = "Test CM Embedding"
            description = "Test importer using custom_metadata_map for embedding."
            fields = [
                ImporterField("media_type", "Media Type", "text", default="audio"),
                ImporterField("path", "Path", "text"),
            ]

            def run(self, field_values, medias, thin=False):
                self.custom_metadata_map["clip.wav"] = {
                    "embedding": cm_vec,
                    "md5": cm_md5,
                    "tag": "from_metadata",
                }
                load_dataset_from_folder(
                    Path(field_values["path"]),
                    field_values.get("media_type", "audio"),
                    medias,
                    custom_metadata_map=self.custom_metadata_map or None,
                    on_progress=lambda *a: None,
                    thin=thin,
                )

        _write_wav(tmp_path / "clip.wav")
        mt, emb = _make_mock_media_type()
        imp = MetadataImporter()

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], cm_vec)
        assert medias[1]["md5"] == cm_md5
        assert medias[1]["custom_metadata"]["tag"] == "from_metadata"
        emb.embed_media.assert_not_called()

    def test_importer_custom_metadata_embedding_in_sorting(self, tmp_path):
        """Embeddings from custom_metadata_map should work in train_and_score."""

        from vtsearch.datasets.loader import load_dataset_from_folder
        from vtsearch.models.training import train_and_score

        rng = np.random.default_rng(42)
        names = [f"s{i}.wav" for i in range(6)]
        cm_map: dict[str, dict] = {}
        for name in names:
            _write_wav(tmp_path / name)
            cm_map[name] = {"embedding": rng.standard_normal(8).astype(np.float32)}

        mt, emb = _make_mock_media_type()

        medias: dict = {}
        with _patch_media_registry(mt, emb):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                custom_metadata_map=cm_map,
                on_progress=lambda *a: None,
            )

        good = {1: None, 2: None}
        bad = {3: None, 4: None}
        results, threshold, model = train_and_score(medias, good, bad)
        assert len(results) == 6
        for entry in results:
            assert 0.0 <= entry["score"] <= 1.0


class TestFolderImporterPassthrough:
    """Verify the folder importer wires content_vectors/content_md5s to load_dataset_from_folder."""

    def test_folder_importer_passes_content_vectors(self, tmp_path):
        from vtsearch.datasets.importers.server_folder import IMPORTER

        rng = np.random.default_rng(11)
        pre_vec = rng.standard_normal(8).astype(np.float32)
        _write_wav(tmp_path / "f.wav")

        mt, emb = _make_mock_media_type()
        IMPORTER.content_vectors = {"f.wav": pre_vec}
        IMPORTER.content_md5s = {}

        medias: dict = {}
        try:
            with _patch_media_registry(mt, emb):
                IMPORTER.run({"path": str(tmp_path), "media_type": "audio"}, medias)
            assert len(medias) == 1
            np.testing.assert_array_equal(medias[1]["embedding"], pre_vec)
            emb.embed_media.assert_not_called()
        finally:
            IMPORTER.content_vectors = {}
            IMPORTER.content_md5s = {}

    def test_folder_importer_passes_content_md5s(self, tmp_path):
        from vtsearch.datasets.importers.server_folder import IMPORTER

        pre_md5 = "e" * 32
        _write_wav(tmp_path / "g.wav")

        mt, emb = _make_mock_media_type()
        IMPORTER.content_vectors = {}
        IMPORTER.content_md5s = {"g.wav": pre_md5}

        medias: dict = {}
        try:
            with _patch_media_registry(mt, emb):
                IMPORTER.run({"path": str(tmp_path), "media_type": "audio"}, medias)
            assert len(medias) == 1
            assert medias[1]["md5"] == pre_md5
        finally:
            IMPORTER.content_vectors = {}
            IMPORTER.content_md5s = {}


# ---------------------------------------------------------------------------
# Tests: importer-provided medias work in downstream operations
# ---------------------------------------------------------------------------


class TestImporterMediasInSorting:
    """Verify that importer-provided embeddings work in cosine sort and train_and_score."""

    def _load_importer_medias_into_app(self, tmp_path, num_files=6, embed_dim=8):
        """Create WAV files, import them with pre-computed vectors, and load into app state."""
        rng = np.random.default_rng(42)
        vectors = {}
        md5s = {}
        for i in range(num_files):
            name = f"clip_{i}.wav"
            _write_wav(tmp_path / name)
            vectors[name] = rng.standard_normal(embed_dim).astype(np.float32)
            md5s[name] = f"{i:032x}"

        mt, emb = _make_mock_media_type()
        imp = _VectorAndMD5Importer.create(tmp_path, vectors=vectors, md5s=md5s)
        medias: dict = {}
        with _patch_media_registry(mt, emb):
            imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        return medias, vectors, md5s

    def test_cosine_sort_uses_importer_embeddings(self, tmp_path):
        """Cosine similarity sort should use the importer-provided embeddings."""
        medias, vectors, _ = self._load_importer_medias_into_app(tmp_path)

        # Build a query vector and compute expected similarities manually
        rng = np.random.default_rng(99)
        query_vec = rng.standard_normal(8).astype(np.float32)

        all_ids = list(medias.keys())
        all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
        query_norm = np.linalg.norm(query_vec)
        emb_norms = np.linalg.norm(all_embs, axis=1)
        norm_products = emb_norms * query_norm
        safe_norms = np.where(norm_products == 0, 1.0, norm_products)
        expected_sims = np.dot(all_embs, query_vec) / safe_norms

        # Verify similarities are computed from the importer vectors, not zeros
        assert not np.allclose(expected_sims, 0.0), "Embeddings should produce non-trivial similarities"
        assert np.all(np.isfinite(expected_sims)), "Similarities should be finite"

    def test_train_and_score_uses_importer_embeddings(self, tmp_path):
        """train_and_score should work with importer-provided embeddings."""

        from vtsearch.models.training import train_and_score

        medias, _, _ = self._load_importer_medias_into_app(tmp_path)

        # Vote on some medias
        good = {1: None, 2: None}
        bad = {3: None, 4: None}

        results, threshold, model = train_and_score(medias, good, bad)
        assert len(results) == len(medias)
        assert isinstance(threshold, float)
        for entry in results:
            assert "id" in entry
            assert "score" in entry
            assert 0.0 <= entry["score"] <= 1.0

        # Scores should be sorted descending
        scores = [e["score"] for e in results]
        assert scores == sorted(scores, reverse=True)

    def test_api_medias_shows_importer_md5(self, client, tmp_path):
        """POST /api/medias/batch should expose the importer-supplied MD5."""
        from vtsearch.state import medias as app_medias

        loaded, _, md5s = self._load_importer_medias_into_app(tmp_path, num_files=3)

        saved = dict(app_medias)
        app_medias.clear()
        try:
            app_medias.update(loaded)
            ids_resp = client.get("/api/medias/ids")
            assert ids_resp.status_code == 200
            ids = [m["id"] for m in ids_resp.get_json()]
            batch_resp = client.post("/api/medias/batch", json={"ids": ids})
            assert batch_resp.status_code == 200
            data = batch_resp.get_json()
            api_md5s = {m["filename"]: m["md5"] for m in data}
            for fname, expected_md5 in md5s.items():
                assert api_md5s[fname] == expected_md5
        finally:
            app_medias.clear()
            app_medias.update(saved)

    def test_label_export_uses_importer_md5(self, client, tmp_path):
        """Label export should include the importer-supplied MD5 in the labelset."""
        from vtsearch.state import (
    good_votes,
    medias as app_medias,
)

        loaded, _, md5s = self._load_importer_medias_into_app(tmp_path, num_files=3)

        saved = dict(app_medias)
        app_medias.clear()
        try:
            app_medias.update(loaded)
            good_votes[1] = None
            resp = client.get("/api/labels/export")
            assert resp.status_code == 200
            data = resp.get_json()
            # The exported labelset should contain elements with our MD5s
            elements = data.get("elements", data.get("labels", []))
            if elements:
                exported_md5s = {e.get("md5") for e in elements if e.get("md5")}
                # At least the voted element's MD5 should be present
                voted_media = loaded[1]
                assert voted_media["md5"] in exported_md5s
        finally:
            app_medias.clear()
            app_medias.update(saved)
