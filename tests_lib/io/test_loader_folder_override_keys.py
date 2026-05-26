"""Override-map ambiguity checks in ``load_dataset_from_folder``.

The loader accepts ``content_vectors`` / ``content_md5s`` /
``custom_metadata_map`` keyed by either the file's relative path or its
basename (relative path wins).  Two situations used to silently
miscompute and are now surfaced:

1. A file has *both* its relative path *and* its basename as keys in the
   same override map with different values - the loader keeps the
   relative-path entry and logs a warning naming both keys.

2. A bare basename key matches multiple files in the folder and not
   every one of those files has its own relative-path entry - the
   loader raises ``ValueError`` rather than silently fanning the same
   value out to every match.

These tests verify both behaviours against the folder loader directly.
"""

from __future__ import annotations

import logging
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

from helpers import make_raw_wav_bytes as _make_wav_bytes


def _make_bulk_embedder(embed_return_dim: int = 3):
    emb = mock.MagicMock()
    emb.name = "fake_bulk"
    emb.media_type_id = "audio"
    emb._model = True
    emb._on_progress = lambda *a, **kw: None

    def _bulk(medias):
        return [np.full(embed_return_dim, float(i), dtype=np.float32) for i, _ in enumerate(medias)]

    emb.embed_media_bulk.side_effect = _bulk
    return emb


def _make_media_type_for_audio():
    mt = mock.MagicMock()
    mt.type_id = "audio"
    mt.file_extensions = ["*.wav"]
    mt.load_media_data.return_value = {"duration": 1.0}
    return mt


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_make_wav_bytes())


def _patches(mt, emb):
    return (
        mock.patch("vtscore.media.get_by_folder_name", return_value=mt),
        mock.patch("vtscore.media.embedders_for_type", return_value=[emb]),
    )


class TestRelPathVsBasenameConflictWarns:
    """When both rel_path and basename keys exist with different values,
    the loader keeps the rel_path entry and logs a warning."""

    def test_content_vectors_conflict_logs_warning(self, tmp_path, caplog):
        from vtscore.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "sub" / "foo.wav")

        rel_vec = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        basename_vec = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()
        medias: dict = {}
        with _patches(mt, emb)[0], _patches(mt, emb)[1]:
            with caplog.at_level(logging.WARNING, logger="vtscore.datasets.loader_folder"):
                load_dataset_from_folder(
                    tmp_path,
                    "audio",
                    medias,
                    content_vectors={"sub/foo.wav": rel_vec, "foo.wav": basename_vec},
                    on_progress=lambda *a: None,
                )

        assert len(medias) == 1
        only = next(iter(medias.values()))
        np.testing.assert_array_equal(only["embedding"], rel_vec)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "content_vectors" in r.getMessage() and "sub/foo.wav" in r.getMessage() and "foo.wav" in r.getMessage()
            for r in warnings
        )

    def test_content_vectors_matching_entries_do_not_warn(self, tmp_path, caplog):
        from vtscore.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "sub" / "foo.wav")

        same = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()
        medias: dict = {}
        with _patches(mt, emb)[0], _patches(mt, emb)[1]:
            with caplog.at_level(logging.WARNING, logger="vtscore.datasets.loader_folder"):
                load_dataset_from_folder(
                    tmp_path,
                    "audio",
                    medias,
                    content_vectors={"sub/foo.wav": same, "foo.wav": same.copy()},
                    on_progress=lambda *a: None,
                )

        assert len(medias) == 1
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_content_md5s_conflict_logs_warning(self, tmp_path, caplog):
        from vtscore.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "sub" / "foo.wav")

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()
        medias: dict = {}
        with _patches(mt, emb)[0], _patches(mt, emb)[1]:
            with caplog.at_level(logging.WARNING, logger="vtscore.datasets.loader_folder"):
                load_dataset_from_folder(
                    tmp_path,
                    "audio",
                    medias,
                    content_md5s={"sub/foo.wav": "a" * 32, "foo.wav": "b" * 32},
                    on_progress=lambda *a: None,
                )

        assert len(medias) == 1
        only = next(iter(medias.values()))
        assert only["md5"] == "a" * 32

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("content_md5s" in r.getMessage() for r in warnings)

    def test_custom_metadata_conflict_logs_warning(self, tmp_path, caplog):
        from vtscore.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "sub" / "foo.wav")

        rel_vec = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        basename_vec = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()
        medias: dict = {}
        with _patches(mt, emb)[0], _patches(mt, emb)[1]:
            with caplog.at_level(logging.WARNING, logger="vtscore.datasets.loader_folder"):
                load_dataset_from_folder(
                    tmp_path,
                    "audio",
                    medias,
                    custom_metadata_map={
                        "sub/foo.wav": {"embedding": rel_vec, "md5": "a" * 32},
                        "foo.wav": {"embedding": basename_vec, "md5": "b" * 32},
                    },
                    on_progress=lambda *a: None,
                )

        assert len(medias) == 1
        only = next(iter(medias.values()))
        np.testing.assert_array_equal(only["embedding"], rel_vec)
        assert only["md5"] == "a" * 32

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("custom_metadata_map" in m and "embeddings" in m for m in warnings)
        assert any("custom_metadata_map" in m and "md5s" in m for m in warnings)


class TestAmbiguousBasenameKeyRaises:
    """A bare basename that matches multiple files (and isn't disambiguated
    by per-file rel_path entries) is a data error, not a fallback."""

    def test_content_vectors_basename_spanning_multiple_files_raises(self, tmp_path):
        from vtscore.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "class_a" / "foo.wav")
        _write_wav(tmp_path / "class_b" / "foo.wav")

        vec = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()
        medias: dict = {}
        with _patches(mt, emb)[0], _patches(mt, emb)[1]:
            with pytest.raises(ValueError, match=r"content_vectors.*bare-basename.*foo\.wav"):
                load_dataset_from_folder(
                    tmp_path,
                    "audio",
                    medias,
                    content_vectors={"foo.wav": vec},
                    on_progress=lambda *a: None,
                )

    def test_content_md5s_basename_spanning_multiple_files_raises(self, tmp_path):
        from vtscore.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "class_a" / "foo.wav")
        _write_wav(tmp_path / "class_b" / "foo.wav")

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()
        medias: dict = {}
        with _patches(mt, emb)[0], _patches(mt, emb)[1]:
            with pytest.raises(ValueError, match=r"content_md5s.*bare-basename"):
                load_dataset_from_folder(
                    tmp_path,
                    "audio",
                    medias,
                    content_md5s={"foo.wav": "a" * 32},
                    on_progress=lambda *a: None,
                )

    def test_custom_metadata_basename_spanning_multiple_files_raises(self, tmp_path):
        from vtscore.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "class_a" / "foo.wav")
        _write_wav(tmp_path / "class_b" / "foo.wav")

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()
        medias: dict = {}
        with _patches(mt, emb)[0], _patches(mt, emb)[1]:
            with pytest.raises(ValueError, match=r"custom_metadata_map.*bare-basename"):
                load_dataset_from_folder(
                    tmp_path,
                    "audio",
                    medias,
                    custom_metadata_map={"foo.wav": {"md5": "a" * 32}},
                    on_progress=lambda *a: None,
                )

    def test_basename_collision_with_full_rel_path_coverage_does_not_raise(self, tmp_path):
        """If every same-basename file has its own rel_path entry, the
        bare-basename key is harmless and should be allowed."""
        from vtscore.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "class_a" / "foo.wav")
        _write_wav(tmp_path / "class_b" / "foo.wav")

        vec_a = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        vec_b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        vec_basename = np.array([99.0, 99.0, 99.0], dtype=np.float32)

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()
        medias: dict = {}
        with _patches(mt, emb)[0], _patches(mt, emb)[1]:
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                content_vectors={
                    "class_a/foo.wav": vec_a,
                    "class_b/foo.wav": vec_b,
                    "foo.wav": vec_basename,
                },
                on_progress=lambda *a: None,
            )

        by_filename = {m["filename"]: m["embedding"] for m in medias.values()}
        np.testing.assert_array_equal(by_filename["class_a/foo.wav"], vec_a)
        np.testing.assert_array_equal(by_filename["class_b/foo.wav"], vec_b)

    def test_unique_basename_in_recursive_scan_uses_basename_fallback(self, tmp_path):
        """A basename key that uniquely identifies one file in a nested
        layout is fine - that's the whole point of the fallback."""
        from vtscore.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "sub" / "only.wav")

        vec = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()
        medias: dict = {}
        with _patches(mt, emb)[0], _patches(mt, emb)[1]:
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                content_vectors={"only.wav": vec},
                on_progress=lambda *a: None,
            )

        assert len(medias) == 1
        np.testing.assert_array_equal(next(iter(medias.values()))["embedding"], vec)
