"""Lazy file enumeration for the chunked folder loader.

The chunked folder loader must stream files (and start building chunks)
without first materialising the whole file list, so a directory tree with
more files than fit in RAM can be scanned.  See
``docs/plans/cli-stream-massive-images.md``.
"""

from __future__ import annotations

from pathlib import Path

from helpers import make_wav_bytes as _make_wav_bytes


def _make_wav_files(folder: Path, n: int) -> None:
    for i in range(n):
        (folder / f"clip_{i:03d}.wav").write_bytes(_make_wav_bytes(frequency=440.0 + i))


class TestGeneratorGlobs:
    def test_iter_rglob_matches_list_version(self, tmp_path):
        from vtscore.security.path_validation import iter_rglob_follow_symlinks, rglob_follow_symlinks

        (tmp_path / "sub").mkdir()
        _make_wav_files(tmp_path, 2)
        _make_wav_files(tmp_path / "sub", 2)

        streamed = {p.name for p in iter_rglob_follow_symlinks(tmp_path, "*.wav")}
        materialised = {p.name for p in rglob_follow_symlinks(tmp_path, "*.wav")}
        assert streamed == materialised
        assert len(streamed) == 4

    def test_iter_glob_top_level_matches_list_version(self, tmp_path):
        from vtscore.security.path_validation import glob_top_level, iter_glob_top_level

        (tmp_path / "sub").mkdir()
        _make_wav_files(tmp_path, 3)
        _make_wav_files(tmp_path / "sub", 2)  # must NOT be matched (no recursion)

        streamed = {p.name for p in iter_glob_top_level(tmp_path, "*.wav")}
        assert streamed == {p.name for p in glob_top_level(tmp_path, "*.wav")}
        assert len(streamed) == 3

    def test_iter_glob_top_level_missing_dir_is_empty(self, tmp_path):
        from vtscore.security.path_validation import iter_glob_top_level

        assert list(iter_glob_top_level(tmp_path / "nope", "*.wav")) == []


class TestChunkedLoaderIsLazy:
    def test_first_chunk_builds_only_chunk_size_medias(self, tmp_path, monkeypatch):
        """Pulling one chunk must build only ``chunk_size`` medias, not all.

        Proves the loader does not materialise/build the whole dataset up
        front when no precomputed-override maps are supplied.
        """
        import vtscore.datasets.loader_folder as lf
        from vtscore.datasets.loader import load_dataset_from_folder_chunked

        _make_wav_files(tmp_path, 20)

        calls = {"n": 0}
        real_build = lf._build_per_file_media

        def _counting_build(**kwargs):
            calls["n"] += 1
            return real_build(**kwargs)

        monkeypatch.setattr(lf, "_build_per_file_media", _counting_build)

        gen = load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=5, thin=True)
        first = next(gen)
        assert len(first) == 5
        # Only the first chunk's worth of files were read/built — the rest of
        # the 20-file tree has not been touched yet.
        assert calls["n"] == 5

    def test_empty_folder_still_raises(self, tmp_path):
        from vtscore.datasets.loader import load_dataset_from_folder_chunked

        try:
            list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=5, thin=True))
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "No audio files found" in str(e)
